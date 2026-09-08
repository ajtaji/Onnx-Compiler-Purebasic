; ============================================================================
; onnx_targets.pbi - native compiler target registry
; ----------------------------------------------------------------------------
; Graph semantics never branch on these records. A target owns only the
; source dialect, launch adapter and optional dense-kernel selection.
; ============================================================================

Enumeration PmoTarget
  #PMO_TARGET_WINDOWS
  #PMO_TARGET_PI4
  #PMO_TARGET_UNOQ
  #PMO_TARGET_PICO
  #PMO_TARGET_PICO2
  #PMO_TARGET_COUNT
EndEnumeration

Enumeration PmoSourceDialect
  #PMO_SOURCE_HOST
  #PMO_SOURCE_PUREMETAL
EndEnumeration

Enumeration PmoLaunchKind
  #PMO_LAUNCH_WINDOWS_FILES
  #PMO_LAUNCH_EXTERNAL_MEMORY
  #PMO_LAUNCH_UEFI_ESP
  #PMO_LAUNCH_EMBEDDED_WEIGHTS
EndEnumeration

Structure PmoTargetProfile
  Id.s
  Label.s
  Description.s
  SourceDialect.i
  Launch.i
  SourceSuffix.s
  MathInclude.s
  TensorInclude.s
  AcceleratorInclude.s
  MatMulProcedure.s
  GemmProcedure.s
  NativeIntegerBytes.i
  RamBytes.q
  RamReserveBytes.q
  FlashBytes.q
  FlashReserveBytes.q
EndStructure

Global Dim PmoTargets.PmoTargetProfile(#PMO_TARGET_COUNT - 1)
Global PmoTargetsReady.i

Procedure PmoInitTargets()
  If PmoTargetsReady : ProcedureReturn : EndIf

  With PmoTargets(#PMO_TARGET_WINDOWS)
    \Id = "windows"
    \Label = "Windows (native x64)"
    \Description = "Host-language source for a native Windows x64 application"
    \SourceDialect = #PMO_SOURCE_HOST
    \Launch = #PMO_LAUNCH_WINDOWS_FILES
    \SourceSuffix = ".pb"
    \TensorInclude = "runtime/tensor_fp32.pmi"
    \MatMulProcedure = "PmTensorMatMul2"
    \GemmProcedure = "PmTensorGemm"
    \NativeIntegerBytes = 8
  EndWith

  With PmoTargets(#PMO_TARGET_PI4)
    \Id = "pi4"
    \Label = "Raspberry Pi 4 (PureMetal AArch64)"
    \Description = "PureMetal source for Raspberry Pi 4 Cortex-A72"
    \SourceDialect = #PMO_SOURCE_PUREMETAL
    \Launch = #PMO_LAUNCH_EXTERNAL_MEMORY
    \SourceSuffix = ".pi4"
    \MathInclude = "runtime/math/math_a64.pmi"
    \TensorInclude = "runtime/tensor_fp32.pmi"
    \AcceleratorInclude = "runtime/tensor_fp32_neon.pmi"
    \MatMulProcedure = "PmTensorMatMul2Neon"
    \GemmProcedure = "PmTensorGemmNeon"
    \NativeIntegerBytes = 8
  EndWith

  With PmoTargets(#PMO_TARGET_UNOQ)
    \Id = "unoq"
    \Label = "Arduino UNO Q (PureMetal AArch64 UEFI)"
    \Description = "PureMetal source for Arduino UNO Q Cortex-A53 UEFI"
    \SourceDialect = #PMO_SOURCE_PUREMETAL
    \Launch = #PMO_LAUNCH_UEFI_ESP
    \SourceSuffix = ".pi4"
    \MathInclude = "runtime/math/math_a64.pmi"
    \TensorInclude = "runtime/tensor_fp32.pmi"
    \AcceleratorInclude = "runtime/tensor_fp32_neon.pmi"
    \MatMulProcedure = "PmTensorMatMul2Neon"
    \GemmProcedure = "PmTensorGemmNeon"
    \NativeIntegerBytes = 8
  EndWith

  With PmoTargets(#PMO_TARGET_PICO)
    \Id = "pico"
    \Label = "Raspberry Pi Pico (RP2040 Cortex-M0+)"
    \Description = "PureMetal source for Raspberry Pi Pico RP2040 with embedded checked weights"
    \SourceDialect = #PMO_SOURCE_PUREMETAL
    \Launch = #PMO_LAUNCH_EMBEDDED_WEIGHTS
    \SourceSuffix = ".pico"
    \MathInclude = "runtime/math/math_m0.pmi"
    \TensorInclude = "runtime/tensor_fp32.pmi"
    \MatMulProcedure = "PmTensorMatMul2"
    \GemmProcedure = "PmTensorGemm"
    \NativeIntegerBytes = 4
    \RamBytes = 264 * 1024
    \RamReserveBytes = 24 * 1024
    \FlashBytes = 2 * 1024 * 1024
    \FlashReserveBytes = 96 * 1024
  EndWith

  With PmoTargets(#PMO_TARGET_PICO2)
    \Id = "pico2"
    \Label = "Raspberry Pi Pico 2 (RP2350 Cortex-M33)"
    \Description = "PureMetal source for Raspberry Pi Pico 2 RP2350 Arm with embedded checked weights"
    \SourceDialect = #PMO_SOURCE_PUREMETAL
    \Launch = #PMO_LAUNCH_EMBEDDED_WEIGHTS
    \SourceSuffix = ".pico2"
    \MathInclude = "runtime/math/math_m33.pmi"
    \TensorInclude = "runtime/tensor_fp32.pmi"
    \MatMulProcedure = "PmTensorMatMul2"
    \GemmProcedure = "PmTensorGemm"
    \NativeIntegerBytes = 4
    \RamBytes = 520 * 1024
    \RamReserveBytes = 40 * 1024
    \FlashBytes = 4 * 1024 * 1024
    \FlashReserveBytes = 128 * 1024
  EndWith

  PmoTargetsReady = #True
EndProcedure

Procedure.i PmoTargetIndex(Id.s)
  Protected Index.i
  PmoInitTargets()
  For Index = 0 To #PMO_TARGET_COUNT - 1
    If LCase(PmoTargets(Index)\Id) = LCase(Id)
      ProcedureReturn Index
    EndIf
  Next
  ProcedureReturn -1
EndProcedure

Procedure.i PmoTargetProfileValid(*Profile.PmoTargetProfile)
  If *Profile = 0 Or *Profile\Id = "" Or *Profile\SourceSuffix = "" Or
     Left(*Profile\SourceSuffix, 1) <> "."
    ProcedureReturn #False
  EndIf
  If *Profile\SourceDialect = #PMO_SOURCE_HOST
    If *Profile\Launch <> #PMO_LAUNCH_WINDOWS_FILES
      ProcedureReturn #False
    EndIf
  ElseIf *Profile\SourceDialect = #PMO_SOURCE_PUREMETAL
    If *Profile\TensorInclude = ""
      ProcedureReturn #False
    EndIf
  Else
    ProcedureReturn #False
  EndIf
  If *Profile\Launch = #PMO_LAUNCH_EMBEDDED_WEIGHTS And
     (*Profile\RamBytes <= 0 Or *Profile\RamReserveBytes < 0 Or
      *Profile\RamReserveBytes >= *Profile\RamBytes Or *Profile\FlashBytes <= 0)
    ProcedureReturn #False
  EndIf
  If *Profile\Launch = #PMO_LAUNCH_EMBEDDED_WEIGHTS And
     (*Profile\FlashReserveBytes <= 0 Or *Profile\FlashReserveBytes >= *Profile\FlashBytes)
    ProcedureReturn #False
  EndIf
  If *Profile\AcceleratorInclude <> "" And
     (*Profile\MatMulProcedure = "" Or *Profile\GemmProcedure = "")
    ProcedureReturn #False
  EndIf
  If *Profile\NativeIntegerBytes <> 4 And *Profile\NativeIntegerBytes <> 8
    ProcedureReturn #False
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoAllTargetsValid()
  Protected Index.i
  PmoInitTargets()
  For Index = 0 To #PMO_TARGET_COUNT - 1
    If PmoTargetProfileValid(@PmoTargets(Index)) = 0
      ProcedureReturn #False
    EndIf
  Next
  ProcedureReturn #True
EndProcedure
