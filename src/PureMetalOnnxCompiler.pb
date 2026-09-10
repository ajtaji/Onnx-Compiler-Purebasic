; ============================================================================
; PureMetalOnnxCompiler.pb - native ONNX ahead-of-time compiler and UI
; ----------------------------------------------------------------------------
; This program is a native executable. Python is not launched or required.
; The first checked vertical slice is the bounded ONNX protobuf/model reader;
; lowering, deterministic packing, source emission and the desktop UI live in
; this same native tool as their modules are completed.
; ============================================================================

EnableExplicit
XIncludeFile "compiler/onnx_model.pbi"
XIncludeFile "compiler/onnx_targets.pbi"
XIncludeFile "compiler/onnx_runtime.pbi"
XIncludeFile "compiler/npy_reader.pbi"
XIncludeFile "compiler/onnx_rank_infer.pbi"
XIncludeFile "compiler/onnx_proto_writer.pbi"
XIncludeFile "compiler/onnx_trace.pbi"
XIncludeFile "compiler/onnx_ir.pbi"
XIncludeFile "compiler/onnx_quant.pbi"
XIncludeFile "compiler/onnx_storage.pbi"
XIncludeFile "compiler/onnx_pack.pbi"
XIncludeFile "compiler/onnx_emit.pbi"
XIncludeFile "compiler/onnx_compile.pbi"
XIncludeFile "compiler/onnx_dynamic_emit.pbi"
XIncludeFile "compiler/onnx_kokoro.pbi"
XIncludeFile "compiler/kokoro_asset_pack.pbi"
XIncludeFile "compiler/onnx_ui.pbi"

Procedure PrintUsage()
  PrintN("PureMetal ONNX Compiler")
  PrintN("")
  PrintN("Usage:")
  PrintN("  PureMetalOnnxCompiler.exe --inspect model.onnx")
  PrintN("  PureMetalOnnxCompiler.exe --self-test model.onnx")
  PrintN("  PureMetalOnnxCompiler.exe --ort-version [onnxruntime library]")
  PrintN("  PureMetalOnnxCompiler.exe --list-targets")
  PrintN("  PureMetalOnnxCompiler.exe --npy-inspect tensor.npy")
  PrintN("  PureMetalOnnxCompiler.exe --trace-model model.onnx output.onnx")
  PrintN("  PureMetalOnnxCompiler.exe --trace-shapes model.onnx --ort-dll path --trace-input NAME=file.npy ...")
  PrintN("  PureMetalOnnxCompiler.exe --lower model.onnx --ort-dll path --trace-input NAME=file.npy ...")
  PrintN("  PureMetalOnnxCompiler.exe --compile model.onnx --output path\\name --target TARGET [--precision fp32|int8|fp16|bf16|int4]")
  PrintN("  PureMetalOnnxCompiler.exe --compile-reusable model.onnx --output path\\name --target windows [--speech-ui]")
  PrintN("  Every --compile emits a resident reusable library; --speech-ui adds a Windows Kokoro application.")
  PrintN("  Pi 4 / UNO Q: --target pi4|unoq --kokoro-text adds a resident text-to-Kokoro adapter (no board I/O).")
  PrintN("  PureMetalOnnxCompiler.exe --kokoro-prepare model.onnx --text TEXT --g2p FILE --voice FILE --output PREFIX")
  PrintN("  PureMetalOnnxCompilerCLI.exe --kokoro-pack-voice RAW.bin --output VOICE.pmvoice [--source-sha256 HEX]")
  PrintN("  PureMetalOnnxCompilerCLI.exe --kokoro-verify-voice VOICE.pmvoice")
  PrintN("")
  PrintN("Run without arguments to open the compiler window (UI wiring follows")
  PrintN("the checked compiler core; it will not shell out to Python).")
EndProcedure

Procedure.s ShapeText(*Value.PmoOnnxValue)
  Protected Result.s
  If *Value = 0 : ProcedureReturn "?" : EndIf
  Result = "["
  ForEach *Value\Dims()
    If Result <> "[" : Result + "," : EndIf
    If *Value\Dims()\HasValue
      Result + Str(*Value\Dims()\Value)
    ElseIf *Value\Dims()\Symbol <> ""
      Result + *Value\Dims()\Symbol
    Else
      Result + "?"
    EndIf
  Next
  ProcedureReturn Result + "]"
EndProcedure

Procedure.i InspectModel(Path.s, Quiet.i)
  Protected Model.PmoOnnxModel
  Protected RawTensors.i
  Protected ExternalTensors.i
  Protected TypedTensors.i
  NewMap Operators.i()
  If PmoOnnxLoad(Path, @Model) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoWireError)
    ProcedureReturn #False
  EndIf
  ForEach Model\Graph\Nodes()
    Operators(Model\Graph\Nodes()\Operation) + 1
  Next
  ForEach Model\Graph\Initializers()
    If Model\Graph\Initializers()\Raw\Bytes > 0 : RawTensors + 1 : EndIf
    If Model\Graph\Initializers()\DataLocation <> 0 : ExternalTensors + 1 : EndIf
    If ListSize(Model\Graph\Initializers()\FloatData()) > 0 Or
       ListSize(Model\Graph\Initializers()\DoubleData()) > 0 Or
       ListSize(Model\Graph\Initializers()\Int32Data()) > 0 Or
       ListSize(Model\Graph\Initializers()\Int64Data()) > 0 Or
       ListSize(Model\Graph\Initializers()\UInt64Data()) > 0
      TypedTensors + 1
    EndIf
  Next
  If Quiet = 0
    PrintN("Model:        " + Model\Path)
    PrintN("Bytes:        " + Str(Model\FileBytes))
    PrintN("IR version:   " + Str(Model\IrVersion))
    PrintN("Graph:        " + Model\Graph\Name)
    PrintN("Nodes:        " + Str(ListSize(Model\Graph\Nodes())))
    PrintN("Initializers: " + Str(ListSize(Model\Graph\Initializers())) +
           " (raw=" + Str(RawTensors) + ", typed=" + Str(TypedTensors) +
           ", external=" + Str(ExternalTensors) + ")")
    PrintN("Value infos:  " + Str(ListSize(Model\Graph\Values())))
    PrintN("Inputs:")
    ForEach Model\Graph\Inputs()
      PrintN("  " + Model\Graph\Inputs()\Name + " type=" +
             Str(Model\Graph\Inputs()\ElementType) + " " + ShapeText(@Model\Graph\Inputs()))
    Next
    PrintN("Outputs:")
    ForEach Model\Graph\Outputs()
      PrintN("  " + Model\Graph\Outputs()\Name + " type=" +
             Str(Model\Graph\Outputs()\ElementType) + " " + ShapeText(@Model\Graph\Outputs()))
    Next
    PrintN("Operators:")
    ForEach Operators()
      PrintN("  " + MapKey(Operators()) + " " + Str(Operators()))
    Next
  EndIf
  PmoOnnxFree(@Model)
  ProcedureReturn #True
EndProcedure

Procedure.i WireSelfTest()
  Protected Cursor.PmoWireCursor
  Protected Value.Quad
  Protected Field.Integer
  Protected Wire.Integer
  Protected Slice.PmoWireSlice
  Protected Shift.i
  Protected Dim Buffer.a(15)
  ; field 1, varint 150; field 2, bytes "abc"; field 3, fixed32 0x78563412
  Buffer(0) = $08 : Buffer(1) = $96 : Buffer(2) = $01
  Buffer(3) = $12 : Buffer(4) = $03 : Buffer(5) = 'a' : Buffer(6) = 'b' : Buffer(7) = 'c'
  Buffer(8) = $1D : Buffer(9) = $12 : Buffer(10) = $34 : Buffer(11) = $56 : Buffer(12) = $78
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 13) = 0 : ProcedureReturn #False : EndIf
  If PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or Field\i <> 1 Or Wire\i <> 0
    ProcedureReturn PmoWireFail("wire self-test lost field 1")
  EndIf
  If PmoWireReadVarint(@Cursor, @Value) = 0 Or Value\q <> 150
    ProcedureReturn PmoWireFail("wire self-test decoded the wrong varint")
  EndIf
  If PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or Field\i <> 2 Or Wire\i <> 2
    ProcedureReturn PmoWireFail("wire self-test lost field 2")
  EndIf
  If PmoWireReadSlice(@Cursor, @Slice) = 0 Or Slice\Bytes <> 3 Or PmoWireUtf8(@Slice) <> "abc"
    ProcedureReturn PmoWireFail("wire self-test decoded the wrong byte slice")
  EndIf
  If PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or Field\i <> 3 Or Wire\i <> 5
    ProcedureReturn PmoWireFail("wire self-test lost field 3")
  EndIf
  If PmoWireSkip(@Cursor, Wire\i) = 0 Or PmoWireRemaining(@Cursor) <> 0
    ProcedureReturn PmoWireFail("wire self-test failed to skip fixed32")
  EndIf
  ; A continuation byte without a terminator must be rejected.
  Buffer(0) = $08 : Buffer(1) = $80
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 2) = 0 Or PmoWireReadKey(@Cursor, @Field, @Wire) = 0
    ProcedureReturn #False
  EndIf
  If PmoWireReadVarint(@Cursor, @Value) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted a truncated varint")
  EndIf

  ; Field zero, group wire types, oversized lengths, truncated fixed values and
  ; a tenth varint byte with payload bits must all fail before reading outside
  ; the enclosing slice.
  Buffer(0) = $00
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 1) = 0 Or PmoWireReadKey(@Cursor, @Field, @Wire) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted field number zero")
  EndIf
  Buffer(0) = $0B
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 1) = 0 Or PmoWireReadKey(@Cursor, @Field, @Wire) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted a protobuf group")
  EndIf
  Buffer(0) = $12 : Buffer(1) = $05 : Buffer(2) = 'a'
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 3) = 0 Or
     PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or PmoWireReadSlice(@Cursor, @Slice) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted an oversized byte slice")
  EndIf
  Buffer(0) = $09 : Buffer(1) = $01
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 2) = 0 Or
     PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or PmoWireSkip(@Cursor, Wire\i) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted a truncated fixed64 field")
  EndIf
  Buffer(0) = $08
  For Shift = 1 To 9 : Buffer(Shift) = $80 : Next
  Buffer(10) = $02
  PmoWireResetError()
  If PmoWireInit(@Cursor, @Buffer(0), 11) = 0 Or
     PmoWireReadKey(@Cursor, @Field, @Wire) = 0 Or PmoWireReadVarint(@Cursor, @Value) <> 0
    ProcedureReturn PmoWireFail("wire self-test accepted a varint wider than 64 bits")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure PrintTargets()
  Protected Index.i
  PmoInitTargets()
  For Index = 0 To #PMO_TARGET_COUNT - 1
    PrintN(PmoTargets(Index)\Id + Chr(9) + PmoTargets(Index)\Label)
  Next
EndProcedure

Procedure.s NpyShapeText(*Array.PmoNpyArray)
  Protected Result.s = "["
  ForEach *Array\Dims()
    If Result <> "[" : Result + "," : EndIf
    Result + Str(*Array\Dims())
  Next
  ProcedureReturn Result + "]"
EndProcedure

Procedure.i InspectNpy(Path.s)
  Protected Array.PmoNpyArray
  If PmoNpyLoad(Path, @Array) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoNpyError)
    ProcedureReturn #False
  EndIf
  PrintN("NumPy: " + Array\Path)
  PrintN("Dtype: " + Array\Descriptor + " (ONNX " + Str(Array\ElementType) + ")")
  PrintN("Shape: " + NpyShapeText(@Array))
  PrintN("Bytes: " + Str(Array\DataBytes))
  PmoNpyFree(@Array)
  ProcedureReturn #True
EndProcedure

Procedure.i WriteTraceModel(ModelPath.s, OutputPath.s)
  Protected Model.PmoOnnxModel
  Protected Buffer.PmoProtoBuffer
  Protected File.i
  Protected Written.q
  NewList Names.s()
  NewMap Checkpoints.i()
  If PmoOnnxLoad(ModelPath, @Model) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoWireError)
    ProcedureReturn #False
  EndIf
  If PmoInferMissingRanks(@Model) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoRankError)
    PmoOnnxFree(@Model)
    ProcedureReturn #False
  EndIf
  If PmoBuildTraceModel(@Model, Names(), Checkpoints(), @Buffer) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoProtoError)
    PmoOnnxFree(@Model)
    ProcedureReturn #False
  EndIf
  File = CreateFile(#PB_Any, OutputPath)
  If File = 0
    PrintN("ONNX COMPILER ERROR: cannot create trace model: " + OutputPath)
    PmoProtoFree(@Buffer) : PmoOnnxFree(@Model)
    ProcedureReturn #False
  EndIf
  Written = WriteData(File, Buffer\Data, Buffer\Length)
  CloseFile(File)
  If Written <> Buffer\Length
    PrintN("ONNX COMPILER ERROR: short write while creating trace model")
    PmoProtoFree(@Buffer) : PmoOnnxFree(@Model)
    ProcedureReturn #False
  EndIf
  PrintN("Trace outputs: " + Str(ListSize(Names())))
  PrintN("Trace model: " + OutputPath + " (" + Str(Buffer\Length) + " bytes)")
  PmoProtoFree(@Buffer) : PmoOnnxFree(@Model)
  ProcedureReturn #True
EndProcedure

Procedure.i TraceShapesCommand(ModelPath.s)
  Protected Model.PmoOnnxModel
  Protected RuntimePath.s
  Protected OptionName.s
  Protected Index.i = 2
  NewList Specs.s()
  NewList Names.s()
  While Index < CountProgramParameters()
    OptionName = ProgramParameter(Index)
    If OptionName = "--ort-dll" And Index + 1 < CountProgramParameters()
      Index + 1 : RuntimePath = ProgramParameter(Index)
    ElseIf OptionName = "--trace-input" And Index + 1 < CountProgramParameters()
      Index + 1 : AddElement(Specs()) : Specs() = ProgramParameter(Index)
    Else
      PrintN("ONNX COMPILER ERROR: unknown or incomplete trace option " + OptionName)
      ProcedureReturn #False
    EndIf
    Index + 1
  Wend
  If PmoOnnxLoad(ModelPath, @Model) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoWireError)
    ProcedureReturn #False
  EndIf
  If PmoTraceShapes(@Model, RuntimePath, Specs(), Names()) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoTraceError)
    PmoOnnxFree(@Model)
    ProcedureReturn #False
  EndIf
  PrintN("ONNX Runtime " + PmoTraceRuntimeVersion)
  PrintN("PASS: traced " + Str(ListSize(Names())) + " concrete tensor shapes")
  ForEach Model\Graph\Outputs()
    PrintN("Output " + Model\Graph\Outputs()\Name + " " + ShapeText(@Model\Graph\Outputs()))
  Next
  PmoOnnxFree(@Model)
  ProcedureReturn #True
EndProcedure

Procedure.i LowerCommand(ModelPath.s)
  Protected Model.PmoOnnxModel
  Protected Ir.PmoIrModel
  Protected RuntimePath.s
  Protected WeightsPath.s
  Protected OptionName.s
  Protected Index.i = 2
  NewList Specs.s()
  NewList Names.s()
  NewMap RuntimeOps.i()
  While Index < CountProgramParameters()
    OptionName = ProgramParameter(Index)
    If OptionName = "--ort-dll" And Index + 1 < CountProgramParameters()
      Index + 1 : RuntimePath = ProgramParameter(Index)
    ElseIf OptionName = "--trace-input" And Index + 1 < CountProgramParameters()
      Index + 1 : AddElement(Specs()) : Specs() = ProgramParameter(Index)
    ElseIf OptionName = "--weights" And Index + 1 < CountProgramParameters()
      Index + 1 : WeightsPath = ProgramParameter(Index)
    ElseIf OptionName = "--external-inputs"
      ; Compatibility option: requests are always external, never packed.
    Else
      PrintN("ONNX COMPILER ERROR: unknown or incomplete lowering option " + OptionName)
      ProcedureReturn #False
    EndIf
    Index + 1
  Wend
  If PmoOnnxLoad(ModelPath, @Model) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoWireError) : ProcedureReturn #False
  EndIf
  If ListSize(Specs()) > 0
    If PmoTraceShapes(@Model, RuntimePath, Specs(), Names()) = 0
      PrintN("ONNX COMPILER ERROR: " + PmoTraceError) : Goto LowerCommandFailed
    EndIf
  EndIf
  If PmoIrPrepare(@Model, @Ir) = 0 Or PmoIrFoldConstants(@Ir) = 0 Or PmoIrPlanArena(@Ir) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoIrError) : Goto LowerCommandFailed
  EndIf
  If WeightsPath <> ""
    If PmoPackWeights(@Ir, WeightsPath) = 0
      If PmoPackError <> "" : PrintN("ONNX COMPILER ERROR: " + PmoPackError) : Else : PrintN("ONNX COMPILER ERROR: cannot pack weights") : EndIf
      Goto LowerCommandFailed
    EndIf
    PrintN("Weight bytes: " + Str(Ir\WeightBytes))
  EndIf
  PrintN("PASS: lowered model")
  PrintN("Runtime nodes: " + Str(ListSize(Ir\Nodes())))
  PrintN("Constants: " + Str(ListSize(Ir\Constants())))
  PrintN("Arena bytes: " + Str(Ir\ArenaBytes))
  ForEach Ir\Nodes()
    RuntimeOps(Ir\Nodes()\Node\Operation) + 1
  Next
  ForEach RuntimeOps()
    PrintN("  " + MapKey(RuntimeOps()) + " " + Str(RuntimeOps()))
  Next
  PmoIrFree(@Ir) : PmoOnnxFree(@Model)
  ProcedureReturn #True

  LowerCommandFailed:
  PmoIrFree(@Ir) : PmoOnnxFree(@Model)
  ProcedureReturn #False
EndProcedure

Procedure.i PackVoiceCommand(Source.s)
  Protected index.i = 2, option.s, value.s, destination.s, expected.s
  Protected seenOutput.i, seenHash.i
  While index < CountProgramParameters()
    option = ProgramParameter(index)
    If index + 1 >= CountProgramParameters()
      PrintN("VOICE PACK ERROR: missing value for " + option) : ProcedureReturn #False
    EndIf
    value = ProgramParameter(index + 1)
    Select option
      Case "--output"
        If seenOutput : PrintN("VOICE PACK ERROR: duplicate --output") : ProcedureReturn #False : EndIf
        seenOutput = #True : destination = value
      Case "--source-sha256"
        If seenHash : PrintN("VOICE PACK ERROR: duplicate --source-sha256") : ProcedureReturn #False : EndIf
        seenHash = #True : expected = value
        If Len(expected) <> 64 : PrintN("VOICE PACK ERROR: SHA-256 must contain 64 hexadecimal digits") : ProcedureReturn #False : EndIf
      Default
        PrintN("VOICE PACK ERROR: unknown option " + option) : ProcedureReturn #False
    EndSelect
    index + 2
  Wend
  If PmoKokoroVoicePack(Source, destination, expected) = 0
    PrintN("VOICE PACK ERROR: " + PmoKokoroVoicePackError) : ProcedureReturn #False
  EndIf
  PrintN("PASS: verified PMVOICE written to " + destination)
  ProcedureReturn #True
EndProcedure

Define Mode.s = ProgramParameter(0)
Define ModelPath.s = ProgramParameter(1)

If Mode = ""
  PmoRunUi()
  End 0
EndIf

OpenConsole()

If Mode = "--inspect"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If InspectModel(ModelPath, #False) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--self-test"
  If WireSelfTest() = 0
    PrintN("FAIL: " + PmoWireError)
    End 1
  EndIf
  If ModelPath <> "" And InspectModel(ModelPath, #True) = 0
    End 1
  EndIf
  PrintN("PASS: bounded protobuf wire reader and native ONNX model inventory")
  End 0
ElseIf Mode = "--ort-version"
  If PmoOrtLoad(ModelPath) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoOrtError)
    End 1
  EndIf
  PrintN("ONNX Runtime " + PmoOrtVersion + " (C API " + Str(#PMO_ORT_API_VERSION) + ")")
  PmoOrtUnload()
  End 0
ElseIf Mode = "--list-targets"
  If PmoAllTargetsValid() = 0
    PrintN("ONNX COMPILER ERROR: invalid target registry")
    End 1
  EndIf
  PrintTargets()
  End 0
ElseIf Mode = "--npy-inspect"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If InspectNpy(ModelPath) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--trace-model"
  If ModelPath = "" Or ProgramParameter(2) = "" : PrintUsage() : End 2 : EndIf
  If WriteTraceModel(ModelPath, ProgramParameter(2)) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--trace-shapes"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If TraceShapesCommand(ModelPath) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--lower"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If LowerCommand(ModelPath) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--compile" Or Mode = "--compile-reusable"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If PmoCompileCommand(ModelPath) = 0
    PrintN("ONNX COMPILER ERROR: " + PmoCompileError)
    End 1
  EndIf
  End 0
ElseIf Mode = "--kokoro-pack-voice"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If PackVoiceCommand(ModelPath) = 0 : End 1 : EndIf
  End 0
ElseIf Mode = "--kokoro-verify-voice"
  If CountProgramParameters() <> 2 : PrintUsage() : End 2 : EndIf
  If PmoKokoroVoiceVerifyFile(ModelPath) = 0
    PrintN("VOICE PACK ERROR: " + PmoKokoroVoicePackError) : End 1
  EndIf
  PrintN("PASS: PMVOICE identity, shape, checksums and finite samples verified")
  End 0
ElseIf Mode = "--kokoro-prepare"
  If ModelPath = "" : PrintUsage() : End 2 : EndIf
  If PmoKokoroPrepareCommand(ModelPath) = 0
    PrintN("KOKORO INPUT ERROR: " + PmoKokoroError)
    End 1
  EndIf
  End 0
Else
  PrintUsage()
  End 2
EndIf
