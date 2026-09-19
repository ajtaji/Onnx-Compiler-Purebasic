; ============================================================================
; node_driver_windows.pbi - the one generic driver for the node-test harness
; ----------------------------------------------------------------------------
; DEVELOPER CHECK ONLY. Nothing here is part of the compiler, and neither
; building nor using the compiler needs it.
;
; node_suite.py writes a two-line stub per case:
;     XIncludeFile "m.pb"                       ; the source the compiler emitted
;     XIncludeFile "<this file>"
; and builds it as a console program. The same driver serves both APIs the
; compiler emits: the fixed-shape PmOnnx... library (caller-owned weights and
; arena) and the runtime-dimension PmModel... library. Which one is present is
; decided at compile time from the procedures the included source defines.
;
; Usage: driver.exe request.bin result.bin weights.pmw
;
; request.bin (little-endian), written by the harness from the case's .pb files:
;   "PMNH" u32 version=1 u32 inputCount
;   per input:  u32 elementType u32 rank i64 dims[rank] i64 byteCount bytes
;   u32 outputCount
;   per output: i64 byteCount (the manifest's bytes; the dynamic API ignores it)
; result.bin:
;   "PMNR" u32 version=1 u32 outputCount
;   per output: u32 elementType u32 rank i64 dims[rank] i64 byteCount bytes
;   The fixed-shape API carries no runtime shape, so it writes elementType 0
;   and rank 0; the harness then takes both from the compiler's manifest.
; A sequence value (runtime-dimension API only) is written in place of a
; tensor record as: u32 $80000000|elementType, u32 elementCount, then per
; element: u32 rank, i64 dims[rank], i64 byteCount, bytes.
; Any failure prints one sentence and exits nonzero; the harness records it.
; ============================================================================

CompilerIf Defined(PmModelExecute, #PB_Procedure)
  #PMH_DYNAMIC = 1
CompilerElseIf Defined(PmOnnxExecuteBound, #PB_Procedure)
  #PMH_DYNAMIC = 0
CompilerElse
  CompilerError "node driver: the included model source defines neither PmModelExecute nor PmOnnxExecuteBound"
CompilerEndIf

Procedure PmhFail(Text.s, Code.i)
  PrintN("NODE DRIVER ERROR: " + Text)
  End Code
EndProcedure

Procedure.i DMaxPmh(A.i, B.i)
  If A > B : ProcedureReturn A : EndIf
  ProcedureReturn B
EndProcedure

Procedure PmhMain()
  Protected RequestPath.s = ProgramParameter(0)
  Protected ResultPath.s = ProgramParameter(1)
  Protected WeightsPath.s = ProgramParameter(2)
  Protected File.i, Magic.l, Version.l, InputCount.l, OutputCount.l
  Protected Index.i, Axis.i, Kind.l, Rank.l, Bytes.q, Got.q, Id.i
  Protected *Target, *Weights, *Arena, WeightBytes.q, Items.l, Item.i
  Protected Dim Dims.i(7)
  If CountProgramParameters() <> 3
    PmhFail("expected three arguments: request file, result file, weights file.", 2)
  EndIf
  File = ReadFile(#PB_Any, RequestPath)
  If File = 0 : PmhFail("cannot open the request file " + RequestPath + ".", 2) : EndIf
  Magic = ReadLong(File) : Version = ReadLong(File)
  If Magic <> $484E4D50 Or Version <> 1
    PmhFail("the request file is not a version 1 node-test request; regenerate it with node_suite.py.", 2)
  EndIf
  InputCount = ReadLong(File)

  CompilerIf #PMH_DYNAMIC
    If PmModelInitialize(WeightsPath) = 0
      PmhFail("model initialization failed: " + DError, 3)
    EndIf
    PmModelResetRequest()
    DError = ""
  CompilerElse
    WeightBytes = FileSize(WeightsPath)
    If WeightBytes <= 0 : PmhFail("cannot find the weights file " + WeightsPath + ".", 3) : EndIf
    *Weights = AllocateMemory(WeightBytes)
    *Arena = AllocateMemory(#PMO_ARENA_ALLOCATION_BYTES)
    If *Weights = 0 Or *Arena = 0 : PmhFail("cannot allocate the weights or the arena.", 3) : EndIf
    Id = ReadFile(#PB_Any, WeightsPath)
    If Id = 0 : PmhFail("cannot open the weights file " + WeightsPath + ".", 3) : EndIf
    If ReadData(Id, *Weights, WeightBytes) <> WeightBytes : CloseFile(Id) : PmhFail("short read of the weights file.", 3) : EndIf
    CloseFile(Id)
    If PmOnnxBindMemory(*Weights, WeightBytes, *Arena, #PMO_ARENA_ALLOCATION_BYTES) = 0
      PmhFail("PmOnnxBindMemory refused the weights or the arena.", 3)
    EndIf
    If InputCount <> PmOnnxInputCount()
      PmhFail("the request carries " + Str(InputCount) + " inputs; the compiled model declares " + Str(PmOnnxInputCount()) + ".", 4)
    EndIf
  CompilerEndIf

  For Index = 0 To InputCount - 1
    Kind = ReadLong(File)
    If Kind & $80000000
      CompilerIf #PMH_DYNAMIC And Defined(DSeqAppendShape, #PB_Procedure)
        Id = PmModelInput(Index)
        If Id = 0 : PmhFail("the compiled model has no input " + Str(Index) + ".", 4) : EndIf
        Items = ReadLong(File)
        If DSeqReset(Id, Kind & $7FFFFFFF) = 0 : PmhFail("input " + Str(Index) + " could not be made a sequence: " + DError, 4) : EndIf
        For Item = 0 To Items - 1
          Rank = ReadLong(File)
          If Rank < 0 Or Rank > 8 : PmhFail("input " + Str(Index) + " element " + Str(Item) + " has rank " + Str(Rank) + "; the driver carries at most eight axes.", 4) : EndIf
          For Axis = 0 To 7 : Dims(Axis) = 0 : Next
          For Axis = 0 To Rank - 1 : Dims(Axis) = ReadQuad(File) : Next
          Bytes = ReadQuad(File)
          *Target = DSeqAppendShape(Id, Kind & $7FFFFFFF, Rank, @Dims(0))
          If *Target = 0 : PmhFail("input " + Str(Index) + " element " + Str(Item) + " could not be appended: " + DError, 4) : EndIf
          If DSeqItemBytes(Id, Item) <> Bytes
            PmhFail("input " + Str(Index) + " element " + Str(Item) + " holds " + Str(DSeqItemBytes(Id, Item)) + " bytes but the request carries " + Str(Bytes) + ".", 4)
          EndIf
          If Bytes > 0 And ReadData(File, *Target, Bytes) <> Bytes : PmhFail("the request file ended inside input " + Str(Index) + ".", 4) : EndIf
        Next
        Continue
      CompilerElse
        PmhFail("input " + Str(Index) + " is a sequence, but the compiled model has no sequence runtime.", 4)
      CompilerEndIf
    EndIf
    Rank = ReadLong(File)
    If Rank < 0 Or Rank > 8 : PmhFail("input " + Str(Index) + " has rank " + Str(Rank) + "; the driver carries at most eight axes.", 4) : EndIf
    For Axis = 0 To 7 : Dims(Axis) = 0 : Next
    For Axis = 0 To Rank - 1 : Dims(Axis) = ReadQuad(File) : Next
    Bytes = ReadQuad(File)
    CompilerIf #PMH_DYNAMIC
      Id = PmModelInput(Index)
      If Id = 0 : PmhFail("the compiled model has no input " + Str(Index) + ".", 4) : EndIf
      If DAlloc(Id, Kind, Rank, @Dims(0)) = 0 : PmhFail("input " + Str(Index) + " could not be shaped: " + DError, 4) : EndIf
      If Dt(Id)\Bytes <> Bytes
        PmhFail("input " + Str(Index) + " holds " + Str(Dt(Id)\Bytes) + " bytes but the request carries " + Str(Bytes) + ".", 4)
      EndIf
      *Target = Dt(Id)\Data
    CompilerElse
      *Target = PmOnnxInputAddress(Index)
      If *Target = 0 : PmhFail("the compiled model has no input " + Str(Index) + ".", 4) : EndIf
    CompilerEndIf
    If Bytes > 0
      Got = ReadData(File, *Target, Bytes)
      If Got <> Bytes : PmhFail("the request file ended inside input " + Str(Index) + ".", 4) : EndIf
    EndIf
  Next
  OutputCount = ReadLong(File)
  Protected Dim OutBytes.q(DMaxPmh(OutputCount, 1))
  For Index = 0 To OutputCount - 1 : OutBytes(Index) = ReadQuad(File) : Next
  CloseFile(File)

  CompilerIf #PMH_DYNAMIC
    If PmModelExecute() = 0 : PmhFail("execution failed: " + DError, 5) : EndIf
  CompilerElse
    If OutputCount <> PmOnnxOutputCount()
      PmhFail("the case expects " + Str(OutputCount) + " outputs; the compiled model declares " + Str(PmOnnxOutputCount()) + ".", 4)
    EndIf
    If PmOnnxExecuteBound() = 0
      PmhFail("execution failed at node " + Str(PmOnnxLastErrorNode()) + ".", 5)
    EndIf
  CompilerEndIf

  File = CreateFile(#PB_Any, ResultPath)
  If File = 0 : PmhFail("cannot create the result file " + ResultPath + ".", 6) : EndIf
  WriteLong(File, $524E4D50) : WriteLong(File, 1) : WriteLong(File, OutputCount)
  For Index = 0 To OutputCount - 1
    CompilerIf #PMH_DYNAMIC
      Id = PmModelOutput(Index)
      If Id = 0 : CloseFile(File) : PmhFail("the compiled model has no output " + Str(Index) + ".", 6) : EndIf
      CompilerIf Defined(DSeqAppendShape, #PB_Procedure)
        If Dt(Id)\Kind = #PMD_KIND_SEQUENCE
          WriteLong(File, $80000000 | DSeqElementKind(Id)) : WriteLong(File, DSeqCount(Id))
          For Item = 0 To DSeqCount(Id) - 1
            WriteLong(File, DSeqItemRank(Id, Item))
            For Axis = 0 To DSeqItemRank(Id, Item) - 1 : WriteQuad(File, DSeqItemDim(Id, Item, Axis)) : Next
            WriteQuad(File, DSeqItemBytes(Id, Item))
            If DSeqItemBytes(Id, Item) > 0 : WriteData(File, DSeqItemData(Id, Item), DSeqItemBytes(Id, Item)) : EndIf
          Next
          Continue
        EndIf
      CompilerEndIf
      Bytes = Dt(Id)\Count * DSize(Dt(Id)\Kind)
      WriteLong(File, Dt(Id)\Kind) : WriteLong(File, Dt(Id)\Rank)
      For Axis = 0 To Dt(Id)\Rank - 1 : WriteQuad(File, Dt(Id)\D[Axis]) : Next
      WriteQuad(File, Bytes)
      If Bytes > 0 : WriteData(File, Dt(Id)\Data, Bytes) : EndIf
    CompilerElse
      *Target = PmOnnxOutputAddress(Index)
      If *Target = 0 : CloseFile(File) : PmhFail("the compiled model has no output " + Str(Index) + ".", 6) : EndIf
      WriteLong(File, 0) : WriteLong(File, 0)
      WriteQuad(File, OutBytes(Index))
      If OutBytes(Index) > 0 : WriteData(File, *Target, OutBytes(Index)) : EndIf
    CompilerEndIf
  Next
  CloseFile(File)
  CompilerIf #PMH_DYNAMIC
    PmModelResetRequest()
    PmModelClose()
  CompilerElse
    PmOnnxUnbindMemory()
  CompilerEndIf
EndProcedure

OpenConsole()
PmhMain()
End 0
