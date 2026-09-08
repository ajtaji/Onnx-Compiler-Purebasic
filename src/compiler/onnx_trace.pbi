; ============================================================================
; onnx_trace.pbi - concrete shape tracing through the native ORT C ABI
; ----------------------------------------------------------------------------
; The runtime is a host-only compilation service. It sees a temporary checked
; graph whose selected intermediates are outputs, returns concrete tensor
; extents for one complete input set, and is then released. Target programs do
; not contain or invoke this bridge.
; ============================================================================

Structure PmoTraceFeed
  Name.s
  Array.PmoNpyArray
  *NameUtf8
  *Value
EndStructure

Global PmoTraceError.s
Global PmoTraceRuntimeVersion.s

Procedure.i PmoTraceFail(Message.s)
  If PmoTraceError = "" : PmoTraceError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoTraceFreeFeeds(List Feeds.PmoTraceFeed())
  Protected Release.PmoOrtReleaseFn = PmoOrtFunction(#PMO_ORT_RELEASE_VALUE)
  ForEach Feeds()
    If Feeds()\Value And Release : Release(Feeds()\Value) : EndIf
    If Feeds()\NameUtf8 : FreeMemory(Feeds()\NameUtf8) : EndIf
    PmoNpyFree(@Feeds()\Array)
  Next
  ClearList(Feeds())
EndProcedure

Procedure.i PmoTraceParseFeed(Spec.s, *Name.String, *Path.String)
  Protected Equals.i = FindString(Spec, "=")
  If Equals <= 1 Or Equals >= Len(Spec)
    ProcedureReturn PmoTraceFail("trace input must be NAME=file.npy: " + Spec)
  EndIf
  *Name\s = Trim(Left(Spec, Equals - 1))
  *Path\s = Trim(Mid(Spec, Equals + 1))
  If *Name\s = "" Or *Path\s = ""
    ProcedureReturn PmoTraceFail("trace input must be NAME=file.npy: " + Spec)
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoTraceValidateFeed(*Input.PmoOnnxValue, *Array.PmoNpyArray)
  Protected Index.i
  If *Input\ElementType <> *Array\ElementType
    ProcedureReturn PmoTraceFail("trace input " + *Input\Name + " has ONNX type " +
                                 Str(*Array\ElementType) + ", expected " + Str(*Input\ElementType))
  EndIf
  If *Input\HasShape And ListSize(*Input\Dims()) <> ListSize(*Array\Dims())
    ProcedureReturn PmoTraceFail("trace input " + *Input\Name + " has the wrong rank")
  EndIf
  If *Input\HasShape
    FirstElement(*Array\Dims())
    Index = 0
    ForEach *Input\Dims()
      If *Input\Dims()\HasValue And *Input\Dims()\Value >= 0 And
         *Input\Dims()\Value <> *Array\Dims()
        ProcedureReturn PmoTraceFail("trace input " + *Input\Name + " dimension " + Str(Index) +
                                     " is " + Str(*Array\Dims()) + ", expected " +
                                     Str(*Input\Dims()\Value))
      EndIf
      NextElement(*Array\Dims())
      Index + 1
    Next
  EndIf
  ClearList(*Input\Dims())
  ForEach *Array\Dims()
    AddElement(*Input\Dims())
    *Input\Dims()\HasValue = #True
    *Input\Dims()\Value = *Array\Dims()
  Next
  *Input\HasShape = #True
  ProcedureReturn #True
EndProcedure

Procedure.i PmoTraceLoadFeeds(*Model.PmoOnnxModel, List Specs.s(), List Feeds.PmoTraceFeed())
  NewMap FeedPointers.i()
  NewMap Initializers.i()
  Protected Name.String
  Protected Path.String
  Protected *Feed.PmoTraceFeed
  Protected Required.i
  ClearList(Feeds())
  PmoTraceError = ""
  PmoTraceRuntimeVersion = ""
  ForEach Specs()
    If PmoTraceParseFeed(Specs(), @Name, @Path) = 0 : Goto PmoTraceLoadFeedsFailed : EndIf
    If FindMapElement(FeedPointers(), Name\s)
      PmoTraceFail("duplicate trace input " + Name\s) : Goto PmoTraceLoadFeedsFailed
    EndIf
    AddElement(Feeds())
    Feeds()\Name = Name\s
    If PmoNpyLoad(Path\s, @Feeds()\Array) = 0
      PmoTraceFail(PmoNpyError) : Goto PmoTraceLoadFeedsFailed
    EndIf
    FeedPointers(Name\s) = @Feeds()
  Next
  ForEach *Model\Graph\Initializers()
    Initializers(*Model\Graph\Initializers()\Name) = #True
  Next
  ForEach *Model\Graph\Inputs()
    If FindMapElement(Initializers(), *Model\Graph\Inputs()\Name) : Continue : EndIf
    Required + 1
    If FindMapElement(FeedPointers(), *Model\Graph\Inputs()\Name) = 0
      PmoTraceFail("missing trace input " + *Model\Graph\Inputs()\Name)
      Goto PmoTraceLoadFeedsFailed
    EndIf
    *Feed = FeedPointers()
    If PmoTraceValidateFeed(@*Model\Graph\Inputs(), @*Feed\Array) = 0
      Goto PmoTraceLoadFeedsFailed
    EndIf
  Next
  If ListSize(Feeds()) <> Required
    ForEach Feeds()
      If FindMapElement(FeedPointers(), Feeds()\Name) And
         FindMapElement(Initializers(), Feeds()\Name) = 0
        ; The exact unknown name is reported below.
      EndIf
      Protected Found.i = #False
      ForEach *Model\Graph\Inputs()
        If *Model\Graph\Inputs()\Name = Feeds()\Name And
           FindMapElement(Initializers(), Feeds()\Name) = 0
          Found = #True : Break
        EndIf
      Next
      If Found = 0
        PmoTraceFail("unknown trace input " + Feeds()\Name)
        Goto PmoTraceLoadFeedsFailed
      EndIf
    Next
    PmoTraceFail("trace input count does not match the graph")
    Goto PmoTraceLoadFeedsFailed
  EndIf
  ProcedureReturn #True

  PmoTraceLoadFeedsFailed:
  PmoTraceFreeFeeds(Feeds())
  ProcedureReturn #False
EndProcedure

Procedure.i PmoTraceCreateOrtValues(*State.PmoOrtSession, List Feeds.PmoTraceFeed(),
                                    *InputNames.Integer, *InputValues.Integer)
  Protected CreateTensor.PmoOrtCreateTensorWithDataFn = PmoOrtFunction(#PMO_ORT_CREATE_TENSOR_WITH_DATA)
  Protected *Shape
  Protected ShapeCount.i
  Protected Index.i
  Protected DimIndex.i
  If CreateTensor = 0 : ProcedureReturn PmoTraceFail("ONNX Runtime tensor function is missing") : EndIf
  ForEach Feeds()
    ShapeCount = ListSize(Feeds()\Array\Dims())
    If ShapeCount > 0
      *Shape = AllocateMemory(ShapeCount * 8)
      If *Shape = 0 : ProcedureReturn PmoTraceFail("cannot allocate trace input shape") : EndIf
      DimIndex = 0
      ForEach Feeds()\Array\Dims()
        PokeQ(*Shape + DimIndex * 8, Feeds()\Array\Dims())
        DimIndex + 1
      Next
    Else
      *Shape = 0
    EndIf
    Feeds()\NameUtf8 = UTF8(Feeds()\Name)
    If Feeds()\NameUtf8 = 0
      If *Shape : FreeMemory(*Shape) : EndIf
      ProcedureReturn PmoTraceFail("cannot encode trace input name")
    EndIf
    If PmoOrtCheck(CreateTensor(*State\MemoryInfo, Feeds()\Array\Data,
                                Feeds()\Array\DataBytes, *Shape, ShapeCount,
                                Feeds()\Array\ElementType, @Feeds()\Value),
                   "cannot bind trace input " + Feeds()\Name) = 0
      If *Shape : FreeMemory(*Shape) : EndIf
      ProcedureReturn PmoTraceFail(PmoOrtError)
    EndIf
    If *Shape : FreeMemory(*Shape) : EndIf
    PokeI(*InputNames + Index * SizeOf(Integer), Feeds()\NameUtf8)
    PokeI(*InputValues + Index * SizeOf(Integer), Feeds()\Value)
    Index + 1
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoTraceApplyShape(*Value.PmoOnnxValue, *OrtValue)
  Protected GetInfo.PmoOrtGetTensorTypeAndShapeFn = PmoOrtFunction(#PMO_ORT_GET_TENSOR_TYPE_AND_SHAPE)
  Protected GetType.PmoOrtGetTensorElementTypeFn = PmoOrtFunction(#PMO_ORT_GET_TENSOR_ELEMENT_TYPE)
  Protected GetRank.PmoOrtGetDimensionsCountFn = PmoOrtFunction(#PMO_ORT_GET_DIMENSIONS_COUNT)
  Protected GetDims.PmoOrtGetDimensionsFn = PmoOrtFunction(#PMO_ORT_GET_DIMENSIONS)
  Protected ReleaseInfo.PmoOrtReleaseFn = PmoOrtFunction(#PMO_ORT_RELEASE_TENSOR_TYPE_AND_SHAPE)
  Protected *Info
  Protected *Dimensions
  Protected ElementType.Long
  Protected Count.Integer
  Protected Index.i
  If GetInfo = 0 Or GetType = 0 Or GetRank = 0 Or GetDims = 0 Or ReleaseInfo = 0
    ProcedureReturn PmoTraceFail("ONNX Runtime shape functions are missing")
  EndIf
  If PmoOrtCheck(GetInfo(*OrtValue, @*Info), "cannot read traced tensor shape") = 0
    ProcedureReturn PmoTraceFail(PmoOrtError)
  EndIf
  If PmoOrtCheck(GetType(*Info, @ElementType), "cannot read traced tensor type") = 0 Or
     PmoOrtCheck(GetRank(*Info, @Count), "cannot read traced tensor rank") = 0
    ReleaseInfo(*Info) : ProcedureReturn PmoTraceFail(PmoOrtError)
  EndIf
  If ElementType\l <> *Value\ElementType
    ReleaseInfo(*Info)
    ProcedureReturn PmoTraceFail("runtime type disagrees for traced value " + *Value\Name)
  EndIf
  If Count\i < 0 Or Count\i > 32
    ReleaseInfo(*Info)
    ProcedureReturn PmoTraceFail("runtime returned an invalid rank for " + *Value\Name)
  EndIf
  If Count\i > 0
    *Dimensions = AllocateMemory(Count\i * 8)
    If *Dimensions = 0
      ReleaseInfo(*Info) : ProcedureReturn PmoTraceFail("cannot allocate traced dimensions")
    EndIf
    If PmoOrtCheck(GetDims(*Info, *Dimensions, Count\i), "cannot read traced dimensions") = 0
      FreeMemory(*Dimensions) : ReleaseInfo(*Info) : ProcedureReturn PmoTraceFail(PmoOrtError)
    EndIf
  EndIf
  ClearList(*Value\Dims())
  For Index = 0 To Count\i - 1
    If PeekQ(*Dimensions + Index * 8) < 0
      If *Dimensions : FreeMemory(*Dimensions) : EndIf
      ReleaseInfo(*Info)
      ProcedureReturn PmoTraceFail("runtime left a negative extent for " + *Value\Name)
    EndIf
    AddElement(*Value\Dims())
    *Value\Dims()\HasValue = #True
    *Value\Dims()\Value = PeekQ(*Dimensions + Index * 8)
  Next
  *Value\HasShape = #True
  If *Dimensions : FreeMemory(*Dimensions) : EndIf
  ReleaseInfo(*Info)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoTraceShapes(*Model.PmoOnnxModel, RuntimePath.s, List Specs.s(),
                           List TraceNames.s())
  Protected TraceModel.PmoProtoBuffer
  Protected State.PmoOrtSession
  NewMap Checkpoints.i()
  NewMap Values.i()
  NewList Feeds.PmoTraceFeed()
  NewList OutputUtf8.i()
  Protected *InputNames
  Protected *InputValues
  Protected *OutputNames
  Protected *Outputs
  Protected Run.PmoOrtRunFn
  Protected ReleaseValue.PmoOrtReleaseFn
  Protected *Encoded
  Protected *Value.PmoOnnxValue
  Protected Index.i
  Protected Result.i
  PmoTraceError = ""
  If PmoInferMissingRanks(*Model) = 0 : ProcedureReturn PmoTraceFail(PmoRankError) : EndIf
  If PmoBuildTraceModel(*Model, TraceNames(), Checkpoints(), @TraceModel) = 0
    ProcedureReturn PmoTraceFail(PmoProtoError)
  EndIf
  If PmoOrtLoad(RuntimePath) = 0
    PmoProtoFree(@TraceModel) : ProcedureReturn PmoTraceFail(PmoOrtError)
  EndIf
  PmoTraceRuntimeVersion = PmoOrtVersion
  If PmoTraceLoadFeeds(*Model, Specs(), Feeds()) = 0 : Goto PmoTraceShapesFailed : EndIf
  If PmoOrtOpenSession(TraceModel\Data, TraceModel\Length, @State) = 0
    PmoTraceFail(PmoOrtError) : Goto PmoTraceShapesFailed
  EndIf
  *InputNames = AllocateMemory(ListSize(Feeds()) * SizeOf(Integer))
  *InputValues = AllocateMemory(ListSize(Feeds()) * SizeOf(Integer))
  *OutputNames = AllocateMemory(ListSize(TraceNames()) * SizeOf(Integer))
  *Outputs = AllocateMemory(ListSize(TraceNames()) * SizeOf(Integer))
  If *InputNames = 0 Or *InputValues = 0 Or *OutputNames = 0 Or *Outputs = 0
    PmoTraceFail("cannot allocate native trace bindings") : Goto PmoTraceShapesFailed
  EndIf
  If PmoTraceCreateOrtValues(@State, Feeds(), *InputNames, *InputValues) = 0
    Goto PmoTraceShapesFailed
  EndIf
  Index = 0
  ForEach TraceNames()
    *Encoded = UTF8(TraceNames())
    If *Encoded = 0 : PmoTraceFail("cannot encode trace output name") : Goto PmoTraceShapesFailed : EndIf
    AddElement(OutputUtf8()) : OutputUtf8() = *Encoded
    PokeI(*OutputNames + Index * SizeOf(Integer), *Encoded)
    Index + 1
  Next
  Run = PmoOrtFunction(#PMO_ORT_RUN)
  If Run = 0 : PmoTraceFail("ONNX Runtime Run function is missing") : Goto PmoTraceShapesFailed : EndIf
  If PmoOrtCheck(Run(State\Session, 0, *InputNames, *InputValues, ListSize(Feeds()),
                      *OutputNames, ListSize(TraceNames()), *Outputs),
                 "native ONNX shape trace failed") = 0
    PmoTraceFail(PmoOrtError) : Goto PmoTraceShapesFailed
  EndIf

  ForEach *Model\Graph\Outputs() : Values(*Model\Graph\Outputs()\Name) = @*Model\Graph\Outputs() : Next
  ForEach *Model\Graph\Values() : Values(*Model\Graph\Values()\Name) = @*Model\Graph\Values() : Next
  Index = 0
  ForEach TraceNames()
    If FindMapElement(Values(), TraceNames()) = 0
      PmoTraceFail("trace returned an unknown value " + TraceNames()) : Goto PmoTraceShapesFailed
    EndIf
    *Value = Values()
    If PmoTraceApplyShape(*Value, PeekI(*Outputs + Index * SizeOf(Integer))) = 0
      Goto PmoTraceShapesFailed
    EndIf
    Index + 1
  Next
  Result = #True

  PmoTraceShapesFailed:
  ReleaseValue = PmoOrtFunction(#PMO_ORT_RELEASE_VALUE)
  If *Outputs And ReleaseValue
    For Index = 0 To ListSize(TraceNames()) - 1
      If PeekI(*Outputs + Index * SizeOf(Integer))
        ReleaseValue(PeekI(*Outputs + Index * SizeOf(Integer)))
      EndIf
    Next
  EndIf
  ForEach OutputUtf8() : FreeMemory(OutputUtf8()) : Next
  If *InputNames : FreeMemory(*InputNames) : EndIf
  If *InputValues : FreeMemory(*InputValues) : EndIf
  If *OutputNames : FreeMemory(*OutputNames) : EndIf
  If *Outputs : FreeMemory(*Outputs) : EndIf
  PmoTraceFreeFeeds(Feeds())
  PmoOrtCloseSession(@State)
  PmoOrtUnload()
  PmoProtoFree(@TraceModel)
  ProcedureReturn Result
EndProcedure
