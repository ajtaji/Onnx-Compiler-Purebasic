; ============================================================================
; onnx_rank_infer.pbi - rank propagation used to build checker-valid traces
; ----------------------------------------------------------------------------
; Runtime tracing resolves value-dependent extents, but ONNX graph outputs must
; have a declared rank before a runtime may load the temporary graph. This pass
; infers ranks only. It never invents extents and it fails if an operator's rank
; cannot be justified from its inputs and attributes.
; ============================================================================

Global PmoRankError.s

Procedure.i PmoRankFail(Message.s)
  If PmoRankError = "" : PmoRankError = Message : EndIf
  ProcedureReturn -1
EndProcedure

Procedure.i PmoRankAttributeInteger(*Node.PmoOnnxNode, Name.s, DefaultValue.i)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name
      ProcedureReturn *Node\Attributes()\IntegerValue
    EndIf
  Next
  ProcedureReturn DefaultValue
EndProcedure

Procedure.i PmoRankAttributeListCount(*Node.PmoOnnxNode, Name.s)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name
      ProcedureReturn ListSize(*Node\Attributes()\Integers())
    EndIf
  Next
  ProcedureReturn -1
EndProcedure

Procedure.s PmoRankInputName(*Node.PmoOnnxNode, Index.i)
  If Index < 0 Or SelectElement(*Node\Inputs(), Index) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Inputs()
EndProcedure

Procedure.i PmoRankOf(Name.s, Map Ranks.i(), *Known.Integer)
  If Name <> "" And FindMapElement(Ranks(), Name)
    *Known\i = #True
    ProcedureReturn Ranks()
  EndIf
  *Known\i = #False
  ProcedureReturn 0
EndProcedure

Procedure.i PmoRankTensorElementCount(Name.s, Map Ranks.i(), Map Extents.q())
  Protected Key.s
  Protected Rank.i
  If FindMapElement(Ranks(), Name) = 0 : ProcedureReturn -1 : EndIf
  Rank = Ranks()
  If Rank = 0 : ProcedureReturn 1 : EndIf
  If Rank <> 1 : ProcedureReturn -1 : EndIf
  Key = Name + Chr(1) + "0"
  If FindMapElement(Extents(), Key) = 0 Or Extents() < 0 : ProcedureReturn -1 : EndIf
  ProcedureReturn Extents()
EndProcedure

Procedure.q PmoRankTensorInteger(Name.s, Index.i, Map TensorPointers.i(), *Known.Integer)
  Protected *Tensor.PmoOnnxTensor
  Protected *At
  *Known\i = #False
  If FindMapElement(TensorPointers(), Name) = 0 : ProcedureReturn 0 : EndIf
  *Tensor = TensorPointers()
  If Index < 0 : ProcedureReturn 0 : EndIf
  Select *Tensor\DataType
    Case 7 ; int64
      If *Tensor\Raw\Bytes >= (Index + 1) * 8
        *Known\i = #True : ProcedureReturn PeekQ(*Tensor\Raw\Data + Index * 8)
      EndIf
      If SelectElement(*Tensor\Int64Data(), Index)
        *Known\i = #True : ProcedureReturn *Tensor\Int64Data()
      EndIf
    Case 6 ; int32
      If *Tensor\Raw\Bytes >= (Index + 1) * 4
        *Known\i = #True : ProcedureReturn PeekL(*Tensor\Raw\Data + Index * 4)
      EndIf
      If SelectElement(*Tensor\Int32Data(), Index)
        *Known\i = #True : ProcedureReturn *Tensor\Int32Data()
      EndIf
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure PmoRankPropagateExtent(*Node.PmoOnnxNode, OutputName.s,
                                 Map Ranks.i(), Map Extents.q(), Map TensorPointers.i())
  Protected InputName.s
  Protected Key.s
  Protected Extent.q
  Protected Total.q
  Protected StartValue.q
  Protected EndValue.q
  Protected StepValue.q = 1
  Protected AxisValue.q = 0
  Protected Known.Integer
  Protected Known2.Integer
  Protected Index.i
  Protected Axis.i
  If OutputName = "" : ProcedureReturn : EndIf
  Select *Node\Operation
    Case "Shape"
      InputName = PmoRankInputName(*Node, 0)
      If FindMapElement(Ranks(), InputName)
        Extents(OutputName + Chr(1) + "0") = Ranks()
      EndIf

    Case "Unsqueeze"
      InputName = PmoRankInputName(*Node, 0)
      If FindMapElement(Ranks(), InputName) And Ranks() = 0 And
         FindMapElement(Ranks(), OutputName) And Ranks() = 1
        Extents(OutputName + Chr(1) + "0") = 1
      EndIf

    Case "Concat"
      Axis = PmoRankAttributeInteger(*Node, "axis", 0)
      If Axis = 0 And FindMapElement(Ranks(), OutputName) And Ranks() = 1
        Total = 0
        ForEach *Node\Inputs()
          Key = *Node\Inputs() + Chr(1) + "0"
          If FindMapElement(Extents(), Key) = 0 Or Extents() < 0
            Total = -1 : Break
          EndIf
          Total + Extents()
        Next
        If Total >= 0 : Extents(OutputName + Chr(1) + "0") = Total : EndIf
      EndIf

    Case "Slice"
      InputName = PmoRankInputName(*Node, 0)
      Key = InputName + Chr(1) + "0"
      If FindMapElement(Ranks(), InputName) And Ranks() = 1 And
         FindMapElement(Extents(), Key) And Extents() >= 0
        Extent = Extents()
        StartValue = PmoRankTensorInteger(PmoRankInputName(*Node, 1), 0, TensorPointers(), @Known)
        EndValue = PmoRankTensorInteger(PmoRankInputName(*Node, 2), 0, TensorPointers(), @Known2)
        If Known\i And Known2\i
          If PmoRankInputName(*Node, 3) <> ""
            AxisValue = PmoRankTensorInteger(PmoRankInputName(*Node, 3), 0, TensorPointers(), @Known)
            If Known\i = 0 : ProcedureReturn : EndIf
          EndIf
          If PmoRankInputName(*Node, 4) <> ""
            StepValue = PmoRankTensorInteger(PmoRankInputName(*Node, 4), 0, TensorPointers(), @Known)
            If Known\i = 0 : ProcedureReturn : EndIf
          EndIf
          If AxisValue < 0 : AxisValue + 1 : EndIf
          If AxisValue = 0 And StepValue > 0
            If StartValue < 0 : StartValue + Extent : EndIf
            If EndValue < 0 : EndValue + Extent : EndIf
            If StartValue < 0 : StartValue = 0 : EndIf
            If StartValue > Extent : StartValue = Extent : EndIf
            If EndValue < 0 : EndValue = 0 : EndIf
            If EndValue > Extent : EndValue = Extent : EndIf
            If EndValue <= StartValue
              Extents(OutputName + Chr(1) + "0") = 0
            Else
              Extents(OutputName + Chr(1) + "0") = (EndValue - StartValue + StepValue - 1) / StepValue
            EndIf
          EndIf
        EndIf
      EndIf
  EndSelect
EndProcedure

Procedure PmoRankRememberValue(*Value.PmoOnnxValue, Map Ranks.i(), Map Extents.q(),
                               Map ValuePointers.i())
  Protected Index.i
  If *Value = 0 Or *Value\Name = "" : ProcedureReturn : EndIf
  ValuePointers(*Value\Name) = *Value
  If *Value\HasShape
    Ranks(*Value\Name) = ListSize(*Value\Dims())
    Index = 0
    ForEach *Value\Dims()
      If *Value\Dims()\HasValue
        Extents(*Value\Name + Chr(1) + Str(Index)) = *Value\Dims()\Value
      Else
        Extents(*Value\Name + Chr(1) + Str(Index)) = -1
      EndIf
      Index + 1
    Next
  EndIf
EndProcedure

Procedure.i PmoRankInferElementType(*Node.PmoOnnxNode, Map Types.i())
  Protected Name.s
  Protected Type.i
  If *Node = 0 : ProcedureReturn 0 : EndIf
  Select *Node\Operation
    Case "Shape", "NonZero"
      ProcedureReturn 7 ; INT64
    Case "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And"
      ProcedureReturn 9 ; BOOL
    Case "Cast"
      ProcedureReturn PmoRankAttributeInteger(*Node, "to", 0)
    Case "Where"
      Name = PmoRankInputName(*Node, 1)
    Default
      Name = PmoRankInputName(*Node, 0)
  EndSelect
  If Name <> "" And FindMapElement(Types(), Name) : Type = Types() : EndIf
  ProcedureReturn Type
EndProcedure

Procedure.i PmoRankInferNode(*Node.PmoOnnxNode, Map Ranks.i(), Map Extents.q())
  Protected Operation.s = *Node\Operation
  Protected Name0.s = PmoRankInputName(*Node, 0)
  Protected Name1.s = PmoRankInputName(*Node, 1)
  Protected Name2.s = PmoRankInputName(*Node, 2)
  Protected Known0.Integer
  Protected Known1.Integer
  Protected Known2.Integer
  Protected Rank0.i = PmoRankOf(Name0, Ranks(), @Known0)
  Protected Rank1.i = PmoRankOf(Name1, Ranks(), @Known1)
  Protected Rank2.i = PmoRankOf(Name2, Ranks(), @Known2)
  Protected Count.i
  Protected KeepDims.i
  Select Operation
    Case "STFT"
      If Known0\i : ProcedureReturn 4 : EndIf

    Case "Reshape"
      Count = PmoRankTensorElementCount(Name1, Ranks(), Extents())
      If Count >= 0 : ProcedureReturn Count : EndIf

    Case "Unsqueeze"
      Count = PmoRankAttributeListCount(*Node, "axes")
      If Count < 0 : Count = PmoRankTensorElementCount(Name1, Ranks(), Extents()) : EndIf
      If Known0\i And Count >= 0 : ProcedureReturn Rank0 + Count : EndIf

    Case "Squeeze"
      Count = PmoRankAttributeListCount(*Node, "axes")
      If Count < 0 : Count = PmoRankTensorElementCount(Name1, Ranks(), Extents()) : EndIf
      If Known0\i And Count >= 0 And Count <= Rank0 : ProcedureReturn Rank0 - Count : EndIf

    Case "Gather"
      If Known0\i And Known1\i : ProcedureReturn Rank0 + Rank1 - 1 : EndIf

    Case "ReduceMean", "ReduceSum"
      If Known0\i
        KeepDims = PmoRankAttributeInteger(*Node, "keepdims", 1)
        If KeepDims : ProcedureReturn Rank0 : EndIf
        Count = PmoRankAttributeListCount(*Node, "axes")
        If Count < 0 : Count = PmoRankTensorElementCount(Name1, Ranks(), Extents()) : EndIf
        If Count >= 0 And Count <= Rank0 : ProcedureReturn Rank0 - Count : EndIf
      EndIf

    Case "Shape"
      ProcedureReturn 1

    Case "NonZero"
      ProcedureReturn 2

    Case "Range", "CumSum"
      ProcedureReturn 1

    Case "MatMul"
      If Known0\i And Known1\i
        If Rank0 = 1 And Rank1 = 1 : ProcedureReturn 0 : EndIf
        If Rank0 > Rank1 : ProcedureReturn Rank0 : EndIf
        ProcedureReturn Rank1
      EndIf

    Case "Gemm"
      ProcedureReturn 2

    Case "LSTM"
      ; Kokoro's LSTM outputs already carry shapes. A missing LSTM rank should
      ; be handled with its output index, not guessed by this single-rank API.
      ProcedureReturn PmoRankFail("LSTM output rank metadata is missing")

    Case "Add", "Sub", "Mul", "Div", "Pow", "And", "Equal", "Greater",
         "GreaterOrEqual", "Less", "LessOrEqual", "Where"
      If Known0\i
        Count = Rank0
        If Known1\i And Rank1 > Count : Count = Rank1 : EndIf
        If Known2\i And Rank2 > Count : Count = Rank2 : EndIf
        ProcedureReturn Count
      EndIf

    Case "Abs", "Atan", "BatchNormalization", "Cast", "Clip", "Cos", "Exp", "Floor",
         "Identity", "LeakyRelu", "Neg", "Pad", "Relu", "Round", "Sigmoid", "Sin",
         "Slice", "Softmax", "Sqrt", "Tanh",
         "Transpose", "Conv", "ConvTranspose", "LayerNormalization",
         "ScatterND"
      If Known0\i : ProcedureReturn Rank0 : EndIf

    Case "Flatten"
      If Known0\i : ProcedureReturn 2 : EndIf

    Case "Concat"
      If Known0\i : ProcedureReturn Rank0 : EndIf

    Case "Expand", "ConstantOfShape"
      Count = PmoRankTensorElementCount(Name1, Ranks(), Extents())
      If Operation = "ConstantOfShape"
        Count = PmoRankTensorElementCount(Name0, Ranks(), Extents())
      EndIf
      If Count >= 0 : ProcedureReturn Count : EndIf

    Case "Resize"
      If Known0\i : ProcedureReturn Rank0 : EndIf
  EndSelect
  ProcedureReturn PmoRankFail("cannot infer rank for " + Operation + " node " + *Node\Name)
EndProcedure

Procedure.i PmoInferMissingRanks(*Model.PmoOnnxModel)
  NewMap Ranks.i()
  NewMap Extents.q()
  NewMap ValuePointers.i()
  NewMap TensorPointers.i()
  NewMap Types.i()
  Protected Index.i
  Protected Rank.i
  Protected *Value.PmoOnnxValue
  Protected OutputName.s
  Protected Missing.i
  Protected OutputIndex.i
  Protected OutputRank.i
  PmoRankError = ""

  ForEach *Model\Graph\Inputs()
    PmoRankRememberValue(@*Model\Graph\Inputs(), Ranks(), Extents(), ValuePointers())
    Types(*Model\Graph\Inputs()\Name) = *Model\Graph\Inputs()\ElementType
  Next
  ForEach *Model\Graph\Outputs()
    PmoRankRememberValue(@*Model\Graph\Outputs(), Ranks(), Extents(), ValuePointers())
    Types(*Model\Graph\Outputs()\Name) = *Model\Graph\Outputs()\ElementType
  Next
  ForEach *Model\Graph\Values()
    PmoRankRememberValue(@*Model\Graph\Values(), Ranks(), Extents(), ValuePointers())
    Types(*Model\Graph\Values()\Name) = *Model\Graph\Values()\ElementType
  Next
  ForEach *Model\Graph\Initializers()
    TensorPointers(*Model\Graph\Initializers()\Name) = @*Model\Graph\Initializers()
    Types(*Model\Graph\Initializers()\Name) = *Model\Graph\Initializers()\DataType
    Ranks(*Model\Graph\Initializers()\Name) = ListSize(*Model\Graph\Initializers()\Dims())
    Index = 0
    ForEach *Model\Graph\Initializers()\Dims()
      Extents(*Model\Graph\Initializers()\Name + Chr(1) + Str(Index)) = *Model\Graph\Initializers()\Dims()
      Index + 1
    Next
  Next

  ForEach *Model\Graph\Nodes()
    Missing = #False
    ForEach *Model\Graph\Nodes()\Outputs()
      OutputName = *Model\Graph\Nodes()\Outputs()
      If OutputName <> ""
        If FindMapElement(ValuePointers(), OutputName) = 0
          Missing = #True : Break
        EndIf
        *Value = ValuePointers()
        If *Value\HasShape = 0 : Missing = #True : Break : EndIf
      EndIf
    Next
    If Missing
      If *Model\Graph\Nodes()\Operation = "LSTM"
        Rank = 4
      Else
        Rank = PmoRankInferNode(@*Model\Graph\Nodes(), Ranks(), Extents())
      EndIf
      If Rank < 0 : ProcedureReturn #False : EndIf
    Else
      Rank = -1
    EndIf
    OutputIndex = 0
    ForEach *Model\Graph\Nodes()\Outputs()
      OutputName = *Model\Graph\Nodes()\Outputs()
      If OutputName = "" : OutputIndex + 1 : Continue : EndIf
      OutputRank = Rank
      If *Model\Graph\Nodes()\Operation = "LSTM" And OutputIndex > 0 : OutputRank = 3 : EndIf
      If FindMapElement(ValuePointers(), OutputName) = 0
        AddElement(*Model\Graph\Values())
        *Model\Graph\Values()\Name = OutputName
        *Model\Graph\Values()\ElementType = PmoRankInferElementType(@*Model\Graph\Nodes(), Types())
        If *Model\Graph\Values()\ElementType = 0
          ProcedureReturn PmoRankFail("cannot infer element type for " + *Model\Graph\Nodes()\Operation + " output " + OutputName)
        EndIf
        *Model\Graph\Values()\HasTensorType = #True
        *Model\Graph\Values()\HasShape = #True
        For Index = 1 To OutputRank : AddElement(*Model\Graph\Values()\Dims()) : Next
        ValuePointers(OutputName) = @*Model\Graph\Values()
        Types(OutputName) = *Model\Graph\Values()\ElementType
      EndIf
      If FindMapElement(ValuePointers(), OutputName)
        *Value = ValuePointers()
        If *Value\HasShape = 0
          *Value\HasShape = #True
          ClearList(*Value\Dims())
          For Index = 1 To OutputRank
            AddElement(*Value\Dims())
          Next
        EndIf
        Ranks(OutputName) = ListSize(*Value\Dims())
        Types(OutputName) = *Value\ElementType
        For Index = 0 To Ranks(OutputName) - 1
          If FindMapElement(Extents(), OutputName + Chr(1) + Str(Index)) = 0
            Extents(OutputName + Chr(1) + Str(Index)) = -1
          EndIf
        Next
        PmoRankPropagateExtent(@*Model\Graph\Nodes(), OutputName, Ranks(), Extents(), TensorPointers())
      EndIf
      OutputIndex + 1
    Next
  Next
  ProcedureReturn #True
EndProcedure
