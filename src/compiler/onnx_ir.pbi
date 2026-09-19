; ============================================================================
; onnx_ir.pbi - checked target-neutral graph IR, folding and arena planning
; ============================================================================

#PMO_WEIGHT_HEADER_BYTES = 128
#PMO_WEIGHT_ALIGNMENT = 64
#PMO_CONSTANT_FOLD_LIMIT = 67108864

Structure PmoIrValue
  Name.s
  ElementType.i
  List Dims.q()
  Elements.q
  Bytes.q
  IsConstant.i
  Alias.s
  Start.i
  Finish.i
  Offset.q
  AllocationBytes.q
EndStructure

Structure PmoIrConstant
  Name.s
  ElementType.i
  List Dims.q()
  Elements.q
  Bytes.q
  *Data
  OwnsData.i
  PackedOffset.q
  StorageKind.i
  *StorageData
  DecodedOffset.q
EndStructure

Structure PmoIrQuantWeight
  Name.s
  ScaleName.s
  ScaleCount.i
  ScaleMode.i
  ScaleStride.q
  OriginalBytes.q
  QuantizedBytes.q
EndStructure

Structure PmoIrNodeRef
  Index.i
  *Node.PmoOnnxNode
EndStructure

Structure PmoIrInterval
  *Value.PmoIrValue
EndStructure

Structure PmoIrBlock
  Offset.q
  Bytes.q
EndStructure

Structure PmoIrModel
  *Source.PmoOnnxModel
  List Values.PmoIrValue()
  Map ValueByName.i()
  List Constants.PmoIrConstant()
  Map ConstantByName.i()
  List QuantWeights.PmoIrQuantWeight()
  Map QuantByName.i()
  List Nodes.PmoIrNodeRef()
  List Inputs.s()
  List Outputs.s()
  List Checkpoints.s()
  Map Aliases.s()
  ArenaBytes.q
  DecodedWeightBytes.q
  ReducedWeightCount.i
  Int8ScratchOffset.q
  Int8ScratchBytes.q
  StftScratchOffset.q
  StftScratchBytes.q
  StftScratchComplex.q
  WeightBytes.q
  WeightDataBytes.q
EndStructure

Global PmoIrError.s

Procedure.i PmoIrFail(Message.s)
  If PmoIrError = "" : PmoIrError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.q PmoIrAlign(Value.q, Alignment.q = #PMO_WEIGHT_ALIGNMENT)
  ProcedureReturn (Value + Alignment - 1) & ~(Alignment - 1)
EndProcedure

Procedure.i PmoIrElementBytes(ElementType.i)
  Select ElementType
    Case 1, 6, 12 : ProcedureReturn 4
    Case 2, 3, 9 : ProcedureReturn 1
    Case 4, 5 : ProcedureReturn 2
    Case 7, 11, 13 : ProcedureReturn 8
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmoIrCompileTimeInput(Operation.s, Position.i)
  Select Operation
    Case "Squeeze", "Unsqueeze", "ReduceMean", "ReduceSum", "CumSum", "ReduceMax", "ReduceProd", "TopK"
      ProcedureReturn Bool(Position = 1)
    Case "ConstantOfShape"
      ProcedureReturn Bool(Position = 0)
    Case "Resize"
      ProcedureReturn Bool(Position >= 1 And Position <= 3)
    Case "Pad"
      ; pads and axes shape the output; constant_value is data, read at run time.
      ProcedureReturn Bool(Position = 1 Or Position = 3)
    Case "STFT"
      ProcedureReturn Bool(Position = 1 Or Position = 3)
  EndSelect
  ProcedureReturn #False
EndProcedure

Procedure.i PmoIrShapeProduct(List Dims.q(), ElementBytes.i, *Elements.Quad, *Bytes.Quad)
  Protected Count.q = 1
  If ElementBytes <= 0 : ProcedureReturn PmoIrFail("unsupported tensor element type") : EndIf
  ForEach Dims()
    If Dims() < 0 : ProcedureReturn PmoIrFail("negative tensor extent") : EndIf
    If Dims() > 0 And Count > $7FFFFFFFFFFFFFFF / Dims()
      ProcedureReturn PmoIrFail("tensor element count overflows 64 bits")
    EndIf
    Count * Dims()
  Next
  If Count > $7FFFFFFFFFFFFFFF / ElementBytes
    ProcedureReturn PmoIrFail("tensor byte count overflows 64 bits")
  EndIf
  *Elements\q = Count
  *Bytes\q = Count * ElementBytes
  ProcedureReturn #True
EndProcedure

Procedure PmoIrFree(*Ir.PmoIrModel)
  If *Ir = 0 : ProcedureReturn : EndIf
  ForEach *Ir\Constants()
    If *Ir\Constants()\StorageData : FreeMemory(*Ir\Constants()\StorageData) : EndIf
    If *Ir\Constants()\OwnsData And *Ir\Constants()\Data
      FreeMemory(*Ir\Constants()\Data)
    EndIf
  Next
  ClearStructure(*Ir, PmoIrModel)
  InitializeStructure(*Ir, PmoIrModel)
EndProcedure

Procedure.i PmoIrAddValue(*Ir.PmoIrModel, *Value.PmoOnnxValue)
  Protected Elements.Quad
  Protected Bytes.Quad
  Protected ElementBytes.i
  Protected *Old.PmoIrValue
  If *Value = 0 Or *Value\Name = "" Or *Value\HasTensorType = 0 Or *Value\HasShape = 0
    ProcedureReturn PmoIrFail("graph value is missing its tensor type or shape")
  EndIf
  ElementBytes = PmoIrElementBytes(*Value\ElementType)
  If ElementBytes = 0
    ProcedureReturn PmoIrFail("tensor " + *Value\Name + " uses unsupported ONNX type " +
                              Str(*Value\ElementType))
  EndIf
  NewList Shape.q()
  ForEach *Value\Dims()
    If *Value\Dims()\HasValue = 0
      ProcedureReturn PmoIrFail("tensor " + *Value\Name + " still has a dynamic extent")
    EndIf
    AddElement(Shape()) : Shape() = *Value\Dims()\Value
  Next
  If PmoIrShapeProduct(Shape(), ElementBytes, @Elements, @Bytes) = 0 : ProcedureReturn #False : EndIf
  If FindMapElement(*Ir\ValueByName(), *Value\Name)
    *Old = *Ir\ValueByName()
    If *Old\ElementType <> *Value\ElementType Or *Old\Elements <> Elements\q Or
       ListSize(*Old\Dims()) <> ListSize(Shape())
      ProcedureReturn PmoIrFail("tensor " + *Value\Name + " has conflicting declarations")
    EndIf
    FirstElement(Shape())
    ForEach *Old\Dims()
      If *Old\Dims() <> Shape()
        ProcedureReturn PmoIrFail("tensor " + *Value\Name + " has conflicting shapes")
      EndIf
      NextElement(Shape())
    Next
    ProcedureReturn #True
  EndIf
  AddElement(*Ir\Values())
  *Ir\Values()\Name = *Value\Name
  *Ir\Values()\ElementType = *Value\ElementType
  *Ir\Values()\Elements = Elements\q
  *Ir\Values()\Bytes = Bytes\q
  *Ir\Values()\Start = $7FFFFFFF
  *Ir\Values()\Finish = -1
  ForEach Shape()
    AddElement(*Ir\Values()\Dims()) : *Ir\Values()\Dims() = Shape()
  Next
  *Ir\ValueByName(*Value\Name) = @*Ir\Values()
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrAddTensorValue(*Ir.PmoIrModel, *Tensor.PmoOnnxTensor)
  Protected Elements.Quad
  Protected Bytes.Quad
  Protected ElementBytes.i = PmoIrElementBytes(*Tensor\DataType)
  Protected *Old.PmoIrValue
  If *Tensor\Name = "" Or ElementBytes = 0
    ProcedureReturn PmoIrFail("initializer has no name or uses an unsupported element type")
  EndIf
  If PmoIrShapeProduct(*Tensor\Dims(), ElementBytes, @Elements, @Bytes) = 0 : ProcedureReturn #False : EndIf
  If FindMapElement(*Ir\ValueByName(), *Tensor\Name)
    *Old = *Ir\ValueByName()
    If *Old\ElementType <> *Tensor\DataType Or *Old\Bytes <> Bytes\q
      ProcedureReturn PmoIrFail("initializer " + *Tensor\Name + " disagrees with its graph declaration")
    EndIf
  Else
    AddElement(*Ir\Values())
    *Old = @*Ir\Values()
    *Old\Name = *Tensor\Name
    *Old\ElementType = *Tensor\DataType
    *Old\Elements = Elements\q : *Old\Bytes = Bytes\q
    *Old\Start = $7FFFFFFF : *Old\Finish = -1
    ForEach *Tensor\Dims()
      AddElement(*Old\Dims()) : *Old\Dims() = *Tensor\Dims()
    Next
    *Ir\ValueByName(*Tensor\Name) = *Old
  EndIf
  *Old\IsConstant = #True
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrMaterializeTensor(*Tensor.PmoOnnxTensor, *Constant.PmoIrConstant)
  Protected Index.i
  Protected *At
  If *Tensor\DataLocation <> 0
    ProcedureReturn PmoIrFail("external tensor " + *Tensor\Name + " must be resolved before lowering")
  EndIf
  If *Tensor\Raw\Bytes > 0
    If *Tensor\Raw\Bytes <> *Constant\Bytes
      ProcedureReturn PmoIrFail("initializer " + *Tensor\Name + " raw byte count disagrees with its shape")
    EndIf
    *Constant\Data = *Tensor\Raw\Data
    ProcedureReturn #True
  EndIf
  ; A zero-element initializer (an exporter's empty roi or axes) holds no data;
  ; whether an operator accepts it is that operator's check (forum 861).
  If *Constant\Elements = 0
    *Constant\Data = 0 : *Constant\OwnsData = #False
    ProcedureReturn #True
  EndIf
  *Constant\Data = AllocateMemory(*Constant\Bytes)
  If *Constant\Data = 0 : ProcedureReturn PmoIrFail("cannot materialize initializer " + *Tensor\Name) : EndIf
  *Constant\OwnsData = #True
  Select *Tensor\DataType
    Case 1
      If ListSize(*Tensor\FloatData()) <> *Constant\Elements : Goto PmoIrMaterializeMismatch : EndIf
      Index = 0 : ForEach *Tensor\FloatData() : PokeF(*Constant\Data + Index * 4, *Tensor\FloatData()) : Index + 1 : Next
    Case 6
      If ListSize(*Tensor\Int32Data()) <> *Constant\Elements : Goto PmoIrMaterializeMismatch : EndIf
      Index = 0 : ForEach *Tensor\Int32Data() : PokeL(*Constant\Data + Index * 4, *Tensor\Int32Data()) : Index + 1 : Next
    Case 7
      If ListSize(*Tensor\Int64Data()) <> *Constant\Elements : Goto PmoIrMaterializeMismatch : EndIf
      Index = 0 : ForEach *Tensor\Int64Data() : PokeQ(*Constant\Data + Index * 8, *Tensor\Int64Data()) : Index + 1 : Next
    Case 11
      If ListSize(*Tensor\DoubleData()) <> *Constant\Elements : Goto PmoIrMaterializeMismatch : EndIf
      Index = 0 : ForEach *Tensor\DoubleData() : PokeD(*Constant\Data + Index * 8, *Tensor\DoubleData()) : Index + 1 : Next
    Case 13
      If ListSize(*Tensor\UInt64Data()) <> *Constant\Elements : Goto PmoIrMaterializeMismatch : EndIf
      Index = 0 : ForEach *Tensor\UInt64Data() : PokeQ(*Constant\Data + Index * 8, *Tensor\UInt64Data()) : Index + 1 : Next
    Default
      Goto PmoIrMaterializeMismatch
  EndSelect
  ProcedureReturn #True

  PmoIrMaterializeMismatch:
  If *Constant\Data : FreeMemory(*Constant\Data) : EndIf
  *Constant\Data = 0 : *Constant\OwnsData = #False
  ProcedureReturn PmoIrFail("initializer " + *Tensor\Name + " has incomplete typed data")
EndProcedure

Procedure.i PmoIrAddInitializer(*Ir.PmoIrModel, *Tensor.PmoOnnxTensor)
  Protected *Value.PmoIrValue
  If PmoIrAddTensorValue(*Ir, *Tensor) = 0 : ProcedureReturn #False : EndIf
  *Value = *Ir\ValueByName(*Tensor\Name)
  AddElement(*Ir\Constants())
  *Ir\Constants()\Name = *Tensor\Name
  *Ir\Constants()\ElementType = *Tensor\DataType
  *Ir\Constants()\Elements = *Value\Elements
  *Ir\Constants()\Bytes = *Value\Bytes
  ForEach *Tensor\Dims()
    AddElement(*Ir\Constants()\Dims()) : *Ir\Constants()\Dims() = *Tensor\Dims()
  Next
  If PmoIrMaterializeTensor(*Tensor, @*Ir\Constants()) = 0 : ProcedureReturn #False : EndIf
  *Ir\ConstantByName(*Tensor\Name) = @*Ir\Constants()
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrPrepare(*Model.PmoOnnxModel, *Ir.PmoIrModel)
  NewMap Initializers.i()
  If *Model = 0 Or *Ir = 0 : ProcedureReturn PmoIrFail("invalid ONNX IR source") : EndIf
  PmoIrFree(*Ir) : PmoIrError = "" : *Ir\Source = *Model
  ForEach *Model\Graph\Inputs()
    If PmoIrAddValue(*Ir, @*Model\Graph\Inputs()) = 0 : Goto PmoIrPrepareFailed : EndIf
  Next
  ForEach *Model\Graph\Values()
    If PmoIrAddValue(*Ir, @*Model\Graph\Values()) = 0 : Goto PmoIrPrepareFailed : EndIf
  Next
  ForEach *Model\Graph\Outputs()
    If PmoIrAddValue(*Ir, @*Model\Graph\Outputs()) = 0 : Goto PmoIrPrepareFailed : EndIf
  Next
  ForEach *Model\Graph\Initializers()
    Initializers(*Model\Graph\Initializers()\Name) = #True
    If PmoIrAddInitializer(*Ir, @*Model\Graph\Initializers()) = 0 : Goto PmoIrPrepareFailed : EndIf
  Next
  ForEach *Model\Graph\Inputs()
    If FindMapElement(Initializers(), *Model\Graph\Inputs()\Name) = 0
      AddElement(*Ir\Inputs()) : *Ir\Inputs() = *Model\Graph\Inputs()\Name
    EndIf
  Next
  ForEach *Model\Graph\Outputs()
    AddElement(*Ir\Outputs()) : *Ir\Outputs() = *Model\Graph\Outputs()\Name
  Next
  ProcedureReturn #True

  PmoIrPrepareFailed:
  PmoIrFree(*Ir)
  ProcedureReturn #False
EndProcedure

Procedure.i PmoIrNewConstant(*Ir.PmoIrModel, Name.s, ElementType.i, List Dims.q(),
                             *Data, Bytes.q)
  Protected Elements.Quad
  Protected Expected.Quad
  Protected ElementBytes.i = PmoIrElementBytes(ElementType)
  Protected *Value.PmoIrValue
  If FindMapElement(*Ir\ConstantByName(), Name)
    ProcedureReturn #True
  EndIf
  If PmoIrShapeProduct(Dims(), ElementBytes, @Elements, @Expected) = 0 : ProcedureReturn #False : EndIf
  If Expected\q <> Bytes : ProcedureReturn PmoIrFail("folded tensor " + Name + " byte count disagrees") : EndIf
  If FindMapElement(*Ir\ValueByName(), Name) = 0
    ProcedureReturn PmoIrFail("folded tensor " + Name + " has no graph value")
  EndIf
  *Value = *Ir\ValueByName()
  If *Value\ElementType <> ElementType Or *Value\Bytes <> Bytes
    ProcedureReturn PmoIrFail("folded tensor " + Name + " disagrees with inferred type or shape")
  EndIf
  AddElement(*Ir\Constants())
  *Ir\Constants()\Name = Name
  *Ir\Constants()\ElementType = ElementType
  *Ir\Constants()\Elements = Elements\q
  *Ir\Constants()\Bytes = Bytes
  ; A folded tensor with no elements (Shape with start at or past end) holds no
  ; data, the same as a zero-element initializer (forum 861).
  If Bytes > 0
    *Ir\Constants()\Data = AllocateMemory(Bytes)
    If *Ir\Constants()\Data = 0 : ProcedureReturn PmoIrFail("cannot allocate folded tensor " + Name) : EndIf
    *Ir\Constants()\OwnsData = #True
    CopyMemory(*Data, *Ir\Constants()\Data, Bytes)
  EndIf
  ForEach Dims()
    AddElement(*Ir\Constants()\Dims()) : *Ir\Constants()\Dims() = Dims()
  Next
  *Ir\ConstantByName(Name) = @*Ir\Constants()
  *Value\IsConstant = #True
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrFoldShape(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected InputName.s
  Protected OutputName.s
  Protected *Value.PmoIrValue
  Protected StartAxis.i = 0
  Protected EndAxis.i
  Protected Index.i
  Protected *Data
  NewList Shape.q()
  If SelectElement(*Node\Inputs(), 0) = 0 Or SelectElement(*Node\Outputs(), 0) = 0 : ProcedureReturn #False : EndIf
  InputName = *Node\Inputs() : OutputName = *Node\Outputs()
  If FindMapElement(*Ir\ValueByName(), InputName) = 0 : ProcedureReturn #False : EndIf
  *Value = *Ir\ValueByName()
  EndAxis = ListSize(*Value\Dims())
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "start" : StartAxis = *Node\Attributes()\IntegerValue : EndIf
    If *Node\Attributes()\Name = "end" : EndAxis = *Node\Attributes()\IntegerValue : EndIf
  Next
  If StartAxis < 0 : StartAxis + ListSize(*Value\Dims()) : EndIf
  If EndAxis < 0 : EndAxis + ListSize(*Value\Dims()) : EndIf
  If StartAxis < 0 : StartAxis = 0 : EndIf
  If EndAxis > ListSize(*Value\Dims()) : EndAxis = ListSize(*Value\Dims()) : EndIf
  If EndAxis < StartAxis : EndAxis = StartAxis : EndIf
  AddElement(Shape()) : Shape() = EndAxis - StartAxis
  *Data = AllocateMemory((EndAxis - StartAxis) * 8)
  If *Data = 0 And EndAxis > StartAxis : ProcedureReturn PmoIrFail("cannot fold Shape") : EndIf
  Index = 0
  SelectElement(*Value\Dims(), StartAxis)
  While Index < EndAxis - StartAxis
    PokeQ(*Data + Index * 8, *Value\Dims())
    NextElement(*Value\Dims()) : Index + 1
  Wend
  If PmoIrNewConstant(*Ir, OutputName, 7, Shape(), *Data, (EndAxis - StartAxis) * 8) = 0
    If *Data : FreeMemory(*Data) : EndIf
    ProcedureReturn #False
  EndIf
  If *Data : FreeMemory(*Data) : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrConstantScalarI64(*Constant.PmoIrConstant, Index.i, *Ok.Integer)
  *Ok\i = #False
  If *Constant = 0 Or Index < 0 Or Index >= *Constant\Elements : ProcedureReturn 0 : EndIf
  Select *Constant\ElementType
    Case 7 : *Ok\i = #True : ProcedureReturn PeekQ(*Constant\Data + Index * 8)
    Case 6 : *Ok\i = #True : ProcedureReturn PeekL(*Constant\Data + Index * 4)
    Case 9 : *Ok\i = #True : ProcedureReturn PeekA(*Constant\Data + Index) & $FF
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmoIrFoldView(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected InputName.s
  Protected OutputName.s
  Protected *Input.PmoIrConstant
  Protected *Output.PmoIrValue
  NewList Shape.q()
  If SelectElement(*Node\Inputs(), 0) = 0 Or SelectElement(*Node\Outputs(), 0) = 0 : ProcedureReturn #False : EndIf
  InputName = *Node\Inputs() : OutputName = *Node\Outputs()
  If FindMapElement(*Ir\ConstantByName(), InputName) = 0 Or
     FindMapElement(*Ir\ValueByName(), OutputName) = 0 : ProcedureReturn #False : EndIf
  *Input = *Ir\ConstantByName(InputName)
  *Output = *Ir\ValueByName(OutputName)
  ForEach *Output\Dims() : AddElement(Shape()) : Shape() = *Output\Dims() : Next
  ProcedureReturn PmoIrNewConstant(*Ir, OutputName, *Output\ElementType, Shape(),
                                   *Input\Data, *Input\Bytes)
EndProcedure

Procedure.i PmoIrFoldGather(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected DataName.s
  Protected IndexName.s
  Protected OutputName.s
  Protected Axis.i
  Protected *Data.PmoIrConstant
  Protected *Indices.PmoIrConstant
  Protected *Output.PmoIrValue
  Protected *Raw
  Protected Selected.q
  Protected Ok.Integer
  Protected ItemBytes.i
  Protected Index.i
  NewList Shape.q()
  If ListSize(*Node\Inputs()) < 2 Or SelectElement(*Node\Inputs(), 0) = 0 : ProcedureReturn #False : EndIf
  DataName = *Node\Inputs() : SelectElement(*Node\Inputs(), 1) : IndexName = *Node\Inputs()
  SelectElement(*Node\Outputs(), 0) : OutputName = *Node\Outputs()
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "axis" : Axis = *Node\Attributes()\IntegerValue : EndIf
  Next
  If Axis <> 0 Or FindMapElement(*Ir\ConstantByName(), DataName) = 0 : ProcedureReturn #False : EndIf
  *Data = *Ir\ConstantByName(DataName)
  If FindMapElement(*Ir\ConstantByName(), IndexName) = 0 : ProcedureReturn #False : EndIf
  *Indices = *Ir\ConstantByName(IndexName)
  If *Indices\ElementType <> 7 Or ListSize(*Data\Dims()) <> 1 : ProcedureReturn #False : EndIf
  If FindMapElement(*Ir\ValueByName(), OutputName) = 0 : ProcedureReturn #False : EndIf
  *Output = *Ir\ValueByName(OutputName)
  ItemBytes = PmoIrElementBytes(*Data\ElementType)
  *Raw = AllocateMemory(*Output\Bytes)
  If *Raw = 0 And *Output\Bytes > 0 : ProcedureReturn PmoIrFail("cannot fold Gather") : EndIf
  For Index = 0 To *Indices\Elements - 1
    Selected = PmoIrConstantScalarI64(*Indices, Index, @Ok)
    If Ok\i = 0 : FreeMemory(*Raw) : ProcedureReturn #False : EndIf
    If Selected < 0 : Selected + *Data\Elements : EndIf
    If Selected < 0 Or Selected >= *Data\Elements
      FreeMemory(*Raw) : ProcedureReturn PmoIrFail("constant Gather index is outside its data tensor")
    EndIf
    CopyMemory(*Data\Data + Selected * ItemBytes, *Raw + Index * ItemBytes, ItemBytes)
  Next
  ForEach *Output\Dims() : AddElement(Shape()) : Shape() = *Output\Dims() : Next
  If PmoIrNewConstant(*Ir, OutputName, *Output\ElementType, Shape(), *Raw, *Output\Bytes) = 0
    If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn #False
  EndIf
  If *Raw : FreeMemory(*Raw) : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrFoldConcat(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected Axis.i
  Protected OutputName.s
  Protected *Output.PmoIrValue
  Protected *Input.PmoIrConstant
  Protected *Raw
  Protected At.q
  NewList Shape.q()
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "axis" : Axis = *Node\Attributes()\IntegerValue : EndIf
  Next
  If Axis <> 0 Or SelectElement(*Node\Outputs(), 0) = 0 : ProcedureReturn #False : EndIf
  OutputName = *Node\Outputs()
  If FindMapElement(*Ir\ValueByName(), OutputName) = 0 : ProcedureReturn #False : EndIf
  *Output = *Ir\ValueByName(OutputName)
  If ListSize(*Output\Dims()) <> 1 : ProcedureReturn #False : EndIf
  *Raw = AllocateMemory(*Output\Bytes)
  If *Raw = 0 And *Output\Bytes > 0 : ProcedureReturn PmoIrFail("cannot fold Concat") : EndIf
  ForEach *Node\Inputs()
    If FindMapElement(*Ir\ConstantByName(), *Node\Inputs()) = 0
      If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn #False
    EndIf
    *Input = *Ir\ConstantByName()
    If *Input\ElementType <> *Output\ElementType Or ListSize(*Input\Dims()) <> 1
      If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn #False
    EndIf
    CopyMemory(*Input\Data, *Raw + At, *Input\Bytes) : At + *Input\Bytes
  Next
  If At <> *Output\Bytes
    If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn PmoIrFail("constant Concat size disagrees")
  EndIf
  ForEach *Output\Dims() : AddElement(Shape()) : Shape() = *Output\Dims() : Next
  If PmoIrNewConstant(*Ir, OutputName, *Output\ElementType, Shape(), *Raw, *Output\Bytes) = 0
    If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn #False
  EndIf
  If *Raw : FreeMemory(*Raw) : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrFoldConstantOfShape(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected ShapeName.s
  Protected OutputName.s
  Protected *Output.PmoIrValue
  Protected *Raw
  Protected Fill.f
  Protected Index.q
  NewList Shape.q()
  If SelectElement(*Node\Inputs(), 0) = 0 Or SelectElement(*Node\Outputs(), 0) = 0 : ProcedureReturn #False : EndIf
  ShapeName = *Node\Inputs() : OutputName = *Node\Outputs()
  If FindMapElement(*Ir\ConstantByName(), ShapeName) = 0 Or
     FindMapElement(*Ir\ValueByName(), OutputName) = 0 : ProcedureReturn #False : EndIf
  *Output = *Ir\ValueByName(OutputName)
  If *Output\ElementType <> 1 : ProcedureReturn #False : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "value" And *Node\Attributes()\HasTensor
      If *Node\Attributes()\Tensor\Raw\Bytes >= 4
        Fill = PeekF(*Node\Attributes()\Tensor\Raw\Data)
      ElseIf FirstElement(*Node\Attributes()\Tensor\FloatData())
        Fill = *Node\Attributes()\Tensor\FloatData()
      EndIf
    EndIf
  Next
  *Raw = AllocateMemory(*Output\Bytes)
  If *Raw = 0 And *Output\Bytes > 0 : ProcedureReturn PmoIrFail("cannot fold ConstantOfShape") : EndIf
  If Fill <> 0.0
    For Index = 0 To *Output\Elements - 1 : PokeF(*Raw + Index * 4, Fill) : Next
  EndIf
  ForEach *Output\Dims() : AddElement(Shape()) : Shape() = *Output\Dims() : Next
  If PmoIrNewConstant(*Ir, OutputName, 1, Shape(), *Raw, *Output\Bytes) = 0
    If *Raw : FreeMemory(*Raw) : EndIf : ProcedureReturn #False
  EndIf
  If *Raw : FreeMemory(*Raw) : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoIrFoldRange(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected StartName.s
  Protected LimitName.s
  Protected DeltaName.s
  Protected OutputName.s
  Protected *Start.PmoIrConstant
  Protected *Limit.PmoIrConstant
  Protected *Delta.PmoIrConstant
  Protected *Output.PmoIrValue
  Protected StartI.q
  Protected LimitI.q
  Protected DeltaI.q
  Protected ValueI.q
  Protected StartF.f
  Protected LimitF.f
  Protected DeltaF.f
  Protected ValueF.f
  Protected Ok.Integer
  Protected Index.q
  Protected *Raw
  NewList Shape.q()
  If ListSize(*Node\Inputs()) <> 3 Or SelectElement(*Node\Inputs(), 0) = 0 Or
     SelectElement(*Node\Outputs(), 0) = 0 : ProcedureReturn #False : EndIf
  SelectElement(*Node\Inputs(), 0) : StartName = *Node\Inputs()
  SelectElement(*Node\Inputs(), 1) : LimitName = *Node\Inputs()
  SelectElement(*Node\Inputs(), 2) : DeltaName = *Node\Inputs()
  SelectElement(*Node\Outputs(), 0) : OutputName = *Node\Outputs()
  If FindMapElement(*Ir\ConstantByName(), StartName) = 0 : ProcedureReturn #False : EndIf
  *Start = *Ir\ConstantByName()
  If FindMapElement(*Ir\ConstantByName(), LimitName) = 0 : ProcedureReturn #False : EndIf
  *Limit = *Ir\ConstantByName()
  If FindMapElement(*Ir\ConstantByName(), DeltaName) = 0 : ProcedureReturn #False : EndIf
  *Delta = *Ir\ConstantByName()
  If FindMapElement(*Ir\ValueByName(), OutputName) = 0 : ProcedureReturn #False : EndIf
  *Output = *Ir\ValueByName()
  If *Start\ElementType <> *Output\ElementType Or *Limit\ElementType <> *Output\ElementType Or
     *Delta\ElementType <> *Output\ElementType Or *Start\Elements <> 1 Or
     *Limit\Elements <> 1 Or *Delta\Elements <> 1
    ProcedureReturn #False
  EndIf
  *Raw = AllocateMemory(*Output\Bytes)
  If *Raw = 0 And *Output\Bytes > 0 : ProcedureReturn PmoIrFail("cannot fold Range") : EndIf
  If *Output\ElementType = 7
    StartI = PmoIrConstantScalarI64(*Start, 0, @Ok) : If Ok\i = 0 : Goto PmoIrFoldRangeNo : EndIf
    LimitI = PmoIrConstantScalarI64(*Limit, 0, @Ok) : If Ok\i = 0 : Goto PmoIrFoldRangeNo : EndIf
    DeltaI = PmoIrConstantScalarI64(*Delta, 0, @Ok) : If Ok\i = 0 Or DeltaI = 0 : Goto PmoIrFoldRangeNo : EndIf
    ValueI = StartI
    For Index = 0 To *Output\Elements - 1
      If (DeltaI > 0 And ValueI >= LimitI) Or (DeltaI < 0 And ValueI <= LimitI)
        Goto PmoIrFoldRangeNo
      EndIf
      PokeQ(*Raw + Index * 8, ValueI) : ValueI + DeltaI
    Next
  ElseIf *Output\ElementType = 1
    StartF = PeekF(*Start\Data) : LimitF = PeekF(*Limit\Data) : DeltaF = PeekF(*Delta\Data)
    If DeltaF = 0.0 : Goto PmoIrFoldRangeNo : EndIf
    ValueF = StartF
    For Index = 0 To *Output\Elements - 1
      If (DeltaF > 0.0 And ValueF >= LimitF) Or (DeltaF < 0.0 And ValueF <= LimitF)
        Goto PmoIrFoldRangeNo
      EndIf
      PokeF(*Raw + Index * 4, ValueF) : ValueF + DeltaF
    Next
  Else
    Goto PmoIrFoldRangeNo
  EndIf
  ForEach *Output\Dims() : AddElement(Shape()) : Shape() = *Output\Dims() : Next
  If PmoIrNewConstant(*Ir, OutputName, *Output\ElementType, Shape(), *Raw, *Output\Bytes) = 0
    Goto PmoIrFoldRangeNo
  EndIf
  If *Raw : FreeMemory(*Raw) : EndIf
  ProcedureReturn #True

  PmoIrFoldRangeNo:
  If *Raw : FreeMemory(*Raw) : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoIrTryFold(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Select *Node\Operation
    Case "Shape" : ProcedureReturn PmoIrFoldShape(*Ir, *Node)
    Case "Gather" : ProcedureReturn PmoIrFoldGather(*Ir, *Node)
    Case "Unsqueeze", "Squeeze", "Reshape", "Identity" : ProcedureReturn PmoIrFoldView(*Ir, *Node)
    Case "Concat" : ProcedureReturn PmoIrFoldConcat(*Ir, *Node)
    Case "ConstantOfShape" : ProcedureReturn PmoIrFoldConstantOfShape(*Ir, *Node)
    Case "Range" : ProcedureReturn PmoIrFoldRange(*Ir, *Node)
  EndSelect
  ProcedureReturn #False
EndProcedure

Procedure.i PmoIrFoldConstants(*Ir.PmoIrModel)
  Protected Index.i
  Protected CanTry.i
  Protected Before.i
  PmoIrError = ""
  ClearList(*Ir\Nodes())
  ForEach *Ir\Source\Graph\Nodes()
    CanTry = Bool(*Ir\Source\Graph\Nodes()\Operation = "Shape")
    If CanTry = 0
      CanTry = #True
      ForEach *Ir\Source\Graph\Nodes()\Inputs()
        If *Ir\Source\Graph\Nodes()\Inputs() <> "" And
           FindMapElement(*Ir\ConstantByName(), *Ir\Source\Graph\Nodes()\Inputs()) = 0
          CanTry = #False : Break
        EndIf
      Next
    EndIf
    Before = MapSize(*Ir\ConstantByName())
    If CanTry : PmoIrTryFold(*Ir, @*Ir\Source\Graph\Nodes()) : EndIf
    If PmoIrError <> "" : ProcedureReturn #False : EndIf
    If MapSize(*Ir\ConstantByName()) = Before
      AddElement(*Ir\Nodes())
      *Ir\Nodes()\Index = Index
      *Ir\Nodes()\Node = @*Ir\Source\Graph\Nodes()
    EndIf
    Index + 1
  Next
  ProcedureReturn #True
EndProcedure

Procedure.s PmoIrRoot(*Ir.PmoIrModel, Name.s)
  Protected Count.i
  While FindMapElement(*Ir\Aliases(), Name)
    Name = *Ir\Aliases()
    Count + 1
    If Count > MapSize(*Ir\Aliases()) : PmoIrFail("tensor alias cycle") : ProcedureReturn "" : EndIf
  Wend
  ProcedureReturn Name
EndProcedure

Procedure PmoIrSortIntervals(Array Items.i(1), Count.i)
  Protected I.i
  Protected J.i
  Protected Temp.i
  Protected *A.PmoIrValue
  Protected *B.PmoIrValue
  For I = 1 To Count - 1
    Temp = Items(I) : J = I - 1
    While J >= 0
      *A = Items(J) : *B = Temp
      If *A\Start < *B\Start Or (*A\Start = *B\Start And *A\Name <= *B\Name) : Break : EndIf
      Items(J + 1) = Items(J) : J - 1
    Wend
    Items(J + 1) = Temp
  Next
EndProcedure

Procedure PmoIrCoalesceBlocks(List Blocks.PmoIrBlock())
  Protected Changed.i = #True
  Protected *A.PmoIrBlock
  Protected *B.PmoIrBlock
  SortStructuredList(Blocks(), #PB_Sort_Ascending, OffsetOf(PmoIrBlock\Offset), TypeOf(PmoIrBlock\Offset))
  If FirstElement(Blocks()) = 0 : ProcedureReturn : EndIf
  *A = @Blocks()
  While NextElement(Blocks())
    *B = @Blocks()
    If *A\Offset + *A\Bytes = *B\Offset
      *A\Bytes + *B\Bytes
      DeleteElement(Blocks())
    Else
      *A = *B
    EndIf
  Wend
EndProcedure

Procedure.i PmoIrPlanArena(*Ir.PmoIrModel)
  Protected *Value.PmoIrValue
  Protected *Root.PmoIrValue
  Protected RootName.s
  Protected NodeNumber.i
  Protected Terminal.i
  Protected Count.i
  Protected Index.i
  Protected Choice.i
  Protected High.q
  Protected Needed.q
  Protected Offset.q
  Protected *Block.PmoIrBlock
  Dim Items.i(ListSize(*Ir\Values()))
  NewList Active.PmoIrValue()
  NewList Blocks.PmoIrBlock()
  ClearMap(*Ir\Aliases())
  ForEach *Ir\Nodes()
    If *Ir\Nodes()\Node\Operation = "Identity" Or *Ir\Nodes()\Node\Operation = "Reshape" Or
       *Ir\Nodes()\Node\Operation = "Flatten" Or *Ir\Nodes()\Node\Operation = "Squeeze" Or
       *Ir\Nodes()\Node\Operation = "Unsqueeze"
      If SelectElement(*Ir\Nodes()\Node\Inputs(), 0) And SelectElement(*Ir\Nodes()\Node\Outputs(), 0)
        If FindMapElement(*Ir\ValueByName(), *Ir\Nodes()\Node\Inputs()) And
           FindMapElement(*Ir\ValueByName(), *Ir\Nodes()\Node\Outputs())
          Protected *Input.PmoIrValue = *Ir\ValueByName(*Ir\Nodes()\Node\Inputs())
          Protected *Output.PmoIrValue = *Ir\ValueByName(*Ir\Nodes()\Node\Outputs())
          If *Input\Elements <> *Output\Elements
            ProcedureReturn PmoIrFail("view node changes its element count")
          EndIf
          *Ir\Aliases(*Output\Name) = *Input\Name
          *Output\Alias = *Input\Name
        EndIf
      EndIf
    EndIf
  Next
  ForEach *Ir\Values()
    *Ir\Values()\Start = $7FFFFFFF : *Ir\Values()\Finish = -1
  Next
  ForEach *Ir\Inputs()
    RootName = PmoIrRoot(*Ir, *Ir\Inputs())
    If FindMapElement(*Ir\ValueByName(), RootName)
      *Root = *Ir\ValueByName() : *Root\Start = -1 : *Root\Finish = -1
    EndIf
  Next
  NodeNumber = 0
  ForEach *Ir\Nodes()
    ForEach *Ir\Nodes()\Node\Inputs()
      If *Ir\Nodes()\Node\Inputs() = "" Or
         FindMapElement(*Ir\ConstantByName(), *Ir\Nodes()\Node\Inputs()) : Continue : EndIf
      RootName = PmoIrRoot(*Ir, *Ir\Nodes()\Node\Inputs())
      If FindMapElement(*Ir\ValueByName(), RootName)
        *Root = *Ir\ValueByName() : If NodeNumber > *Root\Finish : *Root\Finish = NodeNumber : EndIf
      EndIf
    Next
    ForEach *Ir\Nodes()\Node\Outputs()
      If *Ir\Nodes()\Node\Outputs() = "" Or
         FindMapElement(*Ir\ConstantByName(), *Ir\Nodes()\Node\Outputs()) Or
         FindMapElement(*Ir\Aliases(), *Ir\Nodes()\Node\Outputs()) : Continue : EndIf
      RootName = PmoIrRoot(*Ir, *Ir\Nodes()\Node\Outputs())
      If FindMapElement(*Ir\ValueByName(), RootName)
        *Root = *Ir\ValueByName()
        If *Root\Start = $7FFFFFFF : *Root\Start = NodeNumber : EndIf
        If *Root\Finish < NodeNumber : *Root\Finish = NodeNumber : EndIf
      EndIf
    Next
    NodeNumber + 1
  Next
  Terminal = NodeNumber
  ForEach *Ir\Outputs()
    If FindMapElement(*Ir\ConstantByName(), *Ir\Outputs()) : Continue : EndIf
    RootName = PmoIrRoot(*Ir, *Ir\Outputs())
    If FindMapElement(*Ir\ValueByName(), RootName)
      *Root = *Ir\ValueByName() : If *Root\Finish < Terminal : *Root\Finish = Terminal : EndIf
    EndIf
  Next
  ForEach *Ir\Checkpoints()
    If FindMapElement(*Ir\ConstantByName(), *Ir\Checkpoints()) : Continue : EndIf
    RootName = PmoIrRoot(*Ir, *Ir\Checkpoints())
    If FindMapElement(*Ir\ValueByName(), RootName)
      *Root = *Ir\ValueByName() : If *Root\Finish < Terminal : *Root\Finish = Terminal : EndIf
    EndIf
  Next
  ForEach *Ir\Values()
    If *Ir\Values()\IsConstant = 0 And *Ir\Values()\Alias = "" And *Ir\Values()\Start <> $7FFFFFFF
      Items(Count) = @*Ir\Values() : Count + 1
    EndIf
  Next
  PmoIrSortIntervals(Items(), Count)
  For Index = 0 To Count - 1
    *Value = Items(Index)
    ForEach Active()
      If Active()\Finish < *Value\Start
        AddElement(Blocks()) : Blocks()\Offset = Active()\Offset : Blocks()\Bytes = Active()\AllocationBytes
        DeleteElement(Active())
      EndIf
    Next
    PmoIrCoalesceBlocks(Blocks())
    Needed = PmoIrAlign(*Value\Bytes)
    Choice = -1
    ForEach Blocks()
      If Blocks()\Bytes >= Needed : Choice = ListIndex(Blocks()) : Break : EndIf
    Next
    If Choice < 0
      Offset = PmoIrAlign(High)
      High = Offset + Needed
    Else
      SelectElement(Blocks(), Choice)
      Offset = Blocks()\Offset
      If Blocks()\Bytes = Needed
        DeleteElement(Blocks())
      Else
        Blocks()\Offset + Needed : Blocks()\Bytes - Needed
      EndIf
    EndIf
    *Value\Offset = Offset : *Value\AllocationBytes = Needed
    AddElement(Active())
    Active()\Name = *Value\Name : Active()\Start = *Value\Start : Active()\Finish = *Value\Finish
    Active()\Offset = Offset : Active()\AllocationBytes = Needed
  Next
  *Ir\ArenaBytes = PmoIrAlign(High)
  ProcedureReturn #True
EndProcedure
