; ============================================================================
; onnx_quant.pbi - checked target-neutral dynamic INT8 weight lowering
; ----------------------------------------------------------------------------
; Eligible constant linear weights are stored as symmetric signed INT8.  The
; generated kernels dynamically quantize FP32 activations, accumulate products
; in signed 32-bit integers, and dequantize at the FP32 operator boundary.
; This deliberately preserves FP32 graph tensors and nonlinear operators.
; ============================================================================

Global PmoQuantError.s

Procedure.i PmoQuantFail(Message.s)
  If PmoQuantError = "" : PmoQuantError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoQuantWeightPosition(Operation.s, Position.i, *Value.PmoIrValue)
  If *Value = 0 : ProcedureReturn #False : EndIf
  Select Operation
    Case "MatMul", "Gemm"
      ProcedureReturn Bool(Position = 1 And ListSize(*Value\Dims()) >= 2)
    Case "Conv"
      ProcedureReturn Bool(Position = 1 And (ListSize(*Value\Dims()) = 3 Or ListSize(*Value\Dims()) = 4))
    Case "LSTM"
      ProcedureReturn Bool((Position = 1 Or Position = 2) And ListSize(*Value\Dims()) = 3)
  EndSelect
  ProcedureReturn #False
EndProcedure

Procedure.q PmoQuantDim(*Value.PmoIrValue, Index.i)
  If *Value = 0 Or Index < 0 Or SelectElement(*Value\Dims(), Index) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn *Value\Dims()
EndProcedure

Procedure.q PmoQuantReduction(*Node.PmoOnnxNode, Position.i, *Weight.PmoIrValue)
  Protected Rank.i = ListSize(*Weight\Dims())
  Protected Result.q = 1
  Protected Axis.i
  Protected TransB.i
  If *Node = 0 Or *Weight = 0 Or Rank < 2 : ProcedureReturn 0 : EndIf
  Select *Node\Operation
    Case "MatMul"
      ProcedureReturn PmoQuantDim(*Weight, Rank - 2)
    Case "Gemm"
      ForEach *Node\Attributes()
        If *Node\Attributes()\Name = "transB" : TransB = *Node\Attributes()\IntegerValue : EndIf
      Next
      If TransB : ProcedureReturn PmoQuantDim(*Weight, 1) : EndIf
      ProcedureReturn PmoQuantDim(*Weight, 0)
    Case "Conv"
      For Axis = 1 To Rank - 1 : Result * PmoQuantDim(*Weight, Axis) : Next
      ProcedureReturn Result
    Case "LSTM"
      ProcedureReturn PmoQuantDim(*Weight, Rank - 1)
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmoQuantChannelLayout(*Node.PmoOnnxNode, *Weight.PmoIrValue,
                                  *Mode.Integer, *Channels.Integer, *Stride.Quad)
  Protected Rank.i = ListSize(*Weight\Dims())
  Protected Axis.i
  Protected TransB.i
  Protected Count.q
  Protected ChannelStride.q = 1
  If *Node = 0 Or *Weight = 0 Or Rank < 2 : ProcedureReturn #False : EndIf
  Select *Node\Operation
    Case "MatMul"
      Count = PmoQuantDim(*Weight, Rank - 1) : *Mode\i = 1
    Case "Gemm"
      ForEach *Node\Attributes()
        If *Node\Attributes()\Name = "transB" : TransB = *Node\Attributes()\IntegerValue : EndIf
      Next
      If TransB
        Count = PmoQuantDim(*Weight, 0) : ChannelStride = PmoQuantDim(*Weight, 1) : *Mode\i = 2
      Else
        Count = PmoQuantDim(*Weight, 1) : *Mode\i = 1
      EndIf
    Case "Conv"
      Count = PmoQuantDim(*Weight, 0)
      For Axis = 1 To Rank - 1 : ChannelStride * PmoQuantDim(*Weight, Axis) : Next
      *Mode\i = 2
    Case "LSTM"
      Count = PmoQuantDim(*Weight, 0) * PmoQuantDim(*Weight, 1)
      ChannelStride = PmoQuantDim(*Weight, 2) : *Mode\i = 2
    Default
      ProcedureReturn #False
  EndSelect
  If Count <= 0 Or Count > $7FFFFFFF Or ChannelStride <= 0 : ProcedureReturn #False : EndIf
  *Channels\i = Count : *Stride\q = ChannelStride
  ProcedureReturn #True
EndProcedure

Procedure.i PmoQuantizeDynamicWeights(*Ir.PmoIrModel)
  NewMap Uses.i()
  NewMap EligibleUses.i()
  NewMap Reductions.q()
  NewMap ScaleCounts.i()
  NewMap ScaleModes.i()
  NewMap ScaleStrides.q()
  NewList Candidates.s()
  Protected Position.i
  Protected Name.s
  Protected ScaleName.s
  Protected Index.q
  Protected Channel.q
  Protected Quant.i
  Protected Value.f
  Protected Magnitude.f
  Protected Scale.f
  Protected *OldData
  Protected *NewData
  Protected *ScaleData
  Protected Reduction.q
  Protected Bits.q
  Protected Mode.Integer
  Protected Channels.Integer
  Protected Stride.Quad
  Protected *Value.PmoIrValue
  Protected *Constant.PmoIrConstant
  PmoQuantError = ""
  If *Ir = 0 : ProcedureReturn PmoQuantFail("internal INT8 IR is null") : EndIf

  ; A tensor is quantized only when every runtime use is a supported weight
  ; position. Shared eligible weights must also agree on their output-channel
  ; layout, because each output channel owns one symmetric scale.
  ForEach *Ir\Nodes()
    Position = 0
    ForEach *Ir\Nodes()\Node\Inputs()
      Name = *Ir\Nodes()\Node\Inputs()
      If Name <> ""
        Uses(Name) + 1
        If FindMapElement(*Ir\ValueByName(), Name)
          *Value = *Ir\ValueByName()
          If PmoQuantWeightPosition(*Ir\Nodes()\Node\Operation, Position, *Value)
            Mode\i = 0 : Channels\i = 0 : Stride\q = 0
            If PmoQuantChannelLayout(*Ir\Nodes()\Node, *Value, @Mode, @Channels, @Stride) = 0
              ProcedureReturn PmoQuantFail("cannot derive per-output-channel INT8 layout for " + Name)
            EndIf
            If ScaleCounts(Name) <> 0 And (ScaleCounts(Name) <> Channels\i Or
               ScaleModes(Name) <> Mode\i Or ScaleStrides(Name) <> Stride\q)
              ProcedureReturn PmoQuantFail("shared INT8 weight has conflicting output-channel layouts: " + Name)
            EndIf
            ScaleCounts(Name) = Channels\i : ScaleModes(Name) = Mode\i : ScaleStrides(Name) = Stride\q
            EligibleUses(Name) + 1
            Reduction = PmoQuantReduction(*Ir\Nodes()\Node, Position, *Value)
            If Reduction > Reductions(Name) : Reductions(Name) = Reduction : EndIf
          EndIf
        EndIf
      EndIf
      Position + 1
    Next
  Next
  ForEach *Ir\Outputs() : Uses(*Ir\Outputs()) + 1 : Next

  ForEach *Ir\Constants()
    Name = *Ir\Constants()\Name
    If *Ir\Constants()\ElementType = 1 And *Ir\Constants()\Elements > 0 And
       FindMapElement(Uses(), Name) And FindMapElement(EligibleUses(), Name) And
       Uses(Name) = EligibleUses(Name)
      AddElement(Candidates()) : Candidates() = Name
    EndIf
  Next

  ForEach Candidates()
    Name = Candidates()
    If FindMapElement(*Ir\ConstantByName(), Name) = 0
      ProcedureReturn PmoQuantFail("INT8 candidate disappeared: " + Name)
    EndIf
    *Constant = *Ir\ConstantByName()
    If Reductions(Name) > 133144
      ProcedureReturn PmoQuantFail("INT8 accumulator could overflow for weight " + Name +
                                   ": reduction length " + Str(Reductions(Name)) +
                                   " exceeds the checked signed-32-bit limit 133144")
    EndIf
    Channels\i = ScaleCounts(Name) : Mode\i = ScaleModes(Name) : Stride\q = ScaleStrides(Name)
    *ScaleData = AllocateMemory(Channels\i * 4)
    If *ScaleData = 0 : ProcedureReturn PmoQuantFail("cannot allocate INT8 scales for " + Name) : EndIf
    For Index = 0 To *Constant\Elements - 1
      If Mode\i = 1 : Channel = Index % Channels\i : Else : Channel = (Index / Stride\q) % Channels\i : EndIf
      Value = PeekF(*Constant\Data + Index * 4)
      Bits = PeekL(*Constant\Data + Index * 4) & $FFFFFFFF
      If (Bits & $7F800000) = $7F800000
        FreeMemory(*ScaleData)
        ProcedureReturn PmoQuantFail("INT8 weight " + Name + " contains NaN or infinity at element " + Str(Index))
      EndIf
      Magnitude = Abs(Value)
      If Magnitude > PeekF(*ScaleData + Channel * 4) : PokeF(*ScaleData + Channel * 4, Magnitude) : EndIf
    Next
    For Channel = 0 To Channels\i - 1
      Scale = PeekF(*ScaleData + Channel * 4) / 127.0
      If Scale <= 0.0 : Scale = 1.0 : EndIf
      PokeF(*ScaleData + Channel * 4, Scale)
    Next
    *NewData = AllocateMemory(*Constant\Elements)
    If *NewData = 0 : FreeMemory(*ScaleData) : ProcedureReturn PmoQuantFail("cannot allocate INT8 tensor " + Name) : EndIf
    For Index = 0 To *Constant\Elements - 1
      If Mode\i = 1 : Channel = Index % Channels\i : Else : Channel = (Index / Stride\q) % Channels\i : EndIf
      Scale = PeekF(*ScaleData + Channel * 4)
      Quant = Round(PeekF(*Constant\Data + Index * 4) / Scale, #PB_Round_Nearest)
      If Quant < -127 : Quant = -127 : EndIf
      If Quant > 127 : Quant = 127 : EndIf
      PokeA(*NewData + Index, Quant & 255)
    Next
    *OldData = *Constant\Data
    If *Constant\OwnsData And *OldData : FreeMemory(*OldData) : EndIf
    AddElement(*Ir\QuantWeights())
    *Ir\QuantWeights()\Name = Name
    *Ir\QuantWeights()\ScaleName = "@quant/" + Name
    *Ir\QuantWeights()\ScaleCount = Channels\i
    *Ir\QuantWeights()\ScaleMode = Mode\i
    *Ir\QuantWeights()\ScaleStride = Stride\q
    *Ir\QuantWeights()\OriginalBytes = *Constant\Bytes
    *Ir\QuantWeights()\QuantizedBytes = *Constant\Elements
    *Ir\QuantByName(Name) = @*Ir\QuantWeights()
    ScaleName = *Ir\QuantWeights()\ScaleName
    *Constant\Data = *NewData : *Constant\OwnsData = #True
    *Constant\ElementType = 3 : *Constant\Bytes = *Constant\Elements

    AddElement(*Ir\Constants())
    *Ir\Constants()\Name = ScaleName : *Ir\Constants()\ElementType = 1
    AddElement(*Ir\Constants()\Dims()) : *Ir\Constants()\Dims() = Channels\i
    *Ir\Constants()\Elements = Channels\i : *Ir\Constants()\Bytes = Channels\i * 4
    *Ir\Constants()\Data = *ScaleData : *Ir\Constants()\OwnsData = #True
    *Ir\ConstantByName(ScaleName) = @*Ir\Constants()
  Next
  If ListSize(*Ir\QuantWeights()) = 0
    ProcedureReturn PmoQuantFail("INT8 selected, but this graph has no exclusively used MatMul, Gemm, Conv, or LSTM constant weights")
  EndIf
  ProcedureReturn #True
EndProcedure

; Reserve one reusable byte-per-activation workspace. Quantized operators use
; it to convert each FP32 activation once instead of once per output multiply.
; LSTM keeps its input and recurrent vectors quantized at the same time.
Procedure.i PmoQuantPlanScratch(*Ir.PmoIrModel)
  Protected Operation.s
  Protected WeightName.s
  Protected RecurrentName.s
  Protected ActivationName.s
  Protected Needed.q
  Protected Largest.q
  Protected *Activation.PmoIrValue
  Protected *Weight.PmoIrValue
  Protected *Recurrent.PmoIrValue
  If *Ir = 0 : ProcedureReturn PmoQuantFail("internal INT8 scratch IR is null") : EndIf
  ForEach *Ir\Nodes()
    Operation = *Ir\Nodes()\Node\Operation
    WeightName = "" : RecurrentName = "" : ActivationName = ""
    If SelectElement(*Ir\Nodes()\Node\Inputs(), 0) : ActivationName = *Ir\Nodes()\Node\Inputs() : EndIf
    If SelectElement(*Ir\Nodes()\Node\Inputs(), 1) : WeightName = *Ir\Nodes()\Node\Inputs() : EndIf
    If Operation = "LSTM" And SelectElement(*Ir\Nodes()\Node\Inputs(), 2) : RecurrentName = *Ir\Nodes()\Node\Inputs() : EndIf
    Needed = 0
    If (Operation = "MatMul" Or Operation = "Gemm" Or Operation = "Conv") And
       WeightName <> "" And FindMapElement(*Ir\QuantByName(), WeightName) And
       FindMapElement(*Ir\ValueByName(), ActivationName)
      *Activation = *Ir\ValueByName()
      Needed = *Activation\Elements
    ElseIf Operation = "LSTM"
      If (WeightName <> "" And FindMapElement(*Ir\QuantByName(), WeightName)) Or
         (RecurrentName <> "" And FindMapElement(*Ir\QuantByName(), RecurrentName))
        If WeightName <> "" And FindMapElement(*Ir\ValueByName(), WeightName)
          *Weight = *Ir\ValueByName()
          If LastElement(*Weight\Dims()) : Needed = *Weight\Dims() : EndIf
        EndIf
        If RecurrentName <> "" And FindMapElement(*Ir\ValueByName(), RecurrentName)
          *Recurrent = *Ir\ValueByName()
          If LastElement(*Recurrent\Dims()) : Needed + *Recurrent\Dims() : EndIf
        EndIf
      EndIf
    EndIf
    If Needed > Largest : Largest = Needed : EndIf
  Next
  If Largest <= 0 : ProcedureReturn PmoQuantFail("INT8 graph has no activation scratch requirement") : EndIf
  *Ir\Int8ScratchOffset = PmoIrAlign(*Ir\ArenaBytes)
  *Ir\Int8ScratchBytes = PmoIrAlign(Largest)
  *Ir\ArenaBytes = *Ir\Int8ScratchOffset + *Ir\Int8ScratchBytes
  ProcedureReturn #True
EndProcedure
