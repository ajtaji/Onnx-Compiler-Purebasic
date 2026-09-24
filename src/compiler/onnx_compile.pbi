; ============================================================================
; onnx_compile.pbi - native checked ONNX-to-source orchestration
; ----------------------------------------------------------------------------
; The product writes host-language or PureMetal source, packed weights and a
; manifest. It never discovers or launches a downstream language compiler.
; ============================================================================

Global PmoCompileError.s
Global PmoDynamicError.s
Declare.i PmoDynamicCommand(ModelPath.s)
Declare.i PmdExportTargetRuntime(Folder.s,TargetIndex.i,Random.i=0,Ops.i=0)

; Decide BEFORE tracing: a sample execution may discover dimensions, but may
; never turn data-dependent dimensions into the contract of a resident model.
; Shape of a fixed-shape value is constant; its data is not.
Procedure.s PmoCompileRuntimeDimensions(*Model.PmoOnnxModel)
  Protected Known.i, Position.i, Control.i, Name.s, Op.s
  NewMap Constants.i()
  ForEach *Model\Graph\Inputs()
    If *Model\Graph\Inputs()\HasShape = 0 : ProcedureReturn "input rank is unresolved" : EndIf
    ForEach *Model\Graph\Inputs()\Dims()
      If *Model\Graph\Inputs()\Dims()\HasValue = 0 : ProcedureReturn "input dimensions vary at runtime" : EndIf
    Next
  Next
  ForEach *Model\Graph\Initializers()
    Constants(*Model\Graph\Initializers()\Name) = 1
  Next
  ForEach *Model\Graph\Nodes()
    Op = *Model\Graph\Nodes()\Operation : Known = 1
    ; A branch, a trip count or a sequence decides shapes and even the set of
    ; values while the program runs; the fixed-shape planner cannot hold that.
    If FindString("|If|Loop|SequenceEmpty|SequenceConstruct|SequenceInsert|SequenceAt|SequenceLength|SplitToSequence|ConcatFromSequence|", "|" + Op + "|")
      ProcedureReturn Op + " is control flow or a sequence operator, so its values are decided at run time"
    EndIf
    If Op = "NonMaxSuppression" : ProcedureReturn "NonMaxSuppression output size depends on the scores" : EndIf
    ForEach *Model\Graph\Nodes()\Inputs()
      If *Model\Graph\Nodes()\Inputs() <> "" And Constants(*Model\Graph\Nodes()\Inputs()) = 0 : Known = 0 : EndIf
    Next
    If Known = 0
      If Op = "NonZero" : ProcedureReturn "NonZero output size depends on input values" : EndIf
      Position = 0
      ForEach *Model\Graph\Nodes()\Inputs()
        Control = 0
        Select Op
          Case "Range", "ConstantOfShape" : Control = 1
          Case "Reshape", "Expand", "Squeeze", "Unsqueeze", "ReduceMean", "ReduceSum", "CumSum", "ReduceMax", "ReduceProd", "TopK" : Control = Bool(Position = 1)
          Case "Slice" : Control = Bool(Position >= 1)
          Case "Pad" : Control = Bool(Position = 1 Or Position = 3)
          Case "Resize" : Control = Bool(Position >= 1)
          Case "STFT" : Control = Bool(Position = 1 Or Position = 3)
          Case "Split", "Tile", "OneHot", "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp",
               "Compress", "Upsample"
            Control = Bool(Position = 1)
          Case "Dropout" : Control = Bool(Position = 2)
          Case "HannWindow", "HammingWindow", "BlackmanWindow" : Control = Bool(Position = 0)
          Case "DFT" : Control = Bool(Position = 1 Or Position = 2)
        EndSelect
        Name = *Model\Graph\Nodes()\Inputs()
        If Control And Name <> "" And Constants(Name) = 0
          ProcedureReturn Op + " has a runtime shape or indexing control"
        EndIf
        Position + 1
      Next
    EndIf
    ; All earlier tensor shapes are fixed if this scan has not returned.
    If Op = "Shape" Or Op = "Size" : Known = 1 : EndIf
    ForEach *Model\Graph\Nodes()\Outputs()
      Constants(*Model\Graph\Nodes()\Outputs()) = Known
    Next
  Next
  ProcedureReturn ""
EndProcedure

; ABSENT OPTIONAL OUTPUTS THE KERNEL STILL WRITES (forum 988). LSTM's Y, Y_h
; and Y_c are all optional, but its kernels keep the running hidden and cell
; state in Y_h and Y_c and build Y_h from Y, so each needs memory whether or
; not the model reads it. A left-out position (fewer outputs listed, or an
; empty name) gets a name no model can spell, "__pmo_absent_<node>_<position>",
; before either path plans anything; nothing reads it, and the graph's own
; outputs are unchanged. With WithShapes the fixed-shape path also gets its
; declared shape (Y [T, D, B, H], Y_h and Y_c [D, B, H], layout 0) when X and
; W have fixed shapes; otherwise the missing shape is refused later with the
; usual sentence. A model with nothing left out is not touched.
Procedure.i PmoCompileFindDims(*Model.PmoOnnxModel, Name.s, List Dims.q())
  ClearList(Dims())
  ForEach *Model\Graph\Initializers()
    If *Model\Graph\Initializers()\Name = Name
      ForEach *Model\Graph\Initializers()\Dims() : AddElement(Dims()) : Dims() = *Model\Graph\Initializers()\Dims() : Next
      ProcedureReturn #True
    EndIf
  Next
  ForEach *Model\Graph\Inputs()
    If *Model\Graph\Inputs()\Name = Name And *Model\Graph\Inputs()\HasShape
      ForEach *Model\Graph\Inputs()\Dims()
        If *Model\Graph\Inputs()\Dims()\HasValue = 0 : ProcedureReturn #False : EndIf
        AddElement(Dims()) : Dims() = *Model\Graph\Inputs()\Dims()\Value
      Next
      ProcedureReturn #True
    EndIf
  Next
  ForEach *Model\Graph\Values()
    If *Model\Graph\Values()\Name = Name And *Model\Graph\Values()\HasShape
      ForEach *Model\Graph\Values()\Dims()
        If *Model\Graph\Values()\Dims()\HasValue = 0 : ProcedureReturn #False : EndIf
        AddElement(Dims()) : Dims() = *Model\Graph\Values()\Dims()\Value
      Next
      ProcedureReturn #True
    EndIf
  Next
  ProcedureReturn #False
EndProcedure

; The value of a one-element FLOAT, INT32 or INT64 initializer.
Procedure.i PmoCompileInitScalar(*Model.PmoOnnxModel, Name.s, *Value.Double)
  If Name = "" : ProcedureReturn #False : EndIf
  ForEach *Model\Graph\Initializers()
    If *Model\Graph\Initializers()\Name <> Name : Continue : EndIf
    With *Model\Graph\Initializers()
      If \DataLocation <> 0 : ProcedureReturn #False : EndIf
      If \Raw\Bytes > 0
        Select \DataType
          Case 1 : If \Raw\Bytes <> 4 : ProcedureReturn #False : EndIf : *Value\d = PeekF(\Raw\Data)
          Case 6 : If \Raw\Bytes <> 4 : ProcedureReturn #False : EndIf : *Value\d = PeekL(\Raw\Data)
          Case 7 : If \Raw\Bytes <> 8 : ProcedureReturn #False : EndIf : *Value\d = PeekQ(\Raw\Data)
          Default : ProcedureReturn #False
        EndSelect
        ProcedureReturn #True
      EndIf
      Select \DataType
        Case 1 : If ListSize(\FloatData()) <> 1 : ProcedureReturn #False : EndIf : FirstElement(\FloatData()) : *Value\d = \FloatData()
        Case 6 : If ListSize(\Int32Data()) <> 1 : ProcedureReturn #False : EndIf : FirstElement(\Int32Data()) : *Value\d = \Int32Data()
        Case 7 : If ListSize(\Int64Data()) <> 1 : ProcedureReturn #False : EndIf : FirstElement(\Int64Data()) : *Value\d = \Int64Data()
        Default : ProcedureReturn #False
      EndSelect
      ProcedureReturn #True
    EndWith
  Next
  ProcedureReturn #False
EndProcedure

; MelWeightMatrix-17 whose five inputs are initializers becomes an
; initializer before either path runs, computed as the ONNX reference computes
; it: the mel edges in binary32 (2595 log10(1 + f / 700)), the bins
; floor((dft_length + 1) f / sample_rate) and the triangles in binary64, then
; rounded to binary32. "" when done or left for the emitters to refuse; a
; sentence for parameters the definition cannot draw.
Procedure.s PmoCompileFoldMel(*Model.PmoOnnxModel)
  Protected Nb.Double, Dl.Double, Sr.Double, Lo.Double, Hi.Double, Name.s, Out.s, i.i, j.i, K.i, Bins.i, Count.i
  Protected LoMel.f, HiMel.f, T.f, MelStep.f, Fb.d, Num.d, Den.d, Lf.i, Cf.i, Hf.i
  Protected Dim Edge.i(0)
  Protected Dim W.f(0)
  Protected Dim In.s(4)
  ForEach *Model\Graph\Nodes()
    If *Model\Graph\Nodes()\Operation <> "MelWeightMatrix" Or ListSize(*Model\Graph\Nodes()\Inputs()) <> 5 : Continue : EndIf
    If *Model\Graph\Nodes()\Domain <> "" And *Model\Graph\Nodes()\Domain <> "ai.onnx" : Continue : EndIf
    If PmoEmitAttrI(@*Model\Graph\Nodes(), "output_datatype", 1) <> 1 : Continue : EndIf
    i = 0
    ForEach *Model\Graph\Nodes()\Inputs() : In(i) = *Model\Graph\Nodes()\Inputs() : i + 1 : Next
    If FirstElement(*Model\Graph\Nodes()\Outputs()) = 0 : Continue : EndIf
    Out = *Model\Graph\Nodes()\Outputs()
    PushListPosition(*Model\Graph\Nodes())
    If PmoCompileInitScalar(*Model, In(0), @Nb) = 0 Or PmoCompileInitScalar(*Model, In(1), @Dl) = 0 Or PmoCompileInitScalar(*Model, In(2), @Sr) = 0 Or
       PmoCompileInitScalar(*Model, In(3), @Lo) = 0 Or PmoCompileInitScalar(*Model, In(4), @Hi) = 0
      PopListPosition(*Model\Graph\Nodes())
      Continue
    EndIf
    PopListPosition(*Model\Graph\Nodes())
    Bins = Nb\d : K = Int(Dl\d) / 2 + 1
    If Bins < 1 Or Dl\d < 1 Or Sr\d < 1 Or Lo\d < 0 Or Hi\d < Lo\d Or Bins > 4096 Or Dl\d > 65536
      ProcedureReturn "MelWeightMatrix node " + *Model\Graph\Nodes()\Name + ": num_mel_bins = " + StrD(Nb\d) + ", dft_length = " + StrD(Dl\d) + ", sample_rate = " + StrD(Sr\d) +
                      ", lower_edge_hertz = " + StrD(Lo\d) + " and upper_edge_hertz = " + StrD(Hi\d) + " do not describe a filter bank (positive counts, 0 <= lower <= upper)."
    EndIf
    ; the edges in binary32, as numpy computes them from binary32 inputs
    T = Lo\d : T = T / 700.0 : T = 1.0 + T : T = Log10(T) : LoMel = 2595.0 * T
    T = Hi\d : T = T / 700.0 : T = 1.0 + T : T = Log10(T) : HiMel = 2595.0 * T
    MelStep = HiMel - LoMel : MelStep = MelStep / (Bins + 2)
    Dim Edge(Bins + 1)
    For i = 0 To Bins + 1
      Fb = i : Num = MelStep : Fb = Fb * Num : Num = LoMel : Fb = Fb + Num
      Fb = 700.0 * (Pow(10.0, Fb / 2595.0) - 1.0)
      Edge(i) = Round(((Dl\d + 1.0) * Fb) / Sr\d, #PB_Round_Down)
    Next
    Dim W.f(K * Bins)
    For i = 0 To Bins - 1
      Lf = Edge(i) : Cf = Edge(i + 1) : Hf = Edge(i + 2)
      If Lf < 0 Or Cf >= K Or Hf > K
        ProcedureReturn "MelWeightMatrix node " + *Model\Graph\Nodes()\Name + ": band " + Str(i) + " reaches bin " + Str(Hf) + " of a " + Str(K) + "-bin spectrum; upper_edge_hertz must not pass half of sample_rate."
      EndIf
      If Cf = Lf
        W(Cf * Bins + i) = 1.0
      Else
        For j = Lf To Cf : Num = j - Lf : Den = Cf - Lf : W(j * Bins + i) = Num / Den : Next
      EndIf
      If Hf > Cf
        For j = Cf To Hf - 1 : Num = Hf - j : Den = Hf - Cf : W(j * Bins + i) = Num / Den : Next
      EndIf
    Next
    ; the node becomes an initializer holding the matrix
    DeleteElement(*Model\Graph\Nodes())
    LastElement(*Model\Graph\Initializers()) : AddElement(*Model\Graph\Initializers())
    With *Model\Graph\Initializers()
      \Name = Out : \DataType = 1
      AddElement(\Dims()) : \Dims() = K
      AddElement(\Dims()) : \Dims() = Bins
      For i = 0 To K * Bins - 1 : AddElement(\FloatData()) : \FloatData() = W(i) : Next
    EndWith
    Count + 1
  Next
  ProcedureReturn ""
EndProcedure

Procedure PmoCompileNameAbsentOutputs(*Model.PmoOnnxModel, WithShapes.i)
  Protected NodeIndex.i, Position.i, Name.s, Hidden.q, Directions.q, Steps.q, Batch.q, Ok.i, Extent.i
  NewList XDims.q()
  NewList WDims.q()
  ForEach *Model\Graph\Nodes()
    If *Model\Graph\Nodes()\Operation = "LSTM" And (*Model\Graph\Nodes()\Domain = "" Or *Model\Graph\Nodes()\Domain = "ai.onnx")
      While ListSize(*Model\Graph\Nodes()\Outputs()) < 3
        LastElement(*Model\Graph\Nodes()\Outputs()) : AddElement(*Model\Graph\Nodes()\Outputs()) : *Model\Graph\Nodes()\Outputs() = ""
      Wend
      Ok = #False
      If WithShapes And SelectElement(*Model\Graph\Nodes()\Inputs(), 1)
        Name = *Model\Graph\Nodes()\Inputs()
        FirstElement(*Model\Graph\Nodes()\Inputs())
        If PmoCompileFindDims(*Model, *Model\Graph\Nodes()\Inputs(), XDims()) And ListSize(XDims()) = 3 And
           PmoCompileFindDims(*Model, Name, WDims()) And ListSize(WDims()) = 3
          SelectElement(XDims(), 0) : Steps = XDims() : SelectElement(XDims(), 1) : Batch = XDims()
          SelectElement(WDims(), 0) : Directions = WDims() : SelectElement(WDims(), 1)
          Hidden = PmoEmitAttrI(@*Model\Graph\Nodes(), "hidden_size", WDims() / 4)
          Ok = #True
        EndIf
      EndIf
      Position = 0
      ForEach *Model\Graph\Nodes()\Outputs()
        If Position < 3 And *Model\Graph\Nodes()\Outputs() = ""
          Name = "__pmo_absent_" + Str(NodeIndex) + "_" + Str(Position)
          *Model\Graph\Nodes()\Outputs() = Name
          If Ok
            LastElement(*Model\Graph\Values()) : AddElement(*Model\Graph\Values())
            *Model\Graph\Values()\Name = Name : *Model\Graph\Values()\ElementType = 1
            *Model\Graph\Values()\HasTensorType = #True : *Model\Graph\Values()\HasShape = #True
            If Position = 0
              AddElement(*Model\Graph\Values()\Dims()) : *Model\Graph\Values()\Dims()\HasValue = #True : *Model\Graph\Values()\Dims()\Value = Steps
            EndIf
            For Extent = 0 To 2
              AddElement(*Model\Graph\Values()\Dims()) : *Model\Graph\Values()\Dims()\HasValue = #True
              Select Extent
                Case 0 : *Model\Graph\Values()\Dims()\Value = Directions
                Case 1 : *Model\Graph\Values()\Dims()\Value = Batch
                Default : *Model\Graph\Values()\Dims()\Value = Hidden
              EndSelect
            Next
          EndIf
        EndIf
        Position + 1
      Next
    EndIf
    NodeIndex + 1
  Next
EndProcedure

; CastLike-15 is Cast to the element type of its second input. On the
; fixed-shape path every value's type is declared, so the node becomes that
; Cast (the attribute to, the second input dropped) before anything is
; planned; one whose target type the model does not declare is refused by
; the emitter with its sentence.
Procedure PmoCompileRewriteCastLike(*Model.PmoOnnxModel)
  Protected Name.s, Target.i
  ForEach *Model\Graph\Nodes()
    If *Model\Graph\Nodes()\Operation <> "CastLike" Or ListSize(*Model\Graph\Nodes()\Inputs()) <> 2 : Continue : EndIf
    SelectElement(*Model\Graph\Nodes()\Inputs(), 1) : Name = *Model\Graph\Nodes()\Inputs()
    Target = 0
    ForEach *Model\Graph\Initializers()
      If *Model\Graph\Initializers()\Name = Name : Target = *Model\Graph\Initializers()\DataType : EndIf
    Next
    ForEach *Model\Graph\Inputs()
      If *Model\Graph\Inputs()\Name = Name And *Model\Graph\Inputs()\HasTensorType : Target = *Model\Graph\Inputs()\ElementType : EndIf
    Next
    ForEach *Model\Graph\Values()
      If *Model\Graph\Values()\Name = Name And *Model\Graph\Values()\HasTensorType : Target = *Model\Graph\Values()\ElementType : EndIf
    Next
    ForEach *Model\Graph\Outputs()
      If *Model\Graph\Outputs()\Name = Name And *Model\Graph\Outputs()\HasTensorType : Target = *Model\Graph\Outputs()\ElementType : EndIf
    Next
    If Target = 0 : Continue : EndIf
    *Model\Graph\Nodes()\Operation = "Cast"
    DeleteElement(*Model\Graph\Nodes()\Inputs())
    LastElement(*Model\Graph\Nodes()\Attributes()) : AddElement(*Model\Graph\Nodes()\Attributes())
    *Model\Graph\Nodes()\Attributes()\Name = "to"
    *Model\Graph\Nodes()\Attributes()\AttributeType = 2
    *Model\Graph\Nodes()\Attributes()\IntegerValue = Target
  Next
EndProcedure

Procedure.i PmoCompileFail(Message.s)
  If PmoCompileError = "" : PmoCompileError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.s PmoCompileBytes(Value.q)
  If Value >= 1024 * 1024 * 1024
    ProcedureReturn StrD(Value / (1024.0 * 1024.0 * 1024.0), 2) + " GiB"
  ElseIf Value >= 1024 * 1024
    ProcedureReturn StrD(Value / (1024.0 * 1024.0), 2) + " MiB"
  ElseIf Value >= 1024
    ProcedureReturn StrD(Value / 1024.0, 2) + " KiB"
  EndIf
  ProcedureReturn Str(Value) + " B"
EndProcedure

Procedure.s PmoCompileShape(*Value.PmoIrValue)
  Protected Text.s = "["
  If *Value = 0 : ProcedureReturn "?" : EndIf
  ForEach *Value\Dims()
    If Text <> "[" : Text + "," : EndIf
    Text + Str(*Value\Dims())
  Next
  ProcedureReturn Text + "]"
EndProcedure

Procedure.s PmoCompileParent(Path.s)
  Protected Trimmed.s = RTrim(Path, "\/")
  ProcedureReturn GetPathPart(Trimmed)
EndProcedure

; Resolve relative output paths against the caller's directory.
Procedure.s PmoCompileAbsolute(Path.s)
  Protected Base.s
  If Path = "" : ProcedureReturn "" : EndIf
  If Mid(Path, 2, 1) = ":" Or Left(Path, 2) = "\\" Or Left(Path, 1) = "/"
    ProcedureReturn Path
  EndIf
  Base = GetCurrentDirectory()
  If Right(Base, 1) <> "\" And Right(Base, 1) <> "/" : Base + "\" : EndIf
  ProcedureReturn Base + Path
EndProcedure

Procedure.s PmoCompileProjectRoot()
  Protected Candidate.s = GetCurrentDirectory()
  Protected Index.i
  If Right(Candidate, 1) <> "\" : Candidate + "\" : EndIf
  For Index = 0 To 8
    If FileSize(Candidate + "runtime\tensor_fp32.pmi") > 0 Or FileSize(Candidate + "MathLib\onnx\tensor_fp32.pmi") > 0
      ProcedureReturn Candidate
    EndIf
    Candidate = PmoCompileParent(Candidate)
    If Candidate = "" : Break : EndIf
  Next
  Candidate = GetPathPart(ProgramFilename())
  For Index = 0 To 8
    If FileSize(Candidate + "runtime\tensor_fp32.pmi") > 0 Or FileSize(Candidate + "MathLib\onnx\tensor_fp32.pmi") > 0
      ProcedureReturn Candidate
    EndIf
    Candidate = PmoCompileParent(Candidate)
    If Candidate = "" : Break : EndIf
  Next
  ProcedureReturn ""
EndProcedure

Procedure.i PmoCompileDigits(Text.s)
  Protected Index.i
  Protected Code.i
  If Text = "" : ProcedureReturn #False : EndIf
  For Index = 1 To Len(Text)
    Code = Asc(Mid(Text, Index, 1))
    If Code < '0' Or Code > '9' : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoCompileApplyShape(*Model.PmoOnnxModel, Spec.s)
  Protected Split.i = FindString(Spec, "=")
  Protected Name.s
  Protected Shape.s
  Protected Part.s
  Protected Count.i
  Protected Index.i
  Protected *Input.PmoOnnxValue
  If Split <= 1 : ProcedureReturn PmoCompileFail("invalid --shape; expected NAME=DIM,DIM,...") : EndIf
  Name = Trim(Left(Spec, Split - 1)) : Shape = Trim(Mid(Spec, Split + 1))
  ForEach *Model\Graph\Inputs()
    If *Model\Graph\Inputs()\Name = Name : *Input = @*Model\Graph\Inputs() : Break : EndIf
  Next
  If *Input = 0 : ProcedureReturn PmoCompileFail("--shape names no graph input: " + Name) : EndIf
  If Shape = "" : Count = 0 : Else : Count = CountString(Shape, ",") + 1 : EndIf
  If *Input\HasShape And ListSize(*Input\Dims()) > 0 And ListSize(*Input\Dims()) <> Count
    ProcedureReturn PmoCompileFail("--shape rank disagrees for input " + Name)
  EndIf
  ClearList(*Input\Dims())
  For Index = 1 To Count
    Part = Trim(StringField(Shape, Index, ","))
    If PmoCompileDigits(Part) = 0 Or Val(Part) <= 0
      ProcedureReturn PmoCompileFail("--shape extents must be positive decimal integers for input " + Name)
    EndIf
    AddElement(*Input\Dims())
    *Input\Dims()\HasValue = #True : *Input\Dims()\Value = Val(Part) : *Input\Dims()\Symbol = ""
  Next
  *Input\HasTensorType = #True : *Input\HasShape = #True
  ProcedureReturn #True
EndProcedure

Procedure.i PmoCompileSupportedOp(Operation.s)
  If PmoOpsOwns(Operation) : ProcedureReturn #True : EndIf
  ProcedureReturn Bool(FindString("|Add|Sub|Mul|Div|Pow|Relu|LeakyRelu|Sigmoid|Tanh|Exp|Log|Sqrt|Abs|Neg|Sin|Cos|Atan|Floor|Round|MatMul|Gemm|Softmax|ReduceMean|ReduceSum|CumSum|LayerNormalization|BatchNormalization|Conv|ConvTranspose|Clip|Resize|STFT|LSTM|Gather|Cast|Range|Equal|Greater|GreaterOrEqual|Less|LessOrEqual|And|Where|Slice|Expand|Pad|NonZero|ScatterND|Identity|InstanceNormalization|TopK|ScatterElements|Scatter|ReduceMax|ReduceProd|Not|Reshape|Flatten|Squeeze|Unsqueeze|Transpose|Concat|RandomNormal|RandomNormalLike|RandomUniform|RandomUniformLike|", "|" + Operation + "|"))
EndProcedure

Procedure.i PmoCompileValidate(*Ir.PmoIrModel)
  NewMap Produced.i()
  Protected DefaultOpset.q
  Protected NodeIndex.i
  Protected Name.s
  Protected *Value.PmoIrValue
  ForEach *Ir\Source\Opsets()
    If *Ir\Source\Opsets()\Domain = "" Or *Ir\Source\Opsets()\Domain = "ai.onnx"
      DefaultOpset = *Ir\Source\Opsets()\Version
    EndIf
  Next
  If DefaultOpset > 23 Or (DefaultOpset < 7 And PmoNsOpsetCovers(*Ir\Source, DefaultOpset) = 0)
    ProcedureReturn PmoCompileFail("ai.onnx opset " + Str(DefaultOpset) + " is outside the audited range 7..23")
  EndIf
  ForEach *Ir\Constants() : Produced(*Ir\Constants()\Name) = #True : Next
  ForEach *Ir\Inputs() : Produced(*Ir\Inputs()) = #True : Next
  ForEach *Ir\Nodes()
    NodeIndex = *Ir\Nodes()\Index
    If *Ir\Nodes()\Node\Domain <> "" And *Ir\Nodes()\Node\Domain <> "ai.onnx"
      ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " uses unsupported domain " + *Ir\Nodes()\Node\Domain)
    EndIf
    If PmoCompileSupportedOp(*Ir\Nodes()\Node\Operation) = 0
      ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " uses unsupported operator " + *Ir\Nodes()\Node\Operation)
    EndIf
    ForEach *Ir\Nodes()\Node\Inputs()
      Name = *Ir\Nodes()\Node\Inputs()
      If Name <> "" And FindMapElement(Produced(), Name) = 0
        ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " input " + Name + " has no earlier producer")
      EndIf
    Next
    ForEach *Ir\Nodes()\Node\Outputs()
      Name = *Ir\Nodes()\Node\Outputs()
      If Name = "" : Continue : EndIf
      If FindMapElement(Produced(), Name)
        ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " violates single-static-assignment for " + Name)
      EndIf
      If FindMapElement(*Ir\ValueByName(), Name) = 0
        ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " output " + Name + " has no concrete type/shape")
      EndIf
      *Value = *Ir\ValueByName()
      ; UINT8, INT8 and INT32 values are produced by the quantized operators,
      ; by Cast, and passed on unchanged by the operators that only rename
      If *Value\ElementType <> 1 And *Value\ElementType <> 7 And *Value\ElementType <> 9 And
         Not ((*Value\ElementType = 2 Or *Value\ElementType = 3 Or *Value\ElementType = 6) And
              FindString("|QuantizeLinear|DequantizeLinear|DynamicQuantizeLinear|MatMulInteger|QLinearMatMul|ConvInteger|QLinearConv|Cast|Identity|Reshape|Flatten|Squeeze|Unsqueeze|", "|" + *Ir\Nodes()\Node\Operation + "|"))
        ProcedureReturn PmoCompileFail("node " + Str(NodeIndex) + " output " + Name + " uses unsupported runtime type " + Str(*Value\ElementType))
      EndIf
      Produced(Name) = #True
    Next
  Next
  ForEach *Ir\Outputs()
    If FindMapElement(Produced(), *Ir\Outputs()) = 0
      ProcedureReturn PmoCompileFail("graph output has no producer: " + *Ir\Outputs())
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoCompileAddCheckpoints(*Ir.PmoIrModel, List Requested.s())
  NewMap Roots.i()
  Protected Name.s
  Protected Root.s
  If *Ir = 0 : ProcedureReturn PmoCompileFail("internal checkpoint IR is null") : EndIf
  ForEach Requested()
    Name = Trim(Requested())
    If Name = "" Or FindMapElement(*Ir\ValueByName(), Name) = 0
      ProcedureReturn PmoCompileFail("checkpoint is not a concrete graph tensor: " + Name)
    EndIf
    Root = PmoIrRoot(*Ir, Name)
    If Root = "" : ProcedureReturn #False : EndIf
    If FindMapElement(Roots(), Root)
      ProcedureReturn PmoCompileFail("duplicate or aliased checkpoint: " + Name)
    EndIf
    Roots(Root) = #True
    AddElement(*Ir\Checkpoints()) : *Ir\Checkpoints() = Name
  Next
  ProcedureReturn #True
EndProcedure

Procedure.q PmoCompileConstantInteger(*Ir.PmoIrModel, Name.s, *Ok.Integer)
  Protected *Constant.PmoIrConstant
  If *Ok : *Ok\i = #False : EndIf
  If *Ir = 0 Or Name = "" Or FindMapElement(*Ir\ConstantByName(), Name) = 0 : ProcedureReturn 0 : EndIf
  *Constant = *Ir\ConstantByName()
  If *Constant\Elements < 1 Or *Constant\Data = 0 : ProcedureReturn 0 : EndIf
  Select *Constant\ElementType
    Case 7
      If *Ok : *Ok\i = #True : EndIf
      ProcedureReturn PeekQ(*Constant\Data)
    Case 6
      If *Ok : *Ok\i = #True : EndIf
      ProcedureReturn PeekL(*Constant\Data)
  EndSelect
  ProcedureReturn 0
EndProcedure

; ONNX Runtime uses Bluestein for non-power-of-two DFT lengths and radix-2
; work buffers internally. Generated programs receive the same bounded scratch
; contract from the arena planner; tensor code never allocates at runtime.
Procedure.i PmoCompilePlanStftScratch(*Ir.PmoIrModel)
  Protected InputName.s
  Protected FrameLength.q
  Protected ComplexCount.q
  Protected Largest.q
  Protected Bytes.q
  Protected Ok.Integer
  Protected *Window.PmoIrValue
  If *Ir = 0 : ProcedureReturn PmoCompileFail("internal STFT scratch IR is null") : EndIf
  *Ir\StftScratchOffset = 0 : *Ir\StftScratchBytes = 0 : *Ir\StftScratchComplex = 0
  ForEach *Ir\Nodes()
    If *Ir\Nodes()\Node\Operation <> "STFT" : Continue : EndIf
    InputName = ""
    If SelectElement(*Ir\Nodes()\Node\Inputs(), 3) : InputName = *Ir\Nodes()\Node\Inputs() : EndIf
    If InputName <> ""
      FrameLength = PmoCompileConstantInteger(*Ir, InputName, @Ok)
      If Ok\i = 0 : ProcedureReturn PmoCompileFail("STFT frame_length must be a static INT32/INT64 scalar") : EndIf
    Else
      If SelectElement(*Ir\Nodes()\Node\Inputs(), 2) = 0 : ProcedureReturn PmoCompileFail("STFT needs a static window or frame_length") : EndIf
      InputName = *Ir\Nodes()\Node\Inputs()
      If FindMapElement(*Ir\ValueByName(), InputName) = 0 : ProcedureReturn PmoCompileFail("STFT window has no concrete tensor") : EndIf
      *Window = *Ir\ValueByName() : FrameLength = *Window\Elements
    EndIf
    If FrameLength <= 0 Or FrameLength > 1048576
      ProcedureReturn PmoCompileFail("STFT frame_length " + Str(FrameLength) + " is outside the bounded runtime range 1..1048576")
    EndIf
    ComplexCount = 1
    While ComplexCount < FrameLength * 2 - 1
      ComplexCount * 2
    Wend
    If ComplexCount > Largest : Largest = ComplexCount : EndIf
  Next
  If Largest = 0 : ProcedureReturn #True : EndIf
  If Largest > $7FFFFFFFFFFFFFF / 40 : ProcedureReturn PmoCompileFail("STFT scratch size overflow") : EndIf
  Bytes = PmoIrAlign(Largest * 40)
  *Ir\StftScratchOffset = PmoIrAlign(*Ir\ArenaBytes)
  *Ir\StftScratchBytes = Bytes
  *Ir\StftScratchComplex = Largest
  *Ir\ArenaBytes = *Ir\StftScratchOffset + Bytes
  ProcedureReturn #True
EndProcedure

Procedure.s PmoCompileLargest(*Ir.PmoIrModel, Limit.i = 5)
  NewMap Used.i()
  Protected Result.s
  Protected Rank.i
  Protected *Best.PmoIrValue
  Protected *Quant.PmoIrQuantWeight
  For Rank = 1 To Limit
    *Best = 0
    ForEach *Ir\Values()
      If FindMapElement(Used(), *Ir\Values()\Name) = 0
        If *Best = 0 Or *Ir\Values()\Bytes > *Best\Bytes : *Best = @*Ir\Values() : EndIf
      EndIf
    Next
    If *Best = 0 : Break : EndIf
    Used(*Best\Name) = #True
    If Result <> "" : Result + Chr(10) : EndIf
    Result + "  " + *Best\Name + " " + PmoCompileShape(*Best) + " = " + PmoCompileBytes(*Best\Bytes)
    If *Best\IsConstant
      If FindMapElement(*Ir\QuantByName(), *Best\Name)
        *Quant = *Ir\QuantByName()
        Result + " [constant; INT8 pack " + PmoCompileBytes(*Quant\QuantizedBytes) + "]"
      Else
        Result + " [packed constant]"
      EndIf
    Else
      Result + " [activation]"
    EndIf
  Next
  ProcedureReturn Result
EndProcedure

Procedure.i PmoCompileCapacity(*Ir.PmoIrModel, *Profile.PmoTargetProfile)
  Protected RamRequired.q
  Protected FlashRequired.q
  Protected EstimatedCode.q
  Protected Message.s
  If *Profile\Launch <> #PMO_LAUNCH_EMBEDDED_WEIGHTS : ProcedureReturn #True : EndIf
  RamRequired = *Ir\ArenaBytes + *Profile\RamReserveBytes
  EstimatedCode = *Profile\FlashReserveBytes + ListSize(*Ir\Nodes()) * 128
  FlashRequired = *Ir\WeightBytes + EstimatedCode
  If RamRequired <= *Profile\RamBytes And FlashRequired <= *Profile\FlashBytes : ProcedureReturn #True : EndIf
  Message = "model does not fit target " + *Profile\Id + ":"
  If RamRequired > *Profile\RamBytes
    Message + Chr(10) + "  SRAM required " + PmoCompileBytes(RamRequired) + " (arena " +
              PmoCompileBytes(*Ir\ArenaBytes) + " + runtime/stack reserve " +
              PmoCompileBytes(*Profile\RamReserveBytes) + "), available " + PmoCompileBytes(*Profile\RamBytes) +
              ", short by " + PmoCompileBytes(RamRequired - *Profile\RamBytes)
  EndIf
  If FlashRequired > *Profile\FlashBytes
    Message + Chr(10) + "  flash required at least " + PmoCompileBytes(FlashRequired) + " (packed model " +
              PmoCompileBytes(*Ir\WeightBytes) + " + conservative code/metadata " +
              PmoCompileBytes(EstimatedCode) + "), available " + PmoCompileBytes(*Profile\FlashBytes) +
              ", short by at least " + PmoCompileBytes(FlashRequired - *Profile\FlashBytes)
  EndIf
  Message + Chr(10) + "largest tensors:" + Chr(10) + PmoCompileLargest(*Ir)
  If ListSize(*Ir\QuantWeights()) > 0
    Message + Chr(10) + "This model is already using checked INT8 weight/activation kernels. Reduce the bounded graph/model or choose a larger target. Nothing was truncated or emitted."
  Else
    Message + Chr(10) + "Try the explicit INT8 option, reduce the bounded graph/model, or choose a larger target. Nothing was truncated or emitted."
  EndIf
  ProcedureReturn PmoCompileFail(Message)
EndProcedure

Procedure.i PmoCompileValidateTargetIntegers(*Ir.PmoIrModel, *Profile.PmoTargetProfile)
  Protected Index.q
  Protected Value.q
  Protected Operation.s
  Protected OutputName.s
  Protected *Output.PmoIrValue
  Protected *Input.PmoIrValue
  If *Ir = 0 Or *Profile = 0 : ProcedureReturn PmoCompileFail("internal target-integer contract is null") : EndIf
  If *Profile\NativeIntegerBytes <> 4 : ProcedureReturn #True : EndIf

  ForEach *Ir\Constants()
    If *Ir\Constants()\ElementType = 7
      For Index = 0 To *Ir\Constants()\Elements - 1
        Value = PeekQ(*Ir\Constants()\Data + Index * 8)
        If Value < -2147483648 Or Value > 2147483647
          ProcedureReturn PmoCompileFail("target " + *Profile\Id + " cannot represent INT64 tensor " +
                                        *Ir\Constants()\Name + " element " + Str(Index) + " = " + Str(Value) +
                                        "; its checked runtime range is -2147483648..2147483647. Nothing was emitted.")
        EndIf
      Next
    EndIf
  Next

  ; Copy/index operators preserve already-validated values. Arithmetic which
  ; can create a wider result is refused unless it folded to a checked constant.
  ForEach *Ir\Nodes()
    Operation = *Ir\Nodes()\Node\Operation
    ForEach *Ir\Nodes()\Node\Outputs()
      OutputName = *Ir\Nodes()\Node\Outputs()
      If OutputName = "" Or FindMapElement(*Ir\ValueByName(), OutputName) = 0 : Continue : EndIf
      *Output = *Ir\ValueByName()
      If *Output\ElementType <> 7 Or *Output\IsConstant : Continue : EndIf
      If FindString("|Add|Sub|Mul|Div|Pow|Range|CumSum|Sum|PRelu|ReduceL1|ReduceSumSquare|", "|" + Operation + "|")
        ProcedureReturn PmoCompileFail("target " + *Profile\Id + " cannot prove that runtime INT64 " +
                                      Operation + " output " + OutputName +
                                      " stays in its checked signed 32-bit execution range. Nothing was emitted.")
      EndIf
      If Operation = "Cast"
        FirstElement(*Ir\Nodes()\Node\Inputs())
        If FindMapElement(*Ir\ValueByName(), *Ir\Nodes()\Node\Inputs())
          *Input = *Ir\ValueByName()
          If *Input\ElementType <> 9 And *Input\ElementType <> 2 And *Input\ElementType <> 3 And *Input\ElementType <> 6
            ProcedureReturn PmoCompileFail("target " + *Profile\Id + " cannot prove that Cast output " +
                                          OutputName + " stays in its checked signed 32-bit INT64 range. Nothing was emitted.")
          EndIf
        EndIf
      EndIf
    Next
  Next
  ProcedureReturn #True
EndProcedure

Procedure.s PmoCompileJson(Text.s)
  Text = ReplaceString(Text, "\", "\\")
  Text = ReplaceString(Text, Chr(34), "\" + Chr(34))
  Text = ReplaceString(Text, Chr(13), "\r")
  Text = ReplaceString(Text, Chr(10), "\n")
  ProcedureReturn Chr(34) + Text + Chr(34)
EndProcedure

Procedure.s PmoCompileElementType(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "float32"
    Case 3 : ProcedureReturn "int8"
    Case 7 : ProcedureReturn "int64"
    Case 9 : ProcedureReturn "bool"
  EndSelect
  ProcedureReturn "onnx-" + Str(ElementType)
EndProcedure

Procedure.s PmoCompileDimsJson(List Dims.q())
  Protected Result.s = "["
  ForEach Dims()
    If Result <> "[" : Result + "," : EndIf
    Result + Str(Dims())
  Next
  ProcedureReturn Result + "]"
EndProcedure

Procedure.s PmoCompileTensorJson(*Ir.PmoIrModel, Name.s, Index.i)
  Protected *Value.PmoIrValue
  Protected *Root.PmoIrValue
  Protected *Constant.PmoIrConstant
  Protected RootName.s
  Protected Storage.s
  Protected Offset.q
  If FindMapElement(*Ir\ValueByName(), Name) = 0 : ProcedureReturn "{}" : EndIf
  *Value = *Ir\ValueByName()
  RootName = PmoIrRoot(*Ir, Name)
  If FindMapElement(*Ir\ConstantByName(), RootName)
    *Constant = *Ir\ConstantByName()
    Storage = "weights" : Offset = *Constant\PackedOffset
  ElseIf FindMapElement(*Ir\ValueByName(), RootName)
    *Root = *Ir\ValueByName()
    Storage = "arena" : Offset = *Root\Offset
  EndIf
  ProcedureReturn "{" + PmoCompileJson("index") + ":" + Str(Index) + "," +
                  PmoCompileJson("name") + ":" + PmoCompileJson(Name) + "," +
                  PmoCompileJson("type") + ":" + PmoCompileJson(PmoCompileElementType(*Value\ElementType)) + "," +
                  PmoCompileJson("element_type") + ":" + Str(*Value\ElementType) + "," +
                  PmoCompileJson("shape") + ":" + PmoCompileDimsJson(*Value\Dims()) + "," +
                  PmoCompileJson("bytes") + ":" + Str(*Value\Bytes) + "," +
                  PmoCompileJson("storage") + ":" + PmoCompileJson(Storage) + "," +
                  PmoCompileJson("offset") + ":" + Str(Offset) + "}"
EndProcedure

Procedure PmoCompileManifestTensorList(File.i, Key.s, *Ir.PmoIrModel, List Names.s(), TrailingComma.i)
  Protected Index.i
  Protected Suffix.s
  WriteStringN(File, "  " + PmoCompileJson(Key) + ": [", #PB_UTF8)
  ForEach Names()
    Suffix = "," : If ListIndex(Names()) = ListSize(Names()) - 1 : Suffix = "" : EndIf
    WriteStringN(File, "    " + PmoCompileTensorJson(*Ir, Names(), Index) + Suffix, #PB_UTF8)
    Index + 1
  Next
  If TrailingComma : Suffix = "," : Else : Suffix = "" : EndIf
  WriteStringN(File, "  ]" + Suffix, #PB_UTF8)
EndProcedure

Procedure.i PmoCompileManifest(Path.s, *Ir.PmoIrModel, *Profile.PmoTargetProfile,
                               Precision.s, Source.s, Weights.s)
  Protected Scheme.s="FP32 arithmetic and tensors"
  Protected Temp.s = Path + ".tmp"
  Protected File.i
  Protected Hash.s = Fingerprint(*Ir\Source\FileData, *Ir\Source\FileBytes, #PB_Cipher_SHA2, 256)
  Protected Index.i
  Protected Suffix.s
  Protected OriginalBytes.q
  Protected QuantizedBytes.q
  Protected DefaultOpset.q
  If FileSize(Temp) >= 0 : DeleteFile(Temp) : EndIf
  File = CreateFile(#PB_Any, Temp)
  If File = 0 : ProcedureReturn PmoCompileFail("cannot create manifest: " + Temp) : EndIf
  WriteStringN(File, "{", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("format") + ": " + PmoCompileJson("PureMetal ONNX AOT manifest") + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("version") + ": 4,", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("target") + ": " + PmoCompileJson(*Profile\Id) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("native_integer_bytes") + ": " + Str(*Profile\NativeIntegerBytes) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("precision") + ": " + PmoCompileJson(Precision) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("model_sha256") + ": " + PmoCompileJson(LCase(Hash)) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("source") + ": " + PmoCompileJson(Source) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("weights") + ": " + PmoCompileJson(Weights) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("output_kind") + ": " + PmoCompileJson("source-only") + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("execution_contract") + ": " + PmoCompileJson("resident-reusable") + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("shape_contract") + ": " + PmoCompileJson("fixed-dimensions-variable-input-values") + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("embedded_request_inputs") + ": false,", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("weight_bytes") + ": " + Str(*Ir\WeightBytes) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("arena_bytes") + ": " + Str(*Ir\ArenaBytes) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("decoded_weight_reserve_bytes") + ": " + Str(*Ir\DecodedWeightBytes) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("reduced_storage_weight_count") + ": " + Str(*Ir\ReducedWeightCount) + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("storage_decode_contract") + ": " + PmoCompileJson("FP16/BF16/INT4 weights decode once to resident FP32; decoded memory is included in arena_bytes; activations remain FP32") + ",", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("int8_scratch") + ": {" + PmoCompileJson("offset") + ":" + Str(*Ir\Int8ScratchOffset) + "," + PmoCompileJson("bytes") + ":" + Str(*Ir\Int8ScratchBytes) + "},", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("stft_scratch") + ": {" + PmoCompileJson("offset") + ":" + Str(*Ir\StftScratchOffset) + "," + PmoCompileJson("bytes") + ":" + Str(*Ir\StftScratchBytes) + "," + PmoCompileJson("complex_elements") + ":" + Str(*Ir\StftScratchComplex) + "},", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("target_capacity") + ": {" + PmoCompileJson("ram_bytes") + ":" + Str(*Profile\RamBytes) + "," + PmoCompileJson("ram_reserve_bytes") + ":" + Str(*Profile\RamReserveBytes) + "," + PmoCompileJson("flash_bytes") + ":" + Str(*Profile\FlashBytes) + "," + PmoCompileJson("flash_reserve_bytes") + ":" + Str(*Profile\FlashReserveBytes) + "},", #PB_UTF8)
  ForEach *Ir\Source\Opsets()
    If *Ir\Source\Opsets()\Domain = "" Or *Ir\Source\Opsets()\Domain = "ai.onnx" : DefaultOpset = *Ir\Source\Opsets()\Version : EndIf
  Next
  ; THE BUILD REQUIREMENT, IN THE MANIFEST AND NOT ONLY IN A COMMENT.
  ; The emitted source says in words that an Anvil payload needs
  ; --wants-services and a PmOnnxAnvilEnter(x0) call. A person reads that
  ; once; a build step never does. This is the same statement in a form the
  ; step that compiles the payload can check, so "we forgot the flag"
  ; becomes a failing check rather than a silently slower run - and a
  ; silently slower run is the worst shape here, because it still produces
  ; exactly the right bytes and therefore nothing ever notices.
  If *Profile\SourceDialect = #PMO_SOURCE_PUREMETAL And *Profile\NativeIntegerBytes = 8
    WriteStringN(File, "  " + PmoCompileJson("anvil_payload") + ": {" +
                 PmoCompileJson("wants_services") + ": true," +
                 PmoCompileJson("entry_call") + ": " + PmoCompileJson("PmOnnxAnvilEnter") + "," +
                 PmoCompileJson("build_flags") + ": [" + PmoCompileJson("--wants-services") + "," + PmoCompileJson("--entry-returns") + "]," +
                 PmoCompileJson("min_service_entry_count") + ": 192," +
                 PmoCompileJson("without_it") + ": " + PmoCompileJson("every convolution runs on one core; identical bytes, several times slower") +
                 "},", #PB_UTF8)
  EndIf
  If PmoRandomCount(*Ir\Source)
    ; Comparison mode left no random node in the schedule; its outputs are the last inputs.
    Protected RandomScheduled.i
    ForEach *Ir\Nodes() : If PmoRandomIsOp(*Ir\Nodes()\Node\Operation) : RandomScheduled + 1 : EndIf : Next
    WriteStringN(File, "  " + PmoCompileJson("random") + ": " + PmoRandomManifestText(*Ir\Source, Bool(RandomScheduled = 0), ListSize(*Ir\Inputs()) - PmoRandomCount(*Ir\Source), "PmOnnxSetRandomSeed") + ",", #PB_UTF8)
  EndIf
  WriteStringN(File, "  " + PmoCompileJson("ai_onnx_opset") + ": " + Str(DefaultOpset) + ",", #PB_UTF8)
  PmoCompileManifestTensorList(File, "inputs", *Ir, *Ir\Inputs(), #True)
  PmoCompileManifestTensorList(File, "outputs", *Ir, *Ir\Outputs(), #True)
  PmoCompileManifestTensorList(File, "checkpoints", *Ir, *Ir\Checkpoints(), #True)
  ForEach *Ir\QuantWeights() : OriginalBytes + *Ir\QuantWeights()\OriginalBytes : QuantizedBytes + *Ir\QuantWeights()\QuantizedBytes : Next
  WriteStringN(File, "  " + PmoCompileJson("quantization") + ": {", #PB_UTF8)
  WriteStringN(File, "    " + PmoCompileJson("mode") + ": " + PmoCompileJson(Precision) + ",", #PB_UTF8)
  If Precision="int8" : Scheme="per-output-channel symmetric signed INT8 weights, dynamic per-row signed INT8 activations, INT32 accumulation, FP32 operator seams" : EndIf
  If *Ir\ReducedWeightCount : Scheme=UCase(Precision)+" weight storage; decode once into resident FP32 memory; FP32 arithmetic and activations" : EndIf
  WriteStringN(File, "    " + PmoCompileJson("scheme") + ": " + PmoCompileJson(Scheme) + ",", #PB_UTF8)
  WriteStringN(File, "    " + PmoCompileJson("weight_count") + ": " + Str(ListSize(*Ir\QuantWeights())) + ",", #PB_UTF8)
  WriteStringN(File, "    " + PmoCompileJson("fp32_weight_bytes") + ": " + Str(OriginalBytes) + ",", #PB_UTF8)
  WriteStringN(File, "    " + PmoCompileJson("int8_weight_bytes") + ": " + Str(QuantizedBytes) + ",", #PB_UTF8)
  WriteStringN(File, "    " + PmoCompileJson("weights") + ": [", #PB_UTF8)
  ForEach *Ir\QuantWeights()
    Suffix = "," : If ListIndex(*Ir\QuantWeights()) = ListSize(*Ir\QuantWeights()) - 1 : Suffix = "" : EndIf
    WriteStringN(File, "      {" + PmoCompileJson("name") + ":" + PmoCompileJson(*Ir\QuantWeights()\Name) + "," + PmoCompileJson("scale_tensor") + ":" + PmoCompileJson(*Ir\QuantWeights()\ScaleName) + "," + PmoCompileJson("scale_count") + ":" + Str(*Ir\QuantWeights()\ScaleCount) + "," + PmoCompileJson("original_bytes") + ":" + Str(*Ir\QuantWeights()\OriginalBytes) + "," + PmoCompileJson("int8_bytes") + ":" + Str(*Ir\QuantWeights()\QuantizedBytes) + "}" + Suffix, #PB_UTF8)
  Next
  WriteStringN(File, "    ]", #PB_UTF8)
  WriteStringN(File, "  },", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("nodes") + ": [", #PB_UTF8)
  ForEach *Ir\Nodes()
    Suffix = "," : If ListIndex(*Ir\Nodes()) = ListSize(*Ir\Nodes()) - 1 : Suffix = "" : EndIf
    WriteStringN(File, "    {" + PmoCompileJson("index") + ":" + Str(*Ir\Nodes()\Index) + "," + PmoCompileJson("name") + ":" + PmoCompileJson(*Ir\Nodes()\Node\Name) + "," + PmoCompileJson("op") + ":" + PmoCompileJson(*Ir\Nodes()\Node\Operation) + "}" + Suffix, #PB_UTF8)
  Next
  WriteStringN(File, "  ],", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("packed_tensors") + ": [", #PB_UTF8)
  Index = 0
  ForEach *Ir\Constants()
    If *Ir\Constants()\PackedOffset > 0 : Index + 1 : EndIf
  Next
  Protected Seen.i
  ForEach *Ir\Constants()
    If *Ir\Constants()\PackedOffset <= 0 : Continue : EndIf
    Seen + 1 : Suffix = "," : If Seen = Index : Suffix = "" : EndIf
    WriteStringN(File, "    {" + PmoCompileJson("name") + ":" + PmoCompileJson(*Ir\Constants()\Name) + "," + PmoCompileJson("type") + ":" + PmoCompileJson(PmoCompileElementType(*Ir\Constants()\ElementType)) + "," + PmoCompileJson("shape") + ":" + PmoCompileDimsJson(*Ir\Constants()\Dims()) + "," + PmoCompileJson("offset") + ":" + Str(*Ir\Constants()\PackedOffset) + "," + PmoCompileJson("bytes") + ":" + Str(*Ir\Constants()\Bytes) + "," + PmoCompileJson("storage_kind") + ":" + Str(*Ir\Constants()\StorageKind) + "}" + Suffix, #PB_UTF8)
  Next
  WriteStringN(File, "  ],", #PB_UTF8)
  WriteStringN(File, "  " + PmoCompileJson("runtime_nodes") + ": " + Str(ListSize(*Ir\Nodes())), #PB_UTF8)
  WriteStringN(File, "}", #PB_UTF8)
  CloseFile(File)
  If FileSize(Path) >= 0 And DeleteFile(Path) = 0 : DeleteFile(Temp) : ProcedureReturn PmoCompileFail("cannot replace manifest") : EndIf
  If RenameFile(Temp, Path) = 0 : ProcedureReturn PmoCompileFail("cannot publish manifest") : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.q PmoCompileAlign(Value.q, Alignment.q)
  ProcedureReturn (Value + Alignment - 1) & ~(Alignment - 1)
EndProcedure

Procedure.i PmoCompileCommand(ModelPath.s)
  Protected Model.PmoOnnxModel
  Protected Ir.PmoIrModel
  Protected OutputPrefix.s
  Protected TargetId.s
  Protected Precision.s = "fp32"
  Protected RuntimePath.s
  Protected OptionName.s
  Protected Index.i = 2
  Protected SpeechUi.i
  Protected KokoroText.i
  Protected RandomInputs.i
  Protected RandomReason.s
  Protected RuntimeReason.s
  Protected EmitProfile.PmoTargetProfile
  Protected TargetIndex.i
  Protected Root.s
  Protected Source.s
  Protected Weights.s
  Protected PendingWeights.s
  Protected Manifest.s
  Protected RuntimeInclude.s
  Protected WeightsAddress.q
  Protected ArenaAddress.q
  NewList Specs.s()
  NewList Shapes.s()
  NewList TraceNames.s()
  NewList Checkpoints.s()
  PmoCompileError = ""
  While Index < CountProgramParameters()
    OptionName = ProgramParameter(Index)
    If OptionName = "--output" And Index + 1 < CountProgramParameters()
      Index + 1 : OutputPrefix = ProgramParameter(Index)
    ElseIf OptionName = "--target" And Index + 1 < CountProgramParameters()
      Index + 1 : TargetId = ProgramParameter(Index)
    ElseIf OptionName = "--precision" And Index + 1 < CountProgramParameters()
      Index + 1 : Precision = LCase(ProgramParameter(Index))
    ElseIf OptionName = "--ort-dll" And Index + 1 < CountProgramParameters()
      Index + 1 : RuntimePath = ProgramParameter(Index)
    ElseIf OptionName = "--trace-input" And Index + 1 < CountProgramParameters()
      Index + 1 : AddElement(Specs()) : Specs() = ProgramParameter(Index)
    ElseIf OptionName = "--shape" And Index + 1 < CountProgramParameters()
      Index + 1 : AddElement(Shapes()) : Shapes() = ProgramParameter(Index)
    ElseIf OptionName = "--checkpoint" And Index + 1 < CountProgramParameters()
      Index + 1 : AddElement(Checkpoints()) : Checkpoints() = ProgramParameter(Index)
    ElseIf OptionName = "--build"
      ProcedureReturn PmoCompileFail("--build is not supported; this tool emits source code and never invokes a downstream compiler")
    ElseIf OptionName = "--library"
      ; Compatibility spelling: every model is now a resident library.
    ElseIf OptionName = "--speech-ui"
      SpeechUi = #True
    ElseIf OptionName = "--kokoro-text"
      KokoroText = #True
    ElseIf OptionName = "--random-inputs"
      RandomInputs = #True
    ElseIf OptionName = "--simplify"
      ; Checked constant folding and dead shape work removal are always enabled.
    Else
      ProcedureReturn PmoCompileFail("unknown or incomplete compile option " + OptionName)
    EndIf
    Index + 1
  Wend
  If OutputPrefix = "" Or GetPathPart(OutputPrefix) = "" : ProcedureReturn PmoCompileFail("--output must name a prefix in an existing folder") : EndIf
  OutputPrefix = PmoCompileAbsolute(OutputPrefix)
  If FileSize(GetPathPart(OutputPrefix)) <> -2 : ProcedureReturn PmoCompileFail("output folder does not exist: " + GetPathPart(OutputPrefix)) : EndIf
  TargetIndex = PmoTargetIndex(TargetId)
  If TargetIndex < 0 : ProcedureReturn PmoCompileFail("unknown target " + TargetId) : EndIf
  If PmoStoragePrecision(Precision)=0 : ProcedureReturn PmoCompileFail("--precision must be fp32, int8, fp16, bf16, or int4") : EndIf
  If SpeechUi And TargetId <> "windows"
    ProcedureReturn PmoCompileFail("The speech-window adapter requires Windows. Remove --speech-ui when generating a model library for another target.")
  EndIf
  If KokoroText And TargetIndex<>#PMO_TARGET_PI4 And TargetIndex<>#PMO_TARGET_UNOQ
    ProcedureReturn PmoCompileFail("--kokoro-text requires pi4 or unoq; Windows uses --speech-ui. Full Kokoro cannot fit Pico RAM.")
  EndIf
  If PmoOnnxLoad(ModelPath, @Model) = 0 : ProcedureReturn PmoCompileFail(PmoWireError) : EndIf
  RandomReason = PmoCompileFoldMel(@Model)
  If RandomReason <> "" : PmoCompileFail(RandomReason) : Goto PmoCompileCommandFailed : EndIf
  ForEach Shapes()
    If PmoCompileApplyShape(@Model, Shapes()) = 0 : Goto PmoCompileCommandFailed : EndIf
  Next
  ; Random operators are checked on the model itself, before either path is
  ; chosen, so both paths refuse an unclaimed form with the same sentence.
  RandomReason = PmoRandomValidateModel(@Model)
  If RandomReason <> "" : PmoCompileFail(RandomReason) : Goto PmoCompileCommandFailed : EndIf
  If RandomInputs And PmoRandomCount(@Model) = 0
    PmoCompileFail("--random-inputs was given, but the model has no RandomNormal, RandomNormalLike, RandomUniform or RandomUniformLike node to turn into an input. Remove the option.")
    Goto PmoCompileCommandFailed
  EndIf
  RuntimeReason = PmoCompileRuntimeDimensions(@Model)
  If RuntimeReason <> "" Or SpeechUi Or KokoroText
    PmoOnnxFree(@Model)
    If PmoDynamicCommand(ModelPath) = 0 : ProcedureReturn PmoCompileFail(PmoDynamicError) : EndIf
    ProcedureReturn #True
  EndIf
  PmoCompileNameAbsentOutputs(@Model, #True)
  PmoCompileRewriteCastLike(@Model)
  ; The attribute forms both paths share (onnx_forms.pbi): an attribute value
  ; this path does not compute is refused here, before anything is planned,
  ; instead of being ignored by the emitter (forum 859).
  ForEach Model\Graph\Nodes()
    RuntimeReason = PmoFormRefusal(@Model\Graph\Nodes(), #True)
    If RuntimeReason <> ""
      PmoCompileFail(PmoFormSentence(@Model\Graph\Nodes(), RuntimeReason)) : Goto PmoCompileCommandFailed
    EndIf
  Next
  If ListSize(Specs()) > 0
    If PmoTraceShapes(@Model, RuntimePath, Specs(), TraceNames()) = 0
      PmoCompileFail(PmoTraceError) : Goto PmoCompileCommandFailed
    EndIf
  EndIf
  If PmoIrPrepare(@Model, @Ir) = 0 Or PmoIrFoldConstants(@Ir) = 0
    PmoCompileFail(PmoIrError) : Goto PmoCompileCommandFailed
  EndIf
  If RandomInputs : PmoRandomNodesToInputs(@Ir) : EndIf
  If PmoCompileValidate(@Ir) = 0 : Goto PmoCompileCommandFailed : EndIf
  If PmoCompileAddCheckpoints(@Ir, Checkpoints()) = 0 : Goto PmoCompileCommandFailed : EndIf
  If Precision = "int8" And PmoQuantizeDynamicWeights(@Ir) = 0
    PmoCompileFail(PmoQuantError) : Goto PmoCompileCommandFailed
  EndIf
  If PmoIrPlanArena(@Ir) = 0
    PmoCompileFail(PmoIrError) : Goto PmoCompileCommandFailed
  EndIf
  If Precision = "int8" And PmoQuantPlanScratch(@Ir, 64 + 4032 * Bool(PmoTargets(TargetIndex)\NativeIntegerBytes = 8 And PmoTargets(TargetIndex)\SourceDialect = #PMO_SOURCE_PUREMETAL)) = 0
    PmoCompileFail(PmoQuantError) : Goto PmoCompileCommandFailed
  EndIf
  If PmoCompilePlanStftScratch(@Ir) = 0 : Goto PmoCompileCommandFailed : EndIf
  PmoOpsPlanScratch(@Ir)
  If PmoStorageReduce(@Ir,Precision)=0 : PmoCompileFail(PmoQuantError) : Goto PmoCompileCommandFailed : EndIf
  Source = OutputPrefix + PmoTargets(TargetIndex)\SourceSuffix
  Weights = OutputPrefix + ".pmw"
  PendingWeights = Weights + ".pending"
  Manifest = OutputPrefix + ".json"
  If FileSize(PendingWeights) >= 0 : DeleteFile(PendingWeights) : EndIf
  If PmoPackWeights(@Ir, PendingWeights) = 0 : PmoCompileFail(PmoPackError) : Goto PmoCompileCommandFailed : EndIf
  If PmoCompileCapacity(@Ir, @PmoTargets(TargetIndex)) = 0 : Goto PmoCompileCommandFailed : EndIf
  If PmoCompileValidateTargetIntegers(@Ir, @PmoTargets(TargetIndex)) = 0 : Goto PmoCompileCommandFailed : EndIf
  If PmoTargets(TargetIndex)\Launch = #PMO_LAUNCH_EXTERNAL_MEMORY
    WeightsAddress = $40000000
    ArenaAddress = PmoCompileAlign(WeightsAddress + Ir\WeightBytes, $10000)
    If ArenaAddress < WeightsAddress Or ArenaAddress + Ir\ArenaBytes < ArenaAddress
      PmoCompileFail("external-memory address plan overflowed") : Goto PmoCompileCommandFailed
    EndIf
  EndIf
  If PmdExportTargetRuntime(OutputPrefix+".runtime/",TargetIndex,Bool(RandomInputs=0 And PmoRandomCount(@Model)>0),PmoOpsGraphUses(@Model\Graph))=0 : PmoCompileFail(PmoDynamicError) : Goto PmoCompileCommandFailed : EndIf
  CopyStructure(@PmoTargets(TargetIndex),@EmitProfile,PmoTargetProfile)
  If EmitProfile\SourceDialect = #PMO_SOURCE_HOST
    RuntimeInclude = GetFilePart(OutputPrefix)+".runtime/tensor_fp32_windows.pbi"
  Else
    EmitProfile\MathInclude=GetFilePart(OutputPrefix)+".runtime/math.pmi"
    EmitProfile\TensorInclude=GetFilePart(OutputPrefix)+".runtime/tensor_fp32.pmi"
    If EmitProfile\AcceleratorInclude<>"" : EmitProfile\AcceleratorInclude=GetFilePart(OutputPrefix)+".runtime/tensor_fp32_neon.pmi" : EndIf
  EndIf
  If PmoEmitSource(@Ir, @EmitProfile, Source, PendingWeights, RuntimeInclude,
                   WeightsAddress, ArenaAddress) = 0
    PmoCompileFail(PmoEmitError) : Goto PmoCompileCommandFailed
  EndIf
  If FileSize(Weights) >= 0 And DeleteFile(Weights) = 0 : PmoCompileFail("cannot replace packed weights") : Goto PmoCompileCommandFailed : EndIf
  If RenameFile(PendingWeights, Weights) = 0 : PmoCompileFail("cannot publish packed weights") : Goto PmoCompileCommandFailed : EndIf
  PendingWeights = ""
  If PmoCompileManifest(Manifest, @Ir, @PmoTargets(TargetIndex), Precision, Source, Weights) = 0
    Goto PmoCompileCommandFailed
  EndIf
  PrintN("PASS: generated ONNX source for " + TargetId + " (" + Precision + ")")
  PrintN("Source: " + Source)
  PrintN("Weights: " + Weights + " (" + PmoCompileBytes(Ir\WeightBytes) + ")")
  If Precision = "int8" : PrintN("Quantized weights: " + Str(ListSize(Ir\QuantWeights())) + " (per-output-channel signed INT8, dynamic per-row activation scales, INT32 accumulation)") : EndIf
  PrintN("Arena: " + PmoCompileBytes(Ir\ArenaBytes) + " including bounded inputs/outputs")
  PrintN("Manifest: " + Manifest)
  PrintN("Resident model: bind weights/arena once, fill inputs and execute repeatedly, then unbind when finished.")
  PmoIrFree(@Ir) : PmoOnnxFree(@Model)
  ProcedureReturn #True

  PmoCompileCommandFailed:
  If PendingWeights <> "" And FileSize(PendingWeights) >= 0 : DeleteFile(PendingWeights) : EndIf
  PmoIrFree(@Ir) : PmoOnnxFree(@Model)
  ProcedureReturn #False
EndProcedure
