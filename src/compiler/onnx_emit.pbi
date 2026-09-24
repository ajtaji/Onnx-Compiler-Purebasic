; ============================================================================
; onnx_emit.pbi - deterministic host-language/PureMetal source emission
; ----------------------------------------------------------------------------
; This is the native compiler backend.  It consumes the checked, folded IR;
; it neither reparses ONNX at run time nor launches another language tool.
; ============================================================================

XIncludeFile "onnx_emit_random.pbi"

Global PmoEmitError.s
; --threads N (Windows only): the most worker threads the generated program
; may use. 0 when the option is absent: the program then uses every
; processor its process may use, found when the model is bound.
Global PmoThreads.i

Procedure.i PmoEmitFail(Message.s)
  If PmoEmitError = "" : PmoEmitError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoEmitLine(File.i, Text.s = "")
  If WriteStringN(File, Text, #PB_Ascii) = 0
    ProcedureReturn PmoEmitFail("short write while emitting source")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.s PmoEmitInput(*Node.PmoOnnxNode, Index.i)
  If *Node = 0 Or Index < 0 Or SelectElement(*Node\Inputs(), Index) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Inputs()
EndProcedure

Procedure.s PmoEmitOutput(*Node.PmoOnnxNode, Index.i)
  If *Node = 0 Or Index < 0 Or SelectElement(*Node\Outputs(), Index) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Outputs()
EndProcedure

Procedure.i PmoEmitValue(*Ir.PmoIrModel, Name.s)
  If *Ir = 0 Or Name = "" Or FindMapElement(*Ir\ValueByName(), Name) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn *Ir\ValueByName()
EndProcedure

Procedure.i PmoEmitConstant(*Ir.PmoIrModel, Name.s)
  If *Ir = 0 Or Name = "" Or FindMapElement(*Ir\ConstantByName(), Name) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn *Ir\ConstantByName()
EndProcedure

Procedure.i PmoEmitQuantWeight(*Ir.PmoIrModel, Name.s)
  If *Ir = 0 Or Name = "" Or FindMapElement(*Ir\QuantByName(), Name) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn *Ir\QuantByName()
EndProcedure

Procedure.q PmoEmitDim(*Value.PmoIrValue, Index.i)
  If *Value = 0 Or Index < 0 Or SelectElement(*Value\Dims(), Index) = 0 : ProcedureReturn -1 : EndIf
  ProcedureReturn *Value\Dims()
EndProcedure

Procedure.i PmoEmitRank(*Value.PmoIrValue)
  If *Value = 0 : ProcedureReturn -1 : EndIf
  ProcedureReturn ListSize(*Value\Dims())
EndProcedure

Procedure.q PmoEmitProduct(*Value.PmoIrValue, First.i, Last.i)
  Protected Result.q = 1
  Protected Index.i
  If *Value = 0 Or First > Last : ProcedureReturn 1 : EndIf
  For Index = First To Last
    Result * PmoEmitDim(*Value, Index)
  Next
  ProcedureReturn Result
EndProcedure

Procedure.q PmoEmitStride(*Value.PmoIrValue, Axis.i)
  ProcedureReturn PmoEmitProduct(*Value, Axis + 1, PmoEmitRank(*Value) - 1)
EndProcedure

Procedure.q PmoEmitAttrI(*Node.PmoOnnxNode, Name.s, DefaultValue.q)
  If *Node = 0 : ProcedureReturn DefaultValue : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn *Node\Attributes()\IntegerValue : EndIf
  Next
  ProcedureReturn DefaultValue
EndProcedure

Procedure.f PmoEmitAttrF(*Node.PmoOnnxNode, Name.s, DefaultValue.f)
  If *Node = 0 : ProcedureReturn DefaultValue : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn *Node\Attributes()\FloatValue : EndIf
  Next
  ProcedureReturn DefaultValue
EndProcedure

Procedure.s PmoEmitAttrS(*Node.PmoOnnxNode, Name.s, DefaultValue.s)
  If *Node = 0 : ProcedureReturn DefaultValue : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn *Node\Attributes()\StringValue : EndIf
  Next
  ProcedureReturn DefaultValue
EndProcedure

Procedure.i PmoEmitAttrListCount(*Node.PmoOnnxNode, Name.s)
  If *Node = 0 : ProcedureReturn 0 : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn ListSize(*Node\Attributes()\Integers()) : EndIf
  Next
  ProcedureReturn 0
EndProcedure

Procedure.q PmoEmitAttrListI(*Node.PmoOnnxNode, Name.s, Index.i, DefaultValue.q)
  If *Node = 0 : ProcedureReturn DefaultValue : EndIf
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name
      If SelectElement(*Node\Attributes()\Integers(), Index) : ProcedureReturn *Node\Attributes()\Integers() : EndIf
      ProcedureReturn DefaultValue
    EndIf
  Next
  ProcedureReturn DefaultValue
EndProcedure

Procedure.q PmoEmitConstI(*Ir.PmoIrModel, Name.s, Index.q, *Ok.Integer)
  Protected *Constant.PmoIrConstant = PmoEmitConstant(*Ir, Name)
  If *Ok : *Ok\i = #False : EndIf
  If *Constant = 0 Or Index < 0 Or Index >= *Constant\Elements : ProcedureReturn 0 : EndIf
  Select *Constant\ElementType
    Case 7 : If *Ok : *Ok\i = #True : EndIf : ProcedureReturn PeekQ(*Constant\Data + Index * 8)
    Case 6 : If *Ok : *Ok\i = #True : EndIf : ProcedureReturn PeekL(*Constant\Data + Index * 4)
    Case 9 : If *Ok : *Ok\i = #True : EndIf : ProcedureReturn PeekA(*Constant\Data + Index) & 255
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.f PmoEmitConstF(*Ir.PmoIrModel, Name.s, Index.q, *Ok.Integer)
  Protected *Constant.PmoIrConstant = PmoEmitConstant(*Ir, Name)
  If *Ok : *Ok\i = #False : EndIf
  If *Constant = 0 Or *Constant\ElementType <> 1 Or Index < 0 Or Index >= *Constant\Elements
    ProcedureReturn 0.0
  EndIf
  If *Ok : *Ok\i = #True : EndIf
  ProcedureReturn PeekF(*Constant\Data + Index * 4)
EndProcedure

Procedure.s PmoEmitFloat(Value.f)
  Protected Text.s
  Protected Magnitude.d = Abs(Value)
  Protected Normalized.d
  Protected Exponent.i
  If Value >= 3.402823466e+38 : ProcedureReturn "3.402823466e+38" : EndIf
  If Value <= -3.402823466e+38 : ProcedureReturn "-3.402823466e+38" : EndIf
  If Magnitude > 0.0 And (Magnitude >= 10000000.0 Or Magnitude < 0.000001)
    Normalized = Value
    While Abs(Normalized) >= 10.0
      Normalized / 10.0
      Exponent + 1
    Wend
    While Abs(Normalized) < 1.0
      Normalized * 10.0
      Exponent - 1
    Wend
    Text = StrD(Normalized, 9)
    While Right(Text, 1) = "0" : Text = Left(Text, Len(Text) - 1) : Wend
    If Right(Text, 1) = "." : Text + "0" : EndIf
    If Text = "-0.0" : Text = "0.0" : EndIf
    ProcedureReturn Text + "e" + Str(Exponent)
  EndIf
  Text = StrF(Value, 9)
  While Right(Text, 1) = "0" : Text = Left(Text, Len(Text) - 1) : Wend
  If Right(Text, 1) = "." : Text + "0" : EndIf
  If FindString(Text, ".") = 0 And FindString(LCase(Text), "e") = 0 : Text + ".0" : EndIf
  If Text = "-0.0" : Text = "0.0" : EndIf
  ProcedureReturn Text
EndProcedure

Procedure.s PmoEmitGet(ElementType.i, Pointer.s, Index.s)
  Select ElementType
    Case 1 : ProcedureReturn "PmTensorGet(" + Pointer + ", " + Index + ")"
    Case 7 : ProcedureReturn "PmTensorGetI64(" + Pointer + ", " + Index + ")"
    Case 9 : ProcedureReturn "PmTensorGetBool(" + Pointer + ", " + Index + ")"
    Case 2 : ProcedureReturn "(PeekA(" + Pointer + " + " + Index + ") & 255)"
    Case 3 : ProcedureReturn "(((PeekA(" + Pointer + " + " + Index + ") & 255) ! 128) - 128)"
    Case 6 : ProcedureReturn "PeekL(" + Pointer + " + (" + Index + ") * 4)"
  EndSelect
  PmoEmitFail("no generated load for ONNX tensor type " + Str(ElementType))
  ProcedureReturn "0"
EndProcedure

Procedure.s PmoEmitPut(ElementType.i, Pointer.s, Index.s, Value.s)
  Select ElementType
    Case 1 : ProcedureReturn "PmTensorPut(" + Pointer + ", " + Index + ", " + Value + ")"
    Case 7 : ProcedureReturn "PmTensorPutI64(" + Pointer + ", " + Index + ", " + Value + ")"
    Case 9 : ProcedureReturn "PmTensorPutBool(" + Pointer + ", " + Index + ", " + Value + ")"
    Case 2, 3 : ProcedureReturn "PokeA(" + Pointer + " + " + Index + ", (" + Value + ") & 255)"
    Case 6 : ProcedureReturn "PokeL(" + Pointer + " + (" + Index + ") * 4, " + Value + ")"
  EndSelect
  PmoEmitFail("no generated store for ONNX tensor type " + Str(ElementType))
  ProcedureReturn ""
EndProcedure

Procedure.s PmoEmitBroadcastIndex(IndexName.s, *Input.PmoIrValue, *Output.PmoIrValue)
  Protected InputRank.i = PmoEmitRank(*Input)
  Protected OutputRank.i = PmoEmitRank(*Output)
  Protected Delta.i = OutputRank - InputRank
  Protected Axis.i
  Protected InputAxis.i
  Protected Src.q
  Protected Dst.q
  Protected InputStride.q
  Protected OutputStride.q
  Protected Coord.s
  Protected Result.s
  If InputRank < 0 Or OutputRank < 0 Or Delta < 0
    PmoEmitFail("cannot broadcast tensor ranks") : ProcedureReturn "0"
  EndIf
  For Axis = 0 To OutputRank - 1
    InputAxis = Axis - Delta
    Src = 1 : InputStride = 1
    If InputAxis >= 0
      Src = PmoEmitDim(*Input, InputAxis)
      InputStride = PmoEmitStride(*Input, InputAxis)
    EndIf
    Dst = PmoEmitDim(*Output, Axis)
    If Src <> 1 And Src <> Dst
      PmoEmitFail("cannot broadcast a source extent to its output") : ProcedureReturn "0"
    EndIf
    If Src = 1 : Continue : EndIf
    OutputStride = PmoEmitStride(*Output, Axis)
    If Dst = 1
      Coord = "0"
    Else
      Coord = "((" + IndexName + " / " + Str(OutputStride) + ") % " + Str(Dst) + ")"
    EndIf
    If InputStride <> 1 : Coord = "(" + Coord + " * " + Str(InputStride) + ")" : EndIf
    If Result <> "" : Result + " + " : EndIf
    Result + Coord
  Next
  If Result = "" : Result = "0" : EndIf
  ProcedureReturn Result
EndProcedure

Procedure.s PmoEmitAddress(*Ir.PmoIrModel, Name.s)
  Protected Root.s
  Protected *Constant.PmoIrConstant
  Protected *Value.PmoIrValue
  If Name = "" : ProcedureReturn "0" : EndIf
  Root = PmoIrRoot(*Ir, Name)
  *Constant = PmoEmitConstant(*Ir, Root)
  If *Constant
    If *Constant\StorageKind : ProcedureReturn "PmOnnxArenaBase + " + Str(*Constant\DecodedOffset) : EndIf
    If *Constant\PackedOffset <= 0
      PmoEmitFail("runtime constant was not packed: " + Root) : ProcedureReturn "0"
    EndIf
    ProcedureReturn "PmOnnxWeightsBase + " + Str(*Constant\PackedOffset)
  EndIf
  *Value = PmoEmitValue(*Ir, Root)
  If *Value = 0
    PmoEmitFail("runtime value has no allocation: " + Root) : ProcedureReturn "0"
  EndIf
  ProcedureReturn "PmOnnxArenaBase + " + Str(*Value\Offset)
EndProcedure

Procedure.s PmoEmitCopyProcedure(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "PmTensorCopy"
    Case 7 : ProcedureReturn "PmTensorCopyI64"
    Case 9 : ProcedureReturn "PmTensorCopyBool"
  EndSelect
  PmoEmitFail("tensor type has no copy kernel")
  ProcedureReturn ""
EndProcedure

Procedure.i PmoEmitElementBytes(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn 4
    Case 7 : ProcedureReturn 8
    Case 9 : ProcedureReturn 1
  EndSelect
  PmoEmitFail("tensor type has no emitted element width")
  ProcedureReturn 0
EndProcedure

Procedure.i PmoEmitShapeEqual(*A.PmoIrValue, *B.PmoIrValue)
  Protected Index.i
  If *A = 0 Or *B = 0 Or PmoEmitRank(*A) <> PmoEmitRank(*B) : ProcedureReturn #False : EndIf
  For Index = 0 To PmoEmitRank(*A) - 1
    If PmoEmitDim(*A, Index) <> PmoEmitDim(*B, Index) : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitBinaryHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *A.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected IA.s
  Protected IB.s
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Expr.s
  Protected ValueDecl.s = "  Protected v.i"
  If *A = 0 Or *B = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("binary node has missing values") : EndIf
  If *Out\ElementType = 1 And PmoEmitShapeEqual(*A, *Out) And PmoEmitShapeEqual(*B, *Out)
    ProcedureReturn #True
  EndIf
  If *Out\ElementType = 1 And (*A\Elements = 1 Or *B\Elements = 1) : ProcedureReturn #True : EndIf
  IA = PmoEmitBroadcastIndex("i", *A, *Out) : IB = PmoEmitBroadcastIndex("i", *B, *Out)
  If PmoEmitError <> "" : ProcedureReturn #False : EndIf
  If *Out\ElementType = 1 : ValueDecl = "  Protected v.f" : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*a, *b, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, ValueDecl)
  PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  If *Node\Operation = "Pow"
    Expr = "Pow(PmTensorGet(*a, " + IA + "), PmTensorGet(*b, " + IB + "))"
  Else
    Select *Node\Operation
      Case "Add" : Expr = PmoEmitGet(*Out\ElementType, "*a", IA) + " + " + PmoEmitGet(*Out\ElementType, "*b", IB)
      Case "Sub" : Expr = PmoEmitGet(*Out\ElementType, "*a", IA) + " - " + PmoEmitGet(*Out\ElementType, "*b", IB)
      Case "Mul" : Expr = PmoEmitGet(*Out\ElementType, "*a", IA) + " * " + PmoEmitGet(*Out\ElementType, "*b", IB)
      Case "Div" : Expr = PmoEmitGet(*Out\ElementType, "*a", IA) + " / " + PmoEmitGet(*Out\ElementType, "*b", IB)
    EndSelect
  EndIf
  PmoEmitLine(File, "    v = " + Expr)
  PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", "v"))
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitReshapeHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Axis.i
  Protected Extent.q
  Protected AllowZero.i = PmoEmitAttrI(*Node, "allowzero", 0)
  If *Src = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Reshape has missing values") : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *shape, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected raw.i")
  PmoEmitLine(File, "  Protected product.i") : PmoEmitLine(File, "  Protected inferred.i")
  PmoEmitLine(File, "  Protected inferredExtent.i")
  PmoEmitLine(File, "  product = 1 : inferred = 0 : inferredExtent = 0 : i = 0")
  PmoEmitLine(File, "  While i < " + Str(PmoEmitRank(*Out)))
  PmoEmitLine(File, "    raw = PmTensorGetI64(*shape, i)") : PmoEmitLine(File, "    Select i")
  For Axis = 0 To PmoEmitRank(*Out) - 1
    Extent = PmoEmitDim(*Out, Axis)
    PmoEmitLine(File, "      Case " + Str(Axis))
    If AllowZero = 0 And Axis < PmoEmitRank(*Src)
      PmoEmitLine(File, "        If raw = 0 : raw = " + Str(PmoEmitDim(*Src, Axis)) + " : EndIf")
    EndIf
    PmoEmitLine(File, "        If raw = -1")
    PmoEmitLine(File, "          inferred = inferred + 1")
    PmoEmitLine(File, "          inferredExtent = " + Str(Extent))
    PmoEmitLine(File, "        Else")
    PmoEmitLine(File, "          If raw <> " + Str(Extent) + " : PmOnnxRuntimeOk = 0 : EndIf")
    PmoEmitLine(File, "          product = product * raw")
    PmoEmitLine(File, "        EndIf")
  Next
  PmoEmitLine(File, "    EndSelect") : PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "  If inferred > 1 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "  If inferred = 0 And product <> " + Str(*Src\Elements) + " : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "  If inferred = 1 And product * inferredExtent <> " + Str(*Src\Elements) + " : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitTransposeHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Rank.i
  Protected Axis.i
  Protected SourceAxis.i
  Protected Coord.s
  Protected Expr.s
  If *Src = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Transpose has missing values") : EndIf
  Rank = PmoEmitRank(*Src)
  For Axis = 0 To Rank - 1
    If PmoEmitAttrListCount(*Node, "perm")
      SourceAxis = PmoEmitAttrListI(*Node, "perm", Axis, -1)
    Else
      SourceAxis = Rank - 1 - Axis
    EndIf
    If SourceAxis < 0 Or SourceAxis >= Rank : ProcedureReturn PmoEmitFail("invalid Transpose permutation") : EndIf
    If PmoEmitDim(*Out, Axis) = 1
      Coord = "0"
    Else
      Coord = "((i / " + Str(PmoEmitStride(*Out, Axis)) + ") % " + Str(PmoEmitDim(*Out, Axis)) + ")"
    EndIf
    If PmoEmitStride(*Src, SourceAxis) <> 1 : Coord = "(" + Coord + " * " + Str(PmoEmitStride(*Src, SourceAxis)) + ")" : EndIf
    If Expr <> "" : Expr + " + " : EndIf : Expr + Coord
  Next
  If Expr = "" : Expr = "0" : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  i = 0")
  PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*Src\ElementType, "*src", Expr)))
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure

XIncludeFile "onnx_emit_general.pbi"

Procedure.i PmoEmitBatchedMatMulHelper(File.i, *Ir.PmoIrModel, *Profile.PmoTargetProfile,
                                       *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *A.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected M.q
  Protected K.q
  Protected N.q
  Protected Batches.q
  Protected IA.s
  Protected IB.s
  Protected TempA.PmoIrValue
  Protected TempB.PmoIrValue
  Protected TempOut.PmoIrValue
  Protected Rank.i
  Protected *Quant.PmoIrQuantWeight
  If *A = 0 Or *B = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("MatMul has missing values") : EndIf
  If PmoEmitRank(*A) <= 2 And PmoEmitRank(*B) <= 2 : ProcedureReturn #True : EndIf
  ; Broadcast only the batch prefixes. Temporary value records make the same
  ; checked expression builder usable without a second indexing algorithm.
  InitializeStructure(@TempA, PmoIrValue) : InitializeStructure(@TempB, PmoIrValue)
  InitializeStructure(@TempOut, PmoIrValue)
  ; A one-dimensional operand has no batch axes and adds no matrix axis to the
  ; result (MatMul-13 follows numpy.matmul; onnx_emit_general.pbi).
  For Rank = 0 To PmoEmitRank(*A) - 3 : AddElement(TempA\Dims()) : TempA\Dims() = PmoEmitDim(*A, Rank) : Next
  For Rank = 0 To PmoEmitRank(*B) - 3 : AddElement(TempB\Dims()) : TempB\Dims() = PmoEmitDim(*B, Rank) : Next
  For Rank = 0 To PmoEmitMatMulBatchRank(*A, *B, *Out) - 1 : AddElement(TempOut\Dims()) : TempOut\Dims() = PmoEmitDim(*Out, Rank) : Next
  IA = PmoEmitBroadcastIndex("batch", @TempA, @TempOut)
  IB = PmoEmitBroadcastIndex("batch", @TempB, @TempOut)
  If PmoEmitMatMulShape(*Node, *A, *B, @M, @K, @N) = 0 : ProcedureReturn #False : EndIf
  Batches = PmoEmitProduct(*Out, 0, PmoEmitMatMulBatchRank(*A, *B, *Out) - 1)
  PmoEmitLine(File, "Procedure " + ProcName + "(*a, *b, *dst)")
  PmoEmitLine(File, "  Protected batch.i") : PmoEmitLine(File, "  batch = 0")
  PmoEmitLine(File, "  While batch < " + Str(Batches))
  *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1))
  If *Quant
    PmoEmitLine(File, "    PmTensorMatMul2Int8(*a + (" + IA + ") * " + Str(M*K*4) +
                      ", *b + (" + IB + ") * " + Str(K*N) + ", *dst + batch * " + Str(M*N*4) +
                      ", " + Str(M) + ", " + Str(K) + ", " + Str(N) + ", " + PmoEmitAddress(*Ir, *Quant\ScaleName) + ")")
  Else
    PmoEmitLine(File, "    " + *Profile\MatMulProcedure + "(*a + (" + IA + ") * " + Str(M*K*4) +
                      ", *b + (" + IB + ") * " + Str(K*N*4) + ", *dst + batch * " + Str(M*N*4) +
                      ", " + Str(M) + ", " + Str(K) + ", " + Str(N) + ")")
  EndIf
  PmoEmitLine(File, "    batch = batch + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ClearStructure(@TempA, PmoIrValue) : ClearStructure(@TempB, PmoIrValue) : ClearStructure(@TempOut, PmoIrValue)
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitConcatHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Input.PmoIrValue
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Params.s
  Protected Axis.i
  Protected Rank.i
  Protected Outer.q
  Protected Inner.q
  Protected OutAxis.q
  Protected Width.q
  Protected Offset.q
  Protected Index.i
  Protected CopyProc.s
  Protected ItemBytes.i
  If *Out = 0 : ProcedureReturn PmoEmitFail("Concat has no output value") : EndIf
  Rank = PmoEmitRank(*Out) : Axis = PmoEmitAttrI(*Node, "axis", 0)
  If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitFail("Concat axis is outside its rank") : EndIf
  If ListSize(*Node\Inputs()) > 7 : ProcedureReturn PmoEmitFail("Concat has more than seven inputs") : EndIf
  Outer = PmoEmitProduct(*Out, 0, Axis - 1) : Inner = PmoEmitProduct(*Out, Axis + 1, Rank - 1)
  OutAxis = PmoEmitDim(*Out, Axis) : CopyProc = PmoEmitCopyProcedure(*Out\ElementType)
  ItemBytes = PmoEmitElementBytes(*Out\ElementType)
  For Index = 0 To ListSize(*Node\Inputs()) - 1
    If Params <> "" : Params + ", " : EndIf : Params + "*p" + Str(Index)
  Next
  If Params <> "" : Params + ", " : EndIf : Params + "*dst"
  PmoEmitLine(File, "Procedure " + ProcName + "(" + Params + ")")
  PmoEmitLine(File, "  Protected o.i") : PmoEmitLine(File, "  o = 0")
  PmoEmitLine(File, "  While o < " + Str(Outer))
  For Index = 0 To ListSize(*Node\Inputs()) - 1
    *Input = PmoEmitValue(*Ir, PmoEmitInput(*Node, Index))
    If *Input = 0 : ProcedureReturn PmoEmitFail("Concat input has no value") : EndIf
    Width = PmoEmitDim(*Input, Axis)
    PmoEmitLine(File, "    " + CopyProc + "(*p" + Str(Index) + " + o * " +
                      Str(Width * Inner * ItemBytes) + ", *dst + (o * " + Str(OutAxis) + " + " +
                      Str(Offset) + ") * " + Str(Inner * ItemBytes) + ", " + Str(Width * Inner) + ")")
    Offset + Width
  Next
  PmoEmitLine(File, "    o = o + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitGatherHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Data.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Indices.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Axis.i
  Protected Rank.i
  Protected IndexRank.i
  Protected OD.i
  Protected DD.i
  Protected Coord.s
  Protected IndexExpr.s
  Protected BaseExpr.s
  If *Data = 0 Or *Indices = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Gather has missing values") : EndIf
  Rank = PmoEmitRank(*Data) : IndexRank = PmoEmitRank(*Indices) : Axis = PmoEmitAttrI(*Node, "axis", 0)
  If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitFail("Gather axis is outside its rank") : EndIf
  For OD = 0 To PmoEmitRank(*Out) - 1
    If PmoEmitDim(*Out, OD) = 1 : Coord = "0" : Else : Coord = "((i / " + Str(PmoEmitStride(*Out, OD)) + ") % " + Str(PmoEmitDim(*Out, OD)) + ")" : EndIf
    If OD < Axis
      If PmoEmitStride(*Data, OD) <> 1 : Coord = "(" + Coord + " * " + Str(PmoEmitStride(*Data, OD)) + ")" : EndIf
      If BaseExpr <> "" : BaseExpr + " + " : EndIf : BaseExpr + Coord
    ElseIf OD < Axis + IndexRank
      DD = OD - Axis
      If PmoEmitStride(*Indices, DD) <> 1 : Coord = "(" + Coord + " * " + Str(PmoEmitStride(*Indices, DD)) + ")" : EndIf
      If IndexExpr <> "" : IndexExpr + " + " : EndIf : IndexExpr + Coord
    Else
      DD = OD - IndexRank + 1
      If PmoEmitStride(*Data, DD) <> 1 : Coord = "(" + Coord + " * " + Str(PmoEmitStride(*Data, DD)) + ")" : EndIf
      If BaseExpr <> "" : BaseExpr + " + " : EndIf : BaseExpr + Coord
    EndIf
  Next
  If IndexExpr = "" : IndexExpr = "0" : EndIf : If BaseExpr = "" : BaseExpr = "0" : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*data, *indices, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected selected.i")
  PmoEmitLine(File, "  Protected dataIndex.i") : PmoEmitLine(File, "  i = 0")
  PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    selected = PmTensorGetI64(*indices, " + IndexExpr + ")")
  PmoEmitLine(File, "    If selected < 0 : selected = selected + " + Str(PmoEmitDim(*Data, Axis)) + " : EndIf")
  PmoEmitLine(File, "    dataIndex = " + BaseExpr + " + selected * " + Str(PmoEmitStride(*Data, Axis)))
  PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*Data\ElementType, "*data", "dataIndex")))
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitCastHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s(), HostDialect.i = #False)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  If *Src = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Cast has missing values") : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected iv.i")
  PmoEmitLine(File, "  Protected fv.f") : PmoEmitLine(File, "  i = 0")
  PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  If *Src\ElementType = 1
    PmoEmitLine(File, "    fv = " + PmoEmitGet(1, "*src", "i"))
    If *Out\ElementType = 7 Or *Out\ElementType = 2 Or *Out\ElementType = 3 Or *Out\ElementType = 6
      ; Float to integer converts toward zero (the ONNX reference and ONNX
      ; Runtime). The bare-metal dialect's assignment already truncates; the
      ; host language's assignment rounds, so it truncates explicitly (forum 860).
      ; UINT8, INT8 and INT32 keep the low bits, as numpy's conversion does.
      If HostDialect : PmoEmitLine(File, "    iv = Int(fv)") : Else : PmoEmitLine(File, "    iv = fv") : EndIf
      PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", "iv"))
    ElseIf *Out\ElementType = 9
      PmoEmitLine(File, "    " + PmoEmitPut(9, "*dst", "i", "Bool(fv <> 0.0)"))
    Else
      PmoEmitLine(File, "    " + PmoEmitPut(1, "*dst", "i", "fv"))
    EndIf
  Else
    PmoEmitLine(File, "    iv = " + PmoEmitGet(*Src\ElementType, "*src", "i"))
    If *Out\ElementType = 1
      PmoEmitLine(File, "    fv = iv") : PmoEmitLine(File, "    " + PmoEmitPut(1, "*dst", "i", "fv"))
    ElseIf *Out\ElementType = 9
      PmoEmitLine(File, "    " + PmoEmitPut(9, "*dst", "i", "Bool(iv <> 0)"))
    Else
      PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", "iv"))
    EndIf
  EndIf
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitCompareHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *A.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected IA.s
  Protected IB.s
  Protected Condition.s
  Protected Symbol.s
  If *A = 0 Or *B = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("comparison has missing values") : EndIf
  IA = PmoEmitBroadcastIndex("i", *A, *Out) : IB = PmoEmitBroadcastIndex("i", *B, *Out)
  Select *Node\Operation
    Case "Equal" : Symbol = " = "
    Case "Greater" : Symbol = " > "
    Case "GreaterOrEqual" : Symbol = " >= "
    Case "Less" : Symbol = " < "
    Case "LessOrEqual" : Symbol = " <= "
  EndSelect
  If *Node\Operation = "And"
    Condition = "(" + PmoEmitGet(*A\ElementType, "*a", IA) + " <> 0) And (" + PmoEmitGet(*B\ElementType, "*b", IB) + " <> 0)"
  Else
    Condition = PmoEmitGet(*A\ElementType, "*a", IA) + Symbol + PmoEmitGet(*B\ElementType, "*b", IB)
  EndIf
  Condition = "Bool(" + Condition + ")"
  PmoEmitLine(File, "Procedure " + ProcName + "(*a, *b, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  i = 0")
  PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    PmTensorPutBool(*dst, i, " + Condition + ")")
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitWhereHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Condition.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected IC.s
  Protected IX.s
  Protected IY.s
  If *Condition = 0 Or *X = 0 Or *Y = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Where has missing values") : EndIf
  IC = PmoEmitBroadcastIndex("i", *Condition, *Out) : IX = PmoEmitBroadcastIndex("i", *X, *Out) : IY = PmoEmitBroadcastIndex("i", *Y, *Out)
  PmoEmitLine(File, "Procedure " + ProcName + "(*cond, *x, *y, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  i = 0")
  PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    If PmTensorGetBool(*cond, " + IC + ") <> 0")
  PmoEmitLine(File, "      " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*X\ElementType, "*x", IX)))
  PmoEmitLine(File, "    Else")
  PmoEmitLine(File, "      " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*Y\ElementType, "*y", IY)))
  PmoEmitLine(File, "    EndIf") : PmoEmitLine(File, "    i = i + 1")
  PmoEmitLine(File, "  Wend") : PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitRangeHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  If *Out = 0 : ProcedureReturn PmoEmitFail("Range has no output value") : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*firstValue, *limitValue, *deltaValue, *dst)")
  PmoEmitLine(File, "  Protected i.i")
  If *Out\ElementType = 7
    PmoEmitLine(File, "  Protected first.i") : PmoEmitLine(File, "  Protected boundary.i")
    PmoEmitLine(File, "  Protected delta.i")
    PmoEmitLine(File, "  first = PmTensorGetI64(*firstValue, 0)")
    PmoEmitLine(File, "  delta = PmTensorGetI64(*deltaValue, 0)")
    PmoEmitLine(File, "  boundary = PmTensorGetI64(*limitValue, 0)")
    PmoEmitLine(File, "  If delta = 0")
    PmoEmitLine(File, "    PmOnnxRuntimeOk = 0")
    PmoEmitLine(File, "    ProcedureReturn")
    PmoEmitLine(File, "  EndIf")
    PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
    PmoEmitLine(File, "    PmTensorPutI64(*dst, i, first + i * delta)")
  Else
    PmoEmitLine(File, "  Protected first.f") : PmoEmitLine(File, "  Protected boundary.f")
    PmoEmitLine(File, "  Protected delta.f")
    PmoEmitLine(File, "  first = PmTensorGet(*firstValue, 0)")
    PmoEmitLine(File, "  delta = PmTensorGet(*deltaValue, 0)")
    PmoEmitLine(File, "  boundary = PmTensorGet(*limitValue, 0)")
    PmoEmitLine(File, "  If delta = 0.0")
    PmoEmitLine(File, "    PmOnnxRuntimeOk = 0")
    PmoEmitLine(File, "    ProcedureReturn")
    PmoEmitLine(File, "  EndIf")
    PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
    PmoEmitLine(File, "    PmTensorPut(*dst, i, first + i * delta)")
  EndIf
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitExpandHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected SourceIndex.s
  Protected Axis.i
  Protected InputAxis.i
  Protected SrcExtent.q
  Protected OutExtent.q
  If *Src = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Expand has missing values") : EndIf
  SourceIndex = PmoEmitBroadcastIndex("i", *Src, *Out)
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *shape, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected requested.i")
  PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(PmoEmitRank(*Out)))
  For Axis = 0 To PmoEmitRank(*Out) - 1
    InputAxis = Axis - (PmoEmitRank(*Out) - PmoEmitRank(*Src))
    SrcExtent = 1 : If InputAxis >= 0 : SrcExtent = PmoEmitDim(*Src, InputAxis) : EndIf
    OutExtent = PmoEmitDim(*Out, Axis)
    If Axis = 0 : PmoEmitLine(File, "    If i = 0") : Else : PmoEmitLine(File, "    ElseIf i = " + Str(Axis)) : EndIf
    PmoEmitLine(File, "      requested = PmTensorGetI64(*shape, i)")
    PmoEmitLine(File, "      If requested <> " + Str(OutExtent) + " And (requested <> 1 Or " +
                      Str(SrcExtent) + " <> " + Str(OutExtent) + ") : PmOnnxRuntimeOk = 0 : EndIf")
  Next
  If PmoEmitRank(*Out) > 0 : PmoEmitLine(File, "    EndIf") : EndIf
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*Src\ElementType, "*src", SourceIndex)))
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitNonZeroHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Axis.i
  Protected Count.q
  Protected Coord.s
  If *Src = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("NonZero has missing values") : EndIf
  Count = PmoEmitDim(*Out, 1)
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected found.i")
  PmoEmitLine(File, "  Protected value.i") : PmoEmitLine(File, "  i = 0 : found = 0")
  PmoEmitLine(File, "  While i < " + Str(*Src\Elements))
  PmoEmitLine(File, "    value = 0")
  If *Src\ElementType = 1
    PmoEmitLine(File, "    If PmTensorGet(*src, i) <> 0.0 : value = 1 : EndIf")
  ElseIf *Src\ElementType = 7
    PmoEmitLine(File, "    If PmTensorGetI64(*src, i) <> 0 : value = 1 : EndIf")
  Else
    PmoEmitLine(File, "    If PmTensorGetBool(*src, i) <> 0 : value = 1 : EndIf")
  EndIf
  PmoEmitLine(File, "    If value <> 0") : PmoEmitLine(File, "      If found < " + Str(Count))
  For Axis = 0 To PmoEmitRank(*Src) - 1
    If PmoEmitDim(*Src, Axis) = 1 : Coord = "0" : Else : Coord = "((i / " + Str(PmoEmitStride(*Src, Axis)) + ") % " + Str(PmoEmitDim(*Src, Axis)) + ")" : EndIf
    PmoEmitLine(File, "        PmTensorPutI64(*dst, " + Str(Axis * Count) + " + found, " + Coord + ")")
  Next
  PmoEmitLine(File, "      EndIf") : PmoEmitLine(File, "      found = found + 1")
  PmoEmitLine(File, "    EndIf") : PmoEmitLine(File, "    i = i + 1")
  PmoEmitLine(File, "  Wend") : PmoEmitLine(File, "  If found <> " + Str(Count) + " : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitScatterHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Data.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Indices.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Updates.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Depth.q
  Protected Tuples.q
  Protected SliceSize.q
  Protected Axis.i
  Protected Extent.q
  Protected Reduction.s = PmoEmitAttrS(*Node, "reduction", "none")
  Protected Scalar.s = ".f"
  If *Data = 0 Or *Indices = 0 Or *Updates = 0 : ProcedureReturn PmoEmitFail("ScatterND has missing values") : EndIf
  ; ScatterND-18: each update is combined with the element already at its
  ; destination, in index order, so a repeated index accumulates.
  If Reduction <> "none" And *Data\ElementType <> 1 And *Data\ElementType <> 7
    ProcedureReturn PmoEmitFail("ScatterND reduction " + Reduction + " is implemented for FLOAT and INT64 data; this data is ONNX type " + Str(*Data\ElementType))
  EndIf
  If *Data\ElementType = 7 : Scalar = ".i" : EndIf
  Depth = PmoEmitDim(*Indices, PmoEmitRank(*Indices) - 1)
  Tuples = PmoEmitProduct(*Indices, 0, PmoEmitRank(*Indices) - 2)
  SliceSize = PmoEmitProduct(*Data, Depth, PmoEmitRank(*Data) - 1)
  PmoEmitLine(File, "Procedure " + ProcName + "(*data, *indices, *updates, *dst)")
  PmoEmitLine(File, "  Protected item.i") : PmoEmitLine(File, "  Protected axis.i")
  PmoEmitLine(File, "  Protected inner.i") : PmoEmitLine(File, "  Protected selected.i")
  PmoEmitLine(File, "  Protected base.i")
  If Reduction <> "none"
    PmoEmitLine(File, "  Protected current" + Scalar) : PmoEmitLine(File, "  Protected update" + Scalar)
  EndIf
  PmoEmitLine(File, "  " + PmoEmitCopyProcedure(*Data\ElementType) + "(*data, *dst, " + Str(*Data\Elements) + ")")
  PmoEmitLine(File, "  item = 0") : PmoEmitLine(File, "  While item < " + Str(Tuples))
  PmoEmitLine(File, "    base = 0 : axis = 0") : PmoEmitLine(File, "    While axis < " + Str(Depth))
  PmoEmitLine(File, "      selected = PmTensorGetI64(*indices, item * " + Str(Depth) + " + axis)")
  For Axis = 0 To Depth - 1
    Extent = PmoEmitDim(*Data, Axis)
    If Axis = 0 : PmoEmitLine(File, "      If axis = 0") : Else : PmoEmitLine(File, "      ElseIf axis = " + Str(Axis)) : EndIf
    PmoEmitLine(File, "        If selected < 0 : selected = selected + " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "        If selected < 0 Or selected >= " + Str(Extent) + " : PmOnnxRuntimeOk = 0 : selected = 0 : EndIf")
    PmoEmitLine(File, "        base = base + selected * " + Str(PmoEmitStride(*Data, Axis)))
  Next
  PmoEmitLine(File, "      EndIf") : PmoEmitLine(File, "      axis = axis + 1")
  PmoEmitLine(File, "    Wend") : PmoEmitLine(File, "    inner = 0")
  PmoEmitLine(File, "    While inner < " + Str(SliceSize))
  If Reduction = "none"
    PmoEmitLine(File, "      " + PmoEmitPut(*Data\ElementType, "*dst", "base + inner",
                    PmoEmitGet(*Data\ElementType, "*updates", "item * " + Str(SliceSize) + " + inner")))
  Else
    PmoEmitLine(File, "      current = " + PmoEmitGet(*Data\ElementType, "*dst", "base + inner"))
    PmoEmitLine(File, "      update = " + PmoEmitGet(*Data\ElementType, "*updates", "item * " + Str(SliceSize) + " + inner"))
    Select Reduction
      Case "add" : PmoEmitLine(File, "      current = current + update")
      Case "mul" : PmoEmitLine(File, "      current = current * update")
      Case "max" : PmoEmitLine(File, "      If update > current : current = update : EndIf")
      Case "min" : PmoEmitLine(File, "      If update < current : current = update : EndIf")
      Default : ProcedureReturn PmoEmitFail("ScatterND reduction " + Reduction + " reached the emitter unvalidated")
    EndSelect
    PmoEmitLine(File, "      " + PmoEmitPut(*Data\ElementType, "*dst", "base + inner", "current"))
  EndIf
  PmoEmitLine(File, "      inner = inner + 1") : PmoEmitLine(File, "    Wend")
  PmoEmitLine(File, "    item = item + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitSliceHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Starts.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Params.s = "*src, *starts, *ends"
  Protected HasAxes.i = Bool(ListSize(*Node\Inputs()) > 3 And PmoEmitInput(*Node, 3) <> "")
  Protected HasSteps.i = Bool(ListSize(*Node\Inputs()) > 4 And PmoEmitInput(*Node, 4) <> "")
  Protected Count.q
  Protected Axis.i
  Protected Extent.q
  Protected OutExtent.q
  Protected Coord.s
  If *Src = 0 Or *Starts = 0 Or *Out = 0 : ProcedureReturn PmoEmitFail("Slice has missing values") : EndIf
  Count = *Starts\Elements
  If HasAxes : Params + ", *axes" : EndIf
  If HasSteps : Params + ", *steps" : EndIf
  Params + ", *dst"
  PmoEmitLine(File, "Procedure " + ProcName + "(" + Params + ")")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected p.i")
  PmoEmitLine(File, "  Protected axis.i") : PmoEmitLine(File, "  Protected strideStep.i")
  PmoEmitLine(File, "  Protected begin.i") : PmoEmitLine(File, "  Protected finish.i")
  PmoEmitLine(File, "  Protected length.i") : PmoEmitLine(File, "  Protected sourceCoord.i")
  PmoEmitLine(File, "  Protected sourceIndex.i")
  PmoEmitLine(File, "  p = 0") : PmoEmitLine(File, "  While p < " + Str(Count))
  If HasAxes : PmoEmitLine(File, "    axis = PmTensorGetI64(*axes, p)") : Else : PmoEmitLine(File, "    axis = p") : EndIf
  PmoEmitLine(File, "    If axis < 0 : axis = axis + " + Str(PmoEmitRank(*Src)) + " : EndIf")
  PmoEmitLine(File, "    If axis < 0 Or axis >= " + Str(PmoEmitRank(*Src)) + " : PmOnnxRuntimeOk = 0 : axis = 0 : EndIf")
  PmoEmitLine(File, "    begin = PmTensorGetI64(*starts, p)")
  PmoEmitLine(File, "    finish = PmTensorGetI64(*ends, p)")
  If HasSteps : PmoEmitLine(File, "    strideStep = PmTensorGetI64(*steps, p)") : Else : PmoEmitLine(File, "    strideStep = 1") : EndIf
  PmoEmitLine(File, "    If strideStep = 0 : PmOnnxRuntimeOk = 0 : strideStep = 1 : EndIf")
  PmoEmitLine(File, "    Select axis")
  For Axis = 0 To PmoEmitRank(*Src) - 1
    Extent = PmoEmitDim(*Src, Axis) : OutExtent = PmoEmitDim(*Out, Axis)
    PmoEmitLine(File, "      Case " + Str(Axis))
    PmoEmitLine(File, "        If begin < 0 : begin = begin + " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "        If finish < 0 : finish = finish + " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "        If strideStep > 0")
    PmoEmitLine(File, "          If begin < 0 : begin = 0 : EndIf : If begin > " + Str(Extent) + " : begin = " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "          If finish < 0 : finish = 0 : EndIf : If finish > " + Str(Extent) + " : finish = " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "          length = 0 : If finish > begin : length = (finish - begin + strideStep - 1) / strideStep : EndIf")
    PmoEmitLine(File, "        Else")
    PmoEmitLine(File, "          If begin < -1 : begin = -1 : EndIf : If begin >= " + Str(Extent) + " : begin = " + Str(Extent - 1) + " : EndIf")
    PmoEmitLine(File, "          If finish < -1 : finish = -1 : EndIf : If finish >= " + Str(Extent) + " : finish = " + Str(Extent - 1) + " : EndIf")
    PmoEmitLine(File, "          length = 0 : If begin > finish : length = (begin - finish + (0 - strideStep) - 1) / (0 - strideStep) : EndIf")
    PmoEmitLine(File, "        EndIf")
    PmoEmitLine(File, "        If length <> " + Str(OutExtent) + " : PmOnnxRuntimeOk = 0 : EndIf")
  Next
  PmoEmitLine(File, "    EndSelect") : PmoEmitLine(File, "    p = p + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  PmoEmitLine(File, "    sourceIndex = 0")
  For Axis = 0 To PmoEmitRank(*Src) - 1
    Extent = PmoEmitDim(*Src, Axis) : OutExtent = PmoEmitDim(*Out, Axis)
    If OutExtent > 1 : Coord = "((i / " + Str(PmoEmitStride(*Out, Axis)) + ") % " + Str(OutExtent) + ")" : Else : Coord = "0" : EndIf
    PmoEmitLine(File, "    sourceCoord = " + Coord)
    PmoEmitLine(File, "    p = 0") : PmoEmitLine(File, "    While p < " + Str(Count))
    If HasAxes : PmoEmitLine(File, "      axis = PmTensorGetI64(*axes, p)") : Else : PmoEmitLine(File, "      axis = p") : EndIf
    PmoEmitLine(File, "      If axis < 0 : axis = axis + " + Str(PmoEmitRank(*Src)) + " : EndIf")
    PmoEmitLine(File, "      If axis = " + Str(Axis))
    PmoEmitLine(File, "        begin = PmTensorGetI64(*starts, p)")
    If HasSteps : PmoEmitLine(File, "        strideStep = PmTensorGetI64(*steps, p)") : Else : PmoEmitLine(File, "        strideStep = 1") : EndIf
    PmoEmitLine(File, "        If begin < 0 : begin = begin + " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "        If strideStep > 0")
    PmoEmitLine(File, "          If begin < 0 : begin = 0 : EndIf : If begin > " + Str(Extent) + " : begin = " + Str(Extent) + " : EndIf")
    PmoEmitLine(File, "        Else")
    PmoEmitLine(File, "          If begin < -1 : begin = -1 : EndIf : If begin >= " + Str(Extent) + " : begin = " + Str(Extent - 1) + " : EndIf")
    PmoEmitLine(File, "        EndIf")
    PmoEmitLine(File, "        sourceCoord = begin + sourceCoord * strideStep")
    PmoEmitLine(File, "      EndIf") : PmoEmitLine(File, "      p = p + 1") : PmoEmitLine(File, "    Wend")
    PmoEmitLine(File, "    If sourceCoord < 0 Or sourceCoord >= " + Str(Extent) + " : PmOnnxRuntimeOk = 0 : sourceCoord = 0 : EndIf")
    PmoEmitLine(File, "    sourceIndex = sourceIndex + sourceCoord * " + Str(PmoEmitStride(*Src, Axis)))
  Next
  PmoEmitLine(File, "    " + PmoEmitPut(*Out\ElementType, "*dst", "i", PmoEmitGet(*Src\ElementType, "*src", "sourceIndex")))
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

XIncludeFile "onnx_emit_norm_small.pbi"
XIncludeFile "onnx_emit_ops.pbi"

Procedure.i PmoEmitGeneratedHelpers(File.i, *Ir.PmoIrModel, *Profile.PmoTargetProfile, Map Calls.s())
  Protected Op.s
  ClearMap(Calls())
  ForEach *Ir\Nodes()
    Op = *Ir\Nodes()\Node\Operation
    Select Op
      Case "Add", "Sub", "Mul", "Div", "Pow"
        If Op = "Pow" And PmoEmitPowNeedsHelper(*Ir, *Ir\Nodes()\Node)
          If PmoEmitPowHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
        ElseIf PmoEmitBinaryHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0
          ProcedureReturn #False
        EndIf
      Case "Softmax"
        If PmoEmitSoftmaxHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Reshape"
        If PmoEmitReshapeHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Transpose"
        If PmoEmitTransposeHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "MatMul"
        If PmoEmitBatchedMatMulHelper(File, *Ir, *Profile, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Concat"
        If PmoEmitConcatHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Gather"
        If PmoEmitGatherHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Cast"
        If PmoEmitCastHelper(File, *Ir, @*Ir\Nodes(), Calls(), Bool(*Profile\SourceDialect = #PMO_SOURCE_HOST)) = 0 : ProcedureReturn #False : EndIf
      Case "Range"
        If PmoEmitRangeHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And"
        If PmoEmitCompareHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Where"
        If PmoEmitWhereHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Slice"
        If PmoEmitSliceHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "Expand"
        If PmoEmitExpandHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "InstanceNormalization", "TopK", "ScatterElements", "Scatter", "ReduceMax", "ReduceProd", "Not", "Pad"
        If PmoEmitNormSmallHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "NonZero"
        If PmoEmitNonZeroHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Case "ScatterND"
        If PmoEmitScatterHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
      Default
        If PmoOpsOwns(Op)
          If PmoEmitOpsHelper(File, *Ir, @*Ir\Nodes(), Calls()) = 0 : ProcedureReturn #False : EndIf
        EndIf
    EndSelect
  Next
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

; The start padding auto_pad SAME_UPPER or SAME_LOWER gives one Conv axis
; (Conv-11): out = ceil(in / stride), total = max(0, (out - 1) * stride +
; (kernel - 1) * dilation + 1 - in), the odd unit at the end for UPPER.
Procedure.q PmoEmitSameConvBegin(InExtent.q, Kernel.q, Stride.q, Dilation.q, Lower.i)
  Protected OutExtent.q, Total.q
  If Stride <= 0 : ProcedureReturn 0 : EndIf
  OutExtent = (InExtent + Stride - 1) / Stride
  Total = (OutExtent - 1) * Stride + (Kernel - 1) * Dilation + 1 - InExtent
  If Total < 0 : Total = 0 : EndIf
  If Lower : ProcedureReturn Total - Total / 2 : EndIf
  ProcedureReturn Total / 2
EndProcedure

Procedure.i PmoEmitNode(File.i, *Ir.PmoIrModel, *Profile.PmoTargetProfile,
                        *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected Op.s = *Node\Operation
  Protected InputCount.i = ListSize(*Node\Inputs())
  Protected OutputCount.i = ListSize(*Node\Outputs())
  ; Several ONNX operators have optional holes at fixed positions (LSTM uses
  ; inputs 0..6 even when only the first four are present).  Keep a checked
  ; fixed scratch range so absent optional inputs become zero rather than an
  ; out-of-bounds compiler access.
  Dim In.s(31)
  Dim Out.s(31)
  If InputCount > 32 Or OutputCount > 32 : ProcedureReturn PmoEmitFail("operator has more than 32 inputs or outputs") : EndIf
  Protected Index.i
  Protected Name.s
  Protected Params.s
  Protected Count.q
  Protected *A.PmoIrValue
  Protected *B.PmoIrValue
  Protected *C.PmoIrValue
  Protected *Y.PmoIrValue
  Protected Rank.i
  Protected Axis.i
  Protected Outer.q
  Protected Width.q
  Protected M.q
  Protected K.q
  Protected N.q
  Protected Alpha.f
  Protected Beta.f
  Protected Scale.f
  Protected Lo.f = -3.402823466e+38
  Protected Hi.f = 3.402823466e+38
  Protected Ok.Integer
  Protected Mode.s
  Protected Coordinate.s
  Protected Bias.s
  Protected Pads0.q
  Protected Pads1.q
  Protected Stride0.q
  Protected Stride1.q
  Protected Dilation0.q
  Protected Dilation1.q
  Protected Groups.q
  Protected FrameStep.q
  Protected FrameLength.q
  Protected *Quant.PmoIrQuantWeight
  Protected *QuantR.PmoIrQuantWeight
  Protected Directions.q
  Protected Hidden.q
  If *Node = 0 : ProcedureReturn PmoEmitFail("source node is null") : EndIf
  ; Every position a node does not list - an optional output such as LSTM's
  ; Y_h and Y_c or BatchNormalization's running statistics, or an optional
  ; trailing input - is the null address "0", exactly as a listed position
  ; with an empty name is (forum 988: an absent Y_h was written as nothing,
  ; source that did not build).
  For Index = 0 To 31 : In(Index) = "0" : Out(Index) = "0" : Next
  For Index = 0 To InputCount - 1
    Name = PmoEmitInput(*Node, Index)
    If Name <> "" And PmoIrCompileTimeInput(Op, Index) = 0 : In(Index) = PmoEmitAddress(*Ir, Name) : Else : In(Index) = "0" : EndIf
  Next
  For Index = 0 To OutputCount - 1
    Name = PmoEmitOutput(*Node, Index)
    If Name <> "" : Out(Index) = PmoEmitAddress(*Ir, Name) : Else : Out(Index) = "0" : EndIf
  Next
  *Y = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  ; A node emitted as its own procedure may leave its first output absent
  ; (RNN or GRU listing only Y_h): the procedure receives "0" for it.
  If *Y = 0 And FindMapElement(Calls(), Str(*Ref\Index)) = 0 : ProcedureReturn PmoEmitFail("node output has no value") : EndIf
  If *Y : Count = *Y\Elements : EndIf
  PmoEmitLine(File, "  ; " + Str(*Ref\Index) + ": " + *Node\Name + " (" + Op + ")")
  If FindMapElement(Calls(), Str(*Ref\Index))
    For Index = 0 To InputCount - 1
      If In(Index) <> "0" : If Params <> "" : Params + ", " : EndIf : Params + In(Index) : EndIf
    Next
    For Index = 0 To OutputCount - 1
      If Params <> "" : Params + ", " : EndIf : Params + Out(Index)
    Next
    PmoEmitLine(File, "  " + Calls() + "(" + Params + ")")
  ElseIf Op = "Identity" Or Op = "Reshape" Or Op = "Flatten" Or Op = "Squeeze" Or Op = "Unsqueeze"
    ProcedureReturn #True
  ElseIf Op = "Add" Or Op = "Sub" Or Op = "Mul" Or Op = "Div" Or Op = "Pow"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    If *A = 0 Or *B = 0 : ProcedureReturn PmoEmitFail(Op + " has missing inputs") : EndIf
    If *A\Elements = Count And *B\Elements = Count
      Select Op
        Case "Add" : Name = "PmTensorAdd"
        Case "Sub" : Name = "PmTensorSub"
        Case "Mul" : Name = "PmTensorMul"
        Case "Div" : Name = "PmTensorDiv"
        Case "Pow" : Name = "PmTensorPow"
      EndSelect
      PmoEmitLine(File, "  " + Name + "(" + In(0) + ", " + In(1) + ", " + Out(0) + ", " + Str(Count) + ")")
    ElseIf *A\Elements = 1 Or *B\Elements = 1
      Index = 0 : If *B\Elements = 1 : Index = 1 : EndIf
      If Op = "Pow"
        PmoEmitLine(File, "  PmTensorPowScalar(" + In(1 - Index) + ", PmTensorGet(" + In(Index) + ", 0), " +
                          Out(0) + ", " + Str(Count) + ", " + Str(Index) + ")")
      Else
        Select Op
          Case "Add" : Axis = 0
          Case "Sub" : Axis = 1
          Case "Mul" : Axis = 2
          Case "Div" : Axis = 3
        EndSelect
        PmoEmitLine(File, "  PmTensorBinaryScalar(" + In(1 - Index) + ", PmTensorGet(" + In(Index) + ", 0), " +
                          Out(0) + ", " + Str(Count) + ", " + Str(Axis) + ", " + Str(Index) + ")")
      EndIf
    Else
      ProcedureReturn PmoEmitFail("broadcast helper was not generated")
    EndIf
  ElseIf Op = "Relu" Or Op = "LeakyRelu" Or Op = "Sigmoid" Or Op = "Tanh"
    If Op = "LeakyRelu"
      PmoEmitLine(File, "  PmTensorLeakyRelu(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ", " +
                        PmoEmitFloat(PmoEmitAttrF(*Node, "alpha", 0.01)) + ")")
    Else
      PmoEmitLine(File, "  PmTensor" + Op + "(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ")")
    EndIf
  ElseIf Op = "Exp" Or Op = "Log" Or Op = "Sqrt" Or Op = "Abs" Or Op = "Neg"
    Select Op
      Case "Exp" : Axis = 0
      Case "Log" : Axis = 1
      Case "Sqrt" : Axis = 2
      Case "Abs" : Axis = 3
      Case "Neg" : Axis = 4
    EndSelect
    ; PmTensorUnaryMath owns a closed op set and refuses anything outside it by
    ; clearing PmTensorUnaryMathOk and leaving the destination untouched.  A
    ; statically emitted program has no DError, so the flag is armed before the
    ; call and folded into PmOnnxRuntimeOk after it - the same hook
    ; PmTensorInt64Ok uses - and the per-node check emitted at the end of this
    ; procedure reports the node and stops the graph.  Without this the refusal
    ; would be silent and the program would run on with a stale destination.
    PmoEmitLine(File, "  PmTensorUnaryMathOk = 1")
    PmoEmitLine(File, "  PmTensorUnaryMath(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ", " + Str(Axis) + ")")
    PmoEmitLine(File, "  If PmTensorUnaryMathOk = 0 : PmOnnxRuntimeOk = 0 : EndIf")
  ElseIf Op = "Sin" Or Op = "Cos" Or Op = "Atan"
    Select Op : Case "Sin" : Axis = 0 : Case "Cos" : Axis = 1 : Default : Axis = 2 : EndSelect
    PmoEmitLine(File, "  PmTensorTrigOk = 1")
    PmoEmitLine(File, "  PmTensorTrig(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ", " + Str(Axis) + ")")
    PmoEmitLine(File, "  If PmTensorTrigOk = 0 : PmOnnxRuntimeOk = 0 : EndIf")
  ElseIf Op = "Floor"
    PmoEmitLine(File, "  PmTensorFloor(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ")")
  ElseIf Op = "Round"
    PmoEmitLine(File, "  PmTensorRoundEven(" + In(0) + ", " + Out(0) + ", " + Str(Count) + ")")
  ElseIf Op = "Resize"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
    If PmoEmitInput(*Node, 2) <> ""
      *C = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 2))
      If *C = 0 : ProcedureReturn PmoEmitFail("Resize scales are not constant") : EndIf
      Scale = PmoEmitConstF(*Ir, PmoEmitInput(*Node, 2), *C\Elements - 1, @Ok)
    Else
      Scale = PmoEmitDim(*Y, PmoEmitRank(*Y) - 1) / PmoEmitDim(*A, PmoEmitRank(*A) - 1)
    EndIf
    Mode = PmoEmitAttrS(*Node, "mode", "nearest") : Coordinate = PmoEmitAttrS(*Node, "coordinate_transformation_mode", "half_pixel")
    Axis = 0 : If Mode <> "nearest" : Axis = 1 : EndIf
    Index = 1 : If Coordinate = "asymmetric" : Index = 0 : EndIf
    PmoEmitLine(File, "  PmTensorResize1D(" + In(0) + ", " + Out(0) + ", " +
                      Str(PmoEmitProduct(*A, 0, PmoEmitRank(*A) - 2)) + ", " +
                      Str(PmoEmitDim(*A, PmoEmitRank(*A) - 1)) + ", " +
                      Str(PmoEmitDim(*Y, PmoEmitRank(*Y) - 1)) + ", " + Str(Axis) + ", " + Str(Index) + ", " + PmoEmitFloat(Scale) + ")")
  ElseIf Op = "MatMul"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    If *A = 0 Or *B = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "an operand has no tensor.") : EndIf
    If PmoEmitMatMulShape(*Node, *A, *B, @M, @K, @N) = 0 : ProcedureReturn #False : EndIf
    *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1))
    If *Quant
      PmoEmitLine(File, "  PmTensorMatMul2Int8(" + In(0) + ", " + In(1) + ", " + Out(0) +
                        ", " + Str(M) + ", " + Str(K) + ", " + Str(N) + ", " + PmoEmitAddress(*Ir, *Quant\ScaleName) + ")")
    Else
      PmoEmitLine(File, "  " + *Profile\MatMulProcedure + "(" + In(0) + ", " + In(1) + ", " + Out(0) +
                        ", " + Str(M) + ", " + Str(K) + ", " + Str(N) + ")")
    EndIf
  ElseIf Op = "Gemm"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    Axis = PmoEmitAttrI(*Node, "transA", 0) : Index = PmoEmitAttrI(*Node, "transB", 0)
    If Axis : M = PmoEmitDim(*A, 1) : K = PmoEmitDim(*A, 0) : Else : M = PmoEmitDim(*A, 0) : K = PmoEmitDim(*A, 1) : EndIf
    If Index : N = PmoEmitDim(*B, 0) : Else : N = PmoEmitDim(*B, 1) : EndIf
    Alpha = PmoEmitAttrF(*Node, "alpha", 1.0) : Beta = PmoEmitAttrF(*Node, "beta", 1.0)
    Count = 0 : If InputCount > 2 And PmoEmitInput(*Node, 2) <> "" : *C = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2)) : Count = *C\Elements : EndIf
    PmoEmitLine(File, "  PmOnnxGemm\A = " + In(0)) : PmoEmitLine(File, "  PmOnnxGemm\B = " + In(1))
    If Count : PmoEmitLine(File, "  PmOnnxGemm\C = " + In(2)) : Else : PmoEmitLine(File, "  PmOnnxGemm\C = 0") : EndIf : PmoEmitLine(File, "  PmOnnxGemm\Dst = " + Out(0))
    PmoEmitLine(File, "  PmOnnxGemm\M = " + Str(M)) : PmoEmitLine(File, "  PmOnnxGemm\K = " + Str(K))
    PmoEmitLine(File, "  PmOnnxGemm\N = " + Str(N)) : PmoEmitLine(File, "  PmOnnxGemm\TransA = " + Str(Axis))
    PmoEmitLine(File, "  PmOnnxGemm\TransB = " + Str(Index)) : PmoEmitLine(File, "  PmOnnxGemm\Alpha = " + PmoEmitFloat(Alpha))
    PmoEmitLine(File, "  PmOnnxGemm\Beta = " + PmoEmitFloat(Beta)) : PmoEmitLine(File, "  PmOnnxGemm\CCount = " + Str(Count))
    *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1))
    If *Quant : PmoEmitLine(File, "  PmOnnxGemm\WeightScales = " + PmoEmitAddress(*Ir, *Quant\ScaleName)) : Else : PmoEmitLine(File, "  PmOnnxGemm\WeightScales = 0") : EndIf
    PmoEmitLine(File, "  " + *Profile\GemmProcedure + "(@PmOnnxGemm)")
  ElseIf Op = "Softmax"
    ; An axis that is not the last has a generated helper and took the branch
    ; above; this is the contiguous form (onnx_emit_general.pbi).
    If PmoEmitSoftmaxShape(*Ir, *Node, @Outer, @Width, @N) = 0 : ProcedureReturn #False : EndIf
    If N <> 1 : ProcedureReturn PmoEmitNodeFail(*Node, "a non-final axis reached the contiguous kernel without its helper.") : EndIf
    If Outer * Width > 0
      PmoEmitLine(File, "  PmTensorSoftmaxLast(" + In(0) + ", " + Out(0) + ", " + Str(Outer) + ", " + Str(Width) + ")")
    EndIf
  ElseIf Op = "ReduceMean" Or Op = "ReduceSum"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : Rank = PmoEmitRank(*A)
    Axis = Rank - 1
    If InputCount > 1 And PmoEmitInput(*Node, 1) <> "" : Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok) :
    ElseIf PmoEmitAttrListCount(*Node, "axes") : Axis = PmoEmitAttrListI(*Node, "axes", 0, Axis) : EndIf
    If Axis < 0 : Axis + Rank : EndIf
    If Axis <> Rank - 1 : ProcedureReturn PmoEmitFail(Op + " is not on the final axis") : EndIf
    Width = PmoEmitDim(*A, Rank - 1) : Name = "PmTensorReduceMeanLast" : If Op = "ReduceSum" : Name = "PmTensorReduceSumLast" : EndIf
    PmoEmitLine(File, "  " + Name + "(" + In(0) + ", " + Out(0) + ", " + Str(PmoEmitProduct(*A, 0, Rank - 2)) + ", " + Str(Width) + ")")
  ElseIf Op = "CumSum"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : Rank = PmoEmitRank(*A)
    Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok) : If Axis < 0 : Axis + Rank : EndIf
    Outer = PmoEmitProduct(*A, 0, Axis - 1) : Width = PmoEmitDim(*A, Axis)
    Name = "PmTensorCumSumF32" : If *A\ElementType = 7 : Name = "PmTensorCumSumI64" : EndIf
    PmoEmitLine(File, "  " + Name + "(" + In(0) + ", " + Out(0) + ", " + Str(Outer) + ", " + Str(Width) + ", " +
                      Str(PmoEmitProduct(*A, Axis + 1, Rank - 1)) + ", " + Str(PmoEmitAttrI(*Node, "exclusive", 0)) +
                      ", " + Str(PmoEmitAttrI(*Node, "reverse", 0)) + ")")
  ElseIf Op = "LayerNormalization"
    ; Any axis flattens to Outer blocks of Width elements; the optional Mean
    ; and InvStdDev outputs go through the kernel's argument block
    ; (onnx_emit_general.pbi, forum 850 and 851).
    If PmoEmitLayerNormShape(*Ir, *Node, @Outer, @Width) = 0 : ProcedureReturn #False : EndIf
    Bias = "0" : If InputCount > 2 : Bias = In(2) : EndIf
    If PmoEmitOutputNamed(*Node, 1) Or PmoEmitOutputNamed(*Node, 2)
      If Out(1) = "" : Out(1) = "0" : EndIf
      If Out(2) = "" : Out(2) = "0" : EndIf
      PmoEmitLine(File, "  PmOnnxLayerNorm\Src = " + In(0)) : PmoEmitLine(File, "  PmOnnxLayerNorm\Scale = " + In(1))
      PmoEmitLine(File, "  PmOnnxLayerNorm\Bias = " + Bias) : PmoEmitLine(File, "  PmOnnxLayerNorm\Dst = " + Out(0))
      PmoEmitLine(File, "  PmOnnxLayerNorm\Mean = " + Out(1)) : PmoEmitLine(File, "  PmOnnxLayerNorm\InvStdDev = " + Out(2))
      PmoEmitLine(File, "  PmOnnxLayerNorm\Outer = " + Str(Outer)) : PmoEmitLine(File, "  PmOnnxLayerNorm\Width = " + Str(Width))
      PmoEmitLine(File, "  PmOnnxLayerNorm\Epsilon = " + PmoEmitFloat(PmoEmitAttrF(*Node, "epsilon", 0.00001)))
      PmoEmitLine(File, "  PmTensorLayerNorm(@PmOnnxLayerNorm)")
    Else
      PmoEmitLine(File, "  PmTensorLayerNormLast(" + In(0) + ", " + In(1) + ", " + Bias + ", " + Out(0) + ", " +
                        Str(Outer) + ", " + Str(Width) + ", " +
                        PmoEmitFloat(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ")")
    EndIf
  ElseIf Op = "Clip"
    If PmoEmitClipCall(File, *Ir, *Node, In(0), Out(0), In(1), In(2), Count) = 0 : ProcedureReturn #False : EndIf
  ElseIf Op = "BatchNormalization"
    Index = PmoEmitBatchNormForm(*Ir, *Node)
    If Index = 0 : ProcedureReturn #False : EndIf
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
    PmoEmitLine(File, "  PmOnnxBatchNorm\Src = " + In(0)) : PmoEmitLine(File, "  PmOnnxBatchNorm\Scale = " + In(1))
    PmoEmitLine(File, "  PmOnnxBatchNorm\Bias = " + In(2)) : PmoEmitLine(File, "  PmOnnxBatchNorm\Mean = " + In(3))
    PmoEmitLine(File, "  PmOnnxBatchNorm\Variance = " + In(4)) : PmoEmitLine(File, "  PmOnnxBatchNorm\Dst = " + Out(0))
    PmoEmitLine(File, "  PmOnnxBatchNorm\N = " + Str(PmoEmitDim(*A, 0))) : PmoEmitLine(File, "  PmOnnxBatchNorm\C = " + Str(PmoEmitDim(*A, 1)))
    PmoEmitLine(File, "  PmOnnxBatchNorm\Spatial = " + Str(PmoEmitProduct(*A, 2, PmoEmitRank(*A) - 1)))
    PmoEmitLine(File, "  PmOnnxBatchNorm\Epsilon = " + PmoEmitFloat(PmoEmitAttrF(*Node, "epsilon", 0.00001)))
    If Index = 2
      ; training_mode = 1 (BatchNormalization-15): batch statistics, and the
      ; running statistics written to outputs 1 and 2.
      PmoEmitLine(File, "  PmOnnxBatchNorm\RunningMean = " + Out(1)) : PmoEmitLine(File, "  PmOnnxBatchNorm\RunningVariance = " + Out(2))
      PmoEmitLine(File, "  PmOnnxBatchNorm\Momentum = " + PmoEmitFloat(PmoEmitAttrF(*Node, "momentum", 0.9)))
      PmoEmitLine(File, "  PmTensorBatchNormTraining(@PmOnnxBatchNorm)")
    Else
      PmoEmitLine(File, "  PmTensorBatchNorm(@PmOnnxBatchNorm)")
    EndIf
  ElseIf Op = "Conv"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    Bias = "0" : If InputCount > 2 : Bias = In(2) : EndIf : Groups = PmoEmitAttrI(*Node, "group", 1)
    Pads0 = PmoEmitAttrListI(*Node, "pads", 0, 0) : Pads1 = PmoEmitAttrListI(*Node, "pads", 1, 0)
    Stride0 = PmoEmitAttrListI(*Node, "strides", 0, 1) : Stride1 = PmoEmitAttrListI(*Node, "strides", 1, 1)
    Dilation0 = PmoEmitAttrListI(*Node, "dilations", 0, 1) : Dilation1 = PmoEmitAttrListI(*Node, "dilations", 1, 1)
    ; auto_pad (Conv-11): VALID pads nothing; SAME_UPPER and SAME_LOWER pad so
    ; that out = ceil(in / stride), the odd pixel at the end for UPPER and at
    ; the start for LOWER. The kernels take the start padding; the end padding
    ; is implied by the output extent.
    Mode = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
    If Mode = "VALID"
      Pads0 = 0 : Pads1 = 0
    ElseIf Mode = "SAME_UPPER" Or Mode = "SAME_LOWER"
      Pads0 = PmoEmitSameConvBegin(PmoEmitDim(*A, 2), PmoEmitDim(*B, 2), Stride0, Dilation0, Bool(Mode = "SAME_LOWER"))
      If PmoEmitRank(*A) = 4
        Pads1 = PmoEmitSameConvBegin(PmoEmitDim(*A, 3), PmoEmitDim(*B, 3), Stride1, Dilation1, Bool(Mode = "SAME_LOWER"))
      EndIf
    EndIf
    If PmoEmitRank(*A) = 3
      PmoEmitLine(File, "  PmOnnxConv1\Src = " + In(0)) : PmoEmitLine(File, "  PmOnnxConv1\Weight = " + In(1))
      PmoEmitLine(File, "  PmOnnxConv1\Bias = " + Bias) : PmoEmitLine(File, "  PmOnnxConv1\Dst = " + Out(0))
      PmoEmitLine(File, "  PmOnnxConv1\Batches = " + Str(PmoEmitDim(*A, 0))) : PmoEmitLine(File, "  PmOnnxConv1\InChannels = " + Str(PmoEmitDim(*A, 1)))
      PmoEmitLine(File, "  PmOnnxConv1\InWidth = " + Str(PmoEmitDim(*A, 2))) : PmoEmitLine(File, "  PmOnnxConv1\OutChannels = " + Str(PmoEmitDim(*Y, 1)))
      PmoEmitLine(File, "  PmOnnxConv1\OutWidth = " + Str(PmoEmitDim(*Y, 2))) : PmoEmitLine(File, "  PmOnnxConv1\Kernel = " + Str(PmoEmitDim(*B, 2)))
      PmoEmitLine(File, "  PmOnnxConv1\PadLeft = " + Str(Pads0)) : PmoEmitLine(File, "  PmOnnxConv1\Stride = " + Str(Stride0))
      PmoEmitLine(File, "  PmOnnxConv1\Dilation = " + Str(Dilation0)) : PmoEmitLine(File, "  PmOnnxConv1\Groups = " + Str(Groups))
      *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1))
      If *Quant : PmoEmitLine(File, "  PmOnnxConv1\WeightScales = " + PmoEmitAddress(*Ir, *Quant\ScaleName)) : Else : PmoEmitLine(File, "  PmOnnxConv1\WeightScales = 0") : EndIf
      If *Quant And *Profile\AcceleratorInclude <> ""
        PmoEmitLine(File, "  PmTensorConv1DNeon(@PmOnnxConv1)")
      Else
        PmoEmitLine(File, "  PmTensorConv1D(@PmOnnxConv1)")
      EndIf
    Else
      PmoEmitLine(File, "  PmOnnxConv\Src = " + In(0)) : PmoEmitLine(File, "  PmOnnxConv\Weight = " + In(1))
      PmoEmitLine(File, "  PmOnnxConv\Bias = " + Bias) : PmoEmitLine(File, "  PmOnnxConv\Dst = " + Out(0))
      PmoEmitLine(File, "  PmOnnxConv\Batches = " + Str(PmoEmitDim(*A, 0))) : PmoEmitLine(File, "  PmOnnxConv\InChannels = " + Str(PmoEmitDim(*A, 1)))
      PmoEmitLine(File, "  PmOnnxConv\InH = " + Str(PmoEmitDim(*A, 2))) : PmoEmitLine(File, "  PmOnnxConv\InW = " + Str(PmoEmitDim(*A, 3)))
      PmoEmitLine(File, "  PmOnnxConv\OutChannels = " + Str(PmoEmitDim(*Y, 1))) : PmoEmitLine(File, "  PmOnnxConv\OutH = " + Str(PmoEmitDim(*Y, 2)))
      PmoEmitLine(File, "  PmOnnxConv\OutW = " + Str(PmoEmitDim(*Y, 3))) : PmoEmitLine(File, "  PmOnnxConv\KernelH = " + Str(PmoEmitDim(*B, 2)))
      PmoEmitLine(File, "  PmOnnxConv\KernelW = " + Str(PmoEmitDim(*B, 3))) : PmoEmitLine(File, "  PmOnnxConv\PadTop = " + Str(Pads0))
      PmoEmitLine(File, "  PmOnnxConv\PadLeft = " + Str(Pads1)) : PmoEmitLine(File, "  PmOnnxConv\StrideH = " + Str(Stride0))
      PmoEmitLine(File, "  PmOnnxConv\StrideW = " + Str(Stride1)) : PmoEmitLine(File, "  PmOnnxConv\DilationH = " + Str(Dilation0))
      PmoEmitLine(File, "  PmOnnxConv\DilationW = " + Str(Dilation1)) : PmoEmitLine(File, "  PmOnnxConv\Groups = " + Str(Groups))
      *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1))
      If *Quant : PmoEmitLine(File, "  PmOnnxConv\WeightScales = " + PmoEmitAddress(*Ir, *Quant\ScaleName)) : Else : PmoEmitLine(File, "  PmOnnxConv\WeightScales = 0") : EndIf
      PmoEmitLine(File, "  PmTensorConv2D(@PmOnnxConv)")
    EndIf
  ElseIf Op = "ConvTranspose"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    Bias = "0" : If InputCount > 2 : Bias = In(2) : EndIf
    If PmoEmitRank(*A) <> 3
      ProcedureReturn PmoEmitFail("ConvTranspose node " + *Node\Name + " has a rank-" + Str(PmoEmitRank(*A)) + " input; fixed-shape emission implements the one-dimensional ConvTranspose (rank-3 input) only")
    EndIf
    ; auto_pad (ConvTranspose-11): the output extent is in * stride and the
    ; total padding is stride*(in-1) + output_padding + (kernel-1)*dilation + 1
    ; - in*stride; SAME_UPPER puts total/2 at the start, SAME_LOWER the rest.
    Pads0 = PmoEmitAttrListI(*Node, "pads", 0, 0)
    Stride0 = PmoEmitAttrListI(*Node, "strides", 0, 1) : Dilation0 = PmoEmitAttrListI(*Node, "dilations", 0, 1)
    Mode = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
    If Mode = "VALID"
      Pads0 = 0
    ElseIf Mode = "SAME_UPPER" Or Mode = "SAME_LOWER"
      Pads1 = Stride0 * (PmoEmitDim(*A, 2) - 1) + PmoEmitAttrListI(*Node, "output_padding", 0, 0) + (PmoEmitDim(*B, 2) - 1) * Dilation0 + 1 - PmoEmitDim(*A, 2) * Stride0
      If Pads1 < 0
        ProcedureReturn PmoEmitFail("ConvTranspose node " + *Node\Name + " auto_pad " + Mode + " needs a negative total padding (" + Str(Pads1) + "); the specification's output extent in * stride is shorter than the kernel allows")
      EndIf
      If Mode = "SAME_UPPER" : Pads0 = Pads1 / 2 : Else : Pads0 = Pads1 - Pads1 / 2 : EndIf
    EndIf
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Src = " + In(0)) : PmoEmitLine(File, "  PmOnnxConvTranspose1\Weight = " + In(1))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Bias = " + Bias) : PmoEmitLine(File, "  PmOnnxConvTranspose1\Dst = " + Out(0))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Batches = " + Str(PmoEmitDim(*A, 0))) : PmoEmitLine(File, "  PmOnnxConvTranspose1\InChannels = " + Str(PmoEmitDim(*A, 1)))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\InWidth = " + Str(PmoEmitDim(*A, 2))) : PmoEmitLine(File, "  PmOnnxConvTranspose1\OutChannels = " + Str(PmoEmitDim(*Y, 1)))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\OutWidth = " + Str(PmoEmitDim(*Y, 2))) : PmoEmitLine(File, "  PmOnnxConvTranspose1\Kernel = " + Str(PmoEmitDim(*B, 2)))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\PadLeft = " + Str(Pads0))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Stride = " + Str(Stride0))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Dilation = " + Str(Dilation0))
    PmoEmitLine(File, "  PmOnnxConvTranspose1\Groups = " + Str(PmoEmitAttrI(*Node, "group", 1)))
    PmoEmitLine(File, "  PmTensorConvTranspose1D(@PmOnnxConvTranspose1)")
  ElseIf Op = "STFT"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
    FrameStep = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
    If InputCount > 3 And PmoEmitInput(*Node, 3) <> ""
      FrameLength = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 3), 0, @Ok)
    Else
      *C = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
      If *C = 0 : ProcedureReturn PmoEmitFail("STFT window has no static value") : EndIf
      FrameLength = *C\Elements
    EndIf
    Bias = "0" : If InputCount > 2 And PmoEmitInput(*Node, 2) <> "" : Bias = In(2) : EndIf
    PmoEmitLine(File, "  PmOnnxStft\Signal = " + In(0)) : PmoEmitLine(File, "  PmOnnxStft\Window = " + Bias)
    PmoEmitLine(File, "  PmOnnxStft\Dst = " + Out(0)) : PmoEmitLine(File, "  PmOnnxStft\Batches = " + Str(PmoEmitDim(*A, 0)))
    PmoEmitLine(File, "  PmOnnxStft\SignalLength = " + Str(PmoEmitDim(*A, 1))) : PmoEmitLine(File, "  PmOnnxStft\FrameStep = " + Str(FrameStep))
    PmoEmitLine(File, "  PmOnnxStft\FrameLength = " + Str(FrameLength)) : PmoEmitLine(File, "  PmOnnxStft\Frames = " + Str(PmoEmitDim(*Y, 1)))
    PmoEmitLine(File, "  PmOnnxStft\Bins = " + Str(PmoEmitDim(*Y, 2)))
    PmoEmitLine(File, "  PmOnnxStft\Scratch = PmOnnxArenaBase + " + Str(*Ir\StftScratchOffset))
    PmoEmitLine(File, "  PmOnnxStft\ScratchComplex = " + Str(*Ir\StftScratchComplex))
    PmoEmitLine(File, "  If PmTensorStft(@PmOnnxStft) = 0 : PmOnnxRuntimeOk = 0 : EndIf")
  ElseIf Op = "LSTM"
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
    Hidden = PmoEmitAttrI(*Node, "hidden_size", PmoEmitDim(*B, 1) / 4) : Directions = PmoEmitDim(*B, 0)
    For Index = 0 To 6 : If In(Index) = "" : In(Index) = "0" : EndIf : Next
    PmoEmitLine(File, "  PmOnnxLstm\X = " + In(0)) : PmoEmitLine(File, "  PmOnnxLstm\W = " + In(1))
    PmoEmitLine(File, "  PmOnnxLstm\R = " + In(2)) : PmoEmitLine(File, "  PmOnnxLstm\B = " + In(3))
    PmoEmitLine(File, "  PmOnnxLstm\SeqLens = " + In(4)) : PmoEmitLine(File, "  PmOnnxLstm\InitialH = " + In(5))
    PmoEmitLine(File, "  PmOnnxLstm\InitialC = " + In(6)) : PmoEmitLine(File, "  PmOnnxLstm\Y = " + Out(0))
    PmoEmitLine(File, "  PmOnnxLstm\YH = " + Out(1)) : PmoEmitLine(File, "  PmOnnxLstm\YC = " + Out(2))
    PmoEmitLine(File, "  PmOnnxLstm\Sequence = " + Str(PmoEmitDim(*A, 0))) : PmoEmitLine(File, "  PmOnnxLstm\Batch = " + Str(PmoEmitDim(*A, 1)))
    PmoEmitLine(File, "  PmOnnxLstm\InputSize = " + Str(PmoEmitDim(*A, 2))) : PmoEmitLine(File, "  PmOnnxLstm\Hidden = " + Str(Hidden))
    PmoEmitLine(File, "  PmOnnxLstm\Directions = " + Str(Directions))
    *Quant = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1)) : *QuantR = PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 2))
    If *Quant : PmoEmitLine(File, "  PmOnnxLstm\WScales = " + PmoEmitAddress(*Ir, *Quant\ScaleName)) : Else : PmoEmitLine(File, "  PmOnnxLstm\WScales = 0") : EndIf
    If *QuantR : PmoEmitLine(File, "  PmOnnxLstm\RScales = " + PmoEmitAddress(*Ir, *QuantR\ScaleName)) : Else : PmoEmitLine(File, "  PmOnnxLstm\RScales = 0") : EndIf
    PmoEmitLine(File, "  PmTensorLstm(@PmOnnxLstm)")
  ElseIf PmoRandomIsDraw(Op)
    If Count > #PMO_RANDOM_ELEMENT_LIMIT
      ProcedureReturn PmoEmitFail(PmoRandomLabel(*Node, *Ref\Index) + " output has " + Str(Count) + " elements; the generator numbers elements with 32 bits, so at most 4294967295.")
    EndIf
    *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
    If Op = "Multinomial"
      If PmoEmitRank(*A) <> 2 Or PmoEmitDim(*A, 1) < 1
        ProcedureReturn PmoEmitFail(PmoRandomLabel(*Node, *Ref\Index) + " input has rank " + Str(PmoEmitRank(*A)) + "; Multinomial takes [batch_size, class_size] with at least one class.")
      EndIf
      PmoEmitLine(File, "  " + PmoRandomDrawFixedCall(*Node, *Ref\Index, Out(0), In(0), Count, PmoEmitDim(*A, 0), PmoEmitDim(*A, 1)))
    Else
      PmoEmitLine(File, "  " + PmoRandomDrawFixedCall(*Node, *Ref\Index, Out(0), In(0), Count, 0, 0))
    EndIf
  ElseIf PmoRandomIsOp(Op)
    If Count > #PMO_RANDOM_ELEMENT_LIMIT
      ProcedureReturn PmoEmitFail(PmoRandomLabel(*Node, *Ref\Index) + " output has " + Str(Count) + " elements; the generator numbers elements with 32 bits, so at most 4294967295.")
    EndIf
    PmoEmitLine(File, "  " + PmoRandomFixedCall(*Node, *Ref\Index, Out(0), Count))
  Else
    ProcedureReturn PmoEmitFail("validated operator reached no source emitter: " + Op)
  EndIf
  ; An INT8 operator refuses a NaN or an infinity in its activations (fault 1)
  ; and working memory smaller than it needs (fault 2) by leaving its output
  ; unwritten and saying so in PmTensorInt8Fault; that fails the node here.
  If (Op = "MatMul" Or Op = "Gemm" Or Op = "Conv" Or Op = "LSTM") And
     (PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 1)) Or (Op = "LSTM" And PmoEmitQuantWeight(*Ir, PmoEmitInput(*Node, 2))))
    PmoEmitLine(File, "  If PmTensorInt8Fault <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  EndIf
  If Op <> "Identity" And Op <> "Flatten" And Op <> "Squeeze" And Op <> "Unsqueeze"
    PmoEmitLine(File, "  If PmOnnxRuntimeOk = 0 : PmOnnxRuntimeErrorNode = " + Str(*Ref\Index) + " : ProcedureReturn 0 : EndIf")
  EndIf
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.s PmoEmitHex32(Value.l)
  ProcedureReturn "$" + RSet(Hex(Value, #PB_Long), 8, "0")
EndProcedure

Procedure.i PmoEmitReadWeightHeader(Path.s, *Header)
  Protected File.i
  If *Header = 0 : ProcedureReturn PmoEmitFail("internal weight-header destination is null") : EndIf
  File = ReadFile(#PB_Any, Path)
  If File = 0 : ProcedureReturn PmoEmitFail("cannot open packed weights: " + Path) : EndIf
  If Lof(File) < #PMO_WEIGHT_HEADER_BYTES Or
     ReadData(File, *Header, #PMO_WEIGHT_HEADER_BYTES) <> #PMO_WEIGHT_HEADER_BYTES
    CloseFile(File)
    ProcedureReturn PmoEmitFail("packed weights have no complete PMONNXW header")
  EndIf
  CloseFile(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitEmbeddedWeights(File.i, Path.s)
  Protected Input.i
  Protected *Block
  Protected Got.i
  Protected Index.i
  Protected Line.s
  Input = ReadFile(#PB_Any, Path)
  If Input = 0 : ProcedureReturn PmoEmitFail("cannot reopen packed weights for embedding") : EndIf
  *Block = AllocateMemory(4096)
  If *Block = 0 : CloseFile(Input) : ProcedureReturn PmoEmitFail("cannot allocate weight embedding buffer") : EndIf
  PmoEmitLine(File, "DataSection")
  PmoEmitLine(File, "  PmOnnxEmbeddedWeights:")
  Repeat
    Got = ReadData(Input, *Block, 4096)
    If Got <= 0 : Break : EndIf
    Index = 0
    While Index < Got
      If Index % 16 = 0
        If Line <> "" : PmoEmitLine(File, Line) : EndIf
        Line = "    Data.a "
      Else
        Line + ","
      EndIf
      Line + Str(PeekA(*Block + Index) & 255)
      Index + 1
    Wend
    If Line <> "" : PmoEmitLine(File, Line) : Line = "" : EndIf
  ForEver
  PmoEmitLine(File, "EndDataSection")
  PmoEmitLine(File)
  FreeMemory(*Block)
  CloseFile(Input)
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

Procedure.i PmoEmitAddressSelector(File.i, *Ir.PmoIrModel, Kind.s)
  Protected Index.i
  Protected Name.s
  If Kind = "Input"
    PmoEmitLine(File, "Procedure.i PmOnnxInputAddress(index.i)")
    PmoEmitLine(File, "  If PmOnnxArenaBase = 0 : ProcedureReturn 0 : EndIf")
    PmoEmitLine(File, "  Select index")
    Index = 0
    ForEach *Ir\Inputs()
      PmoEmitLine(File, "    Case " + Str(Index) + " : ProcedureReturn " + PmoEmitAddress(*Ir, *Ir\Inputs()))
      Index + 1
    Next
  ElseIf Kind = "Output"
    PmoEmitLine(File, "Procedure.i PmOnnxOutputAddress(index.i)")
    PmoEmitLine(File, "  If PmOnnxArenaBase = 0 Or PmOnnxWeightsBase = 0 : ProcedureReturn 0 : EndIf")
    PmoEmitLine(File, "  Select index")
    Index = 0
    ForEach *Ir\Outputs()
      PmoEmitLine(File, "    Case " + Str(Index) + " : ProcedureReturn " + PmoEmitAddress(*Ir, *Ir\Outputs()))
      Index + 1
    Next
  ElseIf Kind = "Checkpoint"
    PmoEmitLine(File, "Procedure.i PmOnnxCheckpointAddress(index.i)")
    PmoEmitLine(File, "  If PmOnnxArenaBase = 0 Or PmOnnxWeightsBase = 0 : ProcedureReturn 0 : EndIf")
    PmoEmitLine(File, "  Select index")
    Index = 0
    ForEach *Ir\Checkpoints()
      PmoEmitLine(File, "    Case " + Str(Index) + " : ProcedureReturn " + PmoEmitAddress(*Ir, *Ir\Checkpoints()))
      Index + 1
    Next
  Else
    ProcedureReturn PmoEmitFail("unknown generated address-selector kind")
  EndIf
  PmoEmitLine(File, "  EndSelect")
  PmoEmitLine(File, "  ProcedureReturn 0")
  PmoEmitLine(File, "EndProcedure")
  PmoEmitLine(File)
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure


Procedure.i PmoEmitSource(*Ir.PmoIrModel, *Profile.PmoTargetProfile, Destination.s,
                          WeightsPath.s, RuntimeInclude.s, WeightsAddress.q, ArenaAddress.q)
  Protected Temp.s = Destination + ".tmp"
  Protected File.i
  Protected *Header
  Protected Index.i
  Protected *Value.PmoIrValue
  Protected ArenaAllocation.q
  Protected PointerType.s = ".i"
  NewMap Calls.s()
  NewMap UsedOps.i()
  Protected RandomNodes.i
  Protected RandomInclude.s
  PmoEmitError = ""
  If *Ir = 0 Or *Profile = 0 : ProcedureReturn PmoEmitFail("internal emitter input is null") : EndIf
  If *Ir\WeightBytes <= 0 Or FileSize(WeightsPath) <> *Ir\WeightBytes
    ProcedureReturn PmoEmitFail("packed weight size changed before source emission")
  EndIf
  ArenaAllocation = *Ir\ArenaBytes
  UseSHA2Fingerprint()
  If ArenaAllocation < 1 : ArenaAllocation = 1 : EndIf
  If *Profile\SourceDialect = #PMO_SOURCE_HOST : PointerType = "" : EndIf
  ForEach *Ir\Nodes()
    UsedOps(*Ir\Nodes()\Node\Operation) = #True
    If PmoRandomIsOp(*Ir\Nodes()\Node\Operation) : RandomNodes + 1 : EndIf
  Next
  *Header = AllocateMemory(#PMO_WEIGHT_HEADER_BYTES)
  If *Header = 0 : ProcedureReturn PmoEmitFail("cannot allocate PMONNXW header buffer") : EndIf
  If PmoEmitReadWeightHeader(WeightsPath, *Header) = 0 : FreeMemory(*Header) : ProcedureReturn #False : EndIf
  If FileSize(Temp) >= 0 : DeleteFile(Temp) : EndIf
  File = CreateFile(#PB_Any, Temp)
  If File = 0 : FreeMemory(*Header) : ProcedureReturn PmoEmitFail("cannot create temporary source: " + Temp) : EndIf

  PmoEmitLine(File, "; Generated by the PureMetal ONNX ahead-of-time compiler.")
  PmoEmitLine(File, "; Model: " + GetFilePart(*Ir\Source\Path))
  PmoEmitLine(File, "; Backend: " + *Profile\Description)
  PmoEmitLine(File, "; Executable tensor code; no ONNX graph interpreter is present at runtime.")
  If *Profile\SourceDialect = #PMO_SOURCE_PUREMETAL And *Profile\NativeIntegerBytes = 8
    PmoEmitLine(File, "; BUILD THIS AS AN ANVIL PAYLOAD WITH --wants-services, and call")
    PmoEmitLine(File, "; PmOnnxAnvilEnter(x0) as the first statement of Main. Without both, every")
    PmoEmitLine(File, "; convolution runs on one core - the same bytes, several times slower.")
  EndIf
  If *Profile\SourceDialect = #PMO_SOURCE_PUREMETAL : PmoEmitLine(File, "EnableFloatingPoint") : Else : PmoEmitLine(File, "EnableExplicit") : EndIf
  PmoEmitLine(File, "#PMO_USE_BATCHNORM = " + Str(Bool(FindMapElement(UsedOps(), "BatchNormalization"))))
  PmoEmitLine(File, "#PMO_USE_STFT = " + Str(Bool(FindMapElement(UsedOps(), "STFT"))))
  PmoEmitLine(File, "#PMO_USE_RESIZE = " + Str(Bool(FindMapElement(UsedOps(), "Resize"))))
  PmoEmitLine(File, "#PMO_USE_CONVTRANSPOSE = " + Str(Bool(FindMapElement(UsedOps(), "ConvTranspose"))))
  PmoEmitLine(File, "#PMO_USE_CONV = " + Str(Bool(FindMapElement(UsedOps(), "Conv"))))
  PmoEmitLine(File, "#PMO_USE_LSTM = " + Str(Bool(FindMapElement(UsedOps(), "LSTM"))))
  PmoEmitLine(File, "#PMO_USE_INT8 = " + Str(Bool(ListSize(*Ir\QuantWeights()) > 0)))
  PmoEmitLine(File, "#PMO_INT64_SPLIT = " + Str(Bool(*Profile\NativeIntegerBytes = 4)))
  PmoEmitLine(File, "#PMO_INT8_BACKEND = " + Str(1+Bool(*Profile\NativeIntegerBytes=4)+Bool(*Profile\Id="pico2")))
  PmoEmitLine(File)
  If *Profile\MathInclude <> "" : PmoEmitLine(File, "XIncludeFile " + Chr(34) + *Profile\MathInclude + Chr(34)) : EndIf
  If RuntimeInclude <> ""
    If PmoThreads > 0 : PmoEmitLine(File, "#PMO_THREADS = " + Str(PmoThreads)) : EndIf
    PmoEmitLine(File, "XIncludeFile " + Chr(34) + RuntimeInclude + Chr(34))
  Else
    PmoEmitLine(File, "XIncludeFile " + Chr(34) + *Profile\TensorInclude + Chr(34))
  EndIf
  If *Profile\AcceleratorInclude <> "" : PmoEmitLine(File, "XIncludeFile " + Chr(34) + *Profile\AcceleratorInclude + Chr(34)) : EndIf
  If RuntimeInclude <> ""
    PmoEmitLine(File, "XIncludeFile " + Chr(34) + GetPathPart(RuntimeInclude) + "tensor_norm_small_windows.pbi" + Chr(34))
  Else
    PmoEmitLine(File, "XIncludeFile " + Chr(34) + GetPathPart(*Profile\TensorInclude) + "tensor_norm_small.pmi" + Chr(34))
  EndIf
  ; The random operators' generator sits beside the tensor library it writes through.
  If RandomNodes
    RandomInclude = RuntimeInclude : If RandomInclude = "" : RandomInclude = *Profile\TensorInclude : EndIf
    PmoEmitLine(File, "XIncludeFile " + Chr(34) + GetPathPart(RandomInclude) + "tensor_random.pmi" + Chr(34))
  EndIf
  ; The operator-set kernels, one source for every target, only where a node
  ; uses them, so every other model's source is unchanged by them.
  ForEach *Ir\Nodes()
    If PmoOpsOwns(*Ir\Nodes()\Node\Operation) Or *Ir\Nodes()\Node\Operation = "Multinomial"
      RandomInclude = RuntimeInclude : If RandomInclude = "" : RandomInclude = *Profile\TensorInclude : EndIf
      PmoEmitLine(File, "#PMO_OPS_INT32 = " + Str(Bool(*Profile\NativeIntegerBytes = 4)))
      PmoEmitLine(File, "XIncludeFile " + Chr(34) + GetPathPart(RandomInclude) + "tensor_ops.pmi" + Chr(34))
      Break
    EndIf
  Next
  If *Ir\ReducedWeightCount : PmoStorageEmit(File,Bool(*Profile\NativeIntegerBytes=8 And *Profile\SourceDialect=#PMO_SOURCE_PUREMETAL)) : EndIf
  PmoEmitLine(File)
  PmoEmitLine(File, "#PMO_WEIGHT_FILE_BYTES = " + Str(*Ir\WeightBytes))
  PmoEmitLine(File, "#PMO_WEIGHT_DATA_BYTES = " + Str(*Ir\WeightDataBytes))
  PmoEmitLine(File, "#PMO_DEFAULT_WEIGHTS_ADDRESS = " + Str(WeightsAddress))
  PmoEmitLine(File, "#PMO_DEFAULT_ARENA_ADDRESS = " + Str(ArenaAddress))
  PmoEmitLine(File, "#PMO_ARENA_BYTES = " + Str(*Ir\ArenaBytes))
  PmoEmitLine(File, "#PMO_INT8_SCRATCH_OFFSET = " + Str(*Ir\Int8ScratchOffset))
  PmoEmitLine(File, "#PMO_INT8_SCRATCH_BYTES = " + Str(*Ir\Int8ScratchBytes))
  PmoEmitLine(File, "#PMO_STFT_SCRATCH_OFFSET = " + Str(*Ir\StftScratchOffset))
  PmoEmitLine(File, "#PMO_STFT_SCRATCH_BYTES = " + Str(*Ir\StftScratchBytes))
  PmoEmitLine(File, "#PMO_STFT_SCRATCH_COMPLEX = " + Str(*Ir\StftScratchComplex))
  PmoEmitLine(File, "#PMO_ARENA_ALLOCATION_BYTES = " + Str(ArenaAllocation))
  PmoEmitLine(File, "Global PmOnnxWeightsBase.i")
  PmoEmitLine(File, "Global PmOnnxWeightBytes.i")
  PmoEmitLine(File, "Global PmOnnxArenaBase.i")
  PmoEmitLine(File, "Global PmOnnxArenaBytes.i")
  PmoEmitLine(File, "Global PmOnnxRuntimeOk.i")
  PmoEmitLine(File, "Global PmOnnxRuntimeErrorNode.i")
  PmoEmitLine(File, "Global PmOnnxGemm.PmTensorGemmArgs")
  PmoEmitLine(File, "Global PmOnnxBatchNorm.PmTensorBatchNormArgs")
  PmoEmitLine(File, "Global PmOnnxLayerNorm.PmTensorLayerNormArgs")
  PmoEmitLine(File, "Global PmOnnxStft.PmTensorStftArgs")
  PmoEmitLine(File, "Global PmOnnxConvTranspose1.PmTensorConvTranspose1DArgs")
  PmoEmitLine(File, "Global PmOnnxConv1.PmTensorConv1DArgs")
  PmoEmitLine(File, "Global PmOnnxConv.PmTensorConv2DArgs")
  PmoEmitLine(File, "Global PmOnnxLstm.PmTensorLstmArgs")
  PmoEmitLine(File)

  If PmoEmitGeneratedHelpers(File, *Ir, *Profile, Calls()) = 0 : Goto PmoEmitSourceFailed : EndIf
  PmoEmitLine(File, "Procedure.i PmOnnxValidateWeights(*weights" + PointerType + ", bytes.i)")
  PmoEmitLine(File, "  If *weights = 0 Or bytes <> #PMO_WEIGHT_FILE_BYTES : ProcedureReturn 0 : EndIf")
  For Index = 0 To 7
    PmoEmitLine(File, "  If (PeekA(*weights + " + Str(Index) + ") & 255) <> " + Str(PeekA(*Header + Index) & 255) + " : ProcedureReturn 0 : EndIf")
  Next
  PmoEmitLine(File, "  If PeekL(*weights + 8) <> 1 Or PeekL(*weights + 12) <> " + Str(#PMO_WEIGHT_HEADER_BYTES) + " : ProcedureReturn 0 : EndIf")
  PmoEmitLine(File, "  If (PeekL(*weights + 16) & $FFFFFFFF) <> " + PmoEmitHex32(PeekL(*Header + 16)) + " Or (PeekL(*weights + 20) & $FFFFFFFF) <> " + PmoEmitHex32(PeekL(*Header + 20)) + " : ProcedureReturn 0 : EndIf")
  For Index = 0 To 7
    PmoEmitLine(File, "  If (PeekL(*weights + " + Str(56 + Index * 4) + ") & $FFFFFFFF) <> " + PmoEmitHex32(PeekL(*Header + 56 + Index * 4)) + " : ProcedureReturn 0 : EndIf")
  Next
  PmoEmitLine(File, "  If PmTensorCrc32(*weights + " + Str(#PMO_WEIGHT_HEADER_BYTES) + ", #PMO_WEIGHT_DATA_BYTES) <> " + PmoEmitHex32(PeekL(*Header + 40)))
  PmoEmitLine(File, "    ProcedureReturn 0")
  PmoEmitLine(File, "  EndIf")
  PmoEmitLine(File, "  ProcedureReturn 1")
  PmoEmitLine(File, "EndProcedure")
  PmoEmitLine(File)

  PmoEmitLine(File, "Procedure.i PmOnnxBindMemory(*weights" + PointerType + ", weightBytes.i, *arena" + PointerType + ", arenaBytes.i)")
  PmoEmitLine(File, "  If PmOnnxValidateWeights(*weights, weightBytes) = 0 Or *arena = 0 Or arenaBytes < #PMO_ARENA_BYTES : ProcedureReturn 0 : EndIf")
  PmoEmitLine(File, "  PmOnnxWeightsBase = *weights : PmOnnxWeightBytes = weightBytes")
  PmoEmitLine(File, "  PmOnnxArenaBase = *arena : PmOnnxArenaBytes = arenaBytes")
  If RandomNodes : PmoEmitLine(File, "  PmRandomRestart()") : EndIf
  ForEach *Ir\Constants()
    If *Ir\Constants()\StorageKind
      PmoEmitLine(File,"  PmWeightDecode(*weights+"+Str(*Ir\Constants()\PackedOffset)+",*arena+"+Str(*Ir\Constants()\DecodedOffset)+","+Str(*Ir\Constants()\Elements)+","+Str(*Ir\Constants()\StorageKind)+")")
    EndIf
  Next
  If *Ir\Int8ScratchBytes > 0
    PmoEmitLine(File, "  PmTensorInt8Scratch = *arena + #PMO_INT8_SCRATCH_OFFSET")
    PmoEmitLine(File, "  PmTensorInt8ScratchBytes = #PMO_INT8_SCRATCH_BYTES")
  EndIf
  If RuntimeInclude <> "" : PmoEmitLine(File, "  PmPoolStart()") : EndIf
  PmoEmitLine(File, "  ProcedureReturn 1")
  PmoEmitLine(File, "EndProcedure")
  PmoEmitLine(File)
  If RuntimeInclude <> ""
    PmoEmitLine(File, "; Worker threads for the bound model: 0 uses every processor this process may")
    PmoEmitLine(File, "; use; a smaller number lowers it. Returns the count in use (1 while unbound).")
    PmoEmitLine(File, "Procedure.i PmOnnxSetThreads(threads.i) : ProcedureReturn PmPoolSetThreads(threads) : EndProcedure")
    PmoEmitLine(File)
  EndIf
  ; ------------------------------------------------------------------
  ;  THE ANVIL ENTRY, ON THE 64-BIT BARE-METAL TARGETS ONLY.
  ;
  ;  A payload entered by the monitor with the service-table flag set is
  ;  handed that table in x0, and this is the one call that turns it into
  ;  a working four-core convolution. THE HOST OWNS Main() - this file is
  ;  a library and cannot write one - so what is emitted here is the
  ;  single named call the host has to make, as Main's first statement,
  ;  before anything else touches a register.
  ;
  ;  IT IS SAFE NOT TO CALL IT AND SAFE TO CALL IT WITH ZERO. Either way
  ;  every convolution takes the single-core path and produces the same
  ;  bytes, so a host that has not been updated is slower and never wrong.
  ;  That is also why the library does not go looking for the table on its
  ;  own: there is no correct value for it to guess, and guessing an
  ;  address that is not a service table is a wild branch through whatever
  ;  the fifth word of it happens to be.
  ;
  ;  IT RETURNS THE NUMBER OF CORES the monitor will actually run a kernel
  ;  on - 1 when there is no table, when the table is older than the
  ;  parallel group, or when the board has one core. A host that wants to
  ;  report what it got has it in one integer; a host that does not can
  ;  ignore it. It cannot fail, so there is nothing to check.
  ; ------------------------------------------------------------------
  If *Profile\SourceDialect = #PMO_SOURCE_PUREMETAL And *Profile\NativeIntegerBytes = 8
    PmoEmitLine(File, "; Call this FIRST in Main, with the value the monitor left in x0, when this")
    PmoEmitLine(File, "; payload was built --wants-services. Returns the core count; 1 when there")
    PmoEmitLine(File, "; is no parallel service. It never fails and never needs checking.")
    PmoEmitLine(File, "Procedure.i PmOnnxAnvilEnter(services.i)")
    PmoEmitLine(File, "  If PmSetAnvilServices(services) = 0 : ProcedureReturn 1 : EndIf")
    PmoEmitLine(File, "  ProcedureReturn PmParallelCores()")
    PmoEmitLine(File, "EndProcedure")
    PmoEmitLine(File)
  EndIf
  If RandomNodes
    PmoEmitLine(File, "; The model seed of the random operators (32 bits; 0 until set). Setting it")
    PmoEmitLine(File, "; restarts the request numbers, so the next execution repeats request 0.")
    PmoEmitLine(File, "; Returns 0 for a value wider than 32 bits. Nodes with a seed attribute keep it.")
    PmoEmitLine(File, "Procedure.i PmOnnxSetRandomSeed(seed.i)")
    PmoEmitLine(File, "  ProcedureReturn PmRandomSetSeed(seed)")
    PmoEmitLine(File, "EndProcedure")
    PmoEmitLine(File)
  EndIf
  PmoEmitLine(File, "Procedure.i PmOnnxInputCount() : ProcedureReturn " + Str(ListSize(*Ir\Inputs())) + " : EndProcedure")
  PmoEmitLine(File, "Procedure.i PmOnnxOutputCount() : ProcedureReturn " + Str(ListSize(*Ir\Outputs())) + " : EndProcedure")
  PmoEmitLine(File, "Procedure.i PmOnnxCheckpointCount() : ProcedureReturn " + Str(ListSize(*Ir\Checkpoints())) + " : EndProcedure")
  PmoEmitLine(File)
  If PmoEmitAddressSelector(File, *Ir, "Input") = 0 Or PmoEmitAddressSelector(File, *Ir, "Output") = 0 Or
     PmoEmitAddressSelector(File, *Ir, "Checkpoint") = 0 : Goto PmoEmitSourceFailed : EndIf
  PmoEmitLine(File, "Procedure.i PmOnnxLastErrorNode() : ProcedureReturn PmOnnxRuntimeErrorNode : EndProcedure")
  PmoEmitLine(File)
  PmoEmitLine(File, "Procedure.i PmOnnxExecuteBound()")
  If *Profile\NativeIntegerBytes = 4
    PmoEmitLine(File, "  Protected pmoInt64Index.i")
  EndIf
  PmoEmitLine(File, "  If PmOnnxWeightsBase = 0 Or PmOnnxArenaBase = 0 : ProcedureReturn 0 : EndIf")
  PmoEmitLine(File, "  PmOnnxRuntimeOk = 1 : PmOnnxRuntimeErrorNode = 0")
  If *Ir\Int8ScratchBytes > 0 : PmoEmitLine(File, "  PmTensorInt8Fault = 0") : EndIf
  If RandomNodes : PmoEmitLine(File, "  PmRandomBeginRequest()") : EndIf
  If *Profile\NativeIntegerBytes = 4
    PmoEmitLine(File, "  PmTensorInt64Ok = 1")
    ForEach *Ir\Inputs()
      *Value = PmoEmitValue(*Ir, *Ir\Inputs())
      If *Value And *Value\ElementType = 7
        PmoEmitLine(File, "  pmoInt64Index = 0")
        PmoEmitLine(File, "  While pmoInt64Index < " + Str(*Value\Elements))
        PmoEmitLine(File, "    PmTensorGetI64(" + PmoEmitAddress(*Ir, *Ir\Inputs()) + ", pmoInt64Index)")
        PmoEmitLine(File, "    pmoInt64Index = pmoInt64Index + 1")
        PmoEmitLine(File, "  Wend")
      EndIf
    Next
    PmoEmitLine(File, "  If PmTensorInt64Ok = 0 : PmOnnxRuntimeErrorNode = -1 : ProcedureReturn 0 : EndIf")
  EndIf
  ForEach *Ir\Nodes()
    If PmoEmitNode(File, *Ir, *Profile, @*Ir\Nodes(), Calls()) = 0 : Goto PmoEmitSourceFailed : EndIf
    If *Profile\NativeIntegerBytes = 4
      PmoEmitLine(File, "  If PmTensorInt64Ok = 0 : PmOnnxRuntimeOk = 0 : PmOnnxRuntimeErrorNode = " + Str(*Ir\Nodes()\Index) + " : ProcedureReturn 0 : EndIf")
    EndIf
  Next
  PmoEmitLine(File, "  ProcedureReturn PmOnnxRuntimeOk")
  PmoEmitLine(File, "EndProcedure")
  PmoEmitLine(File)
  PmoEmitLine(File, "; Bind once; fill inputs and ExecuteBound repeatedly; copy outputs before reuse.")
  PmoEmitLine(File, "; Caller-owned weights and arena stay resident. One active request at a time.")
  PmoEmitLine(File, "Procedure PmOnnxUnbindMemory()")
  PmoEmitLine(File, "  PmOnnxWeightsBase = 0 : PmOnnxWeightBytes = 0")
  PmoEmitLine(File, "  PmOnnxArenaBase = 0 : PmOnnxArenaBytes = 0")
  PmoEmitLine(File, "  PmOnnxRuntimeOk = 0 : PmOnnxRuntimeErrorNode = 0")
  If *Ir\Int8ScratchBytes > 0
    PmoEmitLine(File, "  PmTensorInt8Scratch = 0 : PmTensorInt8ScratchBytes = 0")
    PmoEmitLine(File, "  PmTensorInt8Release()")
  EndIf
  If RuntimeInclude <> "" : PmoEmitLine(File, "  PmPoolStop()") : EndIf
  PmoEmitLine(File, "EndProcedure")
  If *Profile\Launch = #PMO_LAUNCH_EMBEDDED_WEIGHTS
    If PmoEmitEmbeddedWeights(File, WeightsPath) = 0 : Goto PmoEmitSourceFailed : EndIf
  EndIf
  CloseFile(File) : File = 0
  FreeMemory(*Header) : *Header = 0
  If FileSize(Destination) >= 0 And DeleteFile(Destination) = 0
    PmoEmitFail("cannot replace existing generated source") : Goto PmoEmitSourceFailed
  EndIf
  If RenameFile(Temp, Destination) = 0
    ProcedureReturn PmoEmitFail("cannot publish generated source: " + Destination)
  EndIf
  ProcedureReturn #True

  PmoEmitSourceFailed:
  If File : CloseFile(File) : EndIf
  If *Header : FreeMemory(*Header) : EndIf
  If FileSize(Temp) >= 0 : DeleteFile(Temp) : EndIf
  ProcedureReturn #False
EndProcedure
