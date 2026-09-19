; ============================================================================
; onnx_emit_general.pbi - operator forms the fixed-shape emitter decides from
; the ONNX specification rather than from a fixed kernel call
; ----------------------------------------------------------------------------
; Included by onnx_emit.pbi before its generated helpers. Each section decides
; one operator's shape parameters, element types and optional outputs from the
; specification section it cites (https://onnx.ai/onnx/operators/), or refuses
; the node in one sentence that names the operator, the node and the value, so
; the emitter never computes a different form from the one the model asked for.
; ============================================================================

Procedure.i PmoEmitNodeFail(*Node.PmoOnnxNode, Reason.s)
  Protected Name.s = *Node\Name
  If Name = "" : Name = "(unnamed)" : EndIf
  ProcedureReturn PmoEmitFail(*Node\Operation + " node " + Name + ": " + Reason)
EndProcedure

; The ai.onnx opset the model imports; operators whose attribute defaults or
; semantics changed between versions read it.
Procedure.q PmoEmitOpset(*Ir.PmoIrModel)
  Protected Version.q
  ForEach *Ir\Source\Opsets()
    If *Ir\Source\Opsets()\Domain = "" Or *Ir\Source\Opsets()\Domain = "ai.onnx"
      Version = *Ir\Source\Opsets()\Version
    EndIf
  Next
  ProcedureReturn Version
EndProcedure

Procedure.i PmoEmitOutputNamed(*Node.PmoOnnxNode, Index.i)
  ProcedureReturn Bool(PmoEmitOutput(*Node, Index) <> "")
EndProcedure

; ----------------------------------------------------------------------------
; Clip. Clip-11 and later take min and max as optional scalar inputs, Clip-6
; as attributes. A bound that is a constant is written into the call; a bound
; that arrives while the program runs (a graph input, or the output of another
; node) is read from its tensor when the node runs (forum 848). When min is
; greater than max every element becomes max, which is what the kernel's
; clamp to lo and then to hi gives.
; ----------------------------------------------------------------------------
Procedure.s PmoEmitClipBound(*Ir.PmoIrModel, *Node.PmoOnnxNode, Position.i, Attribute.s, Rail.f, Address.s)
  Protected Name.s = PmoEmitInput(*Node, Position)
  Protected *Bound.PmoIrValue
  Protected Ok.Integer
  Protected Value.f
  If Name = ""
    ProcedureReturn PmoEmitFloat(PmoEmitAttrF(*Node, Attribute, Rail))
  EndIf
  *Bound = PmoEmitValue(*Ir, Name)
  If *Bound = 0 : *Bound = PmoEmitValue(*Ir, PmoIrRoot(*Ir, Name)) : EndIf
  If *Bound = 0
    PmoEmitNodeFail(*Node, "input " + Attribute + " (" + Name + ") has no tensor.")
    ProcedureReturn ""
  EndIf
  If *Bound\ElementType <> 1
    PmoEmitNodeFail(*Node, "input " + Attribute + " is ONNX element type " + Str(*Bound\ElementType) + "; fixed-shape emission clips FLOAT tensors, so the bound must be FLOAT (1).")
    ProcedureReturn ""
  EndIf
  If *Bound\Elements <> 1
    PmoEmitNodeFail(*Node, "input " + Attribute + " holds " + Str(*Bound\Elements) + " elements; the specification requires a scalar.")
    ProcedureReturn ""
  EndIf
  Value = PmoEmitConstF(*Ir, PmoIrRoot(*Ir, Name), 0, @Ok)
  If Ok\i : ProcedureReturn PmoEmitFloat(Value) : EndIf
  ProcedureReturn "PmTensorGet(" + Address + ", 0)"
EndProcedure

Procedure.i PmoEmitClipCall(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, Source.s, Destination.s, MinAddress.s, MaxAddress.s, Count.q)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected Lo.s
  Protected Hi.s
  If *X = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "input has no tensor.") : EndIf
  If *X\ElementType <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "input is ONNX element type " + Str(*X\ElementType) + "; fixed-shape emission implements Clip for FLOAT (1) only.")
  EndIf
  Lo = PmoEmitClipBound(*Ir, *Node, 1, "min", -3.402823466e+38, MinAddress)
  If Lo = "" : ProcedureReturn #False : EndIf
  Hi = PmoEmitClipBound(*Ir, *Node, 2, "max", 3.402823466e+38, MaxAddress)
  If Hi = "" : ProcedureReturn #False : EndIf
  PmoEmitLine(File, "  PmTensorClip(" + Source + ", " + Destination + ", " + Str(Count) + ", " + Lo + ", " + Hi + ")")
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

; ----------------------------------------------------------------------------
; MatMul behaves like numpy.matmul (MatMul-13). A one-dimensional first operand
; is a row [1, K] and a one-dimensional second operand a column [K, 1]; the
; promoted axis is dropped from the result. Everything before the last two
; axes is a broadcast batch (forum 853: a one-dimensional operand used to read
; the extents of axes it does not have).
; ----------------------------------------------------------------------------
Procedure.i PmoEmitMatMulShape(*Node.PmoOnnxNode, *A.PmoIrValue, *B.PmoIrValue, *M.Quad, *K.Quad, *N.Quad)
  Protected RankA.i = PmoEmitRank(*A)
  Protected RankB.i = PmoEmitRank(*B)
  Protected Rows.q
  If RankA < 1 Or RankB < 1
    ProcedureReturn PmoEmitNodeFail(*Node, "an operand is a scalar; the specification requires both operands to have at least one axis.")
  EndIf
  If RankA = 1
    *M\q = 1 : *K\q = PmoEmitDim(*A, 0)
  Else
    *M\q = PmoEmitDim(*A, RankA - 2) : *K\q = PmoEmitDim(*A, RankA - 1)
  EndIf
  If RankB = 1
    Rows = PmoEmitDim(*B, 0) : *N\q = 1
  Else
    Rows = PmoEmitDim(*B, RankB - 2) : *N\q = PmoEmitDim(*B, RankB - 1)
  EndIf
  If Rows <> *K\q
    ProcedureReturn PmoEmitNodeFail(*Node, "the first operand has " + Str(*K\q) + " columns and the second " + Str(Rows) + " rows; the specification requires them to match.")
  EndIf
  ProcedureReturn #True
EndProcedure

; The number of broadcast batch axes in MatMul's result.
Procedure.i PmoEmitMatMulBatchRank(*A.PmoIrValue, *B.PmoIrValue, *Out.PmoIrValue)
  ProcedureReturn PmoEmitRank(*Out) - Bool(PmoEmitRank(*A) > 1) - Bool(PmoEmitRank(*B) > 1)
EndProcedure

; ----------------------------------------------------------------------------
; Softmax. Softmax-13 normalises along the one axis named by axis (default -1).
; Softmax-1 and Softmax-11 coerce the input to two dimensions at axis (default
; 1) and normalise over every element from axis onward. Both reduce to Outer
; blocks of Width elements whose members sit Inner elements apart (forum 852:
; the last axis was always used).
; ----------------------------------------------------------------------------
Procedure.i PmoEmitSoftmaxShape(*Ir.PmoIrModel, *Node.PmoOnnxNode, *Outer.Quad, *Width.Quad, *Inner.Quad)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected Rank.i
  Protected Axis.q
  Protected Given.q
  Protected Opset.q = PmoEmitOpset(*Ir)
  If *X = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "input has no tensor.") : EndIf
  If *X\ElementType <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "input is ONNX element type " + Str(*X\ElementType) + "; fixed-shape emission implements Softmax for FLOAT (1) only.")
  EndIf
  Rank = PmoEmitRank(*X)
  If Opset >= 13 : Given = PmoEmitAttrI(*Node, "axis", -1) : Else : Given = PmoEmitAttrI(*Node, "axis", 1) : EndIf
  Axis = Given : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank
    ProcedureReturn PmoEmitNodeFail(*Node, "axis = " + Str(Given) + " is outside the input's rank " + Str(Rank) + ".")
  EndIf
  *Outer\q = PmoEmitProduct(*X, 0, Axis - 1)
  If Opset >= 13
    *Width\q = PmoEmitDim(*X, Axis) : *Inner\q = PmoEmitProduct(*X, Axis + 1, Rank - 1)
  Else
    *Width\q = PmoEmitProduct(*X, Axis, Rank - 1) : *Inner\q = 1
  EndIf
  ProcedureReturn #True
EndProcedure

; A Softmax whose axis is not the last: the same arithmetic as the runtime's
; PmTensorSoftmaxLast (subtract the block maximum, exponentiate, divide by the
; sum), stepping Inner elements between the members of each block.
Procedure.i PmoEmitSoftmaxHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected Outer.Quad
  Protected Width.Quad
  Protected Inner.Quad
  Protected Member.s
  If PmoEmitSoftmaxShape(*Ir, *Node, @Outer, @Width, @Inner) = 0 : ProcedureReturn #False : EndIf
  If Inner\q = 1 Or Outer\q * Width\q * Inner\q = 0 : ProcedureReturn #True : EndIf
  Member = "base + j * " + Str(Inner\q)
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitLine(File, "  Protected o.i") : PmoEmitLine(File, "  Protected r.i")
  PmoEmitLine(File, "  Protected j.i") : PmoEmitLine(File, "  Protected base.i")
  PmoEmitLine(File, "  Protected v.f") : PmoEmitLine(File, "  Protected mx.f")
  PmoEmitLine(File, "  Protected sum.f")
  PmoEmitLine(File, "  o = 0") : PmoEmitLine(File, "  While o < " + Str(Outer\q))
  PmoEmitLine(File, "    r = 0") : PmoEmitLine(File, "    While r < " + Str(Inner\q))
  PmoEmitLine(File, "      base = o * " + Str(Width\q * Inner\q) + " + r")
  PmoEmitLine(File, "      mx = PmTensorGet(*src, base)")
  PmoEmitLine(File, "      j = 1") : PmoEmitLine(File, "      While j < " + Str(Width\q))
  PmoEmitLine(File, "        v = PmTensorGet(*src, " + Member + ")")
  PmoEmitLine(File, "        If v > mx : mx = v : EndIf")
  PmoEmitLine(File, "        j = j + 1") : PmoEmitLine(File, "      Wend")
  PmoEmitLine(File, "      sum = 0.0")
  PmoEmitLine(File, "      j = 0") : PmoEmitLine(File, "      While j < " + Str(Width\q))
  PmoEmitLine(File, "        v = Exp(PmTensorGet(*src, " + Member + ") - mx)")
  PmoEmitLine(File, "        PmTensorPut(*dst, " + Member + ", v)")
  PmoEmitLine(File, "        sum = sum + v")
  PmoEmitLine(File, "        j = j + 1") : PmoEmitLine(File, "      Wend")
  PmoEmitLine(File, "      j = 0") : PmoEmitLine(File, "      While j < " + Str(Width\q))
  PmoEmitLine(File, "        PmTensorPut(*dst, " + Member + ", PmTensorGet(*dst, " + Member + ") / sum)")
  PmoEmitLine(File, "        j = j + 1") : PmoEmitLine(File, "      Wend")
  PmoEmitLine(File, "      r = r + 1") : PmoEmitLine(File, "    Wend")
  PmoEmitLine(File, "    o = o + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

; ----------------------------------------------------------------------------
; Pow (Pow-15). The base is T (FLOAT or INT64 here, the element types this
; path stores) and the exponent T1, any numeric type; the result has the
; base's type. Each operand is read in its own element type (forum 854: the
; integer bytes were read as FLOAT).
;   FLOAT base:  Pow(x, y) with an integer exponent converted to FLOAT.
;   INT64 base:  an exact integer power by repeated squaring. An exponent
;                given as FLOAT must be a whole number; one that is not has
;                no exact integer answer and stops the program with the node
;                named instead of rounding. A negative exponent gives the
;                integer part of 1 / x^n: 1 for x = 1, +/-1 for x = -1, and 0
;                otherwise; x = 0 with a negative exponent stops the program.
; An unsigned exponent that does not fit a signed integer also stops the
; program. The targets whose INT64 is a checked 32-bit subset refuse a runtime
; INT64 Pow result before anything is emitted (onnx_compile.pbi), so the
; integer power below only runs where INT64 is the native integer.
; ----------------------------------------------------------------------------
Procedure.s PmoEmitPowRead(ElementType.i, Pointer.s, Index.s)
  Select ElementType
    Case 1 : ProcedureReturn "PmTensorGet(" + Pointer + ", " + Index + ")"
    Case 6, 12 : ProcedureReturn "PeekL(" + Pointer + " + (" + Index + ") * 4)"
    Case 7, 13 : ProcedureReturn "PmTensorGetI64(" + Pointer + ", " + Index + ")"
  EndSelect
  ProcedureReturn ""
EndProcedure

Procedure.i PmoEmitPowNeedsHelper(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected *A.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  If *A = 0 Or *B = 0 : ProcedureReturn #False : EndIf
  ProcedureReturn Bool(*A\ElementType <> 1 Or *B\ElementType <> 1)
EndProcedure

Procedure.i PmoEmitPowHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *A.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Out.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index)
  Protected IA.s
  Protected IB.s
  If *A = 0 Or *B = 0 Or *Out = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "an operand or the result has no tensor.") : EndIf
  If *A\ElementType <> 1 And *A\ElementType <> 7
    ProcedureReturn PmoEmitNodeFail(*Node, "the base is ONNX element type " + Str(*A\ElementType) + "; fixed-shape emission implements Pow for a FLOAT (1) or INT64 (7) base.")
  EndIf
  If PmoEmitPowRead(*B\ElementType, "*b", "0") = ""
    ProcedureReturn PmoEmitNodeFail(*Node, "the exponent is ONNX element type " + Str(*B\ElementType) + "; fixed-shape emission reads a FLOAT (1), INT32 (6), INT64 (7), UINT32 (12) or UINT64 (13) exponent.")
  EndIf
  If *Out\ElementType <> *A\ElementType
    ProcedureReturn PmoEmitNodeFail(*Node, "the result is ONNX element type " + Str(*Out\ElementType) + " but the base is " + Str(*A\ElementType) + "; the specification gives the result the base's type.")
  EndIf
  IA = PmoEmitBroadcastIndex("i", *A, *Out) : IB = PmoEmitBroadcastIndex("i", *B, *Out)
  If PmoEmitError <> "" : ProcedureReturn #False : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*a, *b, *dst)")
  PmoEmitLine(File, "  Protected i.i") : PmoEmitLine(File, "  Protected e.i")
  PmoEmitLine(File, "  Protected ef.f") : PmoEmitLine(File, "  Protected back.f")
  If *A\ElementType = 7
    PmoEmitLine(File, "  Protected x.i") : PmoEmitLine(File, "  Protected p.i")
    PmoEmitLine(File, "  Protected r.i")
  EndIf
  PmoEmitLine(File, "  i = 0") : PmoEmitLine(File, "  While i < " + Str(*Out\Elements))
  If *B\ElementType = 1
    PmoEmitLine(File, "    ef = " + PmoEmitPowRead(1, "*b", IB))
  Else
    PmoEmitLine(File, "    e = " + PmoEmitPowRead(*B\ElementType, "*b", IB))
    If *B\ElementType = 12 Or *B\ElementType = 13
      PmoEmitLine(File, "    If e < 0 : PmOnnxRuntimeOk = 0 : e = 0 : EndIf")
    EndIf
    PmoEmitLine(File, "    ef = e")
  EndIf
  If *A\ElementType = 1
    PmoEmitLine(File, "    PmTensorPut(*dst, i, Pow(PmTensorGet(*a, " + IA + "), ef))")
  Else
    If *B\ElementType = 1
      PmoEmitLine(File, "    e = 0")
      PmoEmitLine(File, "    If ef >= -2147483647.0 And ef <= 2147483647.0")
      PmoEmitLine(File, "      e = Int(ef)")
      PmoEmitLine(File, "      back = e")
      PmoEmitLine(File, "      If back <> ef : PmOnnxRuntimeOk = 0 : EndIf")
      PmoEmitLine(File, "    Else")
      PmoEmitLine(File, "      PmOnnxRuntimeOk = 0")
      PmoEmitLine(File, "    EndIf")
    EndIf
    PmoEmitLine(File, "    x = PmTensorGetI64(*a, " + IA + ")")
    PmoEmitLine(File, "    r = 1")
    PmoEmitLine(File, "    If e < 0")
    PmoEmitLine(File, "      If x = 0")
    PmoEmitLine(File, "        PmOnnxRuntimeOk = 0")
    PmoEmitLine(File, "      ElseIf x = -1")
    PmoEmitLine(File, "        If e & 1 : r = -1 : EndIf")
    PmoEmitLine(File, "      ElseIf x <> 1")
    PmoEmitLine(File, "        r = 0")
    PmoEmitLine(File, "      EndIf")
    PmoEmitLine(File, "      e = 0")
    PmoEmitLine(File, "    EndIf")
    PmoEmitLine(File, "    p = x")
    PmoEmitLine(File, "    While e > 0")
    PmoEmitLine(File, "      If e & 1")
    PmoEmitLine(File, "        r = r * p")
    PmoEmitLine(File, "      EndIf")
    PmoEmitLine(File, "      e = e >> 1")
    PmoEmitLine(File, "      If e > 0")
    PmoEmitLine(File, "        p = p * p")
    PmoEmitLine(File, "      EndIf")
    PmoEmitLine(File, "    Wend")
    PmoEmitLine(File, "    PmTensorPutI64(*dst, i, r)")
  EndIf
  PmoEmitLine(File, "    i = i + 1") : PmoEmitLine(File, "  Wend")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn Bool(PmoEmitError = "")
EndProcedure

; ----------------------------------------------------------------------------
; LayerNormalization-17. The normalised axes are [axis, rank - 1] (axis
; default -1), flattened: Outer blocks of Width elements (forum 850: the last
; axis alone was used). Scale and B must be unidirectionally broadcastable to
; X; the form implemented is the one whose extents are X's on every
; normalised axis and 1 (or absent) before axis, so each holds Width elements.
; Mean and InvStdDev have X's shape with every normalised axis set to 1, Outer
; elements each (forum 851: they were never written).
; ----------------------------------------------------------------------------
Procedure.i PmoEmitLayerNormOperand(*Node.PmoOnnxNode, *X.PmoIrValue, *T.PmoIrValue, Label.s, Axis.i)
  Protected Rank.i = PmoEmitRank(*X)
  Protected RankT.i
  Protected Index.i
  Protected XAxis.i
  If *T = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " has no tensor.") : EndIf
  If *T\ElementType <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " is ONNX element type " + Str(*T\ElementType) + "; fixed-shape emission implements LayerNormalization for FLOAT (1) only.")
  EndIf
  RankT = PmoEmitRank(*T)
  If RankT > Rank
    ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " has rank " + Str(RankT) + ", more than X's " + Str(Rank) + ", so it is not unidirectionally broadcastable to X.")
  EndIf
  For Index = 0 To RankT - 1
    XAxis = Rank - RankT + Index
    If XAxis < Axis
      If PmoEmitDim(*T, Index) <> 1
        ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " varies along axis " + Str(XAxis) + ", which is not normalised (axis = " + Str(Axis) + "); fixed-shape emission implements " + Label + " constant across the non-normalised axes.")
      EndIf
    ElseIf PmoEmitDim(*T, Index) <> PmoEmitDim(*X, XAxis)
      ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " has extent " + Str(PmoEmitDim(*T, Index)) + " on axis " + Str(XAxis) + " where X has " + Str(PmoEmitDim(*X, XAxis)) + "; fixed-shape emission implements " + Label + " with X's extent on every normalised axis.")
    EndIf
  Next
  If *T\Elements <> PmoEmitProduct(*X, Axis, Rank - 1)
    ProcedureReturn PmoEmitNodeFail(*Node, "input " + Label + " holds " + Str(*T\Elements) + " elements; fixed-shape emission implements " + Label + " with one element per normalised position (" + Str(PmoEmitProduct(*X, Axis, Rank - 1)) + ").")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitLayerNormShape(*Ir.PmoIrModel, *Node.PmoOnnxNode, *Outer.Quad, *Width.Quad)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Stat.PmoIrValue
  Protected Rank.i
  Protected Given.q = PmoEmitAttrI(*Node, "axis", -1)
  Protected Axis.q
  Protected Index.i
  If *X = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "input X has no tensor.") : EndIf
  If *X\ElementType <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "input X is ONNX element type " + Str(*X\ElementType) + "; fixed-shape emission implements LayerNormalization for FLOAT (1) only.")
  EndIf
  Rank = PmoEmitRank(*X)
  Axis = Given : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank
    ProcedureReturn PmoEmitNodeFail(*Node, "axis = " + Str(Given) + " is outside X's rank " + Str(Rank) + ".")
  EndIf
  If PmoEmitLayerNormOperand(*Node, *X, PmoEmitValue(*Ir, PmoEmitInput(*Node, 1)), "Scale", Axis) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitInput(*Node, 2) <> ""
    If PmoEmitLayerNormOperand(*Node, *X, PmoEmitValue(*Ir, PmoEmitInput(*Node, 2)), "B", Axis) = 0 : ProcedureReturn #False : EndIf
  EndIf
  *Outer\q = PmoEmitProduct(*X, 0, Axis - 1)
  *Width\q = PmoEmitProduct(*X, Axis, Rank - 1)
  For Index = 1 To 2
    If PmoEmitOutputNamed(*Node, Index)
      *Stat = PmoEmitValue(*Ir, PmoEmitOutput(*Node, Index))
      If *Stat = 0 Or *Stat\ElementType <> 1 Or *Stat\Elements <> *Outer\q
        ProcedureReturn PmoEmitNodeFail(*Node, "output " + StringField("Mean|InvStdDev", Index, "|") + " is not a FLOAT tensor of " + Str(*Outer\q) + " elements, the shape the specification gives it.")
      EndIf
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

; ----------------------------------------------------------------------------
; BatchNormalization. BatchNormalization-15 and -14: training_mode 0 has one
; output, Y; training_mode 1 normalises with the batch's own mean and
; population variance and returns running_mean and running_var (forum 856).
; BatchNormalization-9: its four optional outputs are training statistics
; this path does not compute, and BatchNormalization-7's spatial = 0 is the
; deprecated per-element form; both are refused by name.
; Returns 0 on refusal, 1 for inference, 2 for training.
; ----------------------------------------------------------------------------
Procedure.i PmoEmitBatchNormForm(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *T.PmoIrValue
  Protected Opset.q = PmoEmitOpset(*Ir)
  Protected Mode.q = PmoEmitAttrI(*Node, "training_mode", 0)
  Protected Channels.q
  Protected Index.i
  If *X = 0 : ProcedureReturn PmoEmitNodeFail(*Node, "input X has no tensor.") : EndIf
  If *X\ElementType <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "input X is ONNX element type " + Str(*X\ElementType) + "; fixed-shape emission implements BatchNormalization for FLOAT (1) only.")
  EndIf
  If PmoEmitRank(*X) < 2
    ProcedureReturn PmoEmitNodeFail(*Node, "input X has rank " + Str(PmoEmitRank(*X)) + "; the specification requires a batch axis and a channel axis.")
  EndIf
  If PmoEmitAttrI(*Node, "spatial", 1) <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "spatial = " + Str(PmoEmitAttrI(*Node, "spatial", 1)) + " (per-element statistics) is not implemented by fixed-shape emission; spatial 1, the default, is.")
  EndIf
  Channels = PmoEmitDim(*X, 1)
  For Index = 1 To 4
    *T = PmoEmitValue(*Ir, PmoEmitInput(*Node, Index))
    If *T = 0 Or *T\ElementType <> 1 Or *T\Elements <> Channels
      ProcedureReturn PmoEmitNodeFail(*Node, "input " + StringField("scale|B|input_mean|input_var", Index, "|") + " is not a FLOAT tensor of " + Str(Channels) + " elements, one per channel.")
    EndIf
  Next
  If Opset < 14
    For Index = 1 To 4
      If PmoEmitOutputNamed(*Node, Index)
        ProcedureReturn PmoEmitNodeFail(*Node, "output " + Str(Index) + " (" + PmoEmitOutput(*Node, Index) + ") is one of BatchNormalization-9's training statistics, which fixed-shape emission does not compute; name Y only.")
      EndIf
    Next
    ProcedureReturn 1
  EndIf
  If Mode <> 0 And Mode <> 1
    ProcedureReturn PmoEmitNodeFail(*Node, "training_mode = " + Str(Mode) + " is not 0 or 1.")
  EndIf
  If Mode = 0
    If PmoEmitOutputNamed(*Node, 1) Or PmoEmitOutputNamed(*Node, 2)
      ProcedureReturn PmoEmitNodeFail(*Node, "training_mode = 0 names running_mean or running_var; the specification makes those outputs invalid outside training mode.")
    EndIf
    ProcedureReturn 1
  EndIf
  ; The operator's own shape inference requires all three outputs in training
  ; mode, so a node that leaves one out is not a valid model.
  For Index = 1 To 2
    If PmoEmitOutputNamed(*Node, Index) = 0
      ProcedureReturn PmoEmitNodeFail(*Node, "training_mode = 1 without output " + StringField("running_mean|running_var", Index, "|") + "; the specification requires Y, running_mean and running_var in training mode.")
    EndIf
    *T = PmoEmitValue(*Ir, PmoEmitOutput(*Node, Index))
    If *T = 0 Or *T\ElementType <> 1 Or *T\Elements <> Channels
      ProcedureReturn PmoEmitNodeFail(*Node, "output " + StringField("running_mean|running_var", Index, "|") + " is not a FLOAT tensor of " + Str(Channels) + " elements.")
    EndIf
  Next
  ProcedureReturn 2
EndProcedure
