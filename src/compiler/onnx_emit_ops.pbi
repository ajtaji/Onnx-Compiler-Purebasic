; ============================================================================
; onnx_emit_ops.pbi - the operators that complete the ONNX operator set:
; which they are, from which opset each is accepted, and their fixed-shape
; forms
; ----------------------------------------------------------------------------
; Included by onnx_emit.pbi after onnx_emit_norm_small.pbi. For each node
; this writes one generated procedure that fills the parameter arrays of
; runtime/tensor_ops.pmi with the node's static shapes and attributes and
; calls the kernel. Forms the ONNX specification
; (https://onnx.ai/onnx/operators/) allows up to opset 20 are either emitted
; or refused with one sentence naming the operator, the attribute or input
; and the value. The runtime-dimension path implements the same forms; see
; onnx_dynamic_ops.pbi. CastLike becomes Cast before either path plans
; (onnx_compile.pbi) and Size folds to a constant (onnx_ir.pbi).
; ============================================================================

Procedure.i PmoOpsOwns(Operation.s)
  ProcedureReturn Bool(FindString("|Erf|Reciprocal|Ceil|Sign|Softplus|Softsign|Elu|Selu|Celu|HardSigmoid|HardSwish|Mish|Gelu|" +
                                  "ThresholdedRelu|Shrink|IsNaN|IsInf|Min|Max|Sum|Mean|Mod|PRelu|Or|Xor|ReduceMin|ReduceL1|ReduceL2|" +
                                  "ReduceSumSquare|ReduceLogSum|ReduceLogSumExp|ArgMax|ArgMin|LogSoftmax|MaxPool|AveragePool|LpPool|" +
                                  "GlobalMaxPool|GlobalAveragePool|GlobalLpPool|Split|Tile|DepthToSpace|SpaceToDepth|Trilu|" +
                                  "GatherElements|GatherND|OneHot|Einsum|Dropout|CastLike|Size|Tan|Asin|Acos|Sinh|Cosh|Asinh|" +
                                  "Acosh|Atanh|BitwiseNot|BitwiseAnd|BitwiseOr|BitwiseXor|Hardmax|LpNormalization|" +
                                  "MeanVarianceNormalization|LRN|GroupNormalization|EyeLike|Det|Compress|ReverseSequence|Upsample|" +
                                  "RNN|GRU|NonMaxSuppression|RoiAlign|GridSample|", "|" + Operation + "|"))
EndProcedure

; The oldest ai.onnx opset whose definition of an operator is one these
; kernels compute; later definitions add element types, or attributes whose
; defaults keep the older behaviour (checked per node where they do not).
Procedure.i PmoOpsFloor(Operation.s)
  Select Operation
    Case "Softplus", "Softsign", "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp",
         "ArgMax", "ArgMin", "LogSoftmax", "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool", "DepthToSpace",
         "SpaceToDepth", "Size"
      ProcedureReturn 1
    Case "Hardmax", "LpNormalization", "LRN" : ProcedureReturn 1
    Case "LpPool", "GlobalLpPool", "Split" : ProcedureReturn 2
    Case "Reciprocal", "Ceil", "Elu", "Selu", "HardSigmoid", "Tile" : ProcedureReturn 6
    Case "PRelu", "Or", "Xor", "Dropout", "Tan", "Asin", "Acos", "Upsample", "RNN", "GRU" : ProcedureReturn 7
    Case "Min", "Max", "Sum", "Mean" : ProcedureReturn 8
    Case "Erf", "Sign", "Shrink", "IsNaN", "OneHot", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "MeanVarianceNormalization",
         "EyeLike"
      ProcedureReturn 9
    Case "Mod", "ThresholdedRelu", "IsInf", "ReverseSequence", "NonMaxSuppression", "RoiAlign" : ProcedureReturn 10
    Case "GatherElements", "GatherND", "Det", "Compress" : ProcedureReturn 11
    Case "Celu", "Einsum" : ProcedureReturn 12
    Case "HardSwish", "Trilu" : ProcedureReturn 14
    Case "CastLike" : ProcedureReturn 15
    Case "GridSample" : ProcedureReturn 16
    Case "Mish", "BitwiseNot", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "GroupNormalization" : ProcedureReturn 18
    Case "Gelu" : ProcedureReturn 20
  EndSelect
  ProcedureReturn 0
EndProcedure

; 1 when a graph or any graph an If or Loop attribute holds uses one of
; these operators, so the program must include tensor_ops.pmi.
Procedure.i PmoOpsGraphUses(*Graph.PmoOnnxGraph)
  Protected *Sub.PmoOnnxGraph
  If *Graph = 0 : ProcedureReturn #False : EndIf
  ForEach *Graph\Nodes()
    If PmoOpsOwns(*Graph\Nodes()\Operation) And *Graph\Nodes()\Operation <> "CastLike"
      ProcedureReturn #True
    EndIf
    ForEach *Graph\Nodes()\Attributes()
      *Sub = *Graph\Nodes()\Attributes()\Graph
      If *Sub And PmoOpsGraphUses(*Sub) : ProcedureReturn #True : EndIf
    Next
  Next
  ProcedureReturn #False
EndProcedure

; Codes shared by both paths.
Procedure.i PmoOpsUnaryCode(*Node.PmoOnnxNode)
  Select *Node\Operation
    Case "Erf" : ProcedureReturn 0
    Case "Reciprocal" : ProcedureReturn 1
    Case "Ceil" : ProcedureReturn 2
    Case "Sign" : ProcedureReturn 3
    Case "Softplus" : ProcedureReturn 4
    Case "Softsign" : ProcedureReturn 5
    Case "Elu" : ProcedureReturn 6
    Case "Selu" : ProcedureReturn 7
    Case "Celu" : ProcedureReturn 8
    Case "HardSigmoid" : ProcedureReturn 9
    Case "HardSwish" : ProcedureReturn 10
    Case "Mish" : ProcedureReturn 11
    Case "Gelu"
      If PmoEmitAttrS(*Node, "approximate", "none") = "tanh" : ProcedureReturn 13 : EndIf
      ProcedureReturn 12
    Case "ThresholdedRelu" : ProcedureReturn 14
    Case "Shrink" : ProcedureReturn 15
    Case "IsNaN" : ProcedureReturn 16
    Case "IsInf" : ProcedureReturn 17
    Case "Tan" : ProcedureReturn 19
    Case "Asin" : ProcedureReturn 20
    Case "Acos" : ProcedureReturn 21
    Case "Sinh" : ProcedureReturn 22
    Case "Cosh" : ProcedureReturn 23
    Case "Asinh" : ProcedureReturn 24
    Case "Acosh" : ProcedureReturn 25
    Case "Atanh" : ProcedureReturn 26
    Case "BitwiseNot" : ProcedureReturn 27
  EndSelect
  ProcedureReturn -1
EndProcedure

; The attributes an elementwise function takes, and "" or the refusal of
; a value it does not implement.
Procedure.s PmoOpsUnaryAllowed(Operation.s)
  Select Operation
    Case "Elu", "Celu", "ThresholdedRelu" : ProcedureReturn "|alpha|"
    Case "Selu" : ProcedureReturn "|alpha|gamma|"
    Case "HardSigmoid" : ProcedureReturn "|alpha|beta|"
    Case "Shrink" : ProcedureReturn "|bias|lambd|"
    Case "Gelu" : ProcedureReturn "|approximate|"
    Case "IsInf" : ProcedureReturn "|detect_negative|detect_positive|"
  EndSelect
  ProcedureReturn "|"
EndProcedure

Procedure.s PmoOpsUnaryForm(*Node.PmoOnnxNode)
  Protected Text.s
  If *Node\Operation = "Gelu"
    Text = PmoEmitAttrS(*Node, "approximate", "none")
    If Text <> "none" And Text <> "tanh"
      ProcedureReturn "attribute approximate = " + Text + " is not a Gelu approximation; none and tanh are."
    EndIf
  EndIf
  ProcedureReturn ""
EndProcedure

; The binary32 bits of an attribute value, as the kernels load them.
Procedure.s PmoOpsBits(Value.f)
  Protected Bits.l
  PokeF(@Bits, Value)
  ProcedureReturn "$" + RSet(Hex(Bits & $FFFFFFFF), 8, "0")
EndProcedure

; The float parameters of an elementwise function: "PmOpSetBits(@PmOpF(k),
; $bits)" statements, one per entry of the returned list, and the integer
; ones of IsInf.
Procedure PmoOpsUnaryParams(*Node.PmoOnnxNode, List Lines.s())
  Protected F0.f, F1.f, Two.i
  ClearList(Lines())
  Select *Node\Operation
    Case "Elu", "Celu", "ThresholdedRelu" : F0 = PmoEmitAttrF(*Node, "alpha", 1.0)
    Case "Selu" : F0 = PmoEmitAttrF(*Node, "alpha", 1.6732632) : F1 = PmoEmitAttrF(*Node, "gamma", 1.050701) : Two = 1 ; the binary32 defaults
    Case "HardSigmoid" : F0 = PmoEmitAttrF(*Node, "alpha", 0.2) : F1 = PmoEmitAttrF(*Node, "beta", 0.5) : Two = 1
    Case "Shrink" : F0 = PmoEmitAttrF(*Node, "bias", 0.0) : F1 = PmoEmitAttrF(*Node, "lambd", 0.5) : Two = 1
    Case "IsInf"
      AddElement(Lines()) : Lines() = "PmOpI(0) = " + Str(Bool(PmoEmitAttrI(*Node, "detect_positive", 1) <> 0))
      AddElement(Lines()) : Lines() = "PmOpI(1) = " + Str(Bool(PmoEmitAttrI(*Node, "detect_negative", 1) <> 0))
      ProcedureReturn
    Default
      ProcedureReturn
  EndSelect
  AddElement(Lines()) : Lines() = "PmOpSetBits(@PmOpF(0), " + PmoOpsBits(F0) + ")"
  If Two : AddElement(Lines()) : Lines() = "PmOpSetBits(@PmOpF(1), " + PmoOpsBits(F1) + ")" : EndIf
EndProcedure

Procedure.i PmoOpsVariadicCode(*Node.PmoOnnxNode)
  Select *Node\Operation
    Case "Min" : ProcedureReturn 0
    Case "Max" : ProcedureReturn 1
    Case "Sum" : ProcedureReturn 2
    Case "Mean" : ProcedureReturn 3
    Case "Mod" : ProcedureReturn 4 + Bool(PmoEmitAttrI(*Node, "fmod", 0) <> 0)
    Case "Or" : ProcedureReturn 6
    Case "Xor" : ProcedureReturn 7
    Case "PRelu" : ProcedureReturn 8
    Case "BitwiseAnd" : ProcedureReturn 9
    Case "BitwiseOr" : ProcedureReturn 10
    Case "BitwiseXor" : ProcedureReturn 11
  EndSelect
  ProcedureReturn -1
EndProcedure

; The element types a variadic operator computes, as the bits DOpKindOk
; reads (1 FLOAT, 2 INT32, 4 INT64, 8 BOOL).
Procedure.i PmoOpsVariadicKinds(*Node.PmoOnnxNode)
  Select *Node\Operation
    Case "Min", "Max", "Sum", "PRelu" : ProcedureReturn 7
    Case "Mean" : ProcedureReturn 1
    Case "Mod"
      If PmoEmitAttrI(*Node, "fmod", 0) <> 0 : ProcedureReturn 7 : EndIf
      ProcedureReturn 6
    Case "Or", "Xor" : ProcedureReturn 8
    Case "BitwiseAnd", "BitwiseOr", "BitwiseXor" : ProcedureReturn 6
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmoOpsReduceCode(Operation.s)
  Select Operation
    Case "ReduceMin" : ProcedureReturn 0
    Case "ReduceL1" : ProcedureReturn 1
    Case "ReduceL2" : ProcedureReturn 2
    Case "ReduceSumSquare" : ProcedureReturn 3
    Case "ReduceLogSum" : ProcedureReturn 4
    Case "ReduceLogSumExp" : ProcedureReturn 5
  EndSelect
  ProcedureReturn -1
EndProcedure

Procedure.i PmoOpsKindBit(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn 1
    Case 6 : ProcedureReturn 2
    Case 7 : ProcedureReturn 4
    Case 9 : ProcedureReturn 8
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.s PmoOpsKindNames(Bits.i)
  Protected Text.s, Count.i, i.i
  Protected Dim Names.s(3)
  If Bits & 1 : Names(Count) = "FLOAT" : Count + 1 : EndIf
  If Bits & 2 : Names(Count) = "INT32" : Count + 1 : EndIf
  If Bits & 4 : Names(Count) = "INT64" : Count + 1 : EndIf
  If Bits & 8 : Names(Count) = "BOOL" : Count + 1 : EndIf
  For i = 0 To Count - 1
    If i > 0 And i = Count - 1 : Text + " and " : ElseIf i > 0 : Text + ", " : EndIf
    Text + Names(i)
  Next
  ProcedureReturn Text
EndProcedure

; ----------------------------------------------------------------------------
; Einsum's equation, parsed once for both paths. Terms are the input
; subscripts; an ellipsis stands for the leading or middle axes a term does
; not name, and is expanded here to labels of its own (the ranks come from
; the operands: Ranks(k), or -1 where the path does not know them, which
; refuses an ellipsis). Implicit mode (no "->") gives the output the labels
; that appear once, in alphabetical order, after the ellipsis axes.
; Result: Labels (output labels first), the output's label count, and each
; operand's label per axis. "" or the refusal.
; ----------------------------------------------------------------------------
Procedure.s PmoOpsEinsumParse(Equation.s, Count.i, Array Ranks.i(1), Array AxisLabel.i(2), *Labels.Integer, *Kept.Integer)
  Protected Lhs.s, Rhs.s, Term.s, c.s, i.i, k.i, j.i, Explicit.i, Ell.i, EllRank.i, Named.i, Pos.i
  Protected Dim Letter.s(15)
  Protected Dim Uses.i(15)
  Protected Dim TermText.s(7)
  Protected LabelCount.i, Found.i, Order.s, Out.s
  Equation = RemoveString(Equation, " ")
  Explicit = FindString(Equation, "->")
  If Explicit
    Lhs = Left(Equation, Explicit - 1) : Rhs = Mid(Equation, Explicit + 2)
  Else
    Lhs = Equation
  EndIf
  If CountString(Lhs, ",") + 1 <> Count
    ProcedureReturn "the equation names " + Str(CountString(Lhs, ",") + 1) + " operands and the node has " + Str(Count) + "."
  EndIf
  If Count > 4 : ProcedureReturn "it has " + Str(Count) + " operands; one to four are implemented." : EndIf
  EllRank = -1
  For k = 0 To Count - 1
    Term = StringField(Lhs, k + 1, ",")
    Ell = FindString(Term, "...")
    Named = Len(RemoveString(Term, "."))
    If Ell
      If Ranks(k) < 0 : ProcedureReturn "an ellipsis needs the operand ranks, which this path does not know." : EndIf
      If Ranks(k) - Named < 0 : ProcedureReturn "operand " + Str(k) + " has fewer axes than its term names." : EndIf
      If EllRank >= 0 And EllRank <> Ranks(k) - Named And Ranks(k) - Named > 0 And EllRank > 0
        ProcedureReturn "the ellipsis covers different numbers of axes in different operands, which is not implemented."
      EndIf
      If Ranks(k) - Named > EllRank : EllRank = Ranks(k) - Named : EndIf
    ElseIf Ranks(k) >= 0 And Ranks(k) <> Named
      ProcedureReturn "operand " + Str(k) + " has rank " + Str(Ranks(k)) + " and its term names " + Str(Named) + " axes."
    EndIf
    TermText(k) = Term
  Next
  If EllRank < 0 : EllRank = 0 : EndIf
  ; ellipsis axes become the upper-case letters A, B, ... (a term's letters are lower case or upper case)
  For k = 0 To Count - 1
    Term = TermText(k)
    Ell = FindString(Term, "...")
    If Ell
      Named = Len(Term) - 3
      c = ""
      For j = EllRank - (Ranks(k) - Named) To EllRank - 1 : c + Chr(1 + j) : Next
      Term = Left(Term, Ell - 1) + c + Mid(Term, Ell + 3)
    EndIf
    TermText(k) = Term
  Next
  If Explicit
    If FindString(Rhs, "...")
      c = ""
      For j = 0 To EllRank - 1 : c + Chr(1 + j) : Next
      Rhs = ReplaceString(Rhs, "...", c)
    EndIf
  Else
    ; implicit: ellipsis axes, then every label used exactly once, alphabetically
    Rhs = ""
    For j = 0 To EllRank - 1 : Rhs + Chr(1 + j) : Next
    Order = ""
    For i = 65 To 122
      c = Chr(i)
      Found = 0
      For k = 0 To Count - 1 : Found + CountString(TermText(k), c) : Next
      If Found = 1 : Order + c : EndIf
    Next
    Rhs + Order
  EndIf
  ; labels: the output's first, in its order, then the summed ones in first-use order
  LabelCount = 0
  For i = 1 To Len(Rhs)
    c = Mid(Rhs, i, 1)
    For j = 0 To LabelCount - 1
      If Letter(j) = c : ProcedureReturn "the output subscripts name " + c + " twice." : EndIf
    Next
    If LabelCount >= 16 : ProcedureReturn "it names more than sixteen labels." : EndIf
    Letter(LabelCount) = c : LabelCount + 1
  Next
  *Kept\i = LabelCount
  For k = 0 To Count - 1
    For i = 1 To Len(TermText(k))
      c = Mid(TermText(k), i, 1)
      Found = -1
      For j = 0 To LabelCount - 1
        If Letter(j) = c : Found = j : Break : EndIf
      Next
      If Found < 0
        If LabelCount >= 16 : ProcedureReturn "it names more than sixteen labels." : EndIf
        Letter(LabelCount) = c : Found = LabelCount : LabelCount + 1
      EndIf
      If i > 8 : ProcedureReturn "an operand has more than eight axes." : EndIf
      AxisLabel(k, i - 1) = Found
    Next
    Ranks(k) = Len(TermText(k))
  Next
  For i = 1 To Len(Rhs)
    c = Mid(Rhs, i, 1)
    Found = 0
    For k = 0 To Count - 1 : Found + FindString(TermText(k), c) : Next
    If Found = 0 : ProcedureReturn "the output names " + c + ", which no operand has." : EndIf
  Next
  *Labels\i = LabelCount
  ProcedureReturn ""
EndProcedure

; Working memory a node's kernel needs on the fixed-shape path: Det a copy of
; one matrix, RNN and GRU the state of every direction and batch and three
; gate rows. 0 for every other node.
Procedure.q PmoOpsNodeScratch(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected N.q, H.q
  If *X = 0 : ProcedureReturn 0 : EndIf
  Select *Node\Operation
    Case "Det"
      N = PmoEmitDim(*X, PmoEmitRank(*X) - 1)
      ProcedureReturn N * N * 4
    Case "RNN", "GRU"
      *W = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
      If *W = 0 Or PmoEmitRank(*X) <> 3 Or PmoEmitRank(*W) <> 3 : ProcedureReturn 0 : EndIf
      H = PmoEmitDim(*W, 2)
      ProcedureReturn (PmoEmitDim(*W, 0) * PmoEmitDim(*X, 1) * H + 3 * H) * 4
  EndSelect
  ProcedureReturn 0
EndProcedure

; One region, sized for the largest such node, after everything else in the
; arena: nodes run one at a time. A model without such a node keeps its arena.
Procedure PmoOpsPlanScratch(*Ir.PmoIrModel)
  Protected Largest.q, Bytes.q
  *Ir\OpsScratchOffset = 0 : *Ir\OpsScratchBytes = 0
  ForEach *Ir\Nodes()
    If PmoOpsOwns(*Ir\Nodes()\Node\Operation)
      Bytes = PmoOpsNodeScratch(*Ir, *Ir\Nodes()\Node)
      If Bytes > Largest : Largest = Bytes : EndIf
    EndIf
  Next
  If Largest = 0 : ProcedureReturn : EndIf
  *Ir\OpsScratchOffset = PmoIrAlign(*Ir\ArenaBytes)
  *Ir\OpsScratchBytes = PmoIrAlign(Largest)
  *Ir\ArenaBytes = *Ir\OpsScratchOffset + *Ir\OpsScratchBytes
EndProcedure

; ============================================================================
; Fixed-shape emission
; ============================================================================

; The generated procedure's parameter list, the way PmoEmitNode passes
; arguments: every input with a name that is data at run time (compile-time
; inputs are folded into the procedure), then every listed output.
Procedure.s PmoOpsSignature(*Node.PmoOnnxNode, Op.s)
  Protected Text.s, Index.i
  Index = 0
  ForEach *Node\Inputs()
    If *Node\Inputs() <> "" And PmoIrCompileTimeInput(Op, Index) = 0
      If Text <> "" : Text + ", " : EndIf
      Text + "*i" + Str(Index)
    EndIf
    Index + 1
  Next
  Index = 0
  ForEach *Node\Outputs()
    If Text <> "" : Text + ", " : EndIf
    Text + "*o" + Str(Index)
    Index + 1
  Next
  ProcedureReturn "(" + Text + ")"
EndProcedure

; *iK when input K is a runtime input of the node, "0" otherwise.
Procedure.s PmoOpsIn(*Node.PmoOnnxNode, Index.i)
  If PmoEmitInput(*Node, Index) = "" Or PmoIrCompileTimeInput(*Node\Operation, Index) : ProcedureReturn "0" : EndIf
  ProcedureReturn "*i" + Str(Index)
EndProcedure

Procedure.s PmoOpsOut(*Node.PmoOnnxNode, Index.i)
  If PmoEmitOutput(*Node, Index) = "" : ProcedureReturn "0" : EndIf
  ProcedureReturn "*o" + Str(Index)
EndProcedure

Procedure PmoOpsHead(File.i, ProcName.s, *Node.PmoOnnxNode)
  PmoEmitLine(File, "Procedure " + ProcName + PmoOpsSignature(*Node, *Node\Operation))
EndProcedure

Procedure PmoOpsTail(File.i, Checked.i)
  If Checked : PmoEmitLine(File, "  If PmOpStatus <> 0 : PmOnnxRuntimeOk = 0 : EndIf") : EndIf
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
EndProcedure

Procedure.i PmoOpsTypeOk(*Node.PmoOnnxNode, *Value.PmoIrValue, Label.s, Bits.i)
  If *Value = 0 : ProcedureReturn PmoEmitNsFail(*Node, Label + " needs a concrete tensor.") : EndIf
  If (PmoOpsKindBit(*Value\ElementType) & Bits) = 0
    ProcedureReturn PmoEmitNsFail(*Node, Label + " has element type " + PmoEmitNsTypeName(*Value\ElementType) + "; " + *Node\Operation + " is implemented for " + PmoOpsKindNames(Bits) + ".")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOpsOpsetOk(*Node.PmoOnnxNode, Opset.i)
  Protected Floor.i = PmoOpsFloor(*Node\Operation)
  If Opset < Floor
    ProcedureReturn PmoEmitNsFail(*Node, "the model imports ai.onnx opset " + Str(Opset) + ", and " + *Node\Operation + " is implemented from opset " + Str(Floor) + ", the definition it was checked against.")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOpsSameShape(*Node.PmoOnnxNode, *A.PmoIrValue, *Y.PmoIrValue, ElementType.i)
  If *Y = 0 : ProcedureReturn PmoEmitNsFail(*Node, "the output needs a concrete shape.") : EndIf
  If PmoEmitShapeEqual(*A, *Y) = 0 Or *Y\ElementType <> ElementType
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the input's shape and element type " + PmoEmitNsTypeName(ElementType) + ".")
  EndIf
  ProcedureReturn #True
EndProcedure

; Erf ... IsInf
Procedure.i PmoEmitOpsUnary(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Code.i = PmoOpsUnaryCode(*Node), Bits.i = 1, OutType.i, Reason.s
  NewList Lines.s()
  If PmoEmitNsAttributesAllowed(*Node, PmoOpsUnaryAllowed(*Node\Operation), Opset) = 0 : ProcedureReturn #False : EndIf
  Reason = PmoOpsUnaryForm(*Node)
  If Reason <> "" : ProcedureReturn PmoEmitNsFail(*Node, Reason) : EndIf
  If *Node\Operation = "Sign" : Bits = 5 : EndIf
  If *Node\Operation = "BitwiseNot" : Bits = 6 : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", Bits) = 0 : ProcedureReturn #False : EndIf
  OutType = *X\ElementType
  If Code = 16 Or Code = 17 : OutType = 9 : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, OutType) = 0 : ProcedureReturn #False : EndIf
  PmoOpsUnaryParams(*Node, Lines())
  PmoOpsHead(File, ProcName, *Node)
  ForEach Lines() : PmoEmitLine(File, "  " + Lines()) : Next
  PmoEmitLine(File, "  PmOpStatus = 0")
  PmoEmitLine(File, "  PmOpUnary(*i0, *o0, " + Str(*X\Elements) + ", " + Str(Code) + ", " + Str(*X\ElementType) + ")")
  PmoOpsTail(File, #True)
  ProcedureReturn #True
EndProcedure

; Min, Max, Sum, Mean, Mod, PRelu, Or, Xor
Procedure.i PmoEmitOpsVariadic(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *X.PmoIrValue, Rank.i, Count.i, k.i, d.i, Extent.q, Here.q, Kind.i, Code.i = PmoOpsVariadicCode(*Node)
  Protected Allowed.s = "|"
  If *Node\Operation = "Mod" : Allowed = "|fmod|" : EndIf
  If PmoEmitNsAttributesAllowed(*Node, Allowed, Opset) = 0 : ProcedureReturn #False : EndIf
  If *Y = 0 : ProcedureReturn PmoEmitNsFail(*Node, "the output needs a concrete shape.") : EndIf
  Count = ListSize(*Node\Inputs())
  If Count < 1 Or Count > 16 : ProcedureReturn PmoEmitNsFail(*Node, "it has " + Str(Count) + " inputs; one to sixteen are implemented.") : EndIf
  If FindString("|Mod|PRelu|Or|Xor|BitwiseAnd|BitwiseOr|BitwiseXor|", "|" + *Node\Operation + "|") And Count <> 2
    ProcedureReturn PmoEmitNsFail(*Node, "it has " + Str(Count) + " inputs; the specification gives it two.")
  EndIf
  If *Node\Operation = "Mod" And PmoEmitAttrI(*Node, "fmod", 0) = 0
    *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
    If *X And *X\ElementType = 1
      ProcedureReturn PmoEmitNsFail(*Node, "attribute fmod = 0 on FLOAT inputs; the specification requires fmod = 1 for floating-point types.")
    EndIf
  EndIf
  Rank = PmoEmitRank(*Y)
  For k = 0 To Count - 1
    *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, k))
    If *X = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input " + Str(k) + " needs a concrete tensor.") : EndIf
    If PmoOpsTypeOk(*Node, *X, "input " + Str(k), PmoOpsVariadicKinds(*Node)) = 0 : ProcedureReturn #False : EndIf
    If k = 0 : Kind = *X\ElementType : EndIf
    If *X\ElementType <> Kind : ProcedureReturn PmoEmitNsFail(*Node, "its inputs must share one element type.") : EndIf
    If PmoEmitRank(*X) > Rank : ProcedureReturn PmoEmitNsFail(*Node, "input " + Str(k) + " has a higher rank than the declared output.") : EndIf
  Next
  If *Y\ElementType <> Kind : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the inputs' element type.") : EndIf
  ; every output extent is the broadcast of the inputs'
  For d = 0 To Rank - 1
    Extent = 1
    For k = 0 To Count - 1
      *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, k))
      If d >= Rank - PmoEmitRank(*X)
        Here = PmoEmitDim(*X, d - (Rank - PmoEmitRank(*X)))
        If Here <> 1
          If Extent <> 1 And Here <> Extent : ProcedureReturn PmoEmitNsFail(*Node, "input dimensions do not broadcast on output axis " + Str(d) + ".") : EndIf
          Extent = Here
        EndIf
      EndIf
    Next
    If Extent <> PmoEmitDim(*Y, d) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, d)) + "; the inputs broadcast to " + Str(Extent) + ".") : EndIf
    If *Node\Operation = "PRelu"
      *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
      If PmoEmitRank(*X) <> Rank Or PmoEmitDim(*X, d) <> Extent
        ProcedureReturn PmoEmitNsFail(*Node, "slope must broadcast to X (unidirectional broadcasting); here X would be broadcast.")
      EndIf
    EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  For k = 0 To Count - 1
    *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, k))
    PmoEmitLine(File, "  PmOpAddr(" + Str(k) + ") = *i" + Str(k))
    For d = 0 To Rank - 1
      Here = 1
      If d >= Rank - PmoEmitRank(*X) : Here = PmoEmitDim(*X, d - (Rank - PmoEmitRank(*X))) : EndIf
      PmoEmitLine(File, "  PmOpDN(" + Str(k * 8 + d) + ") = " + Str(Here))
    Next
  Next
  For d = 0 To Rank - 1 : PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(PmoEmitDim(*Y, d))) : Next
  PmoEmitLine(File, "  PmOpStatus = 0")
  PmoEmitLine(File, "  PmOpVariadic(*o0, " + Str(Count) + ", " + Str(Rank) + ", " + Str(Code) + ", " + Str(Kind) + ")")
  PmoOpsTail(File, #True)
  ProcedureReturn #True
EndProcedure

; The reduced-axis mask of a reduction from its axes attribute (before 18)
; or constant axes input (18 and later); -1 when refused. NoopOut is set when
; the empty axes set leaves the data as it is.
Procedure.i PmoOpsReduceMask(*Ir.PmoIrModel, *Node.PmoOnnxNode, Rank.i, Opset.i, *NoopOut.Integer)
  Protected *Axes.PmoIrConstant, Count.i, Index.i, Axis.q, Mask.i, Ok.Integer
  *NoopOut\i = 0
  If Opset >= 18
    If PmoEmitNsAttributesAllowed(*Node, "|keepdims|noop_with_empty_axes|", Opset) = 0 : ProcedureReturn -1 : EndIf
    If PmoEmitInput(*Node, 1) <> ""
      *Axes = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
      If *Axes = 0 : PmoEmitNsFail(*Node, "input axes must be constant for fixed-shape emission.") : ProcedureReturn -1 : EndIf
      Count = *Axes\Elements
    EndIf
  Else
    If PmoEmitNsAttributesAllowed(*Node, "|axes|keepdims|", Opset) = 0 : ProcedureReturn -1 : EndIf
    If PmoEmitInput(*Node, 1) <> "" : PmoEmitNsFail(*Node, "input axes is not part of " + *Node\Operation + " before opset 18; axes is an attribute there.") : ProcedureReturn -1 : EndIf
    Count = PmoEmitAttrListCount(*Node, "axes")
  EndIf
  For Index = 0 To Count - 1
    If *Axes : Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), Index, @Ok) : Else : Axis = PmoEmitAttrListI(*Node, "axes", Index, 0) : EndIf
    If Axis < 0 : Axis + Rank : EndIf
    If Axis < 0 Or Axis >= Rank : PmoEmitNsFail(*Node, "axes names an axis outside the data rank " + Str(Rank) + ".") : ProcedureReturn -1 : EndIf
    If (Mask >> Axis) & 1 : PmoEmitNsFail(*Node, "axes names axis " + Str(Axis) + " twice.") : ProcedureReturn -1 : EndIf
    Mask | (1 << Axis)
  Next
  If Count = 0
    If Opset >= 18 And PmoEmitAttrI(*Node, "noop_with_empty_axes", 0) <> 0
      *NoopOut\i = 1
      ProcedureReturn 0
    EndIf
    Mask = (1 << Rank) - 1
  EndIf
  ProcedureReturn Mask
EndProcedure

Procedure.i PmoEmitOpsReduce(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Code.i = PmoOpsReduceCode(*Node\Operation), Rank.i, Mask.i, Noop.Integer, Kept.q, d.i, Bits.i = 1
  If *X = 0 Or *Y = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input data and its output need concrete shapes.") : EndIf
  Rank = PmoEmitRank(*X)
  If Rank > 8 : ProcedureReturn PmoEmitNsFail(*Node, "the input rank " + Str(Rank) + " exceeds eight.") : EndIf
  If Code = 0 Or Code = 1 Or Code = 3 : Bits = 5 : EndIf
  If PmoOpsTypeOk(*Node, *X, "data", Bits) = 0 : ProcedureReturn #False : EndIf
  Mask = PmoOpsReduceMask(*Ir, *Node, Rank, Opset, @Noop)
  If Mask < 0 : ProcedureReturn #False : EndIf
  Kept = 1
  For d = 0 To Rank - 1
    If ((Mask >> d) & 1) = 0 : Kept * PmoEmitDim(*X, d) : EndIf
  Next
  If *Y\Elements <> Kept Or *Y\ElementType <> *X\ElementType
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output holds " + Str(*Y\Elements) + " elements; the reduction gives " + Str(Kept) + ".")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  If Noop\i
    PmoEmitLine(File, "  PmOpCopyBytes(*i0, *o0, " + Str(*X\Bytes) + ")")
    PmoOpsTail(File, #False)
    ProcedureReturn #True
  EndIf
  For d = 0 To Rank - 1 : PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d))) : Next
  PmoEmitLine(File, "  PmOpStatus = 0")
  PmoEmitLine(File, "  PmOpReduce(*i0, *o0, " + Str(Rank) + ", " + Str(Mask) + ", " + Str(Code) + ", " + Str(*X\ElementType) + ")")
  PmoOpsTail(File, #True)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsArg(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Axis.i, Kept.q
  If Opset >= 12
    If PmoEmitNsAttributesAllowed(*Node, "|axis|keepdims|select_last_index|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|axis|keepdims|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *X, "data", 5) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  Axis = PmoEmitAttrI(*Node, "axis", 0) : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", 0)) + " is outside the data rank " + Str(Rank) + ".") : EndIf
  If PmoEmitDim(*X, Axis) < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the axis has no elements, and the index of an extreme of none is undefined.") : EndIf
  Kept = *X\Elements / PmoEmitDim(*X, Axis)
  If *Y = 0 Or *Y\Elements <> Kept Or *Y\ElementType <> 7
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be INT64 with " + Str(Kept) + " elements.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpArg(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(PmoEmitDim(*X, Axis)) + ", " + Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)) + ", " +
                    Str(Bool(*Node\Operation = "ArgMin")) + ", " + Str(Bool(PmoEmitAttrI(*Node, "select_last_index", 0) <> 0)) + ", " + Str(*X\ElementType) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsLogSoftmax(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Axis.i, Written.i
  If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Opset >= 13
    Written = PmoEmitAttrI(*Node, "axis", -1)
  Else
    Written = PmoEmitAttrI(*Node, "axis", 1)
  EndIf
  Axis = Written : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(Written) + " is outside the input rank " + Str(Rank) + ".") : EndIf
  If Opset < 13 And Axis <> Rank - 1
    ProcedureReturn PmoEmitNsFail(*Node, "the model imports opset " + Str(Opset) + ", where LogSoftmax flattens the input to two dimensions at axis " + Str(Axis) +
                                         "; that equals LogSoftmax-13 only on the last axis, which is what is implemented.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpLogSoftmax(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(PmoEmitDim(*X, Axis)) + ", " + Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; The pooling shape rule, shared with the runtime wrapper (DOpPool):
; explicit pads, or SAME_UPPER/SAME_LOWER/VALID; ceil_mode drops a last
; window that would start in the end padding. Returns the output extent, or
; -1 when the window is larger than the padded input; *Begin/*End receive
; the pads used.
Procedure.q PmoOpsPoolExtent(In.q, Kernel.q, Stride.q, Dilation.q, AutoPad.s, CeilMode.i, *Begin.Quad, *End.Quad)
  Protected Span.q = (Kernel - 1) * Dilation + 1, Out.q, Total.q
  If AutoPad = "SAME_UPPER" Or AutoPad = "SAME_LOWER"
    Out = (In + Stride - 1) / Stride
    Total = (Out - 1) * Stride + Span - In
    If Total < 0 : Total = 0 : EndIf
    If AutoPad = "SAME_UPPER" : *Begin\q = Total / 2 : Else : *Begin\q = Total - Total / 2 : EndIf
    *End\q = Total - *Begin\q
    ProcedureReturn Out
  EndIf
  If AutoPad = "VALID" : *Begin\q = 0 : *End\q = 0 : EndIf
  If In + *Begin\q + *End\q < Span : ProcedureReturn -1 : EndIf
  Out = (In + *Begin\q + *End\q - Span) / Stride + 1
  If CeilMode
    If (In + *Begin\q + *End\q - Span) % Stride <> 0 : Out + 1 : EndIf
    If (Out - 1) * Stride >= In + *Begin\q : Out - 1 : EndIf
  EndIf
  ProcedureReturn Out
EndProcedure

Procedure.i PmoEmitOpsPool(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Ind.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected Op.s = *Node\Operation, IsGlobal.i, Mode.i, Allowed.s, s.i, d.i, AutoPad.s, CeilMode.i, P.q
  Protected Kernel.q, Stride.q, Dilation.q, Begin.Quad, EndPad.Quad, Out.q
  IsGlobal = Bool(Left(Op, 6) = "Global")
  If Op = "MaxPool" Or Op = "GlobalMaxPool" : Mode = 0 : ElseIf Op = "AveragePool" Or Op = "GlobalAveragePool" : Mode = 1 : Else : Mode = 2 : EndIf
  Select Op
    Case "MaxPool"
      Allowed = "|auto_pad|kernel_shape|pads|strides|"
      If Opset >= 8 : Allowed + "storage_order|" : EndIf
      If Opset >= 10 : Allowed + "ceil_mode|dilations|" : EndIf
    Case "AveragePool"
      Allowed = "|auto_pad|kernel_shape|pads|strides|count_include_pad|"
      If Opset >= 10 : Allowed + "ceil_mode|" : EndIf
      If Opset >= 19 : Allowed + "dilations|" : EndIf
    Case "LpPool"
      Allowed = "|auto_pad|kernel_shape|pads|strides|p|"
      If Opset >= 18 : Allowed + "ceil_mode|dilations|" : EndIf
    Case "GlobalLpPool" : Allowed = "|p|"
    Default : Allowed = "|"
  EndSelect
  If PmoEmitNsAttributesAllowed(*Node, Allowed, Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  s = PmoEmitRank(*X) - 2
  If s < 1 Or s > 3 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; N x C x D1 ... Ds with one to three spatial axes is implemented.") : EndIf
  If *Y = 0 Or PmoEmitRank(*Y) <> s + 2 Or *Y\ElementType <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT with the input's rank.") : EndIf
  If *Ind And (*Ind\ElementType <> 7 Or PmoEmitShapeEqual(*Ind, *Y) = 0) : ProcedureReturn PmoEmitNsFail(*Node, "output Indices must be INT64 with the shape of Y.") : EndIf
  If IsGlobal = 0 And PmoEmitAttrListCount(*Node, "kernel_shape") <> s
    ProcedureReturn PmoEmitNsFail(*Node, "attribute kernel_shape must hold " + Str(s) + " extents, one per spatial axis.")
  EndIf
  AutoPad = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
  If AutoPad <> "NOTSET" And AutoPad <> "SAME_UPPER" And AutoPad <> "SAME_LOWER" And AutoPad <> "VALID"
    ProcedureReturn PmoEmitNsFail(*Node, "attribute auto_pad = " + AutoPad + " is not a padding mode; NOTSET, SAME_UPPER, SAME_LOWER and VALID are.")
  EndIf
  If AutoPad <> "NOTSET" And PmoEmitAttrListCount(*Node, "pads") > 0
    ProcedureReturn PmoEmitNsFail(*Node, "attribute pads is given with auto_pad = " + AutoPad + "; the specification allows explicit pads only with NOTSET.")
  EndIf
  CeilMode = Bool(PmoEmitAttrI(*Node, "ceil_mode", 0) <> 0)
  P = PmoEmitAttrI(*Node, "p", 2)
  If Mode = 2 And P < 1 : ProcedureReturn PmoEmitNsFail(*Node, "attribute p = " + Str(P) + "; the norm order must be at least one.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(s))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(PmoEmitDim(*X, 1)))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(Mode))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(Bool(PmoEmitAttrI(*Node, "count_include_pad", 0) <> 0)))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(P))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Bool(PmoEmitAttrI(*Node, "storage_order", 0) <> 0)))
  For d = 0 To s - 1
    If IsGlobal
      Kernel = PmoEmitDim(*X, 2 + d) : Stride = 1 : Dilation = 1 : Begin\q = 0 : EndPad\q = 0
    Else
      Kernel = PmoEmitAttrListI(*Node, "kernel_shape", d, 1)
      Stride = PmoEmitAttrListI(*Node, "strides", d, 1)
      Dilation = PmoEmitAttrListI(*Node, "dilations", d, 1)
      Begin\q = PmoEmitAttrListI(*Node, "pads", d, 0)
      EndPad\q = PmoEmitAttrListI(*Node, "pads", d + s, 0)
    EndIf
    If Kernel < 1 Or Stride < 1 Or Dilation < 1 Or Begin\q < 0 Or EndPad\q < 0
      ProcedureReturn PmoEmitNsFail(*Node, "kernel_shape, strides and dilations must be positive and pads not negative on spatial axis " + Str(d) + ".")
    EndIf
    Out = PmoOpsPoolExtent(PmoEmitDim(*X, 2 + d), Kernel, Stride, Dilation, AutoPad, CeilMode, @Begin, @EndPad)
    If Out < 0 : ProcedureReturn PmoEmitNsFail(*Node, "the pooling window on spatial axis " + Str(d) + " is larger than the padded input.") : EndIf
    If Out <> PmoEmitDim(*Y, 2 + d)
      ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on spatial axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, 2 + d)) + "; the pooling shape rule gives " + Str(Out) + ".")
    EndIf
    PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = " + Str(PmoEmitDim(*X, 2 + d)))
    PmoEmitLine(File, "  PmOpI(" + Str(12 + d) + ") = " + Str(Out))
    PmoEmitLine(File, "  PmOpI(" + Str(16 + d) + ") = " + Str(Kernel))
    PmoEmitLine(File, "  PmOpI(" + Str(20 + d) + ") = " + Str(Stride))
    PmoEmitLine(File, "  PmOpI(" + Str(24 + d) + ") = " + Str(Dilation))
    PmoEmitLine(File, "  PmOpI(" + Str(28 + d) + ") = " + Str(Begin\q))
    PmoEmitLine(File, "  PmOpI(" + Str(32 + d) + ") = " + Str(EndPad\q))
  Next
  If PmoEmitDim(*Y, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*Y, 1) <> PmoEmitDim(*X, 1)
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must keep the input's N and C.")
  EndIf
  PmoEmitLine(File, "  PmOpPool(*i0, *o0, " + PmoOpsOut(*Node, 1) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsSplit(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue, *Sizes.PmoIrConstant
  Protected Rank.i, Axis.i, n.i, k.i, Extent.q, Size.q, Offset.q, Total.q, Chunk.q, Ok.Integer, d.i
  Protected Dim Sizes.q(16)
  If Opset >= 18
    If PmoEmitNsAttributesAllowed(*Node, "|axis|num_outputs|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf Opset >= 13
    If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|axis|split|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If *X = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input needs a concrete tensor.") : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  Axis = PmoEmitAttrI(*Node, "axis", 0) : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", 0)) + " is outside the input rank " + Str(Rank) + ".") : EndIf
  n = ListSize(*Node\Outputs())
  If n < 1 Or n > 16 : ProcedureReturn PmoEmitNsFail(*Node, "it has " + Str(n) + " outputs; one to sixteen are implemented.") : EndIf
  Extent = PmoEmitDim(*X, Axis)
  If PmoEmitInput(*Node, 1) <> ""
    If Opset < 13 : ProcedureReturn PmoEmitNsFail(*Node, "input split is not part of Split before opset 13; split is an attribute there.") : EndIf
    *Sizes = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
    If *Sizes = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input split must be constant for fixed-shape emission.") : EndIf
    If *Sizes\Elements <> n : ProcedureReturn PmoEmitNsFail(*Node, "input split holds " + Str(*Sizes\Elements) + " sizes for " + Str(n) + " outputs.") : EndIf
    If PmoEmitAttrI(*Node, "num_outputs", 0) <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "input split and attribute num_outputs are both given; the specification allows one.") : EndIf
    For k = 0 To n - 1 : Sizes(k) = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), k, @Ok) : Next
  ElseIf PmoEmitAttrListCount(*Node, "split") > 0
    If PmoEmitAttrListCount(*Node, "split") <> n : ProcedureReturn PmoEmitNsFail(*Node, "attribute split holds " + Str(PmoEmitAttrListCount(*Node, "split")) + " sizes for " + Str(n) + " outputs.") : EndIf
    For k = 0 To n - 1 : Sizes(k) = PmoEmitAttrListI(*Node, "split", k, 0) : Next
  ElseIf Opset >= 18
    If PmoEmitAttrI(*Node, "num_outputs", 0) <> n : ProcedureReturn PmoEmitNsFail(*Node, "Split-18 needs input split or attribute num_outputs equal to its " + Str(n) + " outputs.") : EndIf
    Chunk = (Extent + n - 1) / n : Total = 0
    For k = 0 To n - 1
      Sizes(k) = Chunk : If Total + Chunk > Extent : Sizes(k) = Extent - Total : EndIf
      If Sizes(k) < 0 : Sizes(k) = 0 : EndIf
      Total + Sizes(k)
    Next
  Else
    If Extent % n <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "without sizes the extent " + Str(Extent) + " must divide into " + Str(n) + " equal parts.") : EndIf
    For k = 0 To n - 1 : Sizes(k) = Extent / n : Next
  EndIf
  Total = 0
  For k = 0 To n - 1
    If Sizes(k) < 0 : ProcedureReturn PmoEmitNsFail(*Node, "a split size is negative.") : EndIf
    Total + Sizes(k)
  Next
  If Total <> Extent : ProcedureReturn PmoEmitNsFail(*Node, "the split sizes add up to " + Str(Total) + "; the axis has " + Str(Extent) + ".") : EndIf
  For k = 0 To n - 1
    *Y = PmoEmitValue(*Ir, PmoEmitOutput(*Node, k))
    If PmoEmitOutput(*Node, k) = "" : Continue : EndIf
    If *Y = 0 Or PmoEmitRank(*Y) <> Rank Or *Y\ElementType <> *X\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "output " + Str(k) + " must have the input's rank and element type.") : EndIf
    For d = 0 To Rank - 1
      Size = PmoEmitDim(*X, d) : If d = Axis : Size = Sizes(k) : EndIf
      If PmoEmitDim(*Y, d) <> Size : ProcedureReturn PmoEmitNsFail(*Node, "output " + Str(k) + " declares extent " + Str(PmoEmitDim(*Y, d)) + " on axis " + Str(d) + "; the split gives " + Str(Size) + ".") : EndIf
    Next
  Next
  PmoOpsHead(File, ProcName, *Node)
  Offset = 0
  For k = 0 To n - 1
    If PmoEmitOutput(*Node, k) <> ""
      PmoEmitLine(File, "  PmOpSlab(*i0, *o" + Str(k) + ", " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(Extent) + ", " + Str(Offset) + ", " + Str(Sizes(k)) + ", " +
                        Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
    EndIf
    Offset + Sizes(k)
  Next
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsTile(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Repeats.PmoIrConstant, Rank.i, d.i, R.q, Ok.Integer
  If PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 : ProcedureReturn #False : EndIf
  *Repeats = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  If *Repeats = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input repeats must be constant for fixed-shape emission.") : EndIf
  Rank = PmoEmitRank(*X)
  If *Repeats\Elements <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "input repeats holds " + Str(*Repeats\Elements) + " counts for an input of rank " + Str(Rank) + ".") : EndIf
  If *Y = 0 Or PmoEmitRank(*Y) <> Rank Or *Y\ElementType <> *X\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the input's rank and element type.") : EndIf
  For d = 0 To Rank - 1
    R = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), d, @Ok)
    If Ok\i = 0 Or R < 0 : ProcedureReturn PmoEmitNsFail(*Node, "input repeats must hold INT64 counts of zero or more.") : EndIf
    If PmoEmitDim(*Y, d) <> PmoEmitDim(*X, d) * R : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, d)) + "; the repeats give " + Str(PmoEmitDim(*X, d) * R) + ".") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  If *X\Elements > 0 And *Y\Elements > 0
    For d = 0 To Rank - 1
      PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d)))
      PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(PmoEmitDim(*Y, d)))
    Next
    PmoEmitLine(File, "  PmOpTile(*i0, *o0, " + Str(Rank) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsDepthSpace(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected ToSpace.i = Bool(*Node\Operation = "DepthToSpace"), Block.q, Mode.s, N.q, C.q, H.q, W.q, i.i
  Protected Dim Perm.i(5)
  Protected Dim View.q(5)
  If ToSpace And Opset >= 11
    If PmoEmitNsAttributesAllowed(*Node, "|blocksize|mode|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|blocksize|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) <> 4 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; the specification's form is N x C x H x W.") : EndIf
  If PmoEmitNsAttributePresent(*Node, "blocksize") = 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute blocksize is required.") : EndIf
  Block = PmoEmitAttrI(*Node, "blocksize", 0)
  Mode = PmoEmitAttrS(*Node, "mode", "DCR")
  If Block < 1 : ProcedureReturn PmoEmitNsFail(*Node, "attribute blocksize = " + Str(Block) + " must be positive.") : EndIf
  If Mode <> "DCR" And Mode <> "CRD" : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = " + Mode + " is not a DepthToSpace mode; DCR and CRD are.") : EndIf
  N = PmoEmitDim(*X, 0) : C = PmoEmitDim(*X, 1) : H = PmoEmitDim(*X, 2) : W = PmoEmitDim(*X, 3)
  If ToSpace
    If C % (Block * Block) <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "channels " + Str(C) + " are not divisible by blocksize squared.") : EndIf
    C / (Block * Block)
    If Mode = "DCR"
      View(0) = N : View(1) = Block : View(2) = Block : View(3) = C : View(4) = H : View(5) = W
      Perm(0) = 0 : Perm(1) = 3 : Perm(2) = 4 : Perm(3) = 1 : Perm(4) = 5 : Perm(5) = 2
    Else
      View(0) = N : View(1) = C : View(2) = Block : View(3) = Block : View(4) = H : View(5) = W
      Perm(0) = 0 : Perm(1) = 1 : Perm(2) = 4 : Perm(3) = 2 : Perm(4) = 5 : Perm(5) = 3
    EndIf
    If *Y = 0 Or PmoEmitRank(*Y) <> 4 Or PmoEmitDim(*Y, 0) <> N Or PmoEmitDim(*Y, 1) <> C Or PmoEmitDim(*Y, 2) <> H * Block Or PmoEmitDim(*Y, 3) <> W * Block Or *Y\ElementType <> *X\ElementType
      ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be " + Str(N) + " x " + Str(C) + " x " + Str(H * Block) + " x " + Str(W * Block) + " of the input's element type.")
    EndIf
  Else
    If H % Block <> 0 Or W % Block <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "height and width must be divisible by blocksize " + Str(Block) + ".") : EndIf
    H / Block : W / Block
    View(0) = N : View(1) = C : View(2) = H : View(3) = Block : View(4) = W : View(5) = Block
    Perm(0) = 0 : Perm(1) = 3 : Perm(2) = 5 : Perm(3) = 1 : Perm(4) = 2 : Perm(5) = 4
    If *Y = 0 Or PmoEmitRank(*Y) <> 4 Or PmoEmitDim(*Y, 0) <> N Or PmoEmitDim(*Y, 1) <> C * Block * Block Or PmoEmitDim(*Y, 2) <> H Or PmoEmitDim(*Y, 3) <> W Or *Y\ElementType <> *X\ElementType
      ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be " + Str(N) + " x " + Str(C * Block * Block) + " x " + Str(H) + " x " + Str(W) + " of the input's element type.")
    EndIf
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  If *X\Elements > 0
    For i = 0 To 5
      PmoEmitLine(File, "  PmOpDA(" + Str(i) + ") = " + Str(View(i)))
      PmoEmitLine(File, "  PmOpI(" + Str(40 + i) + ") = " + Str(Perm(i)))
    Next
    PmoEmitLine(File, "  PmOpPermute(*i0, *o0, 6, " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsTrilu(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *K.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1)), Rank.i
  If PmoEmitNsAttributesAllowed(*Node, "|upper|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, *X\ElementType) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 2 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(Rank) + "; Trilu needs two or more.") : EndIf
  If PmoEmitInput(*Node, 1) <> "" And (*K = 0 Or *K\ElementType <> 7 Or *K\Elements <> 1)
    ProcedureReturn PmoEmitNsFail(*Node, "input k must be one INT64 element.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  If PmoEmitInput(*Node, 1) <> ""
    PmoEmitLine(File, "  PmOpTrilu(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Rank - 3)) + ", " + Str(PmoEmitDim(*X, Rank - 2)) + ", " + Str(PmoEmitDim(*X, Rank - 1)) + ", PmTensorGetI64(*i1, 0), " +
                      Str(Bool(PmoEmitAttrI(*Node, "upper", 1) <> 0)) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
  Else
    PmoEmitLine(File, "  PmOpTrilu(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Rank - 3)) + ", " + Str(PmoEmitDim(*X, Rank - 2)) + ", " + Str(PmoEmitDim(*X, Rank - 1)) + ", 0, " +
                      Str(Bool(PmoEmitAttrI(*Node, "upper", 1) <> 0)) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsGatherElements(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *I.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Axis.i, d.i
  If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "data", 13) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *I, "indices", 4) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 1 Or PmoEmitRank(*I) <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "data and indices must share one rank of at least one.") : EndIf
  Axis = PmoEmitAttrI(*Node, "axis", 0) : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", 0)) + " is outside the data rank " + Str(Rank) + ".") : EndIf
  If PmoOpsSameShape(*Node, *I, *Y, *X\ElementType) = 0 : ProcedureReturn #False : EndIf
  For d = 0 To Rank - 1
    If d <> Axis And PmoEmitDim(*I, d) > PmoEmitDim(*X, d) : ProcedureReturn PmoEmitNsFail(*Node, "indices extend past data on axis " + Str(d) + ".") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  For d = 0 To Rank - 1
    PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d)))
    PmoEmitLine(File, "  PmOpDB(" + Str(d) + ") = " + Str(PmoEmitDim(*I, d)))
  Next
  PmoEmitLine(File, "  If PmOpGatherElements(*i0, *i1, *o0, " + Str(Rank) + ", " + Str(Axis) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ", 7) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsGatherND(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *I.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected R.i, Q.i, B.i, K.q, d.i, j.i, Expected.q
  If Opset >= 12
    If PmoEmitNsAttributesAllowed(*Node, "|batch_dims|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *X, "data", 13) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *I, "indices", 4) = 0 : ProcedureReturn #False : EndIf
  R = PmoEmitRank(*X) : Q = PmoEmitRank(*I) : B = PmoEmitAttrI(*Node, "batch_dims", 0)
  If R < 1 Or Q < 1 Or B < 0 Or B >= Q Or B >= R : ProcedureReturn PmoEmitNsFail(*Node, "data and indices need rank one or more and batch_dims = " + Str(B) + " below both ranks.") : EndIf
  K = PmoEmitDim(*I, Q - 1)
  If K < 1 Or K > R - B : ProcedureReturn PmoEmitNsFail(*Node, "index tuples of " + Str(K) + " entries; one to rank(data) - batch_dims are defined.") : EndIf
  For d = 0 To B - 1
    If PmoEmitDim(*I, d) <> PmoEmitDim(*X, d) : ProcedureReturn PmoEmitNsFail(*Node, "the batch dimensions of data and indices differ on axis " + Str(d) + ".") : EndIf
  Next
  If *Y = 0 Or *Y\ElementType <> *X\ElementType Or PmoEmitRank(*Y) <> Q - 1 + R - K - B : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have rank " + Str(Q - 1 + R - K - B) + " and the data's element type.") : EndIf
  j = 0
  For d = 0 To Q - 2
    If PmoEmitDim(*Y, j) <> PmoEmitDim(*I, d) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(j) + " differs from the specification's.") : EndIf
    j + 1
  Next
  For d = B + K To R - 1
    If PmoEmitDim(*Y, j) <> PmoEmitDim(*X, d) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(j) + " differs from the specification's.") : EndIf
    j + 1
  Next
  PmoOpsHead(File, ProcName, *Node)
  For d = 0 To R - 1 : PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d))) : Next
  For d = 0 To Q - 1 : PmoEmitLine(File, "  PmOpDB(" + Str(d) + ") = " + Str(PmoEmitDim(*I, d))) : Next
  PmoEmitLine(File, "  PmOpI(0) = " + Str(B))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(R))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Q))
  PmoEmitLine(File, "  If PmOpGatherND(*i0, *i1, *o0, " + Str(PmoIrElementBytes(*X\ElementType)) + ", 7) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsOneHot(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *I.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *V.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *D.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  Protected Depth.q, Rank.i, Axis.i, d.i, j.i, Ok.Integer, F.f
  If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *I, "indices", 5) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *V, "values", 13) = 0 : ProcedureReturn #False : EndIf
  If *V\Elements <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "input values must hold two elements, [off_value, on_value].") : EndIf
  If *D = 0 Or *D\Elements <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "input depth must be one constant element for fixed-shape emission.") : EndIf
  If *D\ElementType = 1
    F = PmoEmitConstF(*Ir, PmoEmitInput(*Node, 1), 0, @Ok) : Depth = Int(F)
  Else
    Depth = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
  EndIf
  If Ok\i = 0 Or Depth < 1 : ProcedureReturn PmoEmitNsFail(*Node, "input depth must be a positive number.") : EndIf
  Rank = PmoEmitRank(*I)
  Axis = PmoEmitAttrI(*Node, "axis", -1) : If Axis < 0 : Axis + Rank + 1 : EndIf
  If Axis < 0 Or Axis > Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", -1)) + " is outside the output rank " + Str(Rank + 1) + ".") : EndIf
  If *Y = 0 Or *Y\ElementType <> *V\ElementType Or PmoEmitRank(*Y) <> Rank + 1 : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have rank " + Str(Rank + 1) + " and the element type of values.") : EndIf
  j = 0
  For d = 0 To Rank
    If d = Axis
      If PmoEmitDim(*Y, d) <> Depth : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " must be depth " + Str(Depth) + ".") : EndIf
    Else
      If PmoEmitDim(*Y, d) <> PmoEmitDim(*I, j) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " must be the indices' " + Str(PmoEmitDim(*I, j)) + ".") : EndIf
      j + 1
    EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  Select *V\ElementType
    Case 1
      PmoEmitLine(File, "  PmOpF(0) = PmTensorGet(*i2, 0)")
      PmoEmitLine(File, "  PmOpF(1) = PmTensorGet(*i2, 1)")
    Case 7
      PmoEmitLine(File, "  PmOpI(0) = PmTensorGetI64(*i2, 0)")
      PmoEmitLine(File, "  PmOpI(1) = PmTensorGetI64(*i2, 1)")
    Default
      PmoEmitLine(File, "  PmOpI(0) = PeekA(*i2) & 255")
      PmoEmitLine(File, "  PmOpI(1) = PeekA(*i2 + 1) & 255")
  EndSelect
  PmoEmitLine(File, "  PmOpStatus = 0")
  PmoEmitLine(File, "  PmOpOneHot(*i0, *o0, " + Str(PmoEmitProduct(*I, 0, Axis - 1)) + ", " + Str(Depth) + ", " + Str(PmoEmitProduct(*I, Axis, Rank - 1)) + ", " + Str(*I\ElementType) + ", " + Str(*V\ElementType) + ")")
  PmoOpsTail(File, #True)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsEinsum(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0)), *X.PmoIrValue
  Protected n.i = ListSize(*Node\Inputs()), k.i, a.i, l.i, Reason.s, Labels.Integer, Kept.Integer, Stride.q
  Protected Dim Ranks.i(7)
  Protected Dim AxisLabel.i(7, 15)
  Protected Dim Extent.q(15)
  Protected Dim Strides.q(3, 15)
  If PmoEmitNsAttributesAllowed(*Node, "|equation|", Opset) = 0 : ProcedureReturn #False : EndIf
  If n < 1 Or n > 4 : ProcedureReturn PmoEmitNsFail(*Node, "it has " + Str(n) + " operands; one to four are implemented.") : EndIf
  For k = 0 To n - 1
    *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, k))
    If PmoOpsTypeOk(*Node, *X, "operand " + Str(k), 1) = 0 : ProcedureReturn #False : EndIf
    Ranks(k) = PmoEmitRank(*X)
  Next
  Reason = PmoOpsEinsumParse(PmoEmitAttrS(*Node, "equation", ""), n, Ranks(), AxisLabel(), @Labels, @Kept)
  If Reason <> "" : ProcedureReturn PmoEmitNsFail(*Node, "equation " + PmoEmitAttrS(*Node, "equation", "") + ": " + Reason) : EndIf
  For l = 0 To 15 : Extent(l) = -1 : Next
  For k = 0 To n - 1
    *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, k))
    Stride = 1
    For a = Ranks(k) - 1 To 0 Step -1
      l = AxisLabel(k, a)
      If Extent(l) >= 0 And Extent(l) <> PmoEmitDim(*X, a) : ProcedureReturn PmoEmitNsFail(*Node, "label extents disagree between the operands.") : EndIf
      Extent(l) = PmoEmitDim(*X, a)
      Strides(k, l) + Stride
      Stride * PmoEmitDim(*X, a)
    Next
  Next
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> Kept\i : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT of rank " + Str(Kept\i) + ".") : EndIf
  For l = 0 To Kept\i - 1
    If PmoEmitDim(*Y, l) <> Extent(l) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(l) + " is " + Str(PmoEmitDim(*Y, l)) + "; the equation gives " + Str(Extent(l)) + ".") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(n))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(Labels\i))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Kept\i))
  For l = 0 To Labels\i - 1 : PmoEmitLine(File, "  PmOpI(" + Str(4 + l) + ") = " + Str(Extent(l))) : Next
  For k = 0 To n - 1
    PmoEmitLine(File, "  PmOpAddr(" + Str(k) + ") = *i" + Str(k))
    For l = 0 To Labels\i - 1 : PmoEmitLine(File, "  PmOpI(" + Str(20 + k * 16 + l) + ") = " + Str(Strides(k, l))) : Next
  Next
  PmoEmitLine(File, "  PmOpEinsum(*o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsDropout(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *M.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected Ok.Integer, Training.q, Ratio.f
  If Opset >= 12
    If PmoEmitNsAttributesAllowed(*Node, "|seed|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|ratio|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *X, "data", 13) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitInput(*Node, 2) <> ""
    Training = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 2), 0, @Ok)
    If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input training_mode must be constant for fixed-shape emission.") : EndIf
    If Training <> 0
      Ratio = 0.5
      If PmoEmitInput(*Node, 1) <> "" : Ratio = PmoEmitConstF(*Ir, PmoEmitInput(*Node, 1), 0, @Ok) : EndIf
      If Ratio <> 0.0
        ProcedureReturn PmoEmitNsFail(*Node, "training_mode is true with a nonzero ratio, which draws random masks; inference Dropout (training_mode false) is implemented.")
      EndIf
    EndIf
  EndIf
  If *Y And PmoOpsSameShape(*Node, *X, *Y, *X\ElementType) = 0 : ProcedureReturn #False : EndIf
  If *M And PmoEmitShapeEqual(*M, *X) = 0 : ProcedureReturn PmoEmitNsFail(*Node, "output mask must have the shape of data.") : EndIf
  If *M And *M\ElementType <> 9 And *M\ElementType <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "output mask must be BOOL (FLOAT before opset 10).") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  If *M : PmoEmitLine(File, "  Protected i.i") : EndIf
  If *Y : PmoEmitLine(File, "  PmOpCopyBytes(*i0, *o0, " + Str(*X\Bytes) + ")") : EndIf
  If *M
    PmoEmitLine(File, "  i = 0")
    PmoEmitLine(File, "  While i < " + Str(*M\Elements))
    If *M\ElementType = 9
      PmoEmitLine(File, "    PokeA(*o1 + i, 1)")
    Else
      PmoEmitLine(File, "    PokeL(*o1 + i * 4, $3F800000)")
    EndIf
    PmoEmitLine(File, "    i = i + 1")
    PmoEmitLine(File, "  Wend")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; An axis attribute of a one-input FLOAT operator whose output has the
; input's shape; -1 (after the refusal) when out of range.
Procedure.i PmoOpsAxisForm(*Ir.PmoIrModel, *Node.PmoOnnxNode, Fallback.i, *X.PmoIrValue)
  Protected Rank.i = PmoEmitRank(*X), Axis.i = PmoEmitAttrI(*Node, "axis", Fallback)
  If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank
    PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", Fallback)) + " is outside the input rank " + Str(Rank) + ".")
    ProcedureReturn -1
  EndIf
  ProcedureReturn Axis
EndProcedure

; Hardmax and LpNormalization along an axis.
Procedure.i PmoEmitOpsAxisNorm(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Axis.i, Rank.i, P.q
  If *Node\Operation = "Hardmax"
    If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|axis|p|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If *Node\Operation = "Hardmax"
    If Opset >= 13 : Axis = PmoOpsAxisForm(*Ir, *Node, -1, *X) : Else : Axis = PmoOpsAxisForm(*Ir, *Node, 1, *X) : EndIf
    If Axis < 0 : ProcedureReturn #False : EndIf
    If Opset < 13 And Axis <> Rank - 1
      ProcedureReturn PmoEmitNsFail(*Node, "the model imports opset " + Str(Opset) + ", where Hardmax flattens the input to two dimensions at axis " + Str(Axis) +
                                           "; that equals Hardmax-13 only on the last axis, which is what is implemented.")
    EndIf
  Else
    Axis = PmoOpsAxisForm(*Ir, *Node, -1, *X)
    If Axis < 0 : ProcedureReturn #False : EndIf
    P = PmoEmitAttrI(*Node, "p", 2)
    If P <> 1 And P <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "attribute p = " + Str(P) + "; the specification allows 1 and 2.") : EndIf
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  If *Node\Operation = "Hardmax"
    PmoEmitLine(File, "  PmOpHardmax(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(PmoEmitDim(*X, Axis)) + ", " + Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)) + ")")
  Else
    PmoEmitLine(File, "  PmOpLpNorm(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(PmoEmitDim(*X, Axis)) + ", " + Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)) + ", " + Str(P) + ")")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsMvn(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Count.i, Index.i, Axis.q, Mask.i, d.i
  If PmoEmitNsAttributesAllowed(*Node, "|axes|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  Count = PmoEmitAttrListCount(*Node, "axes")
  If PmoEmitNsAttributePresent(*Node, "axes") = 0
    If Rank < 4 : ProcedureReturn PmoEmitNsFail(*Node, "attribute axes is absent and defaults to [0, 2, 3], which needs an input of rank 4 or more; this one has " + Str(Rank) + ".") : EndIf
    Mask = 1 | 4 | 8
  EndIf
  For Index = 0 To Count - 1
    Axis = PmoEmitAttrListI(*Node, "axes", Index, 0)
    If Axis < 0 : Axis + Rank : EndIf
    If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axes names an axis outside the input rank " + Str(Rank) + ".") : EndIf
    Mask | (1 << Axis)
  Next
  PmoOpsHead(File, ProcName, *Node)
  For d = 0 To Rank - 1 : PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d))) : Next
  PmoEmitLine(File, "  PmOpMvn(*i0, *o0, " + Str(Rank) + ", " + Str(Mask) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsLrn(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Size.q
  If PmoEmitNsAttributesAllowed(*Node, "|alpha|beta|bias|size|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) < 3 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; LRN takes N x C x D1 ... Dk.") : EndIf
  If PmoEmitNsAttributePresent(*Node, "size") = 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute size is required.") : EndIf
  Size = PmoEmitAttrI(*Node, "size", 1)
  If Size < 1 : ProcedureReturn PmoEmitNsFail(*Node, "attribute size = " + Str(Size) + " must be positive.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "alpha", 0.0001)) + ")")
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(1), " + PmoOpsBits(PmoEmitAttrF(*Node, "beta", 0.75)) + ")")
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(2), " + PmoOpsBits(PmoEmitAttrF(*Node, "bias", 1.0)) + ")")
  PmoEmitLine(File, "  PmOpLrn(*i0, *o0, " + Str(PmoEmitDim(*X, 0)) + ", " + Str(PmoEmitDim(*X, 1)) + ", " + Str(PmoEmitProduct(*X, 2, PmoEmitRank(*X) - 1)) + ", " + Str(Size) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsGroupNorm(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *S.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Groups.q
  If Opset > 20 : ProcedureReturn PmoEmitNsFail(*Node, "the model imports opset " + Str(Opset) + "; GroupNormalization-21 scales per channel, and the definition implemented is GroupNormalization-18's, per group.") : EndIf
  If PmoEmitNsAttributesAllowed(*Node, "|epsilon|num_groups|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *S, "scale", 1) = 0 Or PmoOpsTypeOk(*Node, *B, "bias", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) < 3 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; GroupNormalization takes N x C x D1 ... Dn.") : EndIf
  If PmoEmitNsAttributePresent(*Node, "num_groups") = 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute num_groups is required.") : EndIf
  Groups = PmoEmitAttrI(*Node, "num_groups", 1)
  If Groups < 1 Or PmoEmitDim(*X, 1) % Groups <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "num_groups = " + Str(Groups) + " must divide the " + Str(PmoEmitDim(*X, 1)) + " channels.") : EndIf
  If *S\Elements <> Groups Or *B\Elements <> Groups : ProcedureReturn PmoEmitNsFail(*Node, "scale and bias must hold num_groups = " + Str(Groups) + " elements each (GroupNormalization-18).") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ")")
  PmoEmitLine(File, "  PmOpGroupNorm(*i0, *i1, *i2, *o0, " + Str(PmoEmitDim(*X, 0)) + ", " + Str(PmoEmitDim(*X, 1)) + ", " + Str(PmoEmitProduct(*X, 2, PmoEmitRank(*X) - 1)) + ", " + Str(Groups) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsEyeLike(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Kind.i
  If PmoEmitNsAttributesAllowed(*Node, "|dtype|k|", Opset) = 0 : ProcedureReturn #False : EndIf
  If *X = 0 Or PmoEmitRank(*X) <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "the input must be a two-dimensional tensor.") : EndIf
  Kind = PmoEmitAttrI(*Node, "dtype", *X\ElementType)
  If Kind <> 1 And Kind <> 7 And Kind <> 9 : ProcedureReturn PmoEmitNsFail(*Node, "dtype " + PmoEmitNsTypeName(Kind) + " is not implemented by fixed-shape emission; FLOAT, INT64 and BOOL are.") : EndIf
  If *Y = 0 Or PmoEmitShapeEqual(*X, *Y) = 0 Or *Y\ElementType <> Kind : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the input's shape and the element type dtype names.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpEyeLike(*o0, " + Str(PmoEmitDim(*X, 0)) + ", " + Str(PmoEmitDim(*X, 1)) + ", " + Str(PmoEmitAttrI(*Node, "k", 0)) + ", " + Str(Kind) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsDet(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, N.q
  If PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 2 Or PmoEmitDim(*X, Rank - 1) <> PmoEmitDim(*X, Rank - 2) : ProcedureReturn PmoEmitNsFail(*Node, "the input must end in two equal axes, [*, M, M].") : EndIf
  N = PmoEmitDim(*X, Rank - 1)
  If *Y = 0 Or *Y\ElementType <> 1 Or *Y\Elements * N * N <> *X\Elements : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT with the input's leading axes.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  If *Y\Elements > 0
    PmoEmitLine(File, "  PmOpDet(*i0, *o0, " + Str(*Y\Elements) + ", " + Str(N) + ", PmOnnxArenaBase + " + Str(*Ir\OpsScratchOffset) + ")")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsCompress(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *C.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Axis.i, Kept.q, i.q, Width.q, Outer.q, Inner.q, Ok.Integer, d.i, Flat.i
  If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 : ProcedureReturn #False : EndIf
  If *C = 0 Or *C\ElementType <> 9 : ProcedureReturn PmoEmitNsFail(*Node, "input condition must be a constant BOOL tensor for fixed-shape emission, which sizes the output when the source is written.") : EndIf
  Rank = PmoEmitRank(*X)
  Flat = Bool(PmoEmitNsAttributePresent(*Node, "axis") = 0)
  If Flat
    Outer = 1 : Width = *X\Elements : Inner = 1
  Else
    Axis = PmoOpsAxisForm(*Ir, *Node, 0, *X)
    If Axis < 0 : ProcedureReturn #False : EndIf
    Outer = PmoEmitProduct(*X, 0, Axis - 1) : Width = PmoEmitDim(*X, Axis) : Inner = PmoEmitProduct(*X, Axis + 1, Rank - 1)
  EndIf
  If *C\Elements > Width : ProcedureReturn PmoEmitNsFail(*Node, "condition holds " + Str(*C\Elements) + " values for an extent of " + Str(Width) + ".") : EndIf
  For i = 0 To *C\Elements - 1
    If PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), i, @Ok) <> 0 : Kept + 1 : EndIf
  Next
  If *Y = 0 Or *Y\ElementType <> *X\ElementType Or *Y\Elements <> Outer * Kept * Inner : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must hold the " + Str(Outer * Kept * Inner) + " selected elements.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  Protected Dim cond.a(" + Str(*C\Elements) + ")")
  For i = 0 To *C\Elements - 1
    PmoEmitLine(File, "  cond(" + Str(i) + ") = " + Str(Bool(PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), i, @Ok) <> 0)))
  Next
  PmoEmitLine(File, "  PmOpCompress(*i0, @cond(0), *o0, " + Str(Outer) + ", " + Str(Width) + ", " + Str(Inner) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ", " + Str(*C\Elements) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsReverseSequence(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *L.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected TimeAxis.i = PmoEmitAttrI(*Node, "time_axis", 0), BatchAxis.i = PmoEmitAttrI(*Node, "batch_axis", 1)
  If PmoEmitNsAttributesAllowed(*Node, "|batch_axis|time_axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 13) = 0 Or PmoOpsTypeOk(*Node, *L, "sequence_lens", 4) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, *X\ElementType) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) < 2 : ProcedureReturn PmoEmitNsFail(*Node, "the input must have rank two or more.") : EndIf
  If Not ((TimeAxis = 0 And BatchAxis = 1) Or (TimeAxis = 1 And BatchAxis = 0))
    ProcedureReturn PmoEmitNsFail(*Node, "time_axis = " + Str(TimeAxis) + " and batch_axis = " + Str(BatchAxis) + "; the specification allows 0 and 1 or 1 and 0.")
  EndIf
  If *L\Elements <> PmoEmitDim(*X, BatchAxis) : ProcedureReturn PmoEmitNsFail(*Node, "sequence_lens must hold one length per batch, " + Str(PmoEmitDim(*X, BatchAxis)) + ".") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  If PmOpReverseSequence(*i0, *i1, *o0, " + Str(PmoEmitDim(*X, 0)) + ", " + Str(PmoEmitDim(*X, 1)) + ", " + Str(PmoEmitProduct(*X, 2, PmoEmitRank(*X) - 1)) + ", " +
                    Str(Bool(TimeAxis = 0)) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ", 7) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsUpsample(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *S.PmoIrConstant, Mode.s = PmoEmitAttrS(*Node, "mode", "nearest"), Rank.i, d.i, Ok.Integer, Linear.i
  Protected Dim Scale.f(8)
  If Opset > 9 : ProcedureReturn PmoEmitNsFail(*Node, "Upsample is deprecated from opset 10 (use Resize), and the model imports opset " + Str(Opset) + ".") : EndIf
  If Opset >= 9
    If PmoEmitNsAttributesAllowed(*Node, "|mode|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|mode|scales|", Opset) = 0
    ProcedureReturn #False
  EndIf
  If Mode <> "nearest" And Mode <> "linear" : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = " + Mode + " is not an Upsample mode; nearest and linear are.") : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 13) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 1 Or Rank > 8 : ProcedureReturn PmoEmitNsFail(*Node, "the input rank must be one to eight.") : EndIf
  If Opset >= 9
    *S = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
    If *S = 0 Or *S\ElementType <> 1 Or *S\Elements <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "input scales must be a constant FLOAT tensor with one scale per axis.") : EndIf
    For d = 0 To Rank - 1 : Scale(d) = PmoEmitConstF(*Ir, PmoEmitInput(*Node, 1), d, @Ok) : Next
  Else
    d = 0
    ForEach *Node\Attributes()
      If *Node\Attributes()\Name = "scales"
        ForEach *Node\Attributes()\Floats()
          If d < 8 : Scale(d) = *Node\Attributes()\Floats() : EndIf
          d + 1
        Next
      EndIf
    Next
    If d <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute scales must hold one scale per axis.") : EndIf
  EndIf
  Linear = Bool(Mode = "linear")
  For d = 0 To Rank - 1
    If Scale(d) < 1.0 : ProcedureReturn PmoEmitNsFail(*Node, "scale " + StrF(Scale(d)) + " on axis " + Str(d) + " is below 1; Upsample only enlarges.") : EndIf
    If Linear And d < Rank - 2 And Scale(d) <> 1.0 : ProcedureReturn PmoEmitNsFail(*Node, "linear mode is implemented over the last two axes (bilinear); axis " + Str(d) + " has scale " + StrF(Scale(d)) + ".") : EndIf
  Next
  If Linear And (*X\ElementType <> 1 Or Rank < 2) : ProcedureReturn PmoEmitNsFail(*Node, "linear mode is implemented for FLOAT inputs of rank two or more.") : EndIf
  If *Y = 0 Or PmoEmitRank(*Y) <> Rank Or *Y\ElementType <> *X\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the input's rank and element type.") : EndIf
  For d = 0 To Rank - 1
    If PmoEmitDim(*Y, d) <> Int(PmoEmitDim(*X, d) * Scale(d)) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, d)) + "; floor(input * scale) gives " + Str(Int(PmoEmitDim(*X, d) * Scale(d))) + ".") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  For d = 0 To Rank - 1
    PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d)))
    PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(PmoEmitDim(*Y, d)))
    PmoEmitLine(File, "  PmOpSetBits(@PmOpF(" + Str(d) + "), " + PmoOpsBits(Scale(d)) + ")")
  Next
  PmoEmitLine(File, "  PmOpUpsample(*i0, *o0, " + Str(Rank) + ", " + Str(Linear) + ", " + Str(PmoIrElementBytes(*X\ElementType)) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; The activation codes of an RNN or GRU node (0 Sigmoid, 1 Tanh, 2 Relu), two
; per direction, or "" with the refusal's reason.
Procedure.s PmoOpsRecurrentActivations(*Node.PmoOnnxNode, Directions.i, Array Codes.i(1))
  Protected Gru.i = Bool(*Node\Operation = "GRU"), Per.i, Count.i, i.i, Name.s, *A.PmoOnnxAttribute
  Per = 1 + Gru
  For i = 0 To 3
    If Gru
      If i % 2 = 0 : Codes(i) = 0 : Else : Codes(i) = 1 : EndIf
    Else
      Codes(i) = 1
    EndIf
  Next
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "activations" : *A = @*Node\Attributes() : EndIf
  Next
  If *A = 0 : ProcedureReturn "" : EndIf
  Count = ListSize(*A\Strings())
  If Count <> Per * Directions : ProcedureReturn "attribute activations holds " + Str(Count) + " functions; " + *Node\Operation + " takes " + Str(Per) + " per direction, " + Str(Per * Directions) + " here." : EndIf
  i = 0
  ForEach *A\Strings()
    Name = *A\Strings()
    Select LCase(Name)
      Case "sigmoid" : Codes((i / Per) * 2 + i % Per) = 0
      Case "tanh" : Codes((i / Per) * 2 + i % Per) = 1
      Case "relu" : Codes((i / Per) * 2 + i % Per) = 2
      Default : ProcedureReturn "activation " + Name + " is not implemented; Sigmoid, Tanh and Relu are."
    EndSelect
    i + 1
  Next
  ProcedureReturn ""
EndProcedure

Procedure.i PmoEmitOpsRecurrent(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *R.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *L.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
  Protected *H0.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 5))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *YH.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected Gru.i = Bool(*Node\Operation = "GRU"), G.i, D.i, T.q, Bt.q, I.q, H.q, Dir.s, DirCode.i, Reason.s, k.i, Allowed.s
  Protected Dim Codes.i(3)
  G = 1 + 2 * Gru
  Allowed = "|activation_alpha|activation_beta|activations|clip|direction|hidden_size|layout|"
  If Gru : Allowed + "linear_before_reset|" : EndIf
  If PmoEmitNsAttributesAllowed(*Node, Allowed, Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitNsAttributePresent(*Node, "activation_alpha") Or PmoEmitNsAttributePresent(*Node, "activation_beta")
    ProcedureReturn PmoEmitNsFail(*Node, "activation_alpha and activation_beta are not implemented; the activations implemented (Sigmoid, Tanh, Relu) take none.")
  EndIf
  If PmoEmitAttrI(*Node, "layout", 0) <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute layout = 1 (batch first) is not implemented; layout 0 is.") : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *W, "W", 1) = 0 Or PmoOpsTypeOk(*Node, *R, "R", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) <> 3 Or PmoEmitRank(*W) <> 3 Or PmoEmitRank(*R) <> 3 : ProcedureReturn PmoEmitNsFail(*Node, "X, W and R must have rank three.") : EndIf
  T = PmoEmitDim(*X, 0) : Bt = PmoEmitDim(*X, 1) : I = PmoEmitDim(*X, 2) : D = PmoEmitDim(*W, 0) : H = PmoEmitDim(*R, 2)
  Dir = PmoEmitAttrS(*Node, "direction", "forward")
  Select Dir
    Case "forward" : DirCode = 0
    Case "reverse" : DirCode = 1
    Case "bidirectional" : DirCode = 2
    Default : ProcedureReturn PmoEmitNsFail(*Node, "attribute direction = " + Dir + "; forward, reverse and bidirectional are.")
  EndSelect
  If D <> 1 + Bool(DirCode = 2) : ProcedureReturn PmoEmitNsFail(*Node, "W has " + Str(D) + " directions and direction = " + Dir + ".") : EndIf
  If PmoEmitDim(*W, 1) <> G * H Or PmoEmitDim(*W, 2) <> I Or PmoEmitDim(*R, 0) <> D Or PmoEmitDim(*R, 1) <> G * H
    ProcedureReturn PmoEmitNsFail(*Node, "W and R must be [" + Str(D) + ", " + Str(G * H) + ", input size] and [" + Str(D) + ", " + Str(G * H) + ", " + Str(H) + "].")
  EndIf
  If PmoEmitNsAttributePresent(*Node, "hidden_size") And PmoEmitAttrI(*Node, "hidden_size", 0) <> H : ProcedureReturn PmoEmitNsFail(*Node, "hidden_size does not match R.") : EndIf
  If *B And (*B\ElementType <> 1 Or *B\Elements <> D * 2 * G * H) : ProcedureReturn PmoEmitNsFail(*Node, "B must be FLOAT [" + Str(D) + ", " + Str(2 * G * H) + "].") : EndIf
  If *L And (*L\ElementType <> 6 Or *L\Elements <> Bt) : ProcedureReturn PmoEmitNsFail(*Node, "sequence_lens must be INT32 with one length per batch.") : EndIf
  If *H0 And (*H0\ElementType <> 1 Or *H0\Elements <> D * Bt * H) : ProcedureReturn PmoEmitNsFail(*Node, "initial_h must be FLOAT [" + Str(D) + ", " + Str(Bt) + ", " + Str(H) + "].") : EndIf
  If *Y And (*Y\ElementType <> 1 Or *Y\Elements <> T * D * Bt * H) : ProcedureReturn PmoEmitNsFail(*Node, "Y must be FLOAT [" + Str(T) + ", " + Str(D) + ", " + Str(Bt) + ", " + Str(H) + "].") : EndIf
  If *YH And (*YH\ElementType <> 1 Or *YH\Elements <> D * Bt * H) : ProcedureReturn PmoEmitNsFail(*Node, "Y_h must be FLOAT [" + Str(D) + ", " + Str(Bt) + ", " + Str(H) + "].") : EndIf
  Reason = PmoOpsRecurrentActivations(*Node, D, Codes())
  If Reason <> "" : ProcedureReturn PmoEmitNsFail(*Node, Reason) : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(T))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(Bt))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(I))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(H))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(D))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(DirCode))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Gru))
  PmoEmitLine(File, "  PmOpI(7) = " + Str(Bool(PmoEmitAttrI(*Node, "linear_before_reset", 0) <> 0)))
  PmoEmitLine(File, "  PmOpI(8) = " + Str(PmoEmitNsAttributePresent(*Node, "clip")))
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "clip", 0.0)) + ")")
  For k = 0 To 3 : PmoEmitLine(File, "  PmOpI(" + Str(10 + k) + ") = " + Str(Codes(k))) : Next
  PmoEmitLine(File, "  PmOpRecurrent(*i0, *i1, *i2, " + PmoOpsIn(*Node, 3) + ", " + PmoOpsIn(*Node, 4) + ", " + PmoOpsIn(*Node, 5) + ", " + PmoOpsOut(*Node, 0) + ", " + PmoOpsOut(*Node, 1) +
                    ", PmOnnxArenaBase + " + Str(*Ir\OpsScratchOffset) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsRoiAlign(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Rois.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Idx.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Mode.s = PmoEmitAttrS(*Node, "mode", "avg"), Ctm.s, OH.q, OW.q, Half.i
  If Opset >= 16
    If PmoEmitNsAttributesAllowed(*Node, "|coordinate_transformation_mode|mode|output_height|output_width|sampling_ratio|spatial_scale|", Opset) = 0 : ProcedureReturn #False : EndIf
    Ctm = PmoEmitAttrS(*Node, "coordinate_transformation_mode", "half_pixel")
  Else
    If PmoEmitNsAttributesAllowed(*Node, "|mode|output_height|output_width|sampling_ratio|spatial_scale|", Opset) = 0 : ProcedureReturn #False : EndIf
    Ctm = "output_half_pixel"
  EndIf
  If Ctm <> "half_pixel" And Ctm <> "output_half_pixel" : ProcedureReturn PmoEmitNsFail(*Node, "attribute coordinate_transformation_mode = " + Ctm + "; half_pixel and output_half_pixel are.") : EndIf
  If Mode <> "avg" And Mode <> "max" : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = " + Mode + "; avg and max are.") : EndIf
  Half = Bool(Ctm = "half_pixel")
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *Rois, "rois", 1) = 0 Or PmoOpsTypeOk(*Node, *Idx, "batch_indices", 4) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) <> 4 Or PmoEmitRank(*Rois) <> 2 Or PmoEmitDim(*Rois, 1) <> 4 Or *Idx\Elements <> PmoEmitDim(*Rois, 0)
    ProcedureReturn PmoEmitNsFail(*Node, "X must be N x C x H x W, rois [num_rois, 4] and batch_indices [num_rois].")
  EndIf
  OH = PmoEmitAttrI(*Node, "output_height", 1) : OW = PmoEmitAttrI(*Node, "output_width", 1)
  If OH < 1 Or OW < 1 Or PmoEmitAttrI(*Node, "sampling_ratio", 0) < 0 : ProcedureReturn PmoEmitNsFail(*Node, "output_height and output_width must be positive and sampling_ratio not negative.") : EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or *Y\Elements <> PmoEmitDim(*Rois, 0) * PmoEmitDim(*X, 1) * OH * OW : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [num_rois, C, output_height, output_width].") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 1)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(PmoEmitDim(*X, 2)))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(PmoEmitDim(*X, 3)))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(PmoEmitDim(*Rois, 0)))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(OH))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(OW))
  PmoEmitLine(File, "  PmOpI(7) = " + Str(PmoEmitAttrI(*Node, "sampling_ratio", 0)))
  PmoEmitLine(File, "  PmOpI(8) = " + Str(Bool(Mode = "max")))
  PmoEmitLine(File, "  PmOpI(9) = " + Str(Half))
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "spatial_scale", 1.0)) + ")")
  PmoEmitLine(File, "  If PmOpRoiAlign(*i0, *i1, *i2, *o0) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; GridSample's mode and padding codes (PmOpGridSample) for a node, or the
; sentence that refuses it; Spatial is the input rank less two, 0 when the
; rank is not yet known (the runtime-dimension wrapper checks it then).
Procedure.s PmoOpsGridSampleForm(*Node.PmoOnnxNode, Opset.i, Spatial.i, *Mode.Integer, *Pad.Integer)
  Protected Mode.s, Pad.s
  If Opset >= 20
    Mode = PmoEmitAttrS(*Node, "mode", "linear")
    Select Mode
      Case "nearest" : *Mode\i = 0
      Case "linear" : *Mode\i = 1
      Case "cubic" : *Mode\i = 2
      Default : ProcedureReturn "attribute mode = " + Mode + " is not a GridSample-20 mode; nearest, linear and cubic are."
    EndSelect
  Else
    Mode = PmoEmitAttrS(*Node, "mode", "bilinear")
    Select Mode
      Case "nearest" : *Mode\i = 0
      Case "bilinear" : *Mode\i = 1
      Case "bicubic" : *Mode\i = 2
      Default : ProcedureReturn "attribute mode = " + Mode + " is not a GridSample-16 mode; nearest, bilinear and bicubic are."
    EndSelect
  EndIf
  Pad = PmoEmitAttrS(*Node, "padding_mode", "zeros")
  Select Pad
    Case "zeros" : *Pad\i = 0
    Case "border" : *Pad\i = 1
    Case "reflection" : *Pad\i = 2
    Default : ProcedureReturn "attribute padding_mode = " + Pad + "; zeros, border and reflection are."
  EndSelect
  If PmoEmitAttrI(*Node, "align_corners", 0) <> 0 And PmoEmitAttrI(*Node, "align_corners", 0) <> 1
    ProcedureReturn "attribute align_corners = " + Str(PmoEmitAttrI(*Node, "align_corners", 0)) + "; 0 and 1 are."
  EndIf
  If Spatial <> 0
    If Opset < 20 And Spatial <> 2 : ProcedureReturn "GridSample-16 takes a 4-D input [N, C, H, W]; this one has rank " + Str(Spatial + 2) + "." : EndIf
    If Spatial < 1 Or Spatial > 3 : ProcedureReturn "the input has rank " + Str(Spatial + 2) + "; one to three spatial axes (rank 3 to 5) are implemented." : EndIf
    If *Mode\i = 2 And Spatial <> 2 : ProcedureReturn "cubic interpolation is implemented for a 4-D input [N, C, H, W]; this one has rank " + Str(Spatial + 2) + "." : EndIf
  EndIf
  ProcedureReturn ""
EndProcedure

Procedure.i PmoEmitOpsGridSample(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *G.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Reason.s, Mode.Integer, Pad.Integer, R.i, k.i
  If PmoEmitNsAttributesAllowed(*Node, "|align_corners|mode|padding_mode|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *G, "grid", 1) = 0 : ProcedureReturn #False : EndIf
  R = PmoEmitRank(*X) - 2
  If R < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; GridSample takes [N, C, D1 ...].") : EndIf
  Reason = PmoOpsGridSampleForm(*Node, Opset, R, @Mode, @Pad)
  If Reason <> "" : ProcedureReturn PmoEmitNsFail(*Node, Reason) : EndIf
  If PmoEmitRank(*G) <> R + 2 Or PmoEmitDim(*G, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*G, R + 1) <> R
    ProcedureReturn PmoEmitNsFail(*Node, "grid must be [N, output extents ..., " + Str(R) + "] for an input with " + Str(R) + " spatial axes.")
  EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> R + 2 Or PmoEmitDim(*Y, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*Y, 1) <> PmoEmitDim(*X, 1)
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, C, the grid's output extents].")
  EndIf
  For k = 1 To R
    If PmoEmitDim(*Y, k + 1) <> PmoEmitDim(*G, k) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, C, the grid's output extents].") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 1)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(R))
  For k = 0 To 2
    If k < R
      PmoEmitLine(File, "  PmOpI(" + Str(3 + k) + ") = " + Str(PmoEmitDim(*X, k + 2)))
      PmoEmitLine(File, "  PmOpI(" + Str(6 + k) + ") = " + Str(PmoEmitDim(*G, k + 1)))
    EndIf
  Next
  PmoEmitLine(File, "  PmOpI(9) = " + Str(Mode\i))
  PmoEmitLine(File, "  PmOpI(10) = " + Str(Pad\i))
  PmoEmitLine(File, "  PmOpI(11) = " + Str(PmoEmitAttrI(*Node, "align_corners", 0)))
  PmoEmitLine(File, "  PmOpGridSample(*i0, *i1, *o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; One generated procedure for a node of these operators, registered in Calls.
Procedure.i PmoEmitOpsHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index), Opset.i = PmoEmitNsOpset(*Ir), Done.i, *Node.PmoOnnxNode = *Ref\Node
  If PmoOpsOpsetOk(*Node, Opset) = 0 : ProcedureReturn #False : EndIf
  Select *Node\Operation
    Case "Erf", "Reciprocal", "Ceil", "Sign", "Softplus", "Softsign", "Elu", "Selu", "Celu", "HardSigmoid", "HardSwish", "Mish", "Gelu",
         "ThresholdedRelu", "Shrink", "IsNaN", "IsInf", "Tan", "Asin", "Acos", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "BitwiseNot"
      Done = PmoEmitOpsUnary(File, *Ir, *Node, ProcName, Opset)
    Case "Min", "Max", "Sum", "Mean", "Mod", "PRelu", "Or", "Xor", "BitwiseAnd", "BitwiseOr", "BitwiseXor"
      Done = PmoEmitOpsVariadic(File, *Ir, *Node, ProcName, Opset)
    Case "Hardmax", "LpNormalization" : Done = PmoEmitOpsAxisNorm(File, *Ir, *Node, ProcName, Opset)
    Case "MeanVarianceNormalization" : Done = PmoEmitOpsMvn(File, *Ir, *Node, ProcName, Opset)
    Case "LRN" : Done = PmoEmitOpsLrn(File, *Ir, *Node, ProcName, Opset)
    Case "GroupNormalization" : Done = PmoEmitOpsGroupNorm(File, *Ir, *Node, ProcName, Opset)
    Case "EyeLike" : Done = PmoEmitOpsEyeLike(File, *Ir, *Node, ProcName, Opset)
    Case "Det" : Done = PmoEmitOpsDet(File, *Ir, *Node, ProcName, Opset)
    Case "Compress" : Done = PmoEmitOpsCompress(File, *Ir, *Node, ProcName, Opset)
    Case "ReverseSequence" : Done = PmoEmitOpsReverseSequence(File, *Ir, *Node, ProcName, Opset)
    Case "Upsample" : Done = PmoEmitOpsUpsample(File, *Ir, *Node, ProcName, Opset)
    Case "RNN", "GRU" : Done = PmoEmitOpsRecurrent(File, *Ir, *Node, ProcName, Opset)
    Case "RoiAlign" : Done = PmoEmitOpsRoiAlign(File, *Ir, *Node, ProcName, Opset)
    Case "GridSample" : Done = PmoEmitOpsGridSample(File, *Ir, *Node, ProcName, Opset)
    Case "NonMaxSuppression" : ProcedureReturn PmoEmitNsFail(*Node, "its output size depends on the scores, which the fixed-shape path cannot plan.")
    Case "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"
      Done = PmoEmitOpsReduce(File, *Ir, *Node, ProcName, Opset)
    Case "ArgMax", "ArgMin" : Done = PmoEmitOpsArg(File, *Ir, *Node, ProcName, Opset)
    Case "LogSoftmax" : Done = PmoEmitOpsLogSoftmax(File, *Ir, *Node, ProcName, Opset)
    Case "MaxPool", "AveragePool", "LpPool", "GlobalMaxPool", "GlobalAveragePool", "GlobalLpPool"
      Done = PmoEmitOpsPool(File, *Ir, *Node, ProcName, Opset)
    Case "Split" : Done = PmoEmitOpsSplit(File, *Ir, *Node, ProcName, Opset)
    Case "Tile" : Done = PmoEmitOpsTile(File, *Ir, *Node, ProcName, Opset)
    Case "DepthToSpace", "SpaceToDepth" : Done = PmoEmitOpsDepthSpace(File, *Ir, *Node, ProcName, Opset)
    Case "Trilu" : Done = PmoEmitOpsTrilu(File, *Ir, *Node, ProcName, Opset)
    Case "GatherElements" : Done = PmoEmitOpsGatherElements(File, *Ir, *Node, ProcName, Opset)
    Case "GatherND" : Done = PmoEmitOpsGatherND(File, *Ir, *Node, ProcName, Opset)
    Case "OneHot" : Done = PmoEmitOpsOneHot(File, *Ir, *Node, ProcName, Opset)
    Case "Einsum" : Done = PmoEmitOpsEinsum(File, *Ir, *Node, ProcName, Opset)
    Case "Dropout" : Done = PmoEmitOpsDropout(File, *Ir, *Node, ProcName, Opset)
    Case "CastLike" : ProcedureReturn PmoEmitNsFail(*Node, "the element type of target_type is not declared in the model, so fixed-shape emission cannot choose the Cast it stands for.")
    Case "Size" : ProcedureReturn PmoEmitNsFail(*Node, "its input has no fixed shape to count.")
  EndSelect
  If Done = 0 Or PmoEmitError <> "" : ProcedureReturn #False : EndIf
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure
