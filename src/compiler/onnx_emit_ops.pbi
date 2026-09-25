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
                                  "RNN|GRU|NonMaxSuppression|RoiAlign|GridSample|QuantizeLinear|DequantizeLinear|" +
                                  "DynamicQuantizeLinear|MatMulInteger|QLinearMatMul|ConvInteger|QLinearConv|HannWindow|HammingWindow|" +
                                  "BlackmanWindow|DFT|MelWeightMatrix|NegativeLogLikelihoodLoss|SoftmaxCrossEntropyLoss|Col2Im|" +
                                  "CenterCropPad|MaxUnpool|AffineGrid|MaxRoiPool|DeformConv|Unique|Swish|RMSNormalization|CumProd|BitCast|RotaryEmbedding|TensorScatter|Attention|CausalConvWithState|LinearAttention|", "|" + Operation + "|"))
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
    Case "MaxRoiPool" : ProcedureReturn 1
    Case "MaxUnpool" : ProcedureReturn 9
    Case "Col2Im", "CenterCropPad" : ProcedureReturn 18
    Case "DeformConv" : ProcedureReturn 19
    Case "AffineGrid" : ProcedureReturn 20
    Case "Unique" : ProcedureReturn 11
    Case "RMSNormalization" : ProcedureReturn 23
    Case "Swish" : ProcedureReturn 24
    Case "CumProd" : ProcedureReturn 26
    Case "RotaryEmbedding" : ProcedureReturn 23
    Case "Attention" : ProcedureReturn 23
    Case "CausalConvWithState", "LinearAttention" : ProcedureReturn 27
    Case "TensorScatter" : ProcedureReturn 24
    Case "BitCast" : ProcedureReturn 26
    Case "QuantizeLinear", "DequantizeLinear", "MatMulInteger", "QLinearMatMul", "ConvInteger", "QLinearConv" : ProcedureReturn 10
    Case "DynamicQuantizeLinear" : ProcedureReturn 11
    Case "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss" : ProcedureReturn 12
    Case "HannWindow", "HammingWindow", "BlackmanWindow", "DFT", "MelWeightMatrix" : ProcedureReturn 17
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
    ; Multinomial is a random operator whose class choice is a kernel here.
    If (PmoOpsOwns(*Graph\Nodes()\Operation) And *Graph\Nodes()\Operation <> "CastLike") Or *Graph\Nodes()\Operation = "Multinomial"
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
    Case "Swish" : ProcedureReturn 28
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
    Case "Elu", "Celu", "ThresholdedRelu", "Swish" : ProcedureReturn "|alpha|"
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
    Case "Elu", "Celu", "ThresholdedRelu", "Swish" : F0 = PmoEmitAttrF(*Node, "alpha", 1.0)
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
    Case 2 : ProcedureReturn 16
    Case 3 : ProcedureReturn 32
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.s PmoOpsKindNames(Bits.i)
  Protected Text.s, Count.i, i.i
  Protected Dim Names.s(5)
  If Bits & 1 : Names(Count) = "FLOAT" : Count + 1 : EndIf
  If Bits & 2 : Names(Count) = "INT32" : Count + 1 : EndIf
  If Bits & 4 : Names(Count) = "INT64" : Count + 1 : EndIf
  If Bits & 8 : Names(Count) = "BOOL" : Count + 1 : EndIf
  If Bits & 16 : Names(Count) = "UINT8" : Count + 1 : EndIf
  If Bits & 32 : Names(Count) = "INT8" : Count + 1 : EndIf
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
; MelWeightMatrix is folded to a constant before either path runs when its
; five inputs are initializers (onnx_compile.pbi); a node left is refused.
Procedure.s PmoOpsMelSentence()
  ProcedureReturn "MelWeightMatrix is implemented for five initializer inputs, from which the compiler computes the matrix in binary64 as the reference does; a band edge computed in binary32 at run time can fall in another bin."
EndProcedure

; Attention-23/24 (see PmOpAttention for the parameter block).
Procedure.q PmoOpsAttentionTotal(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected *K.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *P.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
  Protected T.q
  If *K = 0 : ProcedureReturn 0 : EndIf
  If PmoEmitRank(*K) = 4 : T = PmoEmitDim(*K, 2) : Else : T = PmoEmitDim(*K, 1) : EndIf
  If PmoEmitInput(*Node, 4) <> "" And *P And PmoEmitRank(*P) = 4 : T + PmoEmitDim(*P, 2) : EndIf
  ProcedureReturn T
EndProcedure


Procedure.q PmoOpsNodeScratch(*Ir.PmoIrModel, *Node.PmoOnnxNode)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected N.q, H.q
  If *X = 0 : ProcedureReturn 0 : EndIf
  Select *Node\Operation
    Case "Det"
      N = PmoEmitDim(*X, PmoEmitRank(*X) - 1)
      ProcedureReturn N * N * 4
    Case "Attention"
      ProcedureReturn (PmoOpsAttentionTotal(*Ir, *Node) + 1) * 4
    Case "LinearAttention"
      *W = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
      H = PmoEmitAttrI(*Node, "kv_num_heads", 1) : N = PmoEmitAttrI(*Node, "q_num_heads", 1)
      If *W = 0 Or H < 1 Or N < 1 Or PmoEmitRank(*X) <> 3 Or PmoEmitRank(*W) <> 3 : ProcedureReturn 0 : EndIf
      ProcedureReturn (PmoEmitDim(*X, 0) * H * (PmoEmitDim(*X, 2) / N) * (PmoEmitDim(*W, 2) / H) + PmoEmitDim(*W, 2) / H + 1) * 4
    Case "SoftmaxCrossEntropyLoss"
      If PmoEmitOutput(*Node, 1) = "" : ProcedureReturn *X\Elements * 4 : EndIf
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

; RMSNormalization-23 (stash_type 1): the axes from `axis` on are normalized;
; scale's extents, leading 1s aside, must be the trailing normalized extents.
Procedure.i PmoOpsRmsScaleOk(*X.PmoIrValue, *S.PmoIrValue, Axis.i)
  Protected r.i = PmoEmitRank(*X), s.i = PmoEmitRank(*S), k.i, f.i = -1
  If s > r - Axis : ProcedureReturn #False : EndIf
  For k = 0 To s - 1
    If f < 0 And PmoEmitDim(*S, k) <> 1 : f = k : EndIf
    If f >= 0 And PmoEmitDim(*S, k) <> PmoEmitDim(*X, r - s + k) : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsRmsNorm(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *S.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Axis.i, n.q
  If PmoEmitNsAttributesAllowed(*Node, "|axis|epsilon|stash_type|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitAttrI(*Node, "stash_type", 1) <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "attribute stash_type = " + Str(PmoEmitAttrI(*Node, "stash_type", 1)) + "; the computation is FLOAT (stash_type 1).") : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *S, "scale", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  Axis = PmoEmitAttrI(*Node, "axis", -1)
  If Axis < 0 : Axis + PmoEmitRank(*X) : EndIf
  If Axis < 0 Or Axis >= PmoEmitRank(*X) : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", -1)) + " is outside the input rank " + Str(PmoEmitRank(*X)) + ".") : EndIf
  If PmoOpsRmsScaleOk(*X, *S, Axis) = 0
    ProcedureReturn PmoEmitNsFail(*Node, "scale must have the trailing normalized extents (leading 1s allowed); a scale that broadcasts inside them is not implemented.")
  EndIf
  n = PmoEmitProduct(*X, Axis, PmoEmitRank(*X) - 1)
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ")")
  PmoEmitLine(File, "  PmOpRmsNorm(*i0, *i1, *o0, " + Str(*X\Elements / n) + ", " + Str(n) + ", " + Str(*S\Elements) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; CumProd-26: the axis a constant input; exclusive and reverse as CumSum.
Procedure.i PmoEmitOpsCumProd(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Ok.Integer, Axis.q, r.i
  If PmoEmitNsAttributesAllowed(*Node, "|exclusive|reverse|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "x", 1 | 2 | 4) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, *X\ElementType) = 0 : ProcedureReturn #False : EndIf
  Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
  If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input axis must be a constant for fixed-shape emission.") : EndIf
  r = PmoEmitRank(*X)
  If Axis < 0 : Axis + r : EndIf
  If Axis < 0 Or Axis >= r : ProcedureReturn PmoEmitNsFail(*Node, "axis is outside the input rank " + Str(r) + ".") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpCumProd(*i0, *o0, " + Str(PmoEmitProduct(*X, 0, Axis - 1)) + ", " + Str(PmoEmitDim(*X, Axis)) + ", " + Str(PmoEmitProduct(*X, Axis + 1, r - 1)) + ", " +
                    Str(*X\ElementType) + ", " + Str(Bool(PmoEmitAttrI(*Node, "exclusive", 0) <> 0)) + ", " + Str(Bool(PmoEmitAttrI(*Node, "reverse", 0) <> 0)) + ")")
  PmoOpsTail(File, #True)
  ProcedureReturn #True
EndProcedure

; BitCast-26: the bytes copied; `to` one of the carried types of the input's width.
Procedure.i PmoOpsKindWidth(Kind.i)
  Select Kind
    Case 1, 6 : ProcedureReturn 4
    Case 7 : ProcedureReturn 8
    Case 2, 3, 9 : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmoEmitOpsBitCast(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Kind.i
  If PmoEmitNsAttributesAllowed(*Node, "|to|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1 | 2 | 4 | 8 | 16 | 32) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitNsAttributePresent(*Node, "to") = 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute to is required.") : EndIf
  Kind = PmoEmitAttrI(*Node, "to", 0)
  If PmoOpsKindWidth(Kind) = 0
    ProcedureReturn PmoEmitNsFail(*Node, "to = " + Str(Kind) + " (" + PmoEmitNsTypeName(Kind) + ") is not implemented; FLOAT, UINT8, INT8, INT32, INT64 and BOOL are.")
  EndIf
  If PmoOpsKindWidth(Kind) <> PmoOpsKindWidth(*X\ElementType)
    ProcedureReturn PmoEmitNsFail(*Node, "to = " + PmoEmitNsTypeName(Kind) + " has another bit width than the input's " + PmoEmitNsTypeName(*X\ElementType) + "; BitCast keeps the width.")
  EndIf
  If PmoOpsSameShape(*Node, *X, *Y, Kind) = 0 : ProcedureReturn #False : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpCopyBytes(*i0, *o0, " + Str(*X\Elements * PmoOpsKindWidth(Kind)) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; RotaryEmbedding-23: X [B, S, H*D] (num_heads) or [B, H, S, D]; caches
; [max_position, rd/2] with position_ids, [B, S, rd/2] without.
Procedure.i PmoEmitOpsRotary(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *C.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Sn.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *P.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Bn.q, Sq.q, Hn.q, Dn.q, Rd.q, HasPos.i, MaxPos.q
  If PmoEmitNsAttributesAllowed(*Node, "|interleaved|num_heads|rotary_embedding_dim|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *C, "cos_cache", 1) = 0 Or PmoOpsTypeOk(*Node, *Sn, "sin_cache", 1) = 0 : ProcedureReturn #False : EndIf
  HasPos = Bool(PmoEmitInput(*Node, 3) <> "")
  If HasPos And PmoOpsTypeOk(*Node, *P, "position_ids", 4) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank = 4
    Bn = PmoEmitDim(*X, 0) : Hn = PmoEmitDim(*X, 1) : Sq = PmoEmitDim(*X, 2) : Dn = PmoEmitDim(*X, 3)
  ElseIf Rank = 3
    Hn = PmoEmitAttrI(*Node, "num_heads", 0)
    If Hn < 1 Or PmoEmitDim(*X, 2) % Hn <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "a rank-3 input needs num_heads dividing its last extent.") : EndIf
    Bn = PmoEmitDim(*X, 0) : Sq = PmoEmitDim(*X, 1) : Dn = PmoEmitDim(*X, 2) / Hn
  Else
    ProcedureReturn PmoEmitNsFail(*Node, "X has rank " + Str(Rank) + "; RotaryEmbedding takes [B, S, hidden] or [B, H, S, D].")
  EndIf
  Rd = PmoEmitAttrI(*Node, "rotary_embedding_dim", 0)
  If Rd = 0 : Rd = Dn : EndIf
  If Rd < 2 Or Rd % 2 <> 0 Or Rd > Dn : ProcedureReturn PmoEmitNsFail(*Node, "rotary_embedding_dim = " + Str(Rd) + " must be even and at most the head size " + Str(Dn) + ".") : EndIf
  If HasPos
    If PmoEmitRank(*C) <> 2 Or PmoEmitDim(*C, 1) <> Rd / 2 Or PmoEmitShapeEqual(*C, *Sn) = 0 Or PmoEmitRank(*P) <> 2 Or PmoEmitDim(*P, 0) <> Bn Or PmoEmitDim(*P, 1) <> Sq
      ProcedureReturn PmoEmitNsFail(*Node, "with position_ids [B, S], the caches must be [max_position, rotary_embedding_dim/2].")
    EndIf
    MaxPos = PmoEmitDim(*C, 0)
  ElseIf PmoEmitRank(*C) <> 3 Or PmoEmitDim(*C, 0) <> Bn Or PmoEmitDim(*C, 1) <> Sq Or PmoEmitDim(*C, 2) <> Rd / 2 Or PmoEmitShapeEqual(*C, *Sn) = 0
    ProcedureReturn PmoEmitNsFail(*Node, "without position_ids, the caches must be [B, S, rotary_embedding_dim/2].")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  If PmOpRotary(*i0, *i1, *i2, " + PmoOpsIn(*Node, 3) + ", *o0, " + Str(Bn) + ", " + Str(Sq) + ", " + Str(Hn) + ", " + Str(Dn) + ", " + Str(Rd) + ", " +
                    Str(Bool(PmoEmitAttrI(*Node, "interleaved", 0) <> 0)) + ", " + Str(Bool(Rank = 4)) + ", " + Str(HasPos) + ", " + Str(MaxPos) + ") <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; TensorScatter-24: past cache, update, optional write_indices; linear or circular.
Procedure.i PmoEmitOpsTensorScatter(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *P.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *U.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Axis.i, d.i, Mode.s
  If PmoEmitNsAttributesAllowed(*Node, "|axis|mode|", Opset) = 0 : ProcedureReturn #False : EndIf
  Mode = PmoEmitAttrS(*Node, "mode", "linear")
  If Mode <> "linear" And Mode <> "circular" : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = " + Mode + "; linear and circular are defined.") : EndIf
  If PmoOpsTypeOk(*Node, *P, "past_cache", 1 | 2 | 4 | 8 | 16 | 32) = 0 Or PmoOpsTypeOk(*Node, *U, "update", 1 | 2 | 4 | 8 | 16 | 32) = 0 : ProcedureReturn #False : EndIf
  If *U\ElementType <> *P\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "update must have the past cache's element type.") : EndIf
  If PmoEmitInput(*Node, 2) <> "" And PmoOpsTypeOk(*Node, *W, "write_indices", 4) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsSameShape(*Node, *P, *Y, *P\ElementType) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*P)
  Axis = PmoEmitAttrI(*Node, "axis", -2)
  If Axis < 0 : Axis + Rank : EndIf
  If Axis < 1 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis must name a sequence axis after the batch axis.") : EndIf
  If PmoEmitRank(*U) <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "update must have the past cache's rank.") : EndIf
  For d = 0 To Rank - 1
    If d <> Axis And PmoEmitDim(*U, d) <> PmoEmitDim(*P, d) : ProcedureReturn PmoEmitNsFail(*Node, "update must match the past cache on every axis but the sequence axis.") : EndIf
  Next
  If PmoEmitDim(*U, Axis) > PmoEmitDim(*P, Axis) : ProcedureReturn PmoEmitNsFail(*Node, "update's sequence extent exceeds the cache's.") : EndIf
  If PmoEmitInput(*Node, 2) <> "" And (PmoEmitRank(*W) <> 1 Or PmoEmitDim(*W, 0) <> PmoEmitDim(*P, 0))
    ProcedureReturn PmoEmitNsFail(*Node, "write_indices must hold one index per batch sample.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  If PmOpTensorScatter(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", *o0, " + Str(PmoEmitProduct(*P, 0, Axis - 1)) + ", " + Str(PmoEmitProduct(*P, 1, Axis - 1)) + ", " +
                    Str(PmoEmitDim(*P, Axis)) + ", " + Str(PmoEmitDim(*U, Axis)) + ", " + Str(PmoEmitProduct(*P, Axis + 1, Rank - 1)) + ", " + Str(PmoOpsKindWidth(*P\ElementType)) + ", " +
                    Str(Bool(Mode = "circular")) + ") <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; Attention-23/24 (see PmOpAttention for the parameter block).
Procedure.i PmoEmitOpsAttention(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *Q.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *K.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *V.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *M.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *PK.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
  Protected *PV.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 5))
  Protected *NP.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 6))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rank.i, Bn.q, Hq.q, Hkv.q, Sq.q, Skv.q, Pl.q, Dn.q, Dv.q, T.q, Mode.i, k.i, Soft.f
  Protected Dim MaskDims.q(3)
  If PmoEmitNsAttributesAllowed(*Node, "|is_causal|kv_num_heads|q_num_heads|qk_matmul_output_mode|scale|softcap|softmax_precision|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitNsAttributePresent(*Node, "softmax_precision") And PmoEmitAttrI(*Node, "softmax_precision", 1) <> 1
    ProcedureReturn PmoEmitNsFail(*Node, "softmax_precision = " + Str(PmoEmitAttrI(*Node, "softmax_precision", 1)) + "; the softmax is FLOAT (1).")
  EndIf
  Mode = PmoEmitAttrI(*Node, "qk_matmul_output_mode", 0)
  If Mode < 0 Or Mode > 3 : ProcedureReturn PmoEmitNsFail(*Node, "qk_matmul_output_mode = " + Str(Mode) + "; 0 to 3 are defined.") : EndIf
  If PmoOpsTypeOk(*Node, *Q, "Q", 1) = 0 Or PmoOpsTypeOk(*Node, *K, "K", 1) = 0 Or PmoOpsTypeOk(*Node, *V, "V", 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*Q)
  If PmoEmitRank(*K) <> Rank Or PmoEmitRank(*V) <> Rank Or (Rank <> 3 And Rank <> 4) : ProcedureReturn PmoEmitNsFail(*Node, "Q, K and V must all be rank 3 or all rank 4.") : EndIf
  Bn = PmoEmitDim(*Q, 0)
  If Rank = 3
    Hq = PmoEmitAttrI(*Node, "q_num_heads", 0) : Hkv = PmoEmitAttrI(*Node, "kv_num_heads", 0)
    If Hq < 1 Or Hkv < 1 Or PmoEmitDim(*Q, 2) % Hq <> 0 Or PmoEmitDim(*K, 2) % Hkv <> 0 Or PmoEmitDim(*V, 2) % Hkv <> 0
      ProcedureReturn PmoEmitNsFail(*Node, "3-D inputs need q_num_heads and kv_num_heads dividing the hidden sizes.")
    EndIf
    Sq = PmoEmitDim(*Q, 1) : Skv = PmoEmitDim(*K, 1) : Dn = PmoEmitDim(*Q, 2) / Hq : Dv = PmoEmitDim(*V, 2) / Hkv
    If PmoEmitDim(*K, 2) / Hkv <> Dn Or PmoEmitDim(*V, 1) <> Skv : ProcedureReturn PmoEmitNsFail(*Node, "K must have Q's head size, and V K's sequence length.") : EndIf
  Else
    Hq = PmoEmitDim(*Q, 1) : Hkv = PmoEmitDim(*K, 1) : Sq = PmoEmitDim(*Q, 2) : Skv = PmoEmitDim(*K, 2) : Dn = PmoEmitDim(*Q, 3) : Dv = PmoEmitDim(*V, 3)
    If PmoEmitDim(*K, 3) <> Dn Or PmoEmitDim(*V, 1) <> Hkv Or PmoEmitDim(*V, 2) <> Skv : ProcedureReturn PmoEmitNsFail(*Node, "K must have Q's head size, and V K's heads and sequence length.") : EndIf
  EndIf
  If PmoEmitDim(*K, 0) <> Bn Or PmoEmitDim(*V, 0) <> Bn Or Hq % Hkv <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "the batch sizes must agree, and q_num_heads must be a multiple of kv_num_heads.") : EndIf
  If Bool(PmoEmitInput(*Node, 4) <> "") <> Bool(PmoEmitInput(*Node, 5) <> "") : ProcedureReturn PmoEmitNsFail(*Node, "past_key and past_value go together.") : EndIf
  If PmoEmitInput(*Node, 4) <> ""
    If PmoEmitInput(*Node, 6) <> "" : ProcedureReturn PmoEmitNsFail(*Node, "nonpad_kv_seqlen does not go with past_key and past_value.") : EndIf
    If PmoOpsTypeOk(*Node, *PK, "past_key", 1) = 0 Or PmoOpsTypeOk(*Node, *PV, "past_value", 1) = 0 : ProcedureReturn #False : EndIf
    If PmoEmitRank(*PK) <> 4 Or PmoEmitRank(*PV) <> 4 Or PmoEmitDim(*PK, 0) <> Bn Or PmoEmitDim(*PK, 1) <> Hkv Or PmoEmitDim(*PK, 3) <> Dn Or
       PmoEmitDim(*PV, 0) <> Bn Or PmoEmitDim(*PV, 1) <> Hkv Or PmoEmitDim(*PV, 2) <> PmoEmitDim(*PK, 2) Or PmoEmitDim(*PV, 3) <> Dv
      ProcedureReturn PmoEmitNsFail(*Node, "past_key and past_value must be [batch, kv_num_heads, past, head_size].")
    EndIf
    Pl = PmoEmitDim(*PK, 2)
  EndIf
  T = Pl + Skv
  For k = 0 To 3 : MaskDims(k) = 1 : Next
  If PmoEmitInput(*Node, 3) <> ""
    If *M = 0 Or (*M\ElementType <> 1 And *M\ElementType <> 9) Or PmoEmitRank(*M) < 1 Or PmoEmitRank(*M) > 4
      ProcedureReturn PmoEmitNsFail(*Node, "attn_mask must be FLOAT or BOOL of rank 1 to 4.")
    EndIf
    For k = 0 To PmoEmitRank(*M) - 1 : MaskDims(4 - PmoEmitRank(*M) + k) = PmoEmitDim(*M, k) : Next
    If (MaskDims(0) <> 1 And MaskDims(0) <> Bn) Or (MaskDims(1) <> 1 And MaskDims(1) <> Hq) Or (MaskDims(2) <> 1 And MaskDims(2) <> Sq) Or MaskDims(3) > T
      ProcedureReturn PmoEmitNsFail(*Node, "attn_mask must broadcast to [batch, q_num_heads, q_sequence, total_sequence].")
    EndIf
  EndIf
  If PmoEmitInput(*Node, 6) <> ""
    If PmoOpsTypeOk(*Node, *NP, "nonpad_kv_seqlen", 4) = 0 : ProcedureReturn #False : EndIf
    If *NP\Elements <> Bn : ProcedureReturn PmoEmitNsFail(*Node, "nonpad_kv_seqlen must hold one length per batch sample.") : EndIf
  EndIf
  If *Y = 0 Or *Y\ElementType <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "the output Y needs a concrete FLOAT shape.") : EndIf
  If (Rank = 3 And (PmoEmitRank(*Y) <> 3 Or PmoEmitDim(*Y, 2) <> Hq * Dv)) Or (Rank = 4 And (PmoEmitRank(*Y) <> 4 Or PmoEmitDim(*Y, 3) <> Dv))
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output Y has the wrong shape.")
  EndIf
  Soft = PmoEmitAttrF(*Node, "softcap", 0.0)
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Bn) + " : PmOpI(1) = " + Str(Hq) + " : PmOpI(2) = " + Str(Hkv) + " : PmOpI(3) = " + Str(Sq) + " : PmOpI(4) = " + Str(Skv))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(Pl) + " : PmOpI(6) = " + Str(Dn) + " : PmOpI(7) = " + Str(Dv) + " : PmOpI(8) = " + Str(Bool(Rank = 3)) + " : PmOpI(9) = " + Str(Bool(PmoEmitAttrI(*Node, "is_causal", 0) <> 0)))
  If PmoEmitInput(*Node, 3) <> "" : k = *M\ElementType : Else : k = 0 : EndIf
  PmoEmitLine(File, "  PmOpI(10) = " + Str(k) + " : PmOpI(11) = " + Str(MaskDims(0)) + " : PmOpI(12) = " + Str(MaskDims(1)) + " : PmOpI(13) = " + Str(MaskDims(2)) + " : PmOpI(14) = " + Str(MaskDims(3)))
  If PmoEmitOutput(*Node, 3) <> "" : k = Mode : Else : k = -1 : EndIf
  PmoEmitLine(File, "  PmOpI(15) = " + Str(Bool(PmoEmitInput(*Node, 6) <> "")) + " : PmOpI(16) = " + Str(k) + " : PmOpI(17) = " + Str(PmoEmitNsAttributePresent(*Node, "scale")) + " : PmOpI(18) = " + Str(Bool(Soft > 0.0)))
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "scale", 1.0)) + ") : PmOpSetBits(@PmOpF(1), " + PmoOpsBits(Soft) + ")")
  PmoEmitLine(File, "  PmOpAttention(*i0, *i1, *i2, " + PmoOpsIn(*Node, 3) + ", " + PmoOpsIn(*Node, 4) + ", " + PmoOpsIn(*Node, 5) + ", " + PmoOpsIn(*Node, 6) + ", *o0, " +
                    PmoOpsOut(*Node, 1) + ", " + PmoOpsOut(*Node, 2) + ", " + PmoOpsOut(*Node, 3) + ", PmOnnxArenaBase + " + Str(*Ir\OpsScratchOffset) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; CausalConvWithState-27.
Procedure.i PmoEmitOpsCausalConv(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *P.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Act.s, Kn.q
  If PmoEmitNsAttributesAllowed(*Node, "|activation|", Opset) = 0 : ProcedureReturn #False : EndIf
  Act = PmoEmitAttrS(*Node, "activation", "none")
  If Act <> "none" And Act <> "silu" And Act <> "swish" : ProcedureReturn PmoEmitNsFail(*Node, "attribute activation = " + Act + "; none, silu and swish are defined.") : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 Or PmoOpsTypeOk(*Node, *W, "weight", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) <> 3 Or PmoEmitRank(*W) <> 3 Or PmoEmitDim(*W, 0) <> PmoEmitDim(*X, 1) Or PmoEmitDim(*W, 1) <> 1
    ProcedureReturn PmoEmitNsFail(*Node, "input must be [B, C, L] and weight [C, 1, K].")
  EndIf
  Kn = PmoEmitDim(*W, 2)
  If PmoEmitInput(*Node, 2) <> "" And (PmoOpsTypeOk(*Node, *B, "bias", 1) = 0 Or *B\Elements <> PmoEmitDim(*X, 1)) : ProcedureReturn PmoEmitNsFail(*Node, "bias must hold one FLOAT per channel.") : EndIf
  If PmoEmitInput(*Node, 3) <> ""
    If PmoOpsTypeOk(*Node, *P, "past_state", 1) = 0 Or PmoEmitRank(*P) <> 3 Or PmoEmitDim(*P, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*P, 1) <> PmoEmitDim(*X, 1) Or PmoEmitDim(*P, 2) <> Kn - 1
      ProcedureReturn PmoEmitNsFail(*Node, "past_state must be FLOAT [B, C, K-1].")
    EndIf
  EndIf
  If PmoOpsSameShape(*Node, *X, *Y, 1) = 0 : ProcedureReturn #False : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpCausalConv(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", " + PmoOpsIn(*Node, 3) + ", *o0, " + PmoOpsOut(*Node, 1) + ", " + Str(PmoEmitDim(*X, 0)) + ", " +
                    Str(PmoEmitDim(*X, 1)) + ", " + Str(PmoEmitDim(*X, 2)) + ", " + Str(Kn) + ", " + Str(Bool(Act <> "none")) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; LinearAttention-27.
Procedure.i PmoOpsLinearRule(*Node.PmoOnnxNode)
  Select PmoEmitAttrS(*Node, "update_rule", "gated_delta")
    Case "linear" : ProcedureReturn 0
    Case "gated" : ProcedureReturn 1
    Case "delta" : ProcedureReturn 2
    Case "gated_delta" : ProcedureReturn 3
  EndSelect
  ProcedureReturn -1
EndProcedure

Procedure.i PmoEmitOpsLinearAttention(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *Q.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *K.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *V.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *P.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *G.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
  Protected *Bt.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 5))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Rule.i, Hq.q, Hkv.q, Bn.q, Tn.q, Dk.q, Dv.q, Gated.i, Delta.i, PerKey.i, PerHead.i, State.s, Scale.f
  If PmoEmitNsAttributesAllowed(*Node, "|chunk_size|kv_num_heads|q_num_heads|scale|update_rule|", Opset) = 0 : ProcedureReturn #False : EndIf
  Rule = PmoOpsLinearRule(*Node)
  If Rule < 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute update_rule = " + PmoEmitAttrS(*Node, "update_rule", "") + "; linear, gated, delta and gated_delta are defined.") : EndIf
  If PmoOpsTypeOk(*Node, *Q, "query", 1) = 0 Or PmoOpsTypeOk(*Node, *K, "key", 1) = 0 Or PmoOpsTypeOk(*Node, *V, "value", 1) = 0 : ProcedureReturn #False : EndIf
  Hq = PmoEmitAttrI(*Node, "q_num_heads", 0) : Hkv = PmoEmitAttrI(*Node, "kv_num_heads", 0)
  If Hq < 1 Or Hkv < 1 Or Hq % Hkv <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "q_num_heads and kv_num_heads are required, q_num_heads a multiple of kv_num_heads.") : EndIf
  If PmoEmitRank(*Q) <> 3 Or PmoEmitRank(*K) <> 3 Or PmoEmitRank(*V) <> 3 : ProcedureReturn PmoEmitNsFail(*Node, "query, key and value must be rank 3 [B, T, heads x size].") : EndIf
  Bn = PmoEmitDim(*Q, 0) : Tn = PmoEmitDim(*Q, 1)
  If PmoEmitDim(*Q, 2) % Hq <> 0 Or PmoEmitDim(*V, 2) % Hkv <> 0 : ProcedureReturn PmoEmitNsFail(*Node, "the head counts must divide the packed extents.") : EndIf
  Dk = PmoEmitDim(*Q, 2) / Hq : Dv = PmoEmitDim(*V, 2) / Hkv
  If PmoEmitDim(*K, 0) <> Bn Or PmoEmitDim(*K, 1) <> Tn Or PmoEmitDim(*K, 2) <> Hkv * Dk Or PmoEmitDim(*V, 0) <> Bn Or PmoEmitDim(*V, 1) <> Tn
    ProcedureReturn PmoEmitNsFail(*Node, "key and value must match the query's batch, length and head size.")
  EndIf
  Gated = Bool(Rule = 1 Or Rule = 3) : Delta = Bool(Rule = 2 Or Rule = 3)
  If Bool(PmoEmitInput(*Node, 4) <> "") <> Gated Or Bool(PmoEmitInput(*Node, 5) <> "") <> Delta
    ProcedureReturn PmoEmitNsFail(*Node, "update_rule " + PmoEmitAttrS(*Node, "update_rule", "gated_delta") + " needs decay exactly when gated and beta exactly for the delta rules.")
  EndIf
  If Gated
    If PmoOpsTypeOk(*Node, *G, "decay", 1) = 0 Or PmoEmitRank(*G) <> 3 Or PmoEmitDim(*G, 0) <> Bn Or PmoEmitDim(*G, 1) <> Tn Or (PmoEmitDim(*G, 2) <> Hkv And PmoEmitDim(*G, 2) <> Hkv * Dk)
      ProcedureReturn PmoEmitNsFail(*Node, "decay must be FLOAT [B, T, kv_num_heads] or [B, T, kv_num_heads x d_k].")
    EndIf
    PerKey = Bool(PmoEmitDim(*G, 2) = Hkv * Dk And Dk <> 1)
  EndIf
  If Delta
    If PmoOpsTypeOk(*Node, *Bt, "beta", 1) = 0 Or PmoEmitRank(*Bt) <> 3 Or PmoEmitDim(*Bt, 0) <> Bn Or PmoEmitDim(*Bt, 1) <> Tn Or (PmoEmitDim(*Bt, 2) <> Hkv And PmoEmitDim(*Bt, 2) <> 1)
      ProcedureReturn PmoEmitNsFail(*Node, "beta must be FLOAT [B, T, kv_num_heads] or [B, T, 1].")
    EndIf
    PerHead = Bool(PmoEmitDim(*Bt, 2) = Hkv And Hkv <> 1)
  EndIf
  If PmoEmitInput(*Node, 3) <> ""
    If PmoOpsTypeOk(*Node, *P, "past_state", 1) = 0 Or *P\Elements <> Bn * Hkv * Dk * Dv Or PmoEmitRank(*P) <> 4
      ProcedureReturn PmoEmitNsFail(*Node, "past_state must be FLOAT [B, kv_num_heads, d_k, d_v].")
    EndIf
  EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> 3 Or PmoEmitDim(*Y, 2) <> Hq * Dv : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [B, T, q_num_heads x d_v].") : EndIf
  Scale = PmoEmitAttrF(*Node, "scale", 0.0)
  If PmoEmitOutput(*Node, 1) <> "" : State = "*o1" : Else : State = "PmOnnxArenaBase + " + Str(*Ir\OpsScratchOffset) : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Bn) + " : PmOpI(1) = " + Str(Tn) + " : PmOpI(2) = " + Str(Hq) + " : PmOpI(3) = " + Str(Hkv) + " : PmOpI(4) = " + Str(Dk) + " : PmOpI(5) = " + Str(Dv))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Gated) + " : PmOpI(7) = " + Str(Delta) + " : PmOpI(8) = " + Str(PerKey) + " : PmOpI(9) = " + Str(PerHead) + " : PmOpI(10) = " + Str(Bool(Scale <> 0.0)))
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(Scale) + ")")
  PmoEmitLine(File, "  PmOpLinearAttention(*i0, *i1, *i2, " + PmoOpsIn(*Node, 3) + ", " + PmoOpsIn(*Node, 4) + ", " + PmoOpsIn(*Node, 5) + ", " + State + ", PmOnnxArenaBase + " +
                    Str(*Ir\OpsScratchOffset + Bn * Hkv * Dk * Dv * 4) + ", *o0)")
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

; ---- the signal operators and the losses -----------------------------------
Procedure.i PmoEmitOpsWindow(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Ok.Integer, Size.q, Kind.i
  If PmoEmitNsAttributesAllowed(*Node, "|output_datatype|periodic|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitAttrI(*Node, "output_datatype", 1) <> 1
    ProcedureReturn PmoEmitNsFail(*Node, "output_datatype " + PmoEmitNsTypeName(PmoEmitAttrI(*Node, "output_datatype", 1)) + " is not implemented; FLOAT is.")
  EndIf
  Size = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 0), 0, @Ok)
  If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input size must be a constant for fixed-shape emission, which sizes the output when the source is written.") : EndIf
  If Size < 1 : ProcedureReturn PmoEmitNsFail(*Node, "size = " + Str(Size) + " must be positive.") : EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> 1 Or *Y\Elements <> Size
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [" + Str(Size) + "].")
  EndIf
  Select *Node\Operation
    Case "HannWindow" : Kind = 0
    Case "HammingWindow" : Kind = 1
    Default : Kind = 2
  EndSelect
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpWindow(*o0, " + Str(Size) + ", " + Str(Kind) + ", " + Str(Bool(PmoEmitAttrI(*Node, "periodic", 1) <> 0)) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; DFT-17 (axis an attribute, default 1) and DFT-20 (axis an input, default -2).
Procedure.i PmoEmitOpsDft(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Ok.Integer, Rank.i, Axis.q, L.q, N.q, Inv.i, Ones.i, Comp.q, Outer.q = 1, Inner.q = 1, d.i, OutAxis.q, OutLast.q, Allowed.s
  Allowed = "|inverse|onesided|"
  If Opset < 20 : Allowed = "|axis|inverse|onesided|" : EndIf
  If PmoEmitNsAttributesAllowed(*Node, Allowed, Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 2 : ProcedureReturn PmoEmitNsFail(*Node, "the input must have rank two or more, its last axis holding the real part (1) or the real and imaginary parts (2).") : EndIf
  Comp = PmoEmitDim(*X, Rank - 1)
  If Comp <> 1 And Comp <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "the input's last axis must be 1 (real) or 2 (complex); it is " + Str(Comp) + ".") : EndIf
  If Opset < 20
    Axis = PmoEmitAttrI(*Node, "axis", 1)
  ElseIf PmoEmitInput(*Node, 2) <> ""
    Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 2), 0, @Ok)
    If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input axis must be a constant for fixed-shape emission.") : EndIf
  Else
    Axis = -2
  EndIf
  If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank - 1 : ProcedureReturn PmoEmitNsFail(*Node, "the signal axis must be one of the input's axes before the last; it is " + Str(Axis) + ".") : EndIf
  L = PmoEmitDim(*X, Axis)
  Inv = Bool(PmoEmitAttrI(*Node, "inverse", 0) <> 0)
  Ones = Bool(PmoEmitAttrI(*Node, "onesided", 0) <> 0)
  If Inv And Ones And Comp <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "an inverse one-sided transform takes a complex (last axis 2) spectrum.") : EndIf
  If PmoEmitInput(*Node, 1) <> ""
    N = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
    If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input dft_length must be a constant for fixed-shape emission.") : EndIf
  ElseIf Inv And Ones
    N = 2 * (L - 1)
  Else
    N = L
  EndIf
  If N < 1 : ProcedureReturn PmoEmitNsFail(*Node, "dft_length = " + Str(N) + " must be positive.") : EndIf
  OutAxis = N : OutLast = 2
  If Ones And Inv = 0 : OutAxis = N / 2 + 1 : EndIf
  If Ones And Inv : OutLast = 1 : EndIf
  For d = 0 To Axis - 1 : Outer * PmoEmitDim(*X, d) : Next
  For d = Axis + 1 To Rank - 2 : Inner * PmoEmitDim(*X, d) : Next
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> Rank Or PmoEmitDim(*Y, Axis) <> OutAxis Or PmoEmitDim(*Y, Rank - 1) <> OutLast
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT with " + Str(OutAxis) + " on the signal axis and " + Str(OutLast) + " last.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Outer))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(L))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Inner))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(Comp))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(N))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(Inv))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Ones))
  PmoEmitLine(File, "  PmOpDft(*i0, *o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; NegativeLogLikelihoodLoss-12/13 and SoftmaxCrossEntropyLoss-12/13.
Procedure.i PmoEmitOpsLoss(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected Sce.i = Bool(*Node\Operation = "SoftmaxCrossEntropyLoss")
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *T.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *LP.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected Red.s = PmoEmitAttrS(*Node, "reduction", "mean"), Code.i, Rank.i, N.q, C.q, Dn.q = 1, d.i, LpAddr.s
  If PmoEmitNsAttributesAllowed(*Node, "|ignore_index|reduction|", Opset) = 0 : ProcedureReturn #False : EndIf
  Select Red
    Case "none" : Code = 0
    Case "sum" : Code = 1
    Case "mean" : Code = 2
    Default : ProcedureReturn PmoEmitNsFail(*Node, "attribute reduction = " + Red + "; none, sum and mean are.")
  EndSelect
  If PmoOpsTypeOk(*Node, *X, "the scores", 1) = 0 Or PmoOpsTypeOk(*Node, *T, "the target", 2 | 4) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 2 : ProcedureReturn PmoEmitNsFail(*Node, "the scores must be [N, C] or [N, C, D1 ...].") : EndIf
  N = PmoEmitDim(*X, 0) : C = PmoEmitDim(*X, 1)
  For d = 2 To Rank - 1 : Dn * PmoEmitDim(*X, d) : Next
  If PmoEmitRank(*T) <> Rank - 1 Or *T\Elements <> N * Dn : ProcedureReturn PmoEmitNsFail(*Node, "the target must be [N, D1 ...], the scores' shape without C.") : EndIf
  If *W And (*W\ElementType <> 1 Or PmoEmitRank(*W) <> 1 Or *W\Elements <> C) : ProcedureReturn PmoEmitNsFail(*Node, "weight must be FLOAT [C].") : EndIf
  If *Y = 0 Or *Y\ElementType <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "the declared loss must be FLOAT.") : EndIf
  If (Code = 0 And PmoEmitShapeEqual(*Y, *T) = 0) Or (Code <> 0 And *Y\Elements <> 1)
    ProcedureReturn PmoEmitNsFail(*Node, "the declared loss must have the target's shape with reduction none, one value otherwise.")
  EndIf
  If Sce = 0 And *LP : ProcedureReturn PmoEmitNsFail(*Node, "NegativeLogLikelihoodLoss has one output.") : EndIf
  If *LP And (*LP\ElementType <> 1 Or PmoEmitShapeEqual(*LP, *X) = 0) : ProcedureReturn PmoEmitNsFail(*Node, "log_prob must be FLOAT with the scores' shape.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  LpAddr = "*i0"
  If Sce
    LpAddr = "PmOnnxArenaBase + " + Str(*Ir\OpsScratchOffset)
    If *LP : LpAddr = "*o1" : EndIf
    PmoEmitLine(File, "  PmOpLogSoftmax(*i0, " + LpAddr + ", " + Str(N) + ", " + Str(C) + ", " + Str(Dn) + ")")
  EndIf
  PmoEmitLine(File, "  PmOpI(0) = " + Str(N))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(C))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Dn))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(*T\ElementType))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(Code))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(PmoEmitNsAttributePresent(*Node, "ignore_index")))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(PmoEmitAttrI(*Node, "ignore_index", 0)))
  PmoEmitLine(File, "  PmOpI(7) = " + Str(Bool(*W <> 0)))
  PmoEmitLine(File, "  If PmOpNllLoss(" + LpAddr + ", *i1, " + PmoOpsIn(*Node, 2) + ", *o0) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; ---- Col2Im, CenterCropPad, MaxUnpool, AffineGrid, MaxRoiPool, DeformConv ----
Procedure.i PmoEmitOpsCol2Im(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Img.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  Protected *Blk.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 2))
  Protected s.i, d.i, Ok.Integer, Bl.q = 1, L.q = 1, C.q, ImgE.q, BlkE.q, Dil.q, PadB.q, PadE.q, Strd.q, Col.q
  Protected Dim Im.q(3)
  If PmoEmitNsAttributesAllowed(*Node, "|dilations|pads|strides|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1) = 0 : ProcedureReturn #False : EndIf
  If *Img = 0 Or *Blk = 0 : ProcedureReturn PmoEmitNsFail(*Node, "inputs image_shape and block_shape must be constants for fixed-shape emission.") : EndIf
  s = *Img\Elements
  If s < 1 Or s > 3 Or *Blk\Elements <> s : ProcedureReturn PmoEmitNsFail(*Node, "image_shape and block_shape must hold one to three extents each, as many as each other.") : EndIf
  If PmoEmitRank(*X) <> 3 : ProcedureReturn PmoEmitNsFail(*Node, "the input must be [N, C * prod(block_shape), L].") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(s))
  For d = 0 To s - 1
    ImgE = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), d, @Ok)
    BlkE = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 2), d, @Ok)
    Dil = PmoEmitAttrListI(*Node, "dilations", d, 1)
    PadB = PmoEmitAttrListI(*Node, "pads", d, 0)
    PadE = PmoEmitAttrListI(*Node, "pads", d + s, 0)
    Strd = PmoEmitAttrListI(*Node, "strides", d, 1)
    If ImgE < 1 Or BlkE < 1 Or Dil < 1 Or Strd < 1 Or PadB < 0 Or PadE < 0
      ProcedureReturn PmoEmitNsFail(*Node, "extents, dilations and strides must be positive and pads not negative on axis " + Str(d) + ".")
    EndIf
    Col = (ImgE + PadB + PadE - (Dil * (BlkE - 1) + 1)) / Strd + 1
    If Col < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the block does not fit the padded image on axis " + Str(d) + ".") : EndIf
    Bl * BlkE : L * Col : Im(d) = ImgE
    PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = " + Str(ImgE))
    PmoEmitLine(File, "  PmOpI(" + Str(12 + d) + ") = " + Str(BlkE))
    PmoEmitLine(File, "  PmOpI(" + Str(16 + d) + ") = " + Str(Dil))
    PmoEmitLine(File, "  PmOpI(" + Str(20 + d) + ") = " + Str(PadB))
    PmoEmitLine(File, "  PmOpI(" + Str(24 + d) + ") = " + Str(Strd))
    PmoEmitLine(File, "  PmOpI(" + Str(28 + d) + ") = " + Str(Col))
  Next
  If PmoEmitDim(*X, 1) % Bl <> 0 Or PmoEmitDim(*X, 2) <> L
    ProcedureReturn PmoEmitNsFail(*Node, "the input must be [N, C * " + Str(Bl) + ", " + Str(L) + "] for these blocks.")
  EndIf
  C = PmoEmitDim(*X, 1) / Bl
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> s + 2 Or PmoEmitDim(*Y, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*Y, 1) <> C
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, " + Str(C) + ", image_shape ...].")
  EndIf
  For d = 0 To s - 1
    If PmoEmitDim(*Y, 2 + d) <> Im(d) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, " + Str(C) + ", image_shape ...].") : EndIf
  Next
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(C))
  PmoEmitLine(File, "  PmOpCol2Im(*i0, *o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsCenterCropPad(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Sh.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  Protected Rank.i, n.i, k.i, Axis.q, ShE.q, Ok.Integer, d.i
  Protected Dim Target.q(8)
  Protected Dim Seen.i(8)
  If PmoEmitNsAttributesAllowed(*Node, "|axes|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "input", 1 | 2 | 4 | 8 | 16 | 32) = 0 : ProcedureReturn #False : EndIf
  If *Sh = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input shape must be a constant for fixed-shape emission.") : EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 1 Or Rank > 8 : ProcedureReturn PmoEmitNsFail(*Node, "the input rank must be one to eight.") : EndIf
  For d = 0 To Rank - 1 : Target(d) = PmoEmitDim(*X, d) : Next
  n = PmoEmitAttrListCount(*Node, "axes")
  If n = 0 : n = Rank : EndIf
  If *Sh\Elements <> n : ProcedureReturn PmoEmitNsFail(*Node, "input shape holds " + Str(*Sh\Elements) + " extents for " + Str(n) + " axes.") : EndIf
  For k = 0 To n - 1
    Axis = k
    If PmoEmitAttrListCount(*Node, "axes") > 0 : Axis = PmoEmitAttrListI(*Node, "axes", k, 0) : EndIf
    If Axis < 0 : Axis + Rank : EndIf
    If Axis < 0 Or Axis >= Rank Or Seen(Axis) : ProcedureReturn PmoEmitNsFail(*Node, "attribute axes names an axis outside the rank or twice.") : EndIf
    Seen(Axis) = 1
    ShE = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), k, @Ok)
    If Ok\i = 0 Or ShE < 0 : ProcedureReturn PmoEmitNsFail(*Node, "input shape must hold extents that are not negative.") : EndIf
    Target(Axis) = ShE
  Next
  If *Y = 0 Or *Y\ElementType <> *X\ElementType Or PmoEmitRank(*Y) <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the input's element type and rank.") : EndIf
  For d = 0 To Rank - 1
    If PmoEmitDim(*Y, d) <> Target(d) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, d)) + "; shape gives " + Str(Target(d)) + ".") : EndIf
  Next
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Rank))
  For d = 0 To Rank - 1
    PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d)))
    PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(Target(d)))
    If Target(d) < PmoEmitDim(*X, d)
      PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = " + Str((PmoEmitDim(*X, d) - Target(d)) / 2))
      PmoEmitLine(File, "  PmOpI(" + Str(16 + d) + ") = 0")
    Else
      PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = 0")
      PmoEmitLine(File, "  PmOpI(" + Str(16 + d) + ") = " + Str((Target(d) - PmoEmitDim(*X, d)) / 2))
    EndIf
  Next
  PmoEmitLine(File, "  PmOpCenterCropPad(*i0, *o0, " + Str(PmoEmitElementBytes(*X\ElementType)) + ")")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsMaxUnpool(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *I.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected s.i, d.i, Rank.i, Inf.q, Ok.Integer, K.q, Strd.q, PadB.q, PadE.q, O.q
  If PmoEmitNsAttributesAllowed(*Node, "|kernel_shape|pads|strides|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *I, "I", 4) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X) : s = Rank - 2
  If s < 1 Or s > 3 : ProcedureReturn PmoEmitNsFail(*Node, "X must be N x C x D1 ... Ds with one to three spatial axes.") : EndIf
  If PmoEmitShapeEqual(*X, *I) = 0 : ProcedureReturn PmoEmitNsFail(*Node, "I must have X's shape.") : EndIf
  If PmoEmitAttrListCount(*Node, "kernel_shape") <> s : ProcedureReturn PmoEmitNsFail(*Node, "attribute kernel_shape must hold " + Str(s) + " extents.") : EndIf
  If PmoEmitInput(*Node, 2) <> "" And PmoEmitConstant(*Ir, PmoEmitInput(*Node, 2)) = 0
    ProcedureReturn PmoEmitNsFail(*Node, "input output_shape must be a constant for fixed-shape emission.")
  EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> Rank : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT with X's rank.") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Rank))
  PmoEmitLine(File, "  PmOpI(1) = 7")
  For d = 0 To Rank - 1
    If d < 2
      Inf = PmoEmitDim(*X, d)
    Else
      K = PmoEmitAttrListI(*Node, "kernel_shape", d - 2, 1)
      Strd = PmoEmitAttrListI(*Node, "strides", d - 2, 1)
      PadB = PmoEmitAttrListI(*Node, "pads", d - 2, 0)
      PadE = PmoEmitAttrListI(*Node, "pads", d - 2 + s, 0)
      Inf = (PmoEmitDim(*X, d) - 1) * Strd - (PadB + PadE) + K
    EndIf
    O = Inf
    If PmoEmitInput(*Node, 2) <> "" : O = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 2), d, @Ok) : EndIf
    If Inf < 1 Or O < Inf Or (d < 2 And O <> Inf)
      ProcedureReturn PmoEmitNsFail(*Node, "output_shape must hold X's first two extents and at least the inferred extent on each spatial axis (" + Str(Inf) + " on axis " + Str(d) + ").")
    EndIf
    If PmoEmitDim(*Y, d) <> O : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, d)) + "; it is " + Str(O) + ".") : EndIf
    PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(PmoEmitDim(*X, d)))
    PmoEmitLine(File, "  PmOpDB(" + Str(d) + ") = " + Str(Inf))
    PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(O))
  Next
  PmoEmitLine(File, "  If PmOpMaxUnpool(*i0, *i1, *o0) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsAffineGrid(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *T.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Sz.PmoIrConstant = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
  Protected r.i, d.i, Ok.Integer, E.q
  If PmoEmitNsAttributesAllowed(*Node, "|align_corners|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *T, "theta", 1) = 0 : ProcedureReturn #False : EndIf
  If *Sz = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input size must be a constant for fixed-shape emission.") : EndIf
  r = *Sz\Elements - 2
  If r < 2 Or r > 3 : ProcedureReturn PmoEmitNsFail(*Node, "size must be [N, C, H, W] or [N, C, D, H, W].") : EndIf
  If PmoEmitRank(*T) <> 3 Or PmoEmitDim(*T, 1) <> r Or PmoEmitDim(*T, 2) <> r + 1 Or PmoEmitDim(*T, 0) <> PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
    ProcedureReturn PmoEmitNsFail(*Node, "theta must be [N, " + Str(r) + ", " + Str(r + 1) + "] with size's N.")
  EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitRank(*Y) <> r + 2 Or PmoEmitDim(*Y, 0) <> PmoEmitDim(*T, 0) Or PmoEmitDim(*Y, r + 1) <> r
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, spatial ..., " + Str(r) + "].")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(PmoEmitDim(*T, 0)))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(r))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Bool(PmoEmitAttrI(*Node, "align_corners", 0) <> 0)))
  For d = 0 To r - 1
    E = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), d + 2, @Ok)
    If E < 1 Or PmoEmitDim(*Y, d + 1) <> E : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, spatial ..., " + Str(r) + "] with size's extents.") : EndIf
    PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = " + Str(E))
  Next
  PmoEmitLine(File, "  PmOpAffineGrid(*i0, *o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsMaxRoiPool(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *R.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Ph.q, Pw.q
  If PmoEmitNsAttributesAllowed(*Node, "|pooled_shape|spatial_scale|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *R, "rois", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitAttrListCount(*Node, "pooled_shape") <> 2 : ProcedureReturn PmoEmitNsFail(*Node, "attribute pooled_shape is required, two extents.") : EndIf
  Ph = PmoEmitAttrListI(*Node, "pooled_shape", 0, 0) : Pw = PmoEmitAttrListI(*Node, "pooled_shape", 1, 0)
  If Ph < 1 Or Pw < 1 : ProcedureReturn PmoEmitNsFail(*Node, "pooled_shape must be positive.") : EndIf
  If PmoEmitRank(*X) <> 4 Or PmoEmitRank(*R) <> 2 Or PmoEmitDim(*R, 1) <> 5 : ProcedureReturn PmoEmitNsFail(*Node, "X must be N x C x H x W and rois [num_rois, 5].") : EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or *Y\Elements <> PmoEmitDim(*R, 0) * PmoEmitDim(*X, 1) * Ph * Pw
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [num_rois, C, " + Str(Ph) + ", " + Str(Pw) + "].")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 1)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(PmoEmitDim(*X, 2)))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(PmoEmitDim(*X, 3)))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(PmoEmitDim(*R, 0)))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(Ph))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Pw))
  PmoEmitLine(File, "  PmOpSetBits(@PmOpF(0), " + PmoOpsBits(PmoEmitAttrF(*Node, "spatial_scale", 1.0)) + ")")
  PmoEmitLine(File, "  If PmOpMaxRoiPool(*i0, *i1, *o0) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsDeformConv(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *W.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Off.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *B.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 3))
  Protected *M.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected G.q, OG.q, C.q, Mo.q, Kh.q, Kw.q, Oh.q, Ow.q, d.i
  Protected Dim St.q(1)
  Protected Dim Dl.q(1)
  Protected Dim Pd.q(3)
  If PmoEmitNsAttributesAllowed(*Node, "|dilations|group|kernel_shape|offset_group|pads|strides|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "X", 1) = 0 Or PmoOpsTypeOk(*Node, *W, "W", 1) = 0 Or PmoOpsTypeOk(*Node, *Off, "offset", 1) = 0 : ProcedureReturn #False : EndIf
  If PmoEmitRank(*X) <> 4 : ProcedureReturn PmoEmitNsFail(*Node, "X has rank " + Str(PmoEmitRank(*X)) + "; two spatial axes (N x C x H x W) are implemented, as in the reference.") : EndIf
  G = PmoEmitAttrI(*Node, "group", 1) : OG = PmoEmitAttrI(*Node, "offset_group", 1)
  C = PmoEmitDim(*X, 1) : Mo = PmoEmitDim(*W, 0) : Kh = PmoEmitDim(*W, 2) : Kw = PmoEmitDim(*W, 3)
  If PmoEmitRank(*W) <> 4 Or G < 1 Or OG < 1 Or C <> G * PmoEmitDim(*W, 1) Or Mo % G <> 0 Or C % OG <> 0
    ProcedureReturn PmoEmitNsFail(*Node, "group and offset_group do not divide the channels as W's shape requires.")
  EndIf
  If PmoEmitAttrListCount(*Node, "kernel_shape") > 0 And (PmoEmitAttrListI(*Node, "kernel_shape", 0, 0) <> Kh Or PmoEmitAttrListI(*Node, "kernel_shape", 1, 0) <> Kw)
    ProcedureReturn PmoEmitNsFail(*Node, "attribute kernel_shape does not match W.")
  EndIf
  For d = 0 To 1
    St(d) = PmoEmitAttrListI(*Node, "strides", d, 1) : Dl(d) = PmoEmitAttrListI(*Node, "dilations", d, 1)
    Pd(d) = PmoEmitAttrListI(*Node, "pads", d, 0) : Pd(d + 2) = PmoEmitAttrListI(*Node, "pads", d + 2, 0)
    If St(d) < 1 Or Dl(d) < 1 Or Pd(d) < 0 Or Pd(d + 2) < 0 : ProcedureReturn PmoEmitNsFail(*Node, "strides and dilations must be positive and pads not negative.") : EndIf
  Next
  Oh = (PmoEmitDim(*X, 2) + Pd(0) + Pd(2) - (Dl(0) * (Kh - 1) + 1)) / St(0) + 1
  Ow = (PmoEmitDim(*X, 3) + Pd(1) + Pd(3) - (Dl(1) * (Kw - 1) + 1)) / St(1) + 1
  If Oh < 1 Or Ow < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the kernel is larger than the padded input.") : EndIf
  If PmoEmitRank(*Off) <> 4 Or PmoEmitDim(*Off, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*Off, 1) <> OG * Kh * Kw * 2 Or PmoEmitDim(*Off, 2) <> Oh Or PmoEmitDim(*Off, 3) <> Ow
    ProcedureReturn PmoEmitNsFail(*Node, "offset must be [N, " + Str(OG * Kh * Kw * 2) + ", " + Str(Oh) + ", " + Str(Ow) + "].")
  EndIf
  If *B And (*B\ElementType <> 1 Or *B\Elements <> Mo) : ProcedureReturn PmoEmitNsFail(*Node, "B must be FLOAT [" + Str(Mo) + "].") : EndIf
  If *M And (*M\ElementType <> 1 Or *M\Elements <> PmoEmitDim(*X, 0) * OG * Kh * Kw * Oh * Ow) : ProcedureReturn PmoEmitNsFail(*Node, "mask must be FLOAT [N, " + Str(OG * Kh * Kw) + ", " + Str(Oh) + ", " + Str(Ow) + "].") : EndIf
  If *Y = 0 Or *Y\ElementType <> 1 Or *Y\Elements <> PmoEmitDim(*X, 0) * Mo * Oh * Ow : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT [N, " + Str(Mo) + ", " + Str(Oh) + ", " + Str(Ow) + "].") : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(C))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(PmoEmitDim(*X, 2)))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(PmoEmitDim(*X, 3)))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(Mo))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(G))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(OG))
  PmoEmitLine(File, "  PmOpI(7) = " + Str(Kh))
  PmoEmitLine(File, "  PmOpI(8) = " + Str(Kw))
  PmoEmitLine(File, "  PmOpI(9) = " + Str(Oh))
  PmoEmitLine(File, "  PmOpI(10) = " + Str(Ow))
  PmoEmitLine(File, "  PmOpI(11) = " + Str(St(0)))
  PmoEmitLine(File, "  PmOpI(12) = " + Str(St(1)))
  PmoEmitLine(File, "  PmOpI(13) = " + Str(Dl(0)))
  PmoEmitLine(File, "  PmOpI(14) = " + Str(Dl(1)))
  PmoEmitLine(File, "  PmOpI(15) = " + Str(Pd(0)))
  PmoEmitLine(File, "  PmOpI(16) = " + Str(Pd(1)))
  PmoEmitLine(File, "  PmOpDeformConv(*i0, *i1, *i2, " + PmoOpsIn(*Node, 3) + ", " + PmoOpsIn(*Node, 4) + ", *o0)")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; ---- the quantized operators ------------------------------------------------
; One value (a scalar or a one-element tensor) or one per element of Extent.
Procedure.i PmoOpsQCount(*V.PmoIrValue, Extent.q)
  If *V = 0 : ProcedureReturn 0 : EndIf
  If *V\Elements = 1 And PmoEmitRank(*V) <= 1 : ProcedureReturn 1 : EndIf
  If PmoEmitRank(*V) = 1 And *V\Elements = Extent : ProcedureReturn 2 : EndIf
  ProcedureReturn 0
EndProcedure

Procedure.i PmoEmitOpsQuantizeLinear(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *S.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Z.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected Quant.i = Bool(*Node\Operation = "QuantizeLinear"), Axis.q, Rank.i, Outer.q = 1, Inner.q = 1, Per.i, d.i, Kind.i, Allowed.s = "|"
  If Opset >= 13 : Allowed = "|axis|" : EndIf
  If Opset >= 19 And Quant : Allowed = "|axis|saturate|" : EndIf
  If PmoEmitNsAttributesAllowed(*Node, Allowed, Opset) = 0 : ProcedureReturn #False : EndIf
  If Quant
    If PmoOpsTypeOk(*Node, *X, "x", 1) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoOpsTypeOk(*Node, *X, "x", 2 | 16 | 32) = 0
    ProcedureReturn #False
  EndIf
  If PmoOpsTypeOk(*Node, *S, "the scale", 1) = 0 : ProcedureReturn #False : EndIf
  Rank = PmoEmitRank(*X)
  Axis = PmoEmitAttrI(*Node, "axis", 1)
  If Axis < 0 : Axis + Rank : EndIf
  Per = 0
  If *S\Elements > 1 Or PmoEmitRank(*S) > 1
    If Opset < 13 : ProcedureReturn PmoEmitNsFail(*Node, "the scale holds " + Str(*S\Elements) + " values; before opset 13 it is one value.") : EndIf
    If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", 1)) + " is outside the input rank " + Str(Rank) + ".") : EndIf
    If PmoOpsQCount(*S, PmoEmitDim(*X, Axis)) <> 2
      ProcedureReturn PmoEmitNsFail(*Node, "the scale must be one value or a 1-D tensor with one value per element of axis " + Str(Axis) + "; blocked quantization belongs to opset 21.")
    EndIf
    Per = 1
  EndIf
  Kind = 2
  If *Z
    If *Node\Operation = "QuantizeLinear"
      If PmoOpsTypeOk(*Node, *Z, "the zero point", 16 | 32) = 0 : ProcedureReturn #False : EndIf
    ElseIf *Z\ElementType <> *X\ElementType
      ProcedureReturn PmoEmitNsFail(*Node, "the zero point must have the element type of x.")
    EndIf
    If *Z\Elements <> *S\Elements : ProcedureReturn PmoEmitNsFail(*Node, "the zero point must hold as many values as the scale.") : EndIf
    Kind = *Z\ElementType
  EndIf
  If Quant
    If *Y = 0 Or *Y\ElementType <> Kind Or PmoEmitShapeEqual(*X, *Y) = 0
      ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have x's shape and the zero point's element type (UINT8 without one).")
    EndIf
  ElseIf *Y = 0 Or *Y\ElementType <> 1 Or PmoEmitShapeEqual(*X, *Y) = 0
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be FLOAT with x's shape.")
  EndIf
  If Per
    For d = 0 To Axis - 1 : Outer * PmoEmitDim(*X, d) : Next
    For d = Axis + 1 To Rank - 1 : Inner * PmoEmitDim(*X, d) : Next
  Else
    Outer = *X\Elements : Inner = 1
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(Outer))
  If Per : PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, Axis))) : Else : PmoEmitLine(File, "  PmOpI(1) = 1") : EndIf
  PmoEmitLine(File, "  PmOpI(2) = " + Str(Inner))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(Per))
  If Quant
    If *Z : PmoEmitLine(File, "  PmOpI(4) = " + Str(Kind)) : Else : PmoEmitLine(File, "  PmOpI(4) = 0") : EndIf
    PmoEmitLine(File, "  PmOpI(5) = " + Str(Kind))
    PmoEmitLine(File, "  PmOpQuantize(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", *o0)")
  Else
    PmoEmitLine(File, "  PmOpI(4) = " + Str(*X\ElementType))
    PmoEmitLine(File, "  PmOpI(5) = " + Str(Bool(*Z <> 0)))
    PmoEmitLine(File, "  PmOpDequantize(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", *o0)")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitOpsDynQuantize(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *S.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected *Z.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 2))
  If PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "x", 1) = 0 : ProcedureReturn #False : EndIf
  If *Y = 0 Or *S = 0 Or *Z = 0 : ProcedureReturn PmoEmitNsFail(*Node, "outputs y, y_scale and y_zero_point are all required.") : EndIf
  If *Y\ElementType <> 2 Or PmoEmitShapeEqual(*X, *Y) = 0 Or *S\ElementType <> 1 Or *S\Elements <> 1 Or *Z\ElementType <> 2 Or *Z\Elements <> 1
    ProcedureReturn PmoEmitNsFail(*Node, "the declared outputs must be UINT8 y with x's shape, a FLOAT y_scale and a UINT8 y_zero_point.")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  If PmOpDynQuantize(*i0, *o0, *o1, *o2, " + Str(*X\Elements) + ") <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; MatMulInteger (Mode 0) and QLinearMatMul (1): the kernel's parameters from
; the operand shapes, or the sentence that refuses them.
Procedure.i PmoEmitOpsQMatMul(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected Mode.i = Bool(*Node\Operation = "QLinearMatMul")
  Protected IA.i = 0, IB.i = 1, IAZ.i = 2, IBZ.i = 3, *A.PmoIrValue, *B.PmoIrValue, *AZ.PmoIrValue, *BZ.PmoIrValue
  Protected *AS.PmoIrValue, *BS.PmoIrValue, *YS.PmoIrValue, *YZ.PmoIrValue, *Y.PmoIrValue
  Protected RA.i, RB.i, M.q, K.q, N.q, KB.q, R.i, d.i, EA.q, EB.q, EY.q, Kind.i, BZMode.i, YRank.i, Text.s
  If Mode : IA = 0 : IB = 3 : IAZ = 2 : IBZ = 5 : EndIf
  *A = PmoEmitValue(*Ir, PmoEmitInput(*Node, IA)) : *B = PmoEmitValue(*Ir, PmoEmitInput(*Node, IB))
  *AZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, IAZ)) : *BZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, IBZ))
  *Y = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  If PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *A, "A", 16 | 32) = 0 Or PmoOpsTypeOk(*Node, *B, "B", 16 | 32) = 0 : ProcedureReturn #False : EndIf
  RA = PmoEmitRank(*A) : RB = PmoEmitRank(*B)
  If RA < 1 Or RB < 1 : ProcedureReturn PmoEmitNsFail(*Node, "A and B must have rank one or more.") : EndIf
  If RA = 1 : M = 1 : K = PmoEmitDim(*A, 0) : Else : M = PmoEmitDim(*A, RA - 2) : K = PmoEmitDim(*A, RA - 1) : EndIf
  If RB = 1 : KB = PmoEmitDim(*B, 0) : N = 1 : Else : KB = PmoEmitDim(*B, RB - 2) : N = PmoEmitDim(*B, RB - 1) : EndIf
  If K <> KB : ProcedureReturn PmoEmitNsFail(*Node, "A has " + Str(K) + " columns and B " + Str(KB) + " rows.") : EndIf
  R = 0
  If RA > 2 : R = RA - 2 : EndIf
  If RB > 2 And RB - 2 > R : R = RB - 2 : EndIf
  If R > 8 : ProcedureReturn PmoEmitNsFail(*Node, "more than eight batch axes.") : EndIf
  If *AZ
    If *AZ\ElementType <> *A\ElementType Or PmoOpsQCount(*AZ, 1) <> 1
      ProcedureReturn PmoEmitNsFail(*Node, "the zero point of A must be one value of A's element type; a per-row zero point is not implemented.")
    EndIf
  EndIf
  BZMode = 0
  If *BZ
    BZMode = PmoOpsQCount(*BZ, N)
    If N = 1 And BZMode = 2 : BZMode = 1 : EndIf
    If *BZ\ElementType <> *B\ElementType Or BZMode = 0
      ProcedureReturn PmoEmitNsFail(*Node, "the zero point of B must be of B's element type, one value or one per column of B.")
    EndIf
  EndIf
  Kind = 6
  If Mode
    *AS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1)) : *BS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
    *YS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 6)) : *YZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, 7))
    If *AZ = 0 Or *BZ = 0 Or *YZ = 0 : ProcedureReturn PmoEmitNsFail(*Node, "the zero points of a, b and y are required.") : EndIf
    If PmoOpsTypeOk(*Node, *AS, "a_scale", 1) = 0 Or PmoOpsTypeOk(*Node, *BS, "b_scale", 1) = 0 Or PmoOpsTypeOk(*Node, *YS, "y_scale", 1) = 0 : ProcedureReturn #False : EndIf
    If PmoOpsQCount(*AS, 1) <> 1 Or PmoOpsQCount(*YS, 1) <> 1 Or PmoOpsQCount(*BS, N) = 0 Or PmoOpsQCount(*YZ, 1) <> 1
      ProcedureReturn PmoEmitNsFail(*Node, "a_scale, y_scale and y_zero_point must be one value each, b_scale one value or one per column of b.")
    EndIf
    If PmoOpsTypeOk(*Node, *YZ, "y_zero_point", 16 | 32) = 0 : ProcedureReturn #False : EndIf
    Kind = *YZ\ElementType
  EndIf
  ; Y: the broadcast batch axes, then M unless A is 1-D, then N unless B is 1-D
  YRank = R
  If RA > 1 : YRank + 1 : EndIf
  If RB > 1 : YRank + 1 : EndIf
  Text = "the declared output must be " + PmoEmitNsTypeName(Kind) + " with numpy matmul's shape."
  If *Y = 0 Or *Y\ElementType <> Kind Or PmoEmitRank(*Y) <> YRank : ProcedureReturn PmoEmitNsFail(*Node, Text) : EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(R))
  For d = 0 To R - 1
    EA = 1 : EB = 1
    If RA - 2 - R + d >= 0 : EA = PmoEmitDim(*A, RA - 2 - R + d) : EndIf
    If RB - 2 - R + d >= 0 : EB = PmoEmitDim(*B, RB - 2 - R + d) : EndIf
    If EA <> EB And EA <> 1 And EB <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "batch axes of A and B do not broadcast.") : EndIf
    EY = EA : If EA = 1 : EY = EB : EndIf
    If PmoEmitDim(*Y, d) <> EY : ProcedureReturn PmoEmitNsFail(*Node, Text) : EndIf
    PmoEmitLine(File, "  PmOpDY(" + Str(d) + ") = " + Str(EY))
    PmoEmitLine(File, "  PmOpDA(" + Str(d) + ") = " + Str(EA))
    PmoEmitLine(File, "  PmOpDB(" + Str(d) + ") = " + Str(EB))
  Next
  If RA > 1 And PmoEmitDim(*Y, R) <> M : ProcedureReturn PmoEmitNsFail(*Node, Text) : EndIf
  If RB > 1 And PmoEmitDim(*Y, YRank - 1) <> N : ProcedureReturn PmoEmitNsFail(*Node, Text) : EndIf
  PmoEmitLine(File, "  PmOpI(1) = " + Str(M))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(K))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(N))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(*A\ElementType))
  PmoEmitLine(File, "  PmOpI(5) = " + Str(*B\ElementType))
  PmoEmitLine(File, "  PmOpI(6) = " + Str(Bool(*AZ <> 0)))
  PmoEmitLine(File, "  PmOpI(7) = " + Str(BZMode))
  PmoEmitLine(File, "  PmOpI(8) = " + Str(Mode))
  If Mode
    PmoEmitLine(File, "  PmOpI(9) = " + Str(Bool(PmoOpsQCount(*BS, N) = 2 And N > 1)))
    PmoEmitLine(File, "  PmOpI(10) = " + Str(Kind))
    PmoEmitLine(File, "  PmOpQMatMul(*i0, *i3, " + PmoOpsIn(*Node, 2) + ", " + PmoOpsIn(*Node, 5) + ", *o0, *i1, *i4, *i6, *i7)")
  Else
    PmoEmitLine(File, "  PmOpI(9) = 0")
    PmoEmitLine(File, "  PmOpI(10) = 0")
    PmoEmitLine(File, "  PmOpQMatMul(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", " + PmoOpsIn(*Node, 3) + ", *o0, 0, 0, 0, 0)")
  EndIf
  PmoOpsTail(File, #False)
  ProcedureReturn #True
EndProcedure

; ConvInteger (Mode 0) and QLinearConv (1).
Procedure.i PmoEmitOpsQConv(File.i, *Ir.PmoIrModel, *Node.PmoOnnxNode, ProcName.s, Opset.i)
  Protected Mode.i = Bool(*Node\Operation = "QLinearConv")
  Protected IX.i = 0, IW.i = 1, IXZ.i = 2, IWZ.i = 3, *X.PmoIrValue, *W.PmoIrValue, *XZ.PmoIrValue, *WZ.PmoIrValue, *Bias.PmoIrValue
  Protected *XS.PmoIrValue, *WS.PmoIrValue, *YS.PmoIrValue, *YZ.PmoIrValue, *Y.PmoIrValue
  Protected s.i, d.i, Group.q, C.q, M.q, Kind.i, WZMode.i, AutoPad.s, Kernel.q, Stride.q, Dilation.q, Begin.Quad, EndPad.Quad, Out.q
  If Mode : IX = 0 : IW = 3 : IXZ = 2 : IWZ = 5 : EndIf
  *X = PmoEmitValue(*Ir, PmoEmitInput(*Node, IX)) : *W = PmoEmitValue(*Ir, PmoEmitInput(*Node, IW))
  *XZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, IXZ)) : *WZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, IWZ))
  *Y = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  If PmoEmitNsAttributesAllowed(*Node, "|auto_pad|dilations|group|kernel_shape|pads|strides|", Opset) = 0 : ProcedureReturn #False : EndIf
  If PmoOpsTypeOk(*Node, *X, "x", 16 | 32) = 0 Or PmoOpsTypeOk(*Node, *W, "w", 16 | 32) = 0 : ProcedureReturn #False : EndIf
  s = PmoEmitRank(*X) - 2
  If s < 1 Or s > 3 : ProcedureReturn PmoEmitNsFail(*Node, "x has rank " + Str(PmoEmitRank(*X)) + "; N x C x D1 ... Ds with one to three spatial axes is implemented.") : EndIf
  If PmoEmitRank(*W) <> s + 2 : ProcedureReturn PmoEmitNsFail(*Node, "w must have x's rank.") : EndIf
  Group = PmoEmitAttrI(*Node, "group", 1)
  C = PmoEmitDim(*X, 1) : M = PmoEmitDim(*W, 0)
  If Group < 1 Or C <> Group * PmoEmitDim(*W, 1) Or M % Group <> 0
    ProcedureReturn PmoEmitNsFail(*Node, "group = " + Str(Group) + " does not divide x's " + Str(C) + " channels and w's " + Str(M) + " filters as w's shape requires.")
  EndIf
  If PmoEmitAttrListCount(*Node, "kernel_shape") > 0
    For d = 0 To s - 1
      If PmoEmitAttrListI(*Node, "kernel_shape", d, 0) <> PmoEmitDim(*W, 2 + d) : ProcedureReturn PmoEmitNsFail(*Node, "attribute kernel_shape does not match w's spatial extents.") : EndIf
    Next
  EndIf
  If *XZ
    If *XZ\ElementType <> *X\ElementType Or PmoOpsQCount(*XZ, 1) <> 1 : ProcedureReturn PmoEmitNsFail(*Node, "the zero point of x must be one value of x's element type.") : EndIf
  EndIf
  WZMode = 0
  If *WZ
    WZMode = PmoOpsQCount(*WZ, M)
    If M = 1 And WZMode = 2 : WZMode = 1 : EndIf
    If *WZ\ElementType <> *W\ElementType Or WZMode = 0 : ProcedureReturn PmoEmitNsFail(*Node, "the zero point of w must be of w's element type, one value or one per output channel.") : EndIf
  EndIf
  Kind = 6
  If Mode
    *XS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1)) : *WS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 4))
    *YS = PmoEmitValue(*Ir, PmoEmitInput(*Node, 6)) : *YZ = PmoEmitValue(*Ir, PmoEmitInput(*Node, 7))
    *Bias = PmoEmitValue(*Ir, PmoEmitInput(*Node, 8))
    If *XZ = 0 Or *WZ = 0 Or *YZ = 0 : ProcedureReturn PmoEmitNsFail(*Node, "the zero points of x, w and y are required.") : EndIf
    If PmoOpsTypeOk(*Node, *XS, "x_scale", 1) = 0 Or PmoOpsTypeOk(*Node, *WS, "w_scale", 1) = 0 Or PmoOpsTypeOk(*Node, *YS, "y_scale", 1) = 0 : ProcedureReturn #False : EndIf
    If PmoOpsQCount(*XS, 1) <> 1 Or PmoOpsQCount(*YS, 1) <> 1 Or PmoOpsQCount(*WS, M) = 0 Or PmoOpsQCount(*YZ, 1) <> 1
      ProcedureReturn PmoEmitNsFail(*Node, "x_scale, y_scale and y_zero_point must be one value each, w_scale one value or one per output channel.")
    EndIf
    If PmoOpsTypeOk(*Node, *YZ, "y_zero_point", 16 | 32) = 0 : ProcedureReturn #False : EndIf
    If *Bias And (*Bias\ElementType <> 6 Or PmoEmitRank(*Bias) <> 1 Or *Bias\Elements <> M) : ProcedureReturn PmoEmitNsFail(*Node, "B must be INT32 with one value per output channel.") : EndIf
    Kind = *YZ\ElementType
  EndIf
  AutoPad = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
  If AutoPad <> "NOTSET" And AutoPad <> "SAME_UPPER" And AutoPad <> "SAME_LOWER" And AutoPad <> "VALID"
    ProcedureReturn PmoEmitNsFail(*Node, "attribute auto_pad = " + AutoPad + " is not a padding mode; NOTSET, SAME_UPPER, SAME_LOWER and VALID are.")
  EndIf
  If AutoPad <> "NOTSET" And PmoEmitAttrListCount(*Node, "pads") > 0
    ProcedureReturn PmoEmitNsFail(*Node, "attribute pads is given with auto_pad = " + AutoPad + "; the specification allows explicit pads only with NOTSET.")
  EndIf
  If *Y = 0 Or *Y\ElementType <> Kind Or PmoEmitRank(*Y) <> s + 2 Or PmoEmitDim(*Y, 0) <> PmoEmitDim(*X, 0) Or PmoEmitDim(*Y, 1) <> M
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output must be " + PmoEmitNsTypeName(Kind) + " [N, " + Str(M) + ", spatial ...].")
  EndIf
  PmoOpsHead(File, ProcName, *Node)
  PmoEmitLine(File, "  PmOpI(0) = " + Str(s))
  PmoEmitLine(File, "  PmOpI(1) = " + Str(PmoEmitDim(*X, 0)))
  PmoEmitLine(File, "  PmOpI(2) = " + Str(C))
  PmoEmitLine(File, "  PmOpI(3) = " + Str(M))
  PmoEmitLine(File, "  PmOpI(4) = " + Str(Group))
  For d = 0 To s - 1
    Kernel = PmoEmitDim(*W, 2 + d)
    Stride = PmoEmitAttrListI(*Node, "strides", d, 1)
    Dilation = PmoEmitAttrListI(*Node, "dilations", d, 1)
    Begin\q = PmoEmitAttrListI(*Node, "pads", d, 0)
    EndPad\q = PmoEmitAttrListI(*Node, "pads", d + s, 0)
    If Kernel < 1 Or Stride < 1 Or Dilation < 1 Or Begin\q < 0 Or EndPad\q < 0
      ProcedureReturn PmoEmitNsFail(*Node, "the kernel, strides and dilations must be positive and pads not negative on spatial axis " + Str(d) + ".")
    EndIf
    Out = PmoOpsPoolExtent(PmoEmitDim(*X, 2 + d), Kernel, Stride, Dilation, AutoPad, 0, @Begin, @EndPad)
    If Out < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the kernel on spatial axis " + Str(d) + " is larger than the padded input.") : EndIf
    If Out <> PmoEmitDim(*Y, 2 + d)
      ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on spatial axis " + Str(d) + " is " + Str(PmoEmitDim(*Y, 2 + d)) + "; the convolution shape rule gives " + Str(Out) + ".")
    EndIf
    PmoEmitLine(File, "  PmOpI(" + Str(8 + d) + ") = " + Str(PmoEmitDim(*X, 2 + d)))
    PmoEmitLine(File, "  PmOpI(" + Str(12 + d) + ") = " + Str(Out))
    PmoEmitLine(File, "  PmOpI(" + Str(16 + d) + ") = " + Str(Kernel))
    PmoEmitLine(File, "  PmOpI(" + Str(20 + d) + ") = " + Str(Stride))
    PmoEmitLine(File, "  PmOpI(" + Str(24 + d) + ") = " + Str(Dilation))
    PmoEmitLine(File, "  PmOpI(" + Str(28 + d) + ") = " + Str(Begin\q))
  Next
  PmoEmitLine(File, "  PmOpI(40) = " + Str(*X\ElementType))
  PmoEmitLine(File, "  PmOpI(41) = " + Str(*W\ElementType))
  PmoEmitLine(File, "  PmOpI(42) = " + Str(Bool(*XZ <> 0)))
  PmoEmitLine(File, "  PmOpI(43) = " + Str(WZMode))
  PmoEmitLine(File, "  PmOpI(44) = " + Str(Bool(*Bias <> 0)))
  PmoEmitLine(File, "  PmOpI(45) = " + Str(Mode))
  If Mode
    PmoEmitLine(File, "  PmOpI(46) = " + Str(Bool(PmoOpsQCount(*WS, M) = 2 And M > 1)))
    PmoEmitLine(File, "  PmOpI(47) = " + Str(Kind))
    PmoEmitLine(File, "  PmOpQConv(*i0, *i3, *i2, *i5, " + PmoOpsIn(*Node, 8) + ", *o0, *i1, *i4, *i6, *i7)")
  Else
    PmoEmitLine(File, "  PmOpI(46) = 0")
    PmoEmitLine(File, "  PmOpI(47) = 0")
    PmoEmitLine(File, "  PmOpQConv(*i0, *i1, " + PmoOpsIn(*Node, 2) + ", " + PmoOpsIn(*Node, 3) + ", 0, *o0, 0, 0, 0, 0)")
  EndIf
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
    Case "RMSNormalization" : Done = PmoEmitOpsRmsNorm(File, *Ir, *Node, ProcName, Opset)
    Case "CumProd" : Done = PmoEmitOpsCumProd(File, *Ir, *Node, ProcName, Opset)
    Case "BitCast" : Done = PmoEmitOpsBitCast(File, *Ir, *Node, ProcName, Opset)
    Case "Attention" : Done = PmoEmitOpsAttention(File, *Ir, *Node, ProcName, Opset)
    Case "CausalConvWithState" : Done = PmoEmitOpsCausalConv(File, *Ir, *Node, ProcName, Opset)
    Case "LinearAttention" : Done = PmoEmitOpsLinearAttention(File, *Ir, *Node, ProcName, Opset)
    Case "RotaryEmbedding" : Done = PmoEmitOpsRotary(File, *Ir, *Node, ProcName, Opset)
    Case "TensorScatter" : Done = PmoEmitOpsTensorScatter(File, *Ir, *Node, ProcName, Opset)
    Case "Erf", "Reciprocal", "Ceil", "Sign", "Softplus", "Softsign", "Elu", "Selu", "Celu", "HardSigmoid", "HardSwish", "Mish", "Gelu", "Swish",
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
    Case "QuantizeLinear", "DequantizeLinear" : Done = PmoEmitOpsQuantizeLinear(File, *Ir, *Node, ProcName, Opset)
    Case "DynamicQuantizeLinear" : Done = PmoEmitOpsDynQuantize(File, *Ir, *Node, ProcName, Opset)
    Case "MatMulInteger", "QLinearMatMul" : Done = PmoEmitOpsQMatMul(File, *Ir, *Node, ProcName, Opset)
    Case "ConvInteger", "QLinearConv" : Done = PmoEmitOpsQConv(File, *Ir, *Node, ProcName, Opset)
    Case "HannWindow", "HammingWindow", "BlackmanWindow" : Done = PmoEmitOpsWindow(File, *Ir, *Node, ProcName, Opset)
    Case "DFT" : Done = PmoEmitOpsDft(File, *Ir, *Node, ProcName, Opset)
    Case "Col2Im" : Done = PmoEmitOpsCol2Im(File, *Ir, *Node, ProcName, Opset)
    Case "CenterCropPad" : Done = PmoEmitOpsCenterCropPad(File, *Ir, *Node, ProcName, Opset)
    Case "MaxUnpool" : Done = PmoEmitOpsMaxUnpool(File, *Ir, *Node, ProcName, Opset)
    Case "AffineGrid" : Done = PmoEmitOpsAffineGrid(File, *Ir, *Node, ProcName, Opset)
    Case "MaxRoiPool" : Done = PmoEmitOpsMaxRoiPool(File, *Ir, *Node, ProcName, Opset)
    Case "DeformConv" : Done = PmoEmitOpsDeformConv(File, *Ir, *Node, ProcName, Opset)
    Case "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss" : Done = PmoEmitOpsLoss(File, *Ir, *Node, ProcName, Opset)
    Case "MelWeightMatrix" : ProcedureReturn PmoEmitNsFail(*Node, PmoOpsMelSentence())
    Case "NonMaxSuppression" : ProcedureReturn PmoEmitNsFail(*Node, "its output size depends on the scores, which the fixed-shape path cannot plan.")
    Case "Unique" : ProcedureReturn PmoEmitNsFail(*Node, "its output size depends on the input values, which the fixed-shape path cannot plan.")
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
