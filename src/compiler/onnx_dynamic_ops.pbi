; ============================================================================
; onnx_dynamic_ops.pbi - runtime-dimension forms of the operators that
; complete the ONNX operator set (onnx_emit_ops.pbi lists them)
; ----------------------------------------------------------------------------
; Validation and call text on the runtime-dimension path. Every form the
; ONNX specification (https://onnx.ai/onnx/operators/) allows for them up to
; opset 20 is either lowered to a call into runtime/tensor_dynamic_ops.pmi
; or refused with one sentence naming the operator, the attribute or input,
; and the value. Shapes are checked when the call runs; the attributes and
; the declared element types are checked here. The opset floors are
; PmoOpsFloor's (the path accepts opsets 11 to 20).
; ============================================================================

Procedure.s PmdOpsAllowed(*Node.PmoOnnxNode, Opset.i)
  Protected Op.s = *Node\Operation
  Select Op
    Case "Mod" : ProcedureReturn "|fmod|"
    Case "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"
      If Opset >= 18 : ProcedureReturn "|keepdims|noop_with_empty_axes|" : EndIf
      ProcedureReturn "|axes|keepdims|"
    Case "ArgMax", "ArgMin"
      If Opset >= 12 : ProcedureReturn "|axis|keepdims|select_last_index|" : EndIf
      ProcedureReturn "|axis|keepdims|"
    Case "LogSoftmax", "GatherElements", "OneHot" : ProcedureReturn "|axis|"
    Case "MaxPool"
      ProcedureReturn "|auto_pad|kernel_shape|pads|strides|storage_order|ceil_mode|dilations|"
    Case "AveragePool"
      If Opset >= 19 : ProcedureReturn "|auto_pad|kernel_shape|pads|strides|count_include_pad|ceil_mode|dilations|" : EndIf
      ProcedureReturn "|auto_pad|kernel_shape|pads|strides|count_include_pad|ceil_mode|"
    Case "LpPool"
      If Opset >= 18 : ProcedureReturn "|auto_pad|kernel_shape|pads|strides|p|ceil_mode|dilations|" : EndIf
      ProcedureReturn "|auto_pad|kernel_shape|pads|strides|p|"
    Case "GlobalLpPool" : ProcedureReturn "|p|"
    Case "Split"
      If Opset >= 18 : ProcedureReturn "|axis|num_outputs|" : EndIf
      If Opset >= 13 : ProcedureReturn "|axis|" : EndIf
      ProcedureReturn "|axis|split|"
    Case "DepthToSpace" : ProcedureReturn "|blocksize|mode|"
    Case "SpaceToDepth" : ProcedureReturn "|blocksize|"
    Case "Trilu" : ProcedureReturn "|upper|"
    Case "GatherND"
      If Opset >= 12 : ProcedureReturn "|batch_dims|" : EndIf
      ProcedureReturn "|"
    Case "Einsum" : ProcedureReturn "|equation|"
    Case "Dropout"
      If Opset >= 12 : ProcedureReturn "|seed|" : EndIf
      ProcedureReturn "|ratio|"
    Case "CastLike"
      If Opset >= 19 : ProcedureReturn "|saturate|" : EndIf
      ProcedureReturn "|"
    Case "Hardmax", "Compress" : ProcedureReturn "|axis|"
    Case "LpNormalization" : ProcedureReturn "|axis|p|"
    Case "MeanVarianceNormalization" : ProcedureReturn "|axes|"
    Case "LRN" : ProcedureReturn "|alpha|beta|bias|size|"
    Case "GroupNormalization" : ProcedureReturn "|epsilon|num_groups|"
    Case "EyeLike" : ProcedureReturn "|dtype|k|"
    Case "ReverseSequence" : ProcedureReturn "|batch_axis|time_axis|"
    Case "RNN" : ProcedureReturn "|activation_alpha|activation_beta|activations|clip|direction|hidden_size|layout|"
    Case "GRU" : ProcedureReturn "|activation_alpha|activation_beta|activations|clip|direction|hidden_size|layout|linear_before_reset|"
    Case "NonMaxSuppression" : ProcedureReturn "|center_point_box|"
    Case "Upsample" : ProcedureReturn "|mode|scales|"  ; so its own refusal, not an attribute's, is the sentence
    Case "GridSample" : ProcedureReturn "|align_corners|mode|padding_mode|"
    Case "QuantizeLinear"
      If Opset >= 19 : ProcedureReturn "|axis|saturate|" : EndIf
      If Opset >= 13 : ProcedureReturn "|axis|" : EndIf
      ProcedureReturn "|"
    Case "DequantizeLinear"
      If Opset >= 13 : ProcedureReturn "|axis|" : EndIf
      ProcedureReturn "|"
    Case "DynamicQuantizeLinear", "MatMulInteger", "QLinearMatMul" : ProcedureReturn "|"
    Case "ConvInteger", "QLinearConv" : ProcedureReturn "|auto_pad|dilations|group|kernel_shape|pads|strides|"
    Case "HannWindow", "HammingWindow", "BlackmanWindow" : ProcedureReturn "|output_datatype|periodic|"
    Case "DFT"
      If Opset >= 20 : ProcedureReturn "|inverse|onesided|" : EndIf
      ProcedureReturn "|axis|inverse|onesided|"
    Case "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss" : ProcedureReturn "|ignore_index|reduction|"
    Case "MelWeightMatrix" : ProcedureReturn "|output_datatype|"
    Case "RoiAlign"
      If Opset >= 16 : ProcedureReturn "|coordinate_transformation_mode|mode|output_height|output_width|sampling_ratio|spatial_scale|" : EndIf
      ProcedureReturn "|mode|output_height|output_width|sampling_ratio|spatial_scale|"
  EndSelect
  ProcedureReturn PmoOpsUnaryAllowed(Op)
EndProcedure

; The declared rank of a top-level graph input or value, -1 when the model
; does not declare its shape.
Procedure.i PmdOpsDeclaredRank(Name.s)
  If *PmdNsModel = 0 Or Name = "" : ProcedureReturn -1 : EndIf
  ForEach *PmdNsModel\Graph\Inputs()
    If *PmdNsModel\Graph\Inputs()\Name = Name And *PmdNsModel\Graph\Inputs()\HasShape : ProcedureReturn ListSize(*PmdNsModel\Graph\Inputs()\Dims()) : EndIf
  Next
  ForEach *PmdNsModel\Graph\Values()
    If *PmdNsModel\Graph\Values()\Name = Name And *PmdNsModel\Graph\Values()\HasShape : ProcedureReturn ListSize(*PmdNsModel\Graph\Values()\Dims()) : EndIf
  Next
  ProcedureReturn -1
EndProcedure

; The first element of a top-level initializer as a number: 1 in *Found when
; the name is an initializer of a type read here (FLOAT, INT32, INT64, BOOL).
Procedure.d PmdOpsInitializerValue(Name.s, *Found.Integer)
  *Found\i = 0
  If *PmdNsModel = 0 Or Name = "" : ProcedureReturn 0 : EndIf
  ForEach *PmdNsModel\Graph\Initializers()
    If *PmdNsModel\Graph\Initializers()\Name <> Name : Continue : EndIf
    With *PmdNsModel\Graph\Initializers()
      If \Raw\Bytes > 0
        Select \DataType
          Case 1 : *Found\i = 1 : ProcedureReturn PeekF(\Raw\Data)
          Case 6 : *Found\i = 1 : ProcedureReturn PeekL(\Raw\Data)
          Case 7 : *Found\i = 1 : ProcedureReturn PeekQ(\Raw\Data)
          Case 9 : *Found\i = 1 : ProcedureReturn PeekA(\Raw\Data)
        EndSelect
      ElseIf \DataType = 1 And FirstElement(\FloatData())
        *Found\i = 1 : ProcedureReturn \FloatData()
      ElseIf (\DataType = 6 Or \DataType = 9) And FirstElement(\Int32Data())
        *Found\i = 1 : ProcedureReturn \Int32Data()
      ElseIf \DataType = 7 And FirstElement(\Int64Data())
        *Found\i = 1 : ProcedureReturn \Int64Data()
      EndIf
    EndWith
    ProcedureReturn 0
  Next
  ProcedureReturn 0
EndProcedure

; Returns 1 when the node's form is implemented, else 0 with PmoDynamicError
; holding the sentence.
Procedure.i PmdOpsValidate(*Node.PmoOnnxNode)
  Protected Op.s = *Node\Operation, Allowed.s = PmdOpsAllowed(*Node, PmdNsOpset), Reason.s, Name.s, Text.s, n.i
  Protected Labels.Integer, Kept.Integer, k.i, Found.Integer, Ratio.d
  Protected Dim Ranks.i(7)
  Protected Dim AxisLabel.i(7, 15)
  Protected Dim Codes.i(3)
  ForEach *Node\Attributes()
    If FindString(Allowed, "|" + *Node\Attributes()\Name + "|") = 0
      Name = *Node\Attributes()\Name
      If Trim(Allowed, "|") = ""
        Reason = "attribute " + Name + " = " + PmdNsValueText(@*Node\Attributes()) + " is not part of " + Op + " as runtime-dimension emission implements it, which takes no attribute."
      Else
        Reason = "attribute " + Name + " = " + PmdNsValueText(@*Node\Attributes()) + " is not part of " + Op + " at opset " + Str(PmdNsOpset) +
                 " as runtime-dimension emission implements it; the attributes it implements are " + ReplaceString(Trim(Allowed, "|"), "|", ", ") + "."
      EndIf
      Break
    EndIf
  Next
  If Reason = "" And PmdNsInputPresent(*Node, 0) = 0 And Op <> "Einsum"
    Reason = "its first input is required."
  EndIf
  If Reason = ""
    Select Op
      Case "Gelu" : Reason = PmoOpsUnaryForm(*Node)
      Case "Sign" : Reason = PmdNsTypeReason(*Node, 0, "input", "|1|6|7|")
      Case "Erf", "Reciprocal", "Ceil", "Softplus", "Softsign", "Elu", "Selu", "Celu", "HardSigmoid", "HardSwish", "Mish",
           "ThresholdedRelu", "Shrink", "IsNaN", "IsInf", "LogSoftmax", "MaxPool", "AveragePool", "LpPool", "GlobalMaxPool",
           "GlobalAveragePool", "GlobalLpPool", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp"
        Reason = PmdNsTypeReason(*Node, 0, "input", "|1|")
      Case "ReduceMin", "ReduceL1", "ReduceSumSquare", "ArgMax", "ArgMin" : Reason = PmdNsTypeReason(*Node, 0, "data", "|1|6|7|")
      Case "Min", "Max", "Sum", "PRelu"
        For k = 0 To ListSize(*Node\Inputs()) - 1
          If Reason = "" : Reason = PmdNsTypeReason(*Node, k, "input " + Str(k), "|1|6|7|") : EndIf
        Next
      Case "Mean"
        For k = 0 To ListSize(*Node\Inputs()) - 1
          If Reason = "" : Reason = PmdNsTypeReason(*Node, k, "input " + Str(k), "|1|") : EndIf
        Next
      Case "Mod"
        If PmoEmitAttrI(*Node, "fmod", 0) = 0
          Reason = PmdNsTypeReason(*Node, 0, "A", "|6|7|")
          If Reason <> "" And PmdNsDeclaredType(PmoEmitInput(*Node, 0)) = 1
            Reason = "attribute fmod = 0 on FLOAT inputs; the specification requires fmod = 1 for floating-point types."
          EndIf
        Else
          Reason = PmdNsTypeReason(*Node, 0, "A", "|1|6|7|")
        EndIf
      Case "Or", "Xor"
        Reason = PmdNsTypeReason(*Node, 0, "A", "|9|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "B", "|9|") : EndIf
      Case "GatherElements"
        Reason = PmdNsTypeReason(*Node, 1, "indices", "|6|7|")
      Case "GatherND"
        Reason = PmdNsTypeReason(*Node, 1, "indices", "|7|")
      Case "OneHot"
        Reason = PmdNsTypeReason(*Node, 0, "indices", "|1|6|7|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 2, "values", "|1|6|7|9|") : EndIf
      Case "Tile"
        If PmdNsInputPresent(*Node, 1) = 0 : Reason = "input repeats is required." : EndIf
      Case "DepthToSpace", "SpaceToDepth"
        If PmoEmitNsAttributePresent(*Node, "blocksize") = 0 : Reason = "attribute blocksize is required." : EndIf
        Text = PmoEmitAttrS(*Node, "mode", "DCR")
        If Reason = "" And Text <> "DCR" And Text <> "CRD" : Reason = "attribute mode = " + Text + " is not a DepthToSpace mode; DCR and CRD are." : EndIf
      Case "Split"
        n = ListSize(*Node\Outputs())
        If n < 1 Or n > 16 : Reason = "it has " + Str(n) + " outputs; one to sixteen are implemented." : EndIf
        If Reason = "" And PmdNsInputPresent(*Node, 1) And PmdNsOpset < 13
          Reason = "input split is not part of Split before opset 13; split is an attribute there."
        EndIf
        If Reason = "" And PmdNsOpset >= 18 And PmdNsInputPresent(*Node, 1) = 0 And PmoEmitAttrI(*Node, "num_outputs", 0) <> n
          Reason = "Split-18 needs input split or attribute num_outputs equal to its " + Str(n) + " outputs."
        EndIf
      Case "Einsum"
        n = ListSize(*Node\Inputs())
        For k = 0 To 7 : Ranks(k) = -1 : Next
        Reason = PmoOpsEinsumParse(PmoEmitAttrS(*Node, "equation", ""), n, Ranks(), AxisLabel(), @Labels, @Kept)
        If Reason <> "" : Reason = "equation " + PmoEmitAttrS(*Node, "equation", "") + ": " + Reason : EndIf
        For k = 0 To n - 1
          If Reason = "" : Reason = PmdNsTypeReason(*Node, k, "operand " + Str(k), "|1|") : EndIf
        Next
      Case "MaxPool", "AveragePool", "LpPool"
        If PmoEmitAttrListCount(*Node, "kernel_shape") < 1 : Reason = "attribute kernel_shape is required." : EndIf
      Case "Tan", "Asin", "Acos", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Hardmax", "LpNormalization", "MeanVarianceNormalization",
           "LRN", "Det"
        Reason = PmdNsTypeReason(*Node, 0, "input", "|1|")
      Case "BitwiseNot" : Reason = PmdNsTypeReason(*Node, 0, "X", "|6|7|")
      Case "BitwiseAnd", "BitwiseOr", "BitwiseXor"
        Reason = PmdNsTypeReason(*Node, 0, "A", "|6|7|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "B", "|6|7|") : EndIf
      Case "Upsample"
        Reason = "Upsample is implemented by fixed-shape emission only - declared extents and a constant scales input, opset 7 to 9 (from 10 it is deprecated for Resize)."
      Case "GroupNormalization"
        Reason = PmdNsTypeReason(*Node, 0, "X", "|1|")
        If Reason = "" And PmoEmitNsAttributePresent(*Node, "num_groups") = 0 : Reason = "attribute num_groups is required." : EndIf
      Case "RNN", "GRU"
        Reason = PmdNsTypeReason(*Node, 0, "X", "|1|")
        If Reason = "" And (PmoEmitNsAttributePresent(*Node, "activation_alpha") Or PmoEmitNsAttributePresent(*Node, "activation_beta"))
          Reason = "activation_alpha and activation_beta are not implemented; the activations implemented (Sigmoid, Tanh, Relu) take none."
        EndIf
        If Reason = "" And PmoEmitAttrI(*Node, "layout", 0) <> 0 : Reason = "attribute layout = 1 (batch first) is not implemented; layout 0 is." : EndIf
        Text = PmoEmitAttrS(*Node, "direction", "forward")
        If Reason = "" And Text <> "forward" And Text <> "reverse" And Text <> "bidirectional"
          Reason = "attribute direction = " + Text + "; forward, reverse and bidirectional are."
        EndIf
        If Reason = ""
          k = 1 + Bool(Text = "bidirectional")
          Reason = PmoOpsRecurrentActivations(*Node, k, Codes())
        EndIf
      Case "HannWindow", "HammingWindow", "BlackmanWindow"
        If PmoEmitAttrI(*Node, "output_datatype", 1) <> 1 : Reason = "output_datatype " + PmdNsTypeName(PmoEmitAttrI(*Node, "output_datatype", 1)) + " is not implemented; FLOAT is." : EndIf
      Case "DFT"
        Reason = PmdNsTypeReason(*Node, 0, "input", "|1|")
      Case "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss"
        Reason = PmdNsTypeReason(*Node, 0, "the scores", "|1|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "the target", "|6|7|") : EndIf
        Text = PmoEmitAttrS(*Node, "reduction", "mean")
        If Reason = "" And Text <> "none" And Text <> "sum" And Text <> "mean" : Reason = "attribute reduction = " + Text + "; none, sum and mean are." : EndIf
      Case "MelWeightMatrix"
        Reason = PmoOpsMelSentence()
      Case "QuantizeLinear"
        Reason = PmdNsTypeReason(*Node, 0, "x", "|1|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "y_scale", "|1|") : EndIf
        If Reason = "" And PmdNsInputPresent(*Node, 2) : Reason = PmdNsTypeReason(*Node, 2, "y_zero_point", "|2|3|") : EndIf
      Case "DequantizeLinear"
        Reason = PmdNsTypeReason(*Node, 0, "x", "|2|3|6|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "x_scale", "|1|") : EndIf
      Case "DynamicQuantizeLinear"
        Reason = PmdNsTypeReason(*Node, 0, "x", "|1|")
        If Reason = "" And PmdNsNamedOutputs(*Node) <> 3 : Reason = "outputs y, y_scale and y_zero_point are all required." : EndIf
      Case "MatMulInteger", "ConvInteger"
        Reason = PmdNsTypeReason(*Node, 0, "the first input", "|2|3|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "the second input", "|2|3|") : EndIf
      Case "QLinearMatMul", "QLinearConv"
        Reason = PmdNsTypeReason(*Node, 0, "the first input", "|2|3|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 3, "the second input", "|2|3|") : EndIf
        For k = 0 To 7
          If Reason = "" And PmdNsInputPresent(*Node, k) = 0 : Reason = "inputs 0 to 7 (both operands, their scales and zero points, y_scale and y_zero_point) are required." : EndIf
        Next
      Case "GridSample"
        Reason = PmdNsTypeReason(*Node, 0, "X", "|1|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "grid", "|1|") : EndIf
        k = PmdOpsDeclaredRank(PmoEmitInput(*Node, 0)) - 2
        If k < 1 : k = 0 : EndIf
        If PmdOpsDeclaredRank(PmoEmitInput(*Node, 0)) >= 0 And PmdOpsDeclaredRank(PmoEmitInput(*Node, 0)) < 3
          If Reason = "" : Reason = "the input has rank " + Str(PmdOpsDeclaredRank(PmoEmitInput(*Node, 0))) + "; GridSample takes [N, C, D1 ...]." : EndIf
        EndIf
        If Reason = "" : Reason = PmoOpsGridSampleForm(*Node, PmdNsOpset, k, @Labels, @Kept) : EndIf
      Case "NonMaxSuppression"
        Reason = PmdNsTypeReason(*Node, 0, "boxes", "|1|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "scores", "|1|") : EndIf
      Case "RoiAlign"
        Reason = PmdNsTypeReason(*Node, 0, "X", "|1|")
        Text = PmoEmitAttrS(*Node, "coordinate_transformation_mode", "half_pixel")
        If Reason = "" And Text <> "half_pixel" And Text <> "output_half_pixel" : Reason = "attribute coordinate_transformation_mode = " + Text + "; half_pixel and output_half_pixel are." : EndIf
        Text = PmoEmitAttrS(*Node, "mode", "avg")
        If Reason = "" And Text <> "avg" And Text <> "max" : Reason = "attribute mode = " + Text + "; avg and max are." : EndIf
    EndSelect
  EndIf
  If Reason = ""
    Select Op
      Case "Hardmax"
        If PmdNsOpset < 13 And PmoEmitAttrI(*Node, "axis", 1) <> -1
          Reason = "the model imports opset " + Str(PmdNsOpset) + ", where Hardmax flattens the input at its axis; runtime-dimension emission implements that only as axis = -1, the last axis, where it equals Hardmax-13."
        EndIf
      Case "LpNormalization"
        If PmoEmitAttrI(*Node, "p", 2) <> 1 And PmoEmitAttrI(*Node, "p", 2) <> 2 : Reason = "attribute p = " + Str(PmoEmitAttrI(*Node, "p", 2)) + "; the specification allows 1 and 2." : EndIf
      Case "LRN"
        If PmoEmitNsAttributePresent(*Node, "size") = 0 Or PmoEmitAttrI(*Node, "size", 1) < 1 : Reason = "attribute size is required and must be positive." : EndIf
      Case "ReverseSequence"
        k = PmoEmitAttrI(*Node, "time_axis", 0)
        n = PmoEmitAttrI(*Node, "batch_axis", 1)
        If Not ((k = 0 And n = 1) Or (k = 1 And n = 0)) : Reason = "time_axis = " + Str(k) + " and batch_axis = " + Str(n) + "; the specification allows 0 and 1 or 1 and 0." : EndIf
      Case "EyeLike"
        n = PmoEmitAttrI(*Node, "dtype", 0)
        If n <> 0 And n <> 1 And n <> 6 And n <> 7 And n <> 9 : Reason = "dtype " + PmdNsTypeName(n) + " is not implemented; FLOAT, INT32, INT64 and BOOL are." : EndIf
      Case "RoiAlign"
        If PmoEmitAttrI(*Node, "output_height", 1) < 1 Or PmoEmitAttrI(*Node, "output_width", 1) < 1 Or PmoEmitAttrI(*Node, "sampling_ratio", 0) < 0
          Reason = "output_height and output_width must be positive and sampling_ratio not negative."
        EndIf
    EndSelect
  EndIf
  If Reason = ""
    Select Op
      Case "MaxPool", "AveragePool", "LpPool"
        Text = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
        If Text <> "NOTSET" And Text <> "SAME_UPPER" And Text <> "SAME_LOWER" And Text <> "VALID"
          Reason = "attribute auto_pad = " + Text + " is not a padding mode; NOTSET, SAME_UPPER, SAME_LOWER and VALID are."
        ElseIf Text <> "NOTSET" And PmoEmitAttrListCount(*Node, "pads") > 0
          Reason = "attribute pads is given with auto_pad = " + Text + "; the specification allows explicit pads only with NOTSET."
        ElseIf PmoEmitAttrListCount(*Node, "kernel_shape") > 3
          Reason = "kernel_shape has " + Str(PmoEmitAttrListCount(*Node, "kernel_shape")) + " extents; one to three spatial axes are implemented."
        ElseIf Op = "LpPool" And PmoEmitAttrI(*Node, "p", 2) < 1
          Reason = "attribute p = " + Str(PmoEmitAttrI(*Node, "p", 2)) + "; the norm order must be at least one."
        EndIf
      Case "LogSoftmax"
        If PmdNsOpset < 13 And PmoEmitAttrI(*Node, "axis", 1) <> -1
          Reason = "the model imports opset " + Str(PmdNsOpset) + ", where LogSoftmax flattens the input at its axis; runtime-dimension emission implements that only as axis = -1, the last axis, where it equals LogSoftmax-13."
        EndIf
      Case "Mod", "PRelu", "Or", "Xor"
        If ListSize(*Node\Inputs()) <> 2 : Reason = "it has " + Str(ListSize(*Node\Inputs())) + " inputs; the specification gives it two." : EndIf
      Case "Min", "Max", "Sum", "Mean"
        If ListSize(*Node\Inputs()) < 1 Or ListSize(*Node\Inputs()) > 16
          Reason = "it has " + Str(ListSize(*Node\Inputs())) + " inputs; one to sixteen are implemented."
        EndIf
      Case "OneHot"
        If PmdNsInputPresent(*Node, 1) = 0 Or PmdNsInputPresent(*Node, 2) = 0 : Reason = "inputs depth and values are required." : EndIf
      Case "Dropout"
        ; a training_mode the model fixes as true, with a nonzero ratio, is
        ; refused now; one that arrives at run time is checked when it runs
        If PmdOpsInitializerValue(PmoEmitInput(*Node, 2), @Found) <> 0.0 And Found\i
          Ratio = 0.5
          If PmdNsInputPresent(*Node, 1)
            Ratio = PmdOpsInitializerValue(PmoEmitInput(*Node, 1), @Found)
            If Found\i = 0 : Ratio = 0.5 : EndIf
          EndIf
          If Ratio <> 0.0
            Reason = "training_mode is true with a nonzero ratio, which draws random masks; inference Dropout (training_mode false) is implemented."
          EndIf
        EndIf
    EndSelect
  EndIf
  If Reason = ""
    Select Op
      Case "Split"
      Case "DynamicQuantizeLinear"
        If ListSize(*Node\Outputs()) <> 3 Or PmdNsNamedOutputs(*Node) <> 3
          Reason = "it declares " + Str(PmdNsNamedOutputs(*Node)) + " named outputs; DynamicQuantizeLinear has three (y, y_scale, y_zero_point)."
        EndIf
      Case "MaxPool", "Dropout", "RNN", "GRU", "SoftmaxCrossEntropyLoss"
        If ListSize(*Node\Outputs()) < 1 Or ListSize(*Node\Outputs()) > 2
          Reason = "it declares " + Str(ListSize(*Node\Outputs())) + " outputs; " + Op + " has one or two."
        EndIf
      Default
        If ListSize(*Node\Outputs()) <> 1 Or PmdNsNamedOutputs(*Node) <> 1
          Reason = "it declares " + Str(ListSize(*Node\Outputs())) + " outputs; " + Op + " has one."
        EndIf
    EndSelect
  EndIf
  If Reason = "" : ProcedureReturn 1 : EndIf
  Name = *Node\Name : If Name = "" : Name = "(unnamed)" : EndIf
  PmoDynamicError = Op + " node " + Name + ": " + Reason
  ProcedureReturn 0
EndProcedure

; The call text for one node (validated by PmdOpsValidate after this).
Procedure.s PmdOpsCall(*Node.PmoOnnxNode, Map Ids.i())
  Protected Op.s = *Node\Operation, Call.s, Pre.s, i.i, k.i, n.i, Code.i, Text.s, s.i, d.i
  Protected Labels.Integer, Kept.Integer
  Protected Dim a.s(7)
  Protected Dim Ranks.i(7)
  Protected Dim AxisLabel.i(7, 15)
  NewList Lines.s()
  For i = 0 To 7 : a(i) = PmdNsId(Ids(), PmoEmitInput(*Node, i)) : Next
  Select Op
    Case "Erf", "Reciprocal", "Ceil", "Sign", "Softplus", "Softsign", "Elu", "Selu", "Celu", "HardSigmoid", "HardSwish", "Mish", "Gelu",
         "ThresholdedRelu", "Shrink", "IsNaN", "IsInf", "Tan", "Asin", "Acos", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "BitwiseNot"
      PmoOpsUnaryParams(*Node, Lines())
      ForEach Lines() : Pre + Lines() + " : " : Next
      Call = Pre + "DOpUnary(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoOpsUnaryCode(*Node)) + ")"
    Case "Hardmax"
      Call = "DOpHardmax(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "axis", -1)) + ")"
    Case "LpNormalization"
      Call = "DOpLpNorm(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "axis", -1)) + "," + Str(PmoEmitAttrI(*Node, "p", 2)) + ")"
    Case "MeanVarianceNormalization"
      n = PmoEmitAttrListCount(*Node, "axes")
      Pre = "PmOpT(0)=" + Str(n) + " : "
      For k = 0 To n - 1 : Pre + "PmOpT(" + Str(1 + k) + ")=" + Str(PmoEmitAttrListI(*Node, "axes", k, 0)) + " : " : Next
      Call = Pre + "DOpMvn(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + ")"
    Case "LRN"
      Pre = "PmOpSetBits(@PmOpF(0)," + PmoOpsBits(PmoEmitAttrF(*Node, "alpha", 0.0001)) + ") : PmOpSetBits(@PmOpF(1)," + PmoOpsBits(PmoEmitAttrF(*Node, "beta", 0.75)) + ") : "
      Pre + "PmOpSetBits(@PmOpF(2)," + PmoOpsBits(PmoEmitAttrF(*Node, "bias", 1.0)) + ") : "
      Call = Pre + "DOpLrn(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "size", 1)) + ")"
    Case "GroupNormalization"
      Call = "PmOpSetBits(@PmOpF(0)," + PmoOpsBits(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ") : DOpGroupNorm(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," +
             a(0) + "," + a(1) + "," + a(2) + "," + Str(PmoEmitAttrI(*Node, "num_groups", 1)) + ")"
    Case "EyeLike"
      Call = "DOpEyeLike(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "dtype", 0)) + "," + Str(PmoEmitAttrI(*Node, "k", 0)) + ")"
    Case "Det"
      Call = "DOpDet(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + ")"
    Case "Compress"
      Call = "DOpCompress(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," +
             Str(PmoEmitNsAttributePresent(*Node, "axis")) + ")"
    Case "ReverseSequence"
      Call = "DOpReverseSequence(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "time_axis", 0)) + "," +
             Str(PmoEmitAttrI(*Node, "batch_axis", 1)) + ")"
    Case "RNN", "GRU"
      Text = PmoEmitAttrS(*Node, "direction", "forward")
      Code = 0
      If Text = "reverse" : Code = 1 : ElseIf Text = "bidirectional" : Code = 2 : EndIf
      For k = 0 To 3 : Ranks(k) = 0 : Next
      PmoOpsRecurrentActivations(*Node, 1 + Bool(Code = 2), Ranks())
      Pre = "PmOpI(5)=" + Str(Code) + " : PmOpI(7)=" + Str(Bool(PmoEmitAttrI(*Node, "linear_before_reset", 0) <> 0)) + " : PmOpI(8)=" +
            Str(PmoEmitNsAttributePresent(*Node, "clip")) + " : PmOpSetBits(@PmOpF(0)," + PmoOpsBits(PmoEmitAttrF(*Node, "clip", 0.0)) + ") : "
      For k = 0 To 3 : Pre + "PmOpI(" + Str(10 + k) + ")=" + Str(Ranks(k)) + " : " : Next
      Call = Pre + "DOpRecurrent(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + PmdNsId(Ids(), PmoEmitOutput(*Node, 1)) + "," + a(0) + "," + a(1) + "," + a(2) + "," +
             a(3) + "," + a(4) + "," + a(5) + "," + Str(Bool(Op = "GRU")) + ")"
    Case "NonMaxSuppression"
      Call = "DOpNms(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + "," + a(3) + "," + a(4) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "center_point_box", 0) <> 0)) + ")"
    Case "RoiAlign"
      Text = "half_pixel"
      If PmdNsOpset < 16 : Text = "output_half_pixel" : EndIf
      Text = PmoEmitAttrS(*Node, "coordinate_transformation_mode", Text)
      Pre = "PmOpI(5)=" + Str(PmoEmitAttrI(*Node, "output_height", 1)) + " : PmOpI(6)=" + Str(PmoEmitAttrI(*Node, "output_width", 1)) + " : PmOpI(7)=" +
            Str(PmoEmitAttrI(*Node, "sampling_ratio", 0)) + " : PmOpI(8)=" + Str(Bool(PmoEmitAttrS(*Node, "mode", "avg") = "max")) + " : PmOpI(9)=" +
            Str(Bool(Text = "half_pixel")) + " : PmOpSetBits(@PmOpF(0)," + PmoOpsBits(PmoEmitAttrF(*Node, "spatial_scale", 1.0)) + ") : "
      Call = Pre + "DOpRoiAlign(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + ")"
    Case "HannWindow", "HammingWindow", "BlackmanWindow"
      Code = 2
      If Op = "HannWindow" : Code = 0 : ElseIf Op = "HammingWindow" : Code = 1 : EndIf
      Call = "DOpWindow(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(Code) + "," + Str(Bool(PmoEmitAttrI(*Node, "periodic", 1) <> 0)) + ")"
    Case "DFT"
      d = -2
      If PmdNsOpset < 20 : d = PmoEmitAttrI(*Node, "axis", 1) : EndIf
      Text = "0"
      If PmdNsOpset >= 20 : Text = a(2) : EndIf
      Call = "DOpDft(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Text + "," + Str(d) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "inverse", 0) <> 0)) + "," + Str(Bool(PmoEmitAttrI(*Node, "onesided", 0) <> 0)) + ")"
    Case "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss"
      Text = PmoEmitAttrS(*Node, "reduction", "mean")
      Code = 2
      If Text = "none" : Code = 0 : ElseIf Text = "sum" : Code = 1 : EndIf
      Call = "DOpLoss(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + PmdNsId(Ids(), PmoEmitOutput(*Node, 1)) + "," + a(0) + "," + a(1) + "," + a(2) + "," +
             Str(Code) + "," + Str(PmoEmitNsAttributePresent(*Node, "ignore_index")) + "," + Str(PmoEmitAttrI(*Node, "ignore_index", 0)) + "," +
             Str(Bool(Op = "SoftmaxCrossEntropyLoss")) + ")"
    Case "QuantizeLinear", "DequantizeLinear"
      Call = "DOpQuantize(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + "," + Str(PmoEmitAttrI(*Node, "axis", 1)) + "," +
             Str(Bool(Op = "QuantizeLinear")) + "," + Str(Bool(PmdNsOpset >= 13)) + ")"
    Case "DynamicQuantizeLinear"
      Call = "DOpDynQuantize(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + PmdNsId(Ids(), PmoEmitOutput(*Node, 1)) + "," +
             PmdNsId(Ids(), PmoEmitOutput(*Node, 2)) + "," + a(0) + ")"
    Case "MatMulInteger"
      Call = "DOpQMatMul(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + "," + a(3) + ",0,0,0,0,0)"
    Case "QLinearMatMul"
      Call = "DOpQMatMul(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(3) + "," + a(2) + "," + a(5) + "," + a(1) + "," +
             a(4) + "," + a(6) + "," + a(7) + ",1)"
    Case "ConvInteger", "QLinearConv"
      Text = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
      Code = 0
      If Text = "SAME_UPPER" : Code = 1 : ElseIf Text = "SAME_LOWER" : Code = 2 : ElseIf Text = "VALID" : Code = 3 : EndIf
      Pre = "PmOpT(1)=" + Str(Code) + " : PmOpT(2)=" + Str(PmoEmitAttrI(*Node, "group", 1)) + " : "
      Pre + "PmOpT(3)=" + Str(PmoEmitAttrListCount(*Node, "strides")) + " : PmOpT(4)=" + Str(PmoEmitAttrListCount(*Node, "dilations")) + " : "
      Pre + "PmOpT(5)=" + Str(PmoEmitAttrListCount(*Node, "pads")) + " : PmOpT(6)=" + Str(PmoEmitAttrListCount(*Node, "kernel_shape")) + " : "
      For d = 0 To 2
        Pre + "PmOpT(" + Str(8 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "strides", d, 1)) + " : "
        Pre + "PmOpT(" + Str(12 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "dilations", d, 1)) + " : "
        Pre + "PmOpT(" + Str(24 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "kernel_shape", d, 0)) + " : "
      Next
      For d = 0 To 5 : Pre + "PmOpT(" + Str(16 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "pads", d, 0)) + " : " : Next
      If Op = "ConvInteger"
        Call = Pre + "DOpQConv(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + "," + a(3) + ",0,0,0,0,0,0)"
      Else
        Call = Pre + "DOpQConv(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(3) + "," + a(2) + "," + a(5) + "," +
               PmdNsId(Ids(), PmoEmitInput(*Node, 8)) + "," + a(1) + "," + a(4) + "," + a(6) + "," + a(7) + ",1)"
      EndIf
    Case "GridSample"
      PmoOpsGridSampleForm(*Node, PmdNsOpset, 0, @Labels, @Kept)
      Call = "DOpGridSample(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(Labels\i) + "," + Str(Kept\i) + "," +
             Str(PmoEmitAttrI(*Node, "align_corners", 0)) + "," + Str(Bool(PmdNsOpset >= 20)) + ")"
    Case "Min", "Max", "Sum", "Mean", "Mod", "PRelu", "Or", "Xor", "BitwiseAnd", "BitwiseOr", "BitwiseXor"
      n = ListSize(*Node\Inputs())
      For k = 0 To n - 1
        If k > 7
          Pre + "PmOpN(" + Str(k) + ")=" + PmdNsId(Ids(), PmoEmitInput(*Node, k)) + " : "
        Else
          Pre + "PmOpN(" + Str(k) + ")=" + a(k) + " : "
        EndIf
      Next
      Call = Pre + "DOpVariadic(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + Str(n) + "," + Str(PmoOpsVariadicCode(*Node)) + "," + Str(PmoOpsVariadicKinds(*Node)) + ")"
    Case "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"
      n = PmoEmitAttrListCount(*Node, "axes")
      Pre = "PmOpT(0)=" + Str(n) + " : "
      For k = 0 To n - 1 : Pre + "PmOpT(" + Str(1 + k) + ")=" + Str(PmoEmitAttrListI(*Node, "axes", k, 0)) + " : " : Next
      Call = Pre + "DOpReduce(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(Bool(PmoEmitAttrI(*Node, "keepdims", 1) <> 0)) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "noop_with_empty_axes", 0) <> 0)) + "," + Str(PmoOpsReduceCode(Op)) + ")"
    Case "ArgMax", "ArgMin"
      Call = "DOpArg(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," + Str(Bool(PmoEmitAttrI(*Node, "keepdims", 1) <> 0)) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "select_last_index", 0) <> 0)) + "," + Str(Bool(Op = "ArgMin")) + ")"
    Case "LogSoftmax"
      Call = "DOpLogSoftmax(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "axis", -1)) + ")"
    Case "MaxPool", "AveragePool", "LpPool", "GlobalMaxPool", "GlobalAveragePool", "GlobalLpPool"
      s = PmoEmitAttrListCount(*Node, "kernel_shape")
      If Left(Op, 6) = "Global" : s = 0 : EndIf
      Text = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
      Code = 0
      If Text = "SAME_UPPER" : Code = 1 : ElseIf Text = "SAME_LOWER" : Code = 2 : ElseIf Text = "VALID" : Code = 3 : EndIf
      Pre = "PmOpT(0)=" + Str(s) + " : PmOpT(1)=" + Str(Code) + " : PmOpT(2)=" + Str(Bool(PmoEmitAttrI(*Node, "ceil_mode", 0) <> 0)) + " : "
      Pre + "PmOpT(3)=" + Str(Bool(PmoEmitAttrI(*Node, "count_include_pad", 0) <> 0)) + " : PmOpT(4)=" + Str(PmoEmitAttrI(*Node, "p", 2)) + " : "
      Pre + "PmOpT(5)=" + Str(Bool(PmoEmitAttrI(*Node, "storage_order", 0) <> 0)) + " : "
      Code = 2
      If Op = "MaxPool" Or Op = "GlobalMaxPool" : Code = 0 : ElseIf Op = "AveragePool" Or Op = "GlobalAveragePool" : Code = 1 : EndIf
      Pre + "PmOpT(6)=" + Str(Code) + " : "
      For d = 0 To s - 1
        Pre + "PmOpT(" + Str(8 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "kernel_shape", d, 1)) + " : "
        Pre + "PmOpT(" + Str(12 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "strides", d, 1)) + " : "
        Pre + "PmOpT(" + Str(16 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "dilations", d, 1)) + " : "
        Pre + "PmOpT(" + Str(20 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "pads", d, 0)) + " : "
        Pre + "PmOpT(" + Str(24 + d) + ")=" + Str(PmoEmitAttrListI(*Node, "pads", d + s, 0)) + " : "
      Next
      Call = Pre + "DOpPool(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + PmdNsId(Ids(), PmoEmitOutput(*Node, 1)) + "," + a(0) + ")"
    Case "Split"
      n = ListSize(*Node\Outputs())
      For k = 0 To n - 1 : Pre + "PmOpN(" + Str(k) + ")=" + PmdNsId(Ids(), PmoEmitOutput(*Node, k)) + " : " : Next
      If PmdNsInputPresent(*Node, 1)
        Pre + "PmOpT(0)=0 : "
      ElseIf PmoEmitAttrListCount(*Node, "split") > 0
        Pre + "PmOpT(0)=1 : "
        For k = 0 To PmoEmitAttrListCount(*Node, "split") - 1 : Pre + "PmOpT(" + Str(1 + k) + ")=" + Str(PmoEmitAttrListI(*Node, "split", k, 0)) + " : " : Next
      ElseIf PmoEmitAttrI(*Node, "num_outputs", 0) <> 0
        Pre + "PmOpT(0)=2 : "
      Else
        Pre + "PmOpT(0)=0 : "
      EndIf
      Call = Pre + "DOpSplit(" + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," + Str(n) + ")"
    Case "Tile"
      Call = "DOpTile(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + ")"
    Case "DepthToSpace", "SpaceToDepth"
      Call = "DOpDepthSpace(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + Str(PmoEmitAttrI(*Node, "blocksize", 0)) + "," +
             Str(Bool(PmoEmitAttrS(*Node, "mode", "DCR") = "CRD")) + "," + Str(Bool(Op = "DepthToSpace")) + ")"
    Case "Trilu"
      Call = "DOpTrilu(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(Bool(PmoEmitAttrI(*Node, "upper", 1) <> 0)) + ")"
    Case "GatherElements"
      Call = "DOpGatherElements(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + ")"
    Case "GatherND"
      Call = "DOpGatherND(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "batch_dims", 0)) + ")"
    Case "OneHot"
      Call = "DOpOneHot(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + "," + a(1) + "," + a(2) + "," + Str(PmoEmitAttrI(*Node, "axis", -1)) + ")"
    Case "Einsum"
      n = ListSize(*Node\Inputs())
      For k = 0 To 7 : Ranks(k) = -1 : Next
      If PmoOpsEinsumParse(PmoEmitAttrS(*Node, "equation", ""), n, Ranks(), AxisLabel(), @Labels, @Kept) = ""
        Pre = "PmOpT(0)=" + Str(n) + " : PmOpT(1)=" + Str(Labels\i) + " : PmOpT(2)=" + Str(Kept\i) + " : "
        For k = 0 To n - 1
          Pre + "PmOpT(" + Str(4 + k) + ")=" + Str(Ranks(k)) + " : PmOpN(" + Str(k) + ")=" + a(k) + " : "
          For d = 0 To Ranks(k) - 1 : Pre + "PmOpT(" + Str(8 + k * 8 + d) + ")=" + Str(AxisLabel(k, d)) + " : " : Next
        Next
      EndIf
      Call = Pre + "DOpEinsum(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + ")"
    Case "Dropout"
      Call = "DOpDropout(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + PmdNsId(Ids(), PmoEmitOutput(*Node, 1)) + "," + a(0) + "," + a(1) + "," + a(2) + ")"
    Case "CastLike"
      Call = "DCast(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + ",Dt(" + a(1) + ")\Kind)"
    Case "Size"
      Call = "DOpSize(" + PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) + "," + a(0) + ")"
  EndSelect
  ProcedureReturn Call
EndProcedure
