; ============================================================================
; onnx_forms.pbi - attribute forms, checked the same way on both lowering paths
; ----------------------------------------------------------------------------
; An operator's attributes are refused only when their VALUE selects behaviour
; the kernel does not compute. An attribute that is merely present, carrying
; the default the ONNX specification defines or a value that has no effect in
; the form being emitted, is accepted. Every refusal is one sentence naming the
; operator, the attribute and the value.
;
; The fixed-shape path (onnx_compile.pbi) and the runtime-dimension path
; (onnx_dynamic_emit.pbi) both call PmoFormRefusal for the operators listed in
; PmoFormOwns, so a form is implemented or refused identically on both. Where
; the two paths genuinely implement different forms (Conv attribute lists
; longer than one axis, LSTM outputs left unnamed) the difference is stated in
; the sentence.
;
; Specification sections: https://onnx.ai/onnx/operators/ - Cast-19, Shape-15,
; LSTM-14, LayerNormalization-17, Conv-11, ConvTranspose-11, Resize-19,
; ScatterND-18, ReduceSum-13, ReduceMean-18.
; ============================================================================

Procedure.i PmoFormOwns(Operation.s)
  ProcedureReturn Bool(FindString("|Cast|Shape|LSTM|LayerNormalization|Conv|ConvTranspose|Resize|ScatterND|ReduceMean|ReduceSum|", "|" + Operation + "|"))
EndProcedure

Procedure.i PmoFormAttribute(*Node.PmoOnnxNode, Name.s)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn @*Node\Attributes() : EndIf
  Next
  ProcedureReturn 0
EndProcedure

; The value of an attribute as a model author would write it.
Procedure.s PmoFormValueText(*Attribute.PmoOnnxAttribute)
  Protected Text.s
  If *Attribute = 0 : ProcedureReturn "" : EndIf
  If ListSize(*Attribute\Integers())
    ForEach *Attribute\Integers()
      If Text <> "" : Text + ", " : EndIf
      Text + Str(*Attribute\Integers())
    Next
    ProcedureReturn "[" + Text + "]"
  EndIf
  If ListSize(*Attribute\Floats())
    ForEach *Attribute\Floats()
      If Text <> "" : Text + ", " : EndIf
      Text + StrF(*Attribute\Floats())
    Next
    ProcedureReturn "[" + Text + "]"
  EndIf
  If ListSize(*Attribute\Strings())
    ForEach *Attribute\Strings()
      If Text <> "" : Text + ", " : EndIf
      Text + *Attribute\Strings()
    Next
    ProcedureReturn "[" + Text + "]"
  EndIf
  Select *Attribute\AttributeType
    Case 1 : ProcedureReturn StrF(*Attribute\FloatValue)
    Case 3 : ProcedureReturn *Attribute\StringValue
    Case 4 : ProcedureReturn "a tensor"
    Case 5 : ProcedureReturn "a graph"
  EndSelect
  If *Attribute\StringValue <> "" : ProcedureReturn *Attribute\StringValue : EndIf
  ProcedureReturn Str(*Attribute\IntegerValue)
EndProcedure

; The first attribute whose name is not in Allowed ("|a|b|"), as a sentence.
Procedure.s PmoFormUnknown(*Node.PmoOnnxNode, Allowed.s, Path.s)
  ForEach *Node\Attributes()
    If FindString(Allowed, "|" + *Node\Attributes()\Name + "|") = 0
      If Trim(Allowed, "|") = ""
        ProcedureReturn "attribute " + *Node\Attributes()\Name + " = " + PmoFormValueText(@*Node\Attributes()) +
                        " is not implemented by " + Path + ", which implements no attribute of this operator."
      EndIf
      ProcedureReturn "attribute " + *Node\Attributes()\Name + " = " + PmoFormValueText(@*Node\Attributes()) +
                      " is not implemented by " + Path + "; the attributes it implements are " +
                      ReplaceString(Trim(Allowed, "|"), "|", ", ") + "."
    EndIf
  Next
  ProcedureReturn ""
EndProcedure

Procedure.i PmoFormInputPresent(*Node.PmoOnnxNode, Index.i)
  If SelectElement(*Node\Inputs(), Index) And *Node\Inputs() <> "" : ProcedureReturn #True : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoFormOutputsPresent(*Node.PmoOnnxNode)
  Protected Count.i
  ForEach *Node\Outputs()
    If *Node\Outputs() <> "" : Count + 1 : EndIf
  Next
  ProcedureReturn Count
EndProcedure

Procedure.i PmoFormPadsAllZero(*Node.PmoOnnxNode)
  Protected *Pads.PmoOnnxAttribute = PmoFormAttribute(*Node, "pads")
  If *Pads = 0 : ProcedureReturn #True : EndIf
  ForEach *Pads\Integers()
    If *Pads\Integers() <> 0 : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

; Returns "" when every attribute of the node selects a form this path emits,
; otherwise the refusal sentence without the node prefix.
Procedure.s PmoFormRefusal(*Node.PmoOnnxNode, FixedShape.i)
  Protected Op.s = *Node\Operation
  Protected Path.s = "runtime-dimension emission"
  Protected Reason.s
  Protected Text.s
  Protected Value.q
  Protected Directions.i
  Protected Index.i
  Protected Mode.s
  Protected *A.PmoOnnxAttribute
  If FixedShape : Path = "fixed-shape emission" : EndIf
  Select Op
    Case "Cast"
      Reason = PmoFormUnknown(*Node, "|to|saturate|", Path)
      *A = PmoFormAttribute(*Node, "to")
      If Reason = "" And *A = 0
        Reason = "attribute to is missing; the specification requires the target element type."
      ElseIf Reason = ""
        Value = *A\IntegerValue
        Select Value
          Case 17, 18, 19, 20
            ; saturate only changes a cast to one of these four types.
            Reason = "to = " + Str(Value) + " is a float8 element type, which " + Path + " does not implement; saturate applies only to float8 casts and is accepted with any value for the element types that are implemented."
          Case 1, 7, 9
          Case 6
            If FixedShape : Reason = "to = 6 (INT32) is not implemented by " + Path + "; it casts to FLOAT (1), INT64 (7) and BOOL (9)." : EndIf
          Default
            Text = "FLOAT (1), INT32 (6), INT64 (7) and BOOL (9)"
            If FixedShape : Text = "FLOAT (1), INT64 (7) and BOOL (9)" : EndIf
            Reason = "to = " + Str(Value) + " is not implemented by " + Path + "; it casts to " + Text + "."
        EndSelect
      EndIf

    Case "Shape"
      ; start and end select a slice of the shape, negative values counting
      ; from the end and both clamped to the rank: implemented on both paths.
      Reason = PmoFormUnknown(*Node, "|start|end|", Path)

    Case "LSTM"
      Reason = PmoFormUnknown(*Node, "|direction|hidden_size|layout|input_forget|activations|activation_alpha|activation_beta|clip|", Path)
      Mode = PmoEmitAttrS(*Node, "direction", "forward")
      If Reason = "" And Mode <> "forward" And Mode <> "bidirectional"
        Reason = "direction = " + Mode + " is not implemented by " + Path + "; forward and bidirectional are."
      EndIf
      Directions = 1 : If Mode = "bidirectional" : Directions = 2 : EndIf
      If Reason = "" And PmoEmitAttrI(*Node, "layout", 0) <> 0
        Reason = "layout = " + Str(PmoEmitAttrI(*Node, "layout", 0)) + " (batch-first tensors) is not implemented by " + Path + "; layout 0, the default, is."
      EndIf
      If Reason = "" And PmoEmitAttrI(*Node, "input_forget", 0) <> 0
        Reason = "input_forget = " + Str(PmoEmitAttrI(*Node, "input_forget", 0)) + " (coupled input and forget gates) is not implemented by " + Path + "; input_forget 0, the default, is."
      EndIf
      *A = PmoFormAttribute(*Node, "clip")
      If Reason = "" And *A
        Reason = "clip = " + PmoFormValueText(*A) + " is not implemented by " + Path + "; the cell is computed without clipping, so the attribute must be absent."
      EndIf
      *A = PmoFormAttribute(*Node, "activations")
      If Reason = "" And *A
        ; The defaults are Sigmoid, Tanh, Tanh per direction. Those take no
        ; alpha or beta, so activation_alpha/beta have no effect with them.
        Text = "Sigmoid, Tanh, Tanh"
        If Directions = 2 : Text + ", Sigmoid, Tanh, Tanh" : EndIf
        If PmoFormValueText(*A) <> "[" + Text + "]"
          Reason = "activations = " + PmoFormValueText(*A) + " is not implemented by " + Path + "; only the default [" + Text + "] is."
        EndIf
      EndIf
      If Reason = "" And PmoFormInputPresent(*Node, 7)
        Reason = "input P (peephole weights) is not implemented by " + Path + "."
      EndIf
      If Reason = "" And FixedShape = 0 And ListSize(*Node\Outputs()) <> 3
        Reason = "it has " + Str(ListSize(*Node\Outputs())) + " outputs; " + Path + " requires all three (Y, Y_h, Y_c) to be named."
      EndIf

    Case "LayerNormalization"
      Reason = PmoFormUnknown(*Node, "|axis|epsilon|stash_type|", Path)
      If Reason = "" And PmoEmitAttrI(*Node, "stash_type", 1) <> 1
        Reason = "stash_type = " + Str(PmoEmitAttrI(*Node, "stash_type", 1)) + " is not implemented by " + Path + "; the statistics are computed in FLOAT, stash_type 1, the default."
      EndIf
      ; axis and the optional Mean/InvStdDev outputs are checked by each path
      ; beside its own kernel call (forum 850 and 851), not here.

    Case "Conv", "ConvTranspose"
      Text = "|auto_pad|pads|strides|dilations|kernel_shape|group|"
      If Op = "ConvTranspose" : Text + "output_padding|output_shape|" : EndIf
      Reason = PmoFormUnknown(*Node, Text, Path)
      Mode = PmoEmitAttrS(*Node, "auto_pad", "NOTSET")
      If Reason = "" And Mode <> "NOTSET" And Mode <> "VALID" And Mode <> "SAME_UPPER" And Mode <> "SAME_LOWER"
        Reason = "auto_pad = " + Mode + " is not an ONNX padding mode; NOTSET, VALID, SAME_UPPER and SAME_LOWER are."
      EndIf
      If Reason = "" And Mode <> "NOTSET" And PmoFormPadsAllZero(*Node) = 0
        Reason = "auto_pad = " + Mode + " is combined with pads = " + PmoFormValueText(PmoFormAttribute(*Node, "pads")) + "; the specification computes the padding from auto_pad, so explicit pads must be absent or zero."
      EndIf
      *A = PmoFormAttribute(*Node, "output_shape")
      If Reason = "" And *A
        Reason = "output_shape = " + PmoFormValueText(*A) + " is not implemented by " + Path + "; give pads and output_padding, or auto_pad, instead."
      EndIf
      If Reason = "" And FixedShape = 0
        For Index = 0 To 4
          Text = StringField("pads|strides|dilations|kernel_shape|output_padding", Index + 1, "|")
          Value = 1 : If Text = "pads" : Value = 2 : EndIf
          If PmoEmitAttrListCount(*Node, Text) > Value
            Reason = Text + " = " + PmoFormValueText(PmoFormAttribute(*Node, Text)) + " describes more than one spatial axis; " + Path + " implements the one-dimensional " + Op + " only."
            Break
          EndIf
        Next
      EndIf

    Case "Resize"
      Reason = PmoFormUnknown(*Node, "|mode|coordinate_transformation_mode|nearest_mode|cubic_coeff_a|exclude_outside|extrapolation_value|antialias|keep_aspect_ratio_policy|axes|", Path)
      Mode = PmoEmitAttrS(*Node, "mode", "nearest")
      If Reason = "" And Mode <> "nearest" And Mode <> "linear"
        Reason = "mode = " + Mode + " is not implemented by " + Path + "; nearest and linear are."
      EndIf
      Text = PmoEmitAttrS(*Node, "nearest_mode", "round_prefer_floor")
      If Reason = "" And Mode = "nearest" And Text <> "floor"
        ; nearest_mode has no effect in linear mode, so it is checked for nearest only.
        Reason = "nearest_mode = " + Text + " is not implemented by " + Path + " for mode nearest; floor is."
        If PmoFormAttribute(*Node, "nearest_mode") = 0
          Reason = "nearest_mode is absent, so it takes the specification's default round_prefer_floor, which " + Path + " does not implement for mode nearest; floor is."
        EndIf
      EndIf
      Text = PmoEmitAttrS(*Node, "coordinate_transformation_mode", "half_pixel")
      If Reason = "" And Text <> "half_pixel" And Text <> "asymmetric"
        Reason = "coordinate_transformation_mode = " + Text + " is not implemented by " + Path + "; half_pixel and asymmetric are."
      EndIf
      ; cubic_coeff_a acts in cubic mode only, extrapolation_value with
      ; tf_crop_and_resize only, and roi (input 1) with tf_crop_and_resize only:
      ; none of those modes is accepted above, so their values change nothing.
      ; antialias 1 filters a linear downscale and is undefined for nearest (the
      ; reference implementation refuses it), so only 0 is accepted. The
      ; reference shows exclude_outside 1 changing nearest with half_pixel at
      ; the edges, so only 0 is accepted for it as well.
      If Reason = "" And PmoEmitAttrI(*Node, "antialias", 0) <> 0
        Reason = "antialias = " + Str(PmoEmitAttrI(*Node, "antialias", 0)) + " is not implemented by " + Path + "; antialias 0, the default, is."
      EndIf
      If Reason = "" And PmoEmitAttrI(*Node, "exclude_outside", 0) <> 0
        Reason = "exclude_outside = " + Str(PmoEmitAttrI(*Node, "exclude_outside", 0)) + " is not implemented by " + Path + "; exclude_outside 0, the default, is."
      EndIf
      Text = PmoEmitAttrS(*Node, "keep_aspect_ratio_policy", "stretch")
      If Reason = "" And Text <> "stretch" And PmoFormInputPresent(*Node, 3)
        Reason = "keep_aspect_ratio_policy = " + Text + " is not implemented by " + Path + " when sizes are given; stretch, the default, is."
      EndIf
      *A = PmoFormAttribute(*Node, "axes")
      If Reason = "" And *A
        Reason = "axes = " + PmoFormValueText(*A) + " is not implemented by " + Path + "; scales or sizes must give every axis."
      EndIf

    Case "ScatterND"
      Reason = PmoFormUnknown(*Node, "|reduction|", Path)
      Mode = PmoEmitAttrS(*Node, "reduction", "none")
      If Reason = "" And Mode <> "none" And Mode <> "add" And Mode <> "mul" And Mode <> "max" And Mode <> "min"
        Reason = "reduction = " + Mode + " is not an ONNX ScatterND reduction; none, add, mul, max and min are, and all five are implemented."
      EndIf

    Case "ReduceMean", "ReduceSum"
      Text = "|keepdims|noop_with_empty_axes|"
      If FixedShape : Text = "|keepdims|noop_with_empty_axes|axes|" : EndIf
      Reason = PmoFormUnknown(*Node, Text, Path)
      ; With axes given, noop_with_empty_axes has no effect, whatever its value.
      ; The runtime-dimension path reduces over any axes set, the empty one
      ; included (DReduceAxes, DReduceNoopEmpty); the fixed-shape path's
      ; kernel reduces one axis.
      If Reason = "" And FixedShape And PmoFormInputPresent(*Node, 1) = 0 And PmoFormAttribute(*Node, "axes") = 0
        If PmoEmitAttrI(*Node, "noop_with_empty_axes", 0)
          Reason = "axes is absent and noop_with_empty_axes = 1 selects the identity, which " + Path + " does not implement; give the axis to reduce."
        Else
          Reason = "axes is absent, which reduces over every axis; " + Path + " implements a reduction over the last axis, so give that axis."
        EndIf
      EndIf
  EndSelect
  ProcedureReturn Reason
EndProcedure

; The refusal as it is reported: operator, node name, then the reason.
Procedure.s PmoFormSentence(*Node.PmoOnnxNode, Reason.s)
  Protected Name.s = *Node\Name
  If Name = "" : Name = "(unnamed)" : EndIf
  ProcedureReturn *Node\Operation + " node " + Name + ": " + Reason
EndProcedure
