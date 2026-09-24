; ============================================================================
; onnx_dynamic_norm_small.pbi - runtime-dimension forms of
; InstanceNormalization, TopK, ScatterElements, ReduceMax, ReduceProd, Not,
; Identity and Pad
; ----------------------------------------------------------------------------
; Validation and call text for these operators on the runtime-dimension path.
; Every form the ONNX specification (https://onnx.ai/onnx/operators/) allows
; for them up to opset 20 is either lowered to a call into
; runtime/tensor_dynamic_norm_small_*.p* or refused with one sentence naming
; the operator, the attribute or input, and the value.
;
; OPSET FLOORS. Each operator is accepted from the oldest ai.onnx opset whose
; definition of it is the one these kernels compute, through 20:
;   InstanceNormalization 6   (InstanceNormalization-1 carried consumed_inputs)
;   TopK 10                   (TopK-1 took k as an attribute; TopK-10 has no
;                              largest/sorted and leaves ties unordered, so
;                              the index order computed here is one it allows)
;   ScatterElements 11        (reduction add/mul from 16, max/min from 18)
;   ReduceMax, ReduceProd 1   (axes attribute through 17, axes input from 18)
;   Not 1, Identity 1         (tensors; sequences and optionals are refused)
;   Pad 11                    (Pad-2 took pads as an attribute; axes input
;                              from 18, mode wrap from 19)
; Every other operator keeps the runtime-dimension path's opset 20 surface.
; ============================================================================

Global PmdNsOpset.i
Global *PmdNsModel.PmoOnnxModel

Procedure.i PmdNsOwns(Operation.s)
  ProcedureReturn PmoNsOwns(Operation)
EndProcedure

Procedure.i PmdNsFloor(Operation.s)
  If PmoNsOwns(Operation) : ProcedureReturn PmoNsFloor(Operation, #False) : EndIf
  If PmoOpsOwns(Operation) : ProcedureReturn PmoOpsFloor(Operation) : EndIf
  ; Every other operator: the audited floors in onnx_control.pbi (20 when unlisted).
  ProcedureReturn PmcOperatorFloor(Operation)
EndProcedure

; The ai.onnx opset the model imports, or 0 when it imports none.
Procedure.i PmdNsModelOpset(*Model.PmoOnnxModel)
  Protected Version.i
  ForEach *Model\Opsets()
    If *Model\Opsets()\Domain = "" Or *Model\Opsets()\Domain = "ai.onnx"
      Version = *Model\Opsets()\Version
    EndIf
  Next
  ProcedureReturn Version
EndProcedure

; "" when every node of the graph is defined, at the model's ai.onnx opset,
; the way this lowering computes it; otherwise the sentence that refuses it.
Procedure.s PmdNsOpsetRefusal(*Model.PmoOnnxModel)
  Protected Version.i = PmdNsModelOpset(*Model), Operation.s, Floor.i, Name.s
  *PmdNsModel = *Model
  If Version = 20 : ProcedureReturn "" : EndIf
  If Version > 20 Or Version < 1
    ProcedureReturn "The model imports ai.onnx opset " + Str(Version) + "; runtime-dimension emission implements the opset 20 operator definitions only."
  EndIf
  ForEach *Model\Graph\Nodes()
    ; a node the Scan lowering added implements the Scan, whose floor was audited
    If FindMapElement(PmcLowered(), Str(@*Model\Graph\Nodes())) : Continue : EndIf
    Operation = *Model\Graph\Nodes()\Operation
    Floor = PmdNsFloor(Operation)
    If Version < Floor
      If PmdNsOwns(Operation)
        Name = *Model\Graph\Nodes()\Name : If Name = "" : Name = "(unnamed)" : EndIf
        ProcedureReturn Operation + " node " + Name + ": the model imports ai.onnx opset " + Str(Version) +
                        ", whose definition of " + Operation + " differs from the one runtime-dimension emission implements, which is current from opset " + Str(Floor) + " through 20."
      EndIf
      ProcedureReturn "The model imports ai.onnx opset " + Str(Version) + "; runtime-dimension emission implements the opset 20 definition of " + Operation + " only."
    EndIf
  Next
  ProcedureReturn ""
EndProcedure

Procedure.s PmdNsValueText(*Attribute.PmoOnnxAttribute)
  Protected Text.s
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
  If *Attribute\StringValue <> "" : ProcedureReturn *Attribute\StringValue : EndIf
  If *Attribute\HasTensor : ProcedureReturn "a tensor" : EndIf
  If *Attribute\AttributeType = 1 : ProcedureReturn StrF(*Attribute\FloatValue) : EndIf
  ProcedureReturn Str(*Attribute\IntegerValue)
EndProcedure

Procedure.i PmdNsInputPresent(*Node.PmoOnnxNode, Index.i)
  If SelectElement(*Node\Inputs(), Index) And *Node\Inputs() <> "" : ProcedureReturn #True : EndIf
  ProcedureReturn #False
EndProcedure

; The declared ONNX element type of a top-level tensor, or 0 when the model
; does not declare it (an intermediate without value_info, a subgraph value).
Procedure.i PmdNsDeclaredType(Name.s)
  If *PmdNsModel = 0 Or Name = "" : ProcedureReturn 0 : EndIf
  ForEach *PmdNsModel\Graph\Initializers()
    If *PmdNsModel\Graph\Initializers()\Name = Name : ProcedureReturn *PmdNsModel\Graph\Initializers()\DataType : EndIf
  Next
  ForEach *PmdNsModel\Graph\Inputs()
    If *PmdNsModel\Graph\Inputs()\Name = Name : ProcedureReturn *PmdNsModel\Graph\Inputs()\ElementType : EndIf
  Next
  ForEach *PmdNsModel\Graph\Values()
    If *PmdNsModel\Graph\Values()\Name = Name : ProcedureReturn *PmdNsModel\Graph\Values()\ElementType : EndIf
  Next
  ProcedureReturn 0
EndProcedure

Procedure.s PmdNsTypeName(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "FLOAT"
    Case 2 : ProcedureReturn "UINT8"
    Case 3 : ProcedureReturn "INT8"
    Case 4 : ProcedureReturn "UINT16"
    Case 5 : ProcedureReturn "INT16"
    Case 6 : ProcedureReturn "INT32"
    Case 7 : ProcedureReturn "INT64"
    Case 8 : ProcedureReturn "STRING"
    Case 9 : ProcedureReturn "BOOL"
    Case 10 : ProcedureReturn "FLOAT16"
    Case 11 : ProcedureReturn "DOUBLE"
    Case 12 : ProcedureReturn "UINT32"
    Case 13 : ProcedureReturn "UINT64"
    Case 16 : ProcedureReturn "BFLOAT16"
  EndSelect
  ProcedureReturn "element type " + Str(ElementType)
EndProcedure

; "" when input Index of the node is undeclared or of a type in Allowed
; ("|1|7|"), else the sentence's reason naming the type and the types implemented.
Procedure.s PmdNsTypeReason(*Node.PmoOnnxNode, Index.i, Label.s, Allowed.s)
  Protected Declared.i = PmdNsDeclaredType(PmoEmitInput(*Node, Index)), Names.s, i.i
  If Declared = 0 Or FindString(Allowed, "|" + Str(Declared) + "|") : ProcedureReturn "" : EndIf
  For i = 2 To CountString(Allowed, "|")
    If i = CountString(Allowed, "|") And i > 2
      Names + " and "
    ElseIf Names <> ""
      Names + ", "
    EndIf
    Names + PmdNsTypeName(Val(StringField(Allowed, i, "|")))
  Next
  ProcedureReturn "input " + Label + " has element type " + PmdNsTypeName(Declared) + "; runtime-dimension emission implements " +
                  *Node\Operation + " for " + Names + "."
EndProcedure

Procedure.i PmdNsNamedOutputs(*Node.PmoOnnxNode)
  Protected Count.i
  ForEach *Node\Outputs()
    If *Node\Outputs() <> "" : Count + 1 : EndIf
  Next
  ProcedureReturn Count
EndProcedure

; Returns 1 when the node's form is implemented, else 0 with PmoDynamicError
; holding the sentence. The reason is decided first and reported once, after
; every Select has closed.
Procedure.i PmdNsValidate(*Node.PmoOnnxNode)
  Protected Op.s = *Node\Operation, Allowed.s = "|", Reason.s, Mode.s, Name.s
  Select Op
    Case "InstanceNormalization"
      Allowed = "|epsilon|"
      If PmdNsInputPresent(*Node, 1) = 0 Or PmdNsInputPresent(*Node, 2) = 0
        Reason = "inputs scale and B are both required by the specification."
      EndIf
    Case "TopK"
      Allowed = "|axis|largest|sorted|"
      If PmdNsOpset < 11 : Allowed = "|axis|" : EndIf
      If PmdNsInputPresent(*Node, 1) = 0
        Reason = "input K is required; the k attribute belongs to TopK before opset 10."
      EndIf
    Case "Scatter"
      Allowed = "|axis|"
      If PmdNsOpset > 10 : Reason = "Scatter is deprecated from opset 11 (use ScatterElements), and the model imports opset " + Str(PmdNsOpset) + "." : EndIf
    Case "ScatterElements"
      Allowed = "|axis|reduction|"
      Mode = PmoEmitAttrS(*Node, "reduction", "none")
      If Mode <> "none" And Mode <> "add" And Mode <> "mul" And Mode <> "max" And Mode <> "min"
        Reason = "attribute reduction = " + Mode + " is not a ScatterElements reduction; none, add, mul, max and min are."
      ElseIf (Mode = "add" Or Mode = "mul") And PmdNsOpset < 16
        Reason = "attribute reduction = " + Mode + " is not part of ScatterElements before opset 16, and the model imports opset " + Str(PmdNsOpset) + "."
      ElseIf (Mode = "max" Or Mode = "min") And PmdNsOpset < 18
        Reason = "attribute reduction = " + Mode + " is not part of ScatterElements before opset 18, and the model imports opset " + Str(PmdNsOpset) + "."
      EndIf
    Case "ReduceMax", "ReduceProd"
      If PmdNsOpset >= 18
        Allowed = "|keepdims|noop_with_empty_axes|"
      Else
        Allowed = "|axes|keepdims|"
        If PmdNsInputPresent(*Node, 1)
          Reason = "input axes is not part of " + Op + " before opset 18, and the model imports opset " + Str(PmdNsOpset) + "; axes is an attribute there."
        EndIf
      EndIf
    Case "Pad"
      Allowed = "|mode|"
      Mode = PmoEmitAttrS(*Node, "mode", "constant")
      If Mode <> "constant" And Mode <> "reflect" And Mode <> "edge" And Mode <> "wrap"
        Reason = "attribute mode = " + Mode + " is not a Pad mode; constant, reflect, edge and wrap are."
      ElseIf Mode = "wrap" And PmdNsOpset < 19
        Reason = "attribute mode = wrap is not part of Pad before opset 19, and the model imports opset " + Str(PmdNsOpset) + "."
      ElseIf PmdNsInputPresent(*Node, 3) And PmdNsOpset < 18
        Reason = "input axes is not part of Pad before opset 18, and the model imports opset " + Str(PmdNsOpset) + "."
      ElseIf PmdNsInputPresent(*Node, 1) = 0
        Reason = "input pads is required."
      EndIf
  EndSelect
  If Reason = ""
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
  EndIf
  If Reason = "" And PmdNsInputPresent(*Node, 0) = 0
    Reason = "its first input is required."
  EndIf
  ; Element types the model declares; an undeclared one is checked when it runs.
  If Reason = ""
    Select Op
      Case "InstanceNormalization" : Reason = PmdNsTypeReason(*Node, 0, "input", "|1|")
      Case "TopK" : Reason = PmdNsTypeReason(*Node, 0, "X", "|1|6|7|")
      Case "ScatterElements", "Scatter"
        Reason = PmdNsTypeReason(*Node, 0, "data", "|1|6|7|9|")
        If Reason = "" : Reason = PmdNsTypeReason(*Node, 1, "indices", "|6|7|") : EndIf
      Case "ReduceMax" : Reason = PmdNsTypeReason(*Node, 0, "data", "|1|6|7|9|")
      Case "ReduceProd" : Reason = PmdNsTypeReason(*Node, 0, "data", "|1|6|7|")
      Case "Not" : Reason = PmdNsTypeReason(*Node, 0, "X", "|9|")
      Case "Identity" : Reason = PmdNsTypeReason(*Node, 0, "input", "|1|3|6|7|9|")
      Case "Pad" : Reason = PmdNsTypeReason(*Node, 0, "data", "|1|6|7|9|")
    EndSelect
  EndIf
  If Reason = ""
    If Op = "TopK"
      If ListSize(*Node\Outputs()) <> 2 : Reason = "it declares " + Str(ListSize(*Node\Outputs())) + " outputs; TopK has Values and Indices." : EndIf
    ElseIf ListSize(*Node\Outputs()) <> 1 Or PmdNsNamedOutputs(*Node) <> 1
      Reason = "it declares " + Str(ListSize(*Node\Outputs())) + " outputs; " + Op + " has one."
    EndIf
  EndIf
  If Reason = "" : ProcedureReturn 1 : EndIf
  Name = *Node\Name : If Name = "" : Name = "(unnamed)" : EndIf
  PmoDynamicError = Op + " node " + Name + ": " + Reason
  ProcedureReturn 0
EndProcedure

Procedure.s PmdNsId(Map Ids.i(), Name.s)
  If Name = "" : ProcedureReturn "0" : EndIf
  If FindMapElement(Ids(), Name) = 0 : PmoDynamicError = "Tensor has no producer: " + Name : ProcedureReturn "0" : EndIf
  ProcedureReturn Str(Ids())
EndProcedure

; The call text for one validated node.
Procedure.s PmdNsCall(*Node.PmoOnnxNode, Map Ids.i())
  Protected Op.s = *Node\Operation, Call.s, Mode.s, Axes.s, Code.i
  Protected Dim a.s(3)
  Protected i.i, y0.s, y1.s
  For i = 0 To 3 : a(i) = PmdNsId(Ids(), PmoEmitInput(*Node, i)) : Next
  y0 = PmdNsId(Ids(), PmoEmitOutput(*Node, 0)) : y1 = PmdNsId(Ids(), PmoEmitOutput(*Node, 1))
  Select Op
    Case "InstanceNormalization"
      Call = "DInstanceNorm(" + y0 + "," + a(0) + "," + a(1) + "," + a(2) + "," + PmoEmitFloat(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ")"
    Case "TopK"
      Call = "DTopK(" + y0 + "," + y1 + "," + a(0) + "," + a(1) + "," + Str(PmoEmitAttrI(*Node, "axis", -1)) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "largest", 1) <> 0)) + "," + Str(Bool(PmoEmitAttrI(*Node, "sorted", 1) <> 0)) + ")"
    Case "ScatterElements", "Scatter"
      Mode = PmoEmitAttrS(*Node, "reduction", "none")
      Code = 0
      If Mode = "add" : Code = 1 : ElseIf Mode = "mul" : Code = 2 : ElseIf Mode = "max" : Code = 3 : ElseIf Mode = "min" : Code = 4 : EndIf
      Call = "DScatterElements(" + y0 + "," + a(0) + "," + a(1) + "," + a(2) + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," + Str(Code) + ")"
    Case "ReduceMax", "ReduceProd"
      For i = 0 To PmoEmitAttrListCount(*Node, "axes") - 1
        If i : Axes + "," : EndIf
        Axes + Str(PmoEmitAttrListI(*Node, "axes", i, 0))
      Next
      Call = "DReduceSelect(" + y0 + "," + a(0) + "," + a(1) + "," + Chr(34) + Axes + Chr(34) + "," +
             Str(Bool(PmoEmitAttrI(*Node, "keepdims", 1) <> 0)) + "," + Str(Bool(PmoEmitAttrI(*Node, "noop_with_empty_axes", 0) <> 0)) + "," + Str(Bool(Op = "ReduceProd")) + ")"
    Case "Not"
      Call = "DNot(" + y0 + "," + a(0) + ")"
    Case "Identity"
      Call = "DIdentity(" + y0 + "," + a(0) + ")"
    Case "Pad"
      Mode = PmoEmitAttrS(*Node, "mode", "constant")
      Code = 0
      If Mode = "reflect" : Code = 1 : ElseIf Mode = "edge" : Code = 2 : ElseIf Mode = "wrap" : Code = 3 : EndIf
      If PmdNsInputPresent(*Node, 3)
        Call = "DPadAxes(" + y0 + "," + a(0) + "," + a(1) + "," + a(2) + "," + a(3) + "," + Str(Code) + ")"
      Else
        Call = "DPad(" + y0 + "," + a(0) + "," + a(1) + "," + a(2) + "," + Str(Code) + ")"
      EndIf
  EndSelect
  ProcedureReturn Call
EndProcedure
