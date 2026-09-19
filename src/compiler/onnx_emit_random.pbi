; ============================================================================
; onnx_emit_random.pbi - the ONNX random operators in both lowering paths
; ----------------------------------------------------------------------------
; RandomUniform, RandomUniformLike, RandomNormal and RandomNormalLike
; (ai.onnx since version 1). The specification leaves the generator open;
; this compiler specifies it (runtime/tensor_random.pmi) so one seed gives
; the same FLOAT values on every target. Here: the validator's sentences,
; the emitted calls for the fixed-shape and runtime-dimension paths, the
; seed procedure, the manifest entry, and the comparison mode
; (--random-inputs) that turns every random node into a model input the
; caller fills.
;
; A random node is never constant: it is never folded and never scheduled
; with the bind-time constant work, because every execution draws again.
; ============================================================================

#PMO_RANDOM_ELEMENT_LIMIT = 4294967295

Procedure.s PmoRandomOutputName(*Node.PmoOnnxNode)
  If FirstElement(*Node\Outputs()) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Outputs()
EndProcedure

Procedure.i PmoRandomIsOp(Operation.s)
  ProcedureReturn Bool(Operation = "RandomUniform" Or Operation = "RandomUniformLike" Or
                       Operation = "RandomNormal" Or Operation = "RandomNormalLike")
EndProcedure

Procedure.i PmoRandomIsLike(Operation.s)
  ProcedureReturn Bool(Operation = "RandomUniformLike" Or Operation = "RandomNormalLike")
EndProcedure

Procedure.i PmoRandomIsNormal(Operation.s)
  ProcedureReturn Bool(Operation = "RandomNormal" Or Operation = "RandomNormalLike")
EndProcedure

Procedure.i PmoRandomAttr(*Node.PmoOnnxNode, Name.s)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn @*Node\Attributes() : EndIf
  Next
  ProcedureReturn 0
EndProcedure

Procedure.l PmoRandomBits(Value.f)
  ProcedureReturn PeekL(@Value)
EndProcedure

Procedure.s PmoRandomTypeName(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "FLOAT"
    Case 2 : ProcedureReturn "UINT8"
    Case 3 : ProcedureReturn "INT8"
    Case 4 : ProcedureReturn "UINT16"
    Case 5 : ProcedureReturn "INT16"
    Case 6 : ProcedureReturn "INT32"
    Case 7 : ProcedureReturn "INT64"
    Case 9 : ProcedureReturn "BOOL"
    Case 10 : ProcedureReturn "FLOAT16"
    Case 11 : ProcedureReturn "DOUBLE"
    Case 12 : ProcedureReturn "UINT32"
    Case 13 : ProcedureReturn "UINT64"
  EndSelect
  ProcedureReturn "type " + Str(ElementType)
EndProcedure

Procedure.s PmoRandomLabel(*Node.PmoOnnxNode, NodeIndex.i)
  ProcedureReturn "node " + Str(NodeIndex) + " (" + *Node\Operation + ")"
EndProcedure

; The claimed forms, and a sentence naming the operator, the attribute and
; the value for every form that is not. InputType is the element type of a
; Like node's input when the caller knows it, 0 when it does not.
Procedure.s PmoRandomValidate(*Node.PmoOnnxNode, NodeIndex.i, InputType.i)
  Protected Op.s = *Node\Operation
  Protected Label.s = PmoRandomLabel(*Node, NodeIndex)
  Protected Name.s
  Protected Allowed.s
  Protected Bits.l
  Protected Extent.q
  Protected Elements.q = 1
  Protected Axis.i
  Protected InputCount.i
  Protected *Attr.PmoOnnxAttribute
  If PmoRandomIsNormal(Op)
    Allowed = "|mean|scale|seed|dtype|"
  Else
    Allowed = "|low|high|seed|dtype|"
  EndIf
  If PmoRandomIsLike(Op) = 0 : Allowed + "shape|" : EndIf
  ForEach *Node\Attributes()
    Name = *Node\Attributes()\Name
    If FindString(Allowed, "|" + Name + "|") = 0
      ProcedureReturn Label + " has attribute " + Name + ", which the ONNX specification does not define for " + Op + "; nothing was emitted."
    EndIf
    ; An If chain, not a Select on the name: a string Select must not be
    ; left from inside a Case.
    If Name = "dtype"
      If *Node\Attributes()\AttributeType <> 0 And *Node\Attributes()\AttributeType <> 2
        ProcedureReturn Label + " attribute dtype is stored as attribute type " + Str(*Node\Attributes()\AttributeType) + "; the ONNX specification defines it as INT."
      EndIf
    ElseIf Name = "shape"
      If *Node\Attributes()\AttributeType <> 0 And *Node\Attributes()\AttributeType <> 7
        ProcedureReturn Label + " attribute shape is stored as attribute type " + Str(*Node\Attributes()\AttributeType) + "; the ONNX specification defines it as INTS."
      EndIf
    Else
      If *Node\Attributes()\AttributeType <> 0 And *Node\Attributes()\AttributeType <> 1
        ProcedureReturn Label + " attribute " + Name + " is stored as attribute type " + Str(*Node\Attributes()\AttributeType) + "; the ONNX specification defines it as FLOAT."
      EndIf
      If Name <> "seed"
        Bits = PmoRandomBits(*Node\Attributes()\FloatValue)
        If (Bits & $7F800000) = $7F800000
          ProcedureReturn Label + " attribute " + Name + " = " + StrF(*Node\Attributes()\FloatValue) + " is not a finite number; the generator needs a finite " + Name + "."
        EndIf
      EndIf
    EndIf
  Next
  *Attr = PmoRandomAttr(*Node, "dtype")
  If *Attr
    Select *Attr\IntegerValue
      Case 1
      Case 10, 11
        ProcedureReturn Label + " attribute dtype = " + Str(*Attr\IntegerValue) + " (" + PmoRandomTypeName(*Attr\IntegerValue) + ") is not implemented: this compiler generates FLOAT (dtype 1) random tensors only. Re-export the node with dtype 1, or follow it with a Cast."
      Default
        ProcedureReturn Label + " attribute dtype = " + Str(*Attr\IntegerValue) + " is not an output type the ONNX specification allows for " + Op + " (FLOAT16 10, FLOAT 1 or DOUBLE 11)."
    EndSelect
  ElseIf PmoRandomIsLike(Op) And InputType <> 0 And InputType <> 1
    ProcedureReturn Label + " has no dtype attribute, so its output takes its input's element type " + PmoRandomTypeName(InputType) + " (" + Str(InputType) + "); this compiler generates FLOAT (1) random tensors only. Add dtype = 1 to the node."
  EndIf
  ForEach *Node\Inputs()
    If *Node\Inputs() <> "" : InputCount + 1 : EndIf
  Next
  If PmoRandomIsLike(Op)
    If InputCount <> 1 Or ListSize(*Node\Inputs()) <> 1
      ProcedureReturn Label + " has " + Str(ListSize(*Node\Inputs())) + " inputs; " + Op + " takes exactly one, the tensor whose shape it copies."
    EndIf
  Else
    If ListSize(*Node\Inputs()) <> 0
      ProcedureReturn Label + " has " + Str(ListSize(*Node\Inputs())) + " inputs; " + Op + " takes none."
    EndIf
    *Attr = PmoRandomAttr(*Node, "shape")
    If *Attr = 0
      ProcedureReturn Label + " has no shape attribute, which the ONNX specification requires for " + Op + "."
    EndIf
    If ListSize(*Attr\Integers()) > 8
      ProcedureReturn Label + " attribute shape has " + Str(ListSize(*Attr\Integers())) + " extents; generated tensors carry at most eight axes."
    EndIf
    Axis = 0
    ForEach *Attr\Integers()
      Extent = *Attr\Integers()
      If Extent < 0
        ProcedureReturn Label + " attribute shape extent " + Str(Axis) + " = " + Str(Extent) + " is negative."
      EndIf
      If Extent > 0 And Elements > #PMO_RANDOM_ELEMENT_LIMIT / Extent
        ProcedureReturn Label + " attribute shape describes more than 4294967295 elements; the generator numbers elements with 32 bits."
      EndIf
      Elements * Extent
      Axis + 1
    Next
  EndIf
  If ListSize(*Node\Outputs()) <> 1 Or PmoRandomOutputName(*Node) = ""
    ProcedureReturn Label + " has " + Str(ListSize(*Node\Outputs())) + " outputs; " + Op + " has exactly one."
  EndIf
  ProcedureReturn ""
EndProcedure

; The element type of a named value, from the graph's declarations: inputs,
; value_info, outputs, initializers. 0 when the model does not say.
Procedure.i PmoRandomDeclaredType(*Model.PmoOnnxModel, Name.s)
  ForEach *Model\Graph\Inputs()
    If *Model\Graph\Inputs()\Name = Name And *Model\Graph\Inputs()\HasTensorType : ProcedureReturn *Model\Graph\Inputs()\ElementType : EndIf
  Next
  ForEach *Model\Graph\Values()
    If *Model\Graph\Values()\Name = Name And *Model\Graph\Values()\HasTensorType : ProcedureReturn *Model\Graph\Values()\ElementType : EndIf
  Next
  ForEach *Model\Graph\Outputs()
    If *Model\Graph\Outputs()\Name = Name And *Model\Graph\Outputs()\HasTensorType : ProcedureReturn *Model\Graph\Outputs()\ElementType : EndIf
  Next
  ForEach *Model\Graph\Initializers()
    If *Model\Graph\Initializers()\Name = Name : ProcedureReturn *Model\Graph\Initializers()\DataType : EndIf
  Next
  ProcedureReturn 0
EndProcedure

; Every random node of the top-level graph, before either path is chosen.
Procedure.s PmoRandomValidateModel(*Model.PmoOnnxModel)
  Protected Index.i
  Protected Message.s
  Protected InputType.i
  Protected Declared.i
  Protected *Attr.PmoOnnxAttribute
  ForEach *Model\Graph\Nodes()
    If PmoRandomIsOp(*Model\Graph\Nodes()\Operation) And (*Model\Graph\Nodes()\Domain = "" Or *Model\Graph\Nodes()\Domain = "ai.onnx")
      InputType = 0
      If PmoRandomIsLike(*Model\Graph\Nodes()\Operation) And FirstElement(*Model\Graph\Nodes()\Inputs())
        InputType = PmoRandomDeclaredType(*Model, *Model\Graph\Nodes()\Inputs())
      EndIf
      Message = PmoRandomValidate(@*Model\Graph\Nodes(), Index, InputType)
      If Message <> "" : ProcedureReturn Message : EndIf
      ; A declared output type other than FLOAT contradicts the node.
      FirstElement(*Model\Graph\Nodes()\Outputs())
      Declared = PmoRandomDeclaredType(*Model, *Model\Graph\Nodes()\Outputs())
      If Declared <> 0 And Declared <> 1
        ProcedureReturn PmoRandomLabel(@*Model\Graph\Nodes(), Index) + " output " + *Model\Graph\Nodes()\Outputs() + " is declared " + PmoRandomTypeName(Declared) + " (" + Str(Declared) + "); this compiler generates FLOAT (1) random tensors only."
      EndIf
    EndIf
    Index + 1
  Next
  ProcedureReturn ""
EndProcedure

Procedure.i PmoRandomCount(*Model.PmoOnnxModel)
  Protected Count.i
  ForEach *Model\Graph\Nodes()
    If PmoRandomIsOp(*Model\Graph\Nodes()\Operation) : Count + 1 : EndIf
  Next
  ProcedureReturn Count
EndProcedure

; The eight integer arguments every generated call carries after its tensors:
; node, hasSeed, seedBits, kind, p1Bits, p2Bits. Kind bit 0 is the
; distribution; bit 1 asks the runtime to require a FLOAT input because the
; node has no dtype attribute.
Procedure.s PmoRandomArguments(*Node.PmoOnnxNode, NodeIndex.i)
  Protected *Attr.PmoOnnxAttribute
  Protected Kind.i
  Protected HasSeed.i
  Protected SeedBits.l
  Protected P1.f
  Protected P2.f = 1.0
  If PmoRandomIsNormal(*Node\Operation)
    Kind = 1
    *Attr = PmoRandomAttr(*Node, "mean") : If *Attr : P1 = *Attr\FloatValue : EndIf
    *Attr = PmoRandomAttr(*Node, "scale") : If *Attr : P2 = *Attr\FloatValue : EndIf
  Else
    *Attr = PmoRandomAttr(*Node, "low") : If *Attr : P1 = *Attr\FloatValue : EndIf
    *Attr = PmoRandomAttr(*Node, "high") : If *Attr : P2 = *Attr\FloatValue : EndIf
  EndIf
  If PmoRandomIsLike(*Node\Operation) And PmoRandomAttr(*Node, "dtype") = 0 : Kind = Kind | 2 : EndIf
  *Attr = PmoRandomAttr(*Node, "seed")
  If *Attr : HasSeed = 1 : SeedBits = PmoRandomBits(*Attr\FloatValue) : EndIf
  ProcedureReturn Str(NodeIndex) + "," + Str(HasSeed) + "," + Str(SeedBits) + "," + Str(Kind) + "," +
                  Str(PmoRandomBits(P1)) + "," + Str(PmoRandomBits(P2))
EndProcedure

; Runtime-dimension call text. Output and input are tensor ids; AsInput is
; the comparison mode, where the node checks the tensor the caller supplied.
Procedure.s PmoRandomDynamicCall(*Node.PmoOnnxNode, NodeIndex.i, OutputId.s, InputId.s, AsInput.i)
  Protected Text.s
  Protected Axis.i
  Protected *Attr.PmoOnnxAttribute
  If PmoRandomIsLike(*Node\Operation)
    If AsInput : ProcedureReturn "DRandomFed(" + OutputId + "," + InputId + ",0)" : EndIf
    ProcedureReturn "DRandomLike(" + OutputId + "," + InputId + "," + PmoRandomArguments(*Node, NodeIndex) + ")"
  EndIf
  *Attr = PmoRandomAttr(*Node, "shape")
  If *Attr
    ForEach *Attr\Integers()
      Text + "PmRandomDims(" + Str(Axis) + ")=" + Str(*Attr\Integers()) + Chr(10)
      Axis + 1
    Next
  EndIf
  If AsInput : ProcedureReturn Text + "DRandomFed(" + OutputId + ",0," + Str(Axis) + ")" : EndIf
  ProcedureReturn Text + "DRandomShape(" + OutputId + "," + Str(Axis) + "," + PmoRandomArguments(*Node, NodeIndex) + ")"
EndProcedure

; Fixed-shape call text: the destination address and element count are
; known when the source is written.
Procedure.s PmoRandomFixedCall(*Node.PmoOnnxNode, NodeIndex.i, Destination.s, Count.q)
  Protected Arguments.s = PmoRandomArguments(*Node, NodeIndex)
  ; The FLOAT-input requirement was settled from the model's declarations.
  Protected Kind.s = StringField(Arguments, 4, ",")
  Arguments = StringField(Arguments, 1, ",") + "," + StringField(Arguments, 2, ",") + "," + StringField(Arguments, 3, ",") + "," +
              Str(Val(Kind) & 1) + "," + StringField(Arguments, 5, ",") + "," + StringField(Arguments, 6, ",")
  ProcedureReturn "If PmRandomNodeBits(" + Destination + ", " + Str(Count) + ", " + ReplaceString(Arguments, ",", ", ") + ") = 0 : PmOnnxRuntimeOk = 0 : EndIf"
EndProcedure

; Comparison mode on the fixed-shape path: each random node leaves the
; schedule and its output becomes a model input, after the graph's own
; inputs and in node order.
Procedure.i PmoRandomNodesToInputs(*Ir.PmoIrModel)
  Protected Moved.i
  ForEach *Ir\Nodes()
    If PmoRandomIsOp(*Ir\Nodes()\Node\Operation)
      AddElement(*Ir\Inputs())
      *Ir\Inputs() = PmoRandomOutputName(*Ir\Nodes()\Node)
      DeleteElement(*Ir\Nodes())
      Moved + 1
    EndIf
  Next
  LastElement(*Ir\Inputs())
  ProcedureReturn Moved
EndProcedure

Procedure.s PmoRandomJsonText(Text.s)
  Text = ReplaceString(Text, "\", "\\")
  Text = ReplaceString(Text, Chr(34), "\" + Chr(34))
  ProcedureReturn Chr(34) + Text + Chr(34)
EndProcedure

Procedure.s PmoRandomHex(Value.l)
  ProcedureReturn "0x" + RSet(Hex(Value, #PB_Long), 8, "0")
EndProcedure

; One manifest object, as JSON text, shared by both manifests.
; FirstInput is the model-input index of the first random node in
; comparison mode.
Procedure.s PmoRandomManifestText(*Model.PmoOnnxModel, AsInputs.i, FirstInput.i, SeedProcedure.s)
  Protected Text.s
  Protected Index.i
  Protected Nodes.s
  Protected InputIndex.i = FirstInput
  Protected *Attr.PmoOnnxAttribute
  Protected P1.f
  Protected P2.f
  Protected First.s
  Protected Second.s
  If AsInputs
    Text = "{" + PmoRandomJsonText("mode") + ":" + PmoRandomJsonText("inputs") + ","
    Text + PmoRandomJsonText("note") + ":" + PmoRandomJsonText("--random-inputs: every random node is a model input the caller fills; nothing is generated and the inputs differ from the ONNX model's") + ","
  Else
    Text = "{" + PmoRandomJsonText("mode") + ":" + PmoRandomJsonText("generated") + ","
    Text + PmoRandomJsonText("set_seed_procedure") + ":" + PmoRandomJsonText(SeedProcedure) + ","
    Text + PmoRandomJsonText("model_seed_default") + ":0,"
    Text + PmoRandomJsonText("model_seed_bits") + ":32,"
    Text + PmoRandomJsonText("request_numbering") + ":" + PmoRandomJsonText("0 after the model is bound or its seed is set; one more for every execution that starts the graph") + ","
  EndIf
  Text + PmoRandomJsonText("generator") + ":" + PmoRandomJsonText("threefry2x32-20") + ","
  Text + PmoRandomJsonText("node_key") + ":" + PmoRandomJsonText("threefry(key=(S,0), counter=(node index, d)); S = seed attribute bits with d=1, else model seed with d=0") + ","
  Text + PmoRandomJsonText("element") + ":" + PmoRandomJsonText("threefry(key=node key, counter=(element index, request number))") + ","
  Text + PmoRandomJsonText("uniform_transform") + ":" + PmoRandomJsonText("low + (high - low) * ((w0 >> 8) * 2^-24)") + ","
  Text + PmoRandomJsonText("normal_transform") + ":" + PmoRandomJsonText("Box-Muller cosine branch in basic binary32 operations; |z| <= 5.77") + ","
  Text + PmoRandomJsonText("contract") + ":" + PmoRandomJsonText("the same FLOAT values for one seed, request number and shape on every target") + ","
  ForEach *Model\Graph\Nodes()
    If PmoRandomIsOp(*Model\Graph\Nodes()\Operation)
      P1 = 0.0 : P2 = 1.0
      If PmoRandomIsNormal(*Model\Graph\Nodes()\Operation)
        First = "mean" : Second = "scale"
      Else
        First = "low" : Second = "high"
      EndIf
      *Attr = PmoRandomAttr(@*Model\Graph\Nodes(), First) : If *Attr : P1 = *Attr\FloatValue : EndIf
      *Attr = PmoRandomAttr(@*Model\Graph\Nodes(), Second) : If *Attr : P2 = *Attr\FloatValue : EndIf
      If Nodes <> "" : Nodes + "," : EndIf
      Nodes + "{" + PmoRandomJsonText("index") + ":" + Str(Index) + ","
      Nodes + PmoRandomJsonText("name") + ":" + PmoRandomJsonText(*Model\Graph\Nodes()\Name) + ","
      Nodes + PmoRandomJsonText("op") + ":" + PmoRandomJsonText(*Model\Graph\Nodes()\Operation) + ","
      Nodes + PmoRandomJsonText("output") + ":" + PmoRandomJsonText(PmoRandomOutputName(@*Model\Graph\Nodes())) + ","
      Nodes + PmoRandomJsonText(First) + ":" + StrF(P1, 9) + "," + PmoRandomJsonText(First + "_bits") + ":" + PmoRandomJsonText(PmoRandomHex(PmoRandomBits(P1))) + ","
      Nodes + PmoRandomJsonText(Second) + ":" + StrF(P2, 9) + "," + PmoRandomJsonText(Second + "_bits") + ":" + PmoRandomJsonText(PmoRandomHex(PmoRandomBits(P2))) + ","
      *Attr = PmoRandomAttr(@*Model\Graph\Nodes(), "seed")
      If *Attr
        Nodes + PmoRandomJsonText("seed") + ":" + PmoRandomJsonText("attribute") + "," + PmoRandomJsonText("seed_bits") + ":" + PmoRandomJsonText(PmoRandomHex(PmoRandomBits(*Attr\FloatValue))) + ","
      Else
        Nodes + PmoRandomJsonText("seed") + ":" + PmoRandomJsonText("model") + ","
      EndIf
      If AsInputs
        Nodes + PmoRandomJsonText("input_index") + ":" + Str(InputIndex) + ","
        InputIndex + 1
      EndIf
      Nodes + PmoRandomJsonText("dtype") + ":1}"
    EndIf
    Index + 1
  Next
  ProcedureReturn Text + PmoRandomJsonText("nodes") + ":[" + Nodes + "]}"
EndProcedure

; Copies a parsed JSON value into a value of the runtime-dimension manifest,
; which is built with the JSON library rather than written as text.
Procedure PmoRandomJsonCopy(Source.i, Destination.i)
  Protected Target.i
  Protected Index.i
  Protected Number.d
  Select JSONType(Source)
    Case #PB_JSON_Object
      Target = SetJSONObject(Destination)
      If ExamineJSONMembers(Source)
        While NextJSONMember(Source)
          PmoRandomJsonCopy(JSONMemberValue(Source), AddJSONMember(Target, JSONMemberKey(Source)))
        Wend
      EndIf
    Case #PB_JSON_Array
      Target = SetJSONArray(Destination)
      For Index = 0 To JSONArraySize(Source) - 1
        PmoRandomJsonCopy(GetJSONElement(Source, Index), AddJSONElement(Target))
      Next
    Case #PB_JSON_String
      SetJSONString(Destination, GetJSONString(Source))
    Case #PB_JSON_Boolean
      SetJSONBoolean(Destination, GetJSONBoolean(Source))
    Case #PB_JSON_Number
      Number = GetJSONDouble(Source)
      If Number = Round(Number, #PB_Round_Down) And Abs(Number) < 9007199254740992.0
        SetJSONQuad(Destination, GetJSONQuad(Source))
      Else
        SetJSONDouble(Destination, Number)
      EndIf
    Default
      SetJSONNull(Destination)
  EndSelect
EndProcedure

; Adds "random" to a manifest object built with the JSON library. Returns
; #False only if the text did not parse, which would be a defect here.
Procedure.i PmoRandomManifestMember(Object.i, Text.s)
  Protected Parsed.i = ParseJSON(#PB_Any, Text)
  If Parsed = 0 : ProcedureReturn #False : EndIf
  PmoRandomJsonCopy(JSONValue(Parsed), AddJSONMember(Object, "random"))
  FreeJSON(Parsed)
  ProcedureReturn #True
EndProcedure
