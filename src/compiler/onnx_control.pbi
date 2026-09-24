; ============================================================================
; onnx_control.pbi - If, Loop and the sequence operators on the
; runtime-dimension path
; ----------------------------------------------------------------------------
; Operator semantics are taken from the ONNX operator specification
; (https://onnx.ai/onnx/operators/): If-11/13/16/19, Loop-11/13/16/19,
; SequenceEmpty-11, SequenceConstruct-11, SequenceInsert-11, SequenceAt-11,
; SequenceLength-11, SequenceErase-11, SplitToSequence-11 and
; ConcatFromSequence-11.
;
; Three stages, all on the runtime-dimension path:
;
; 1. PmcPrepareModel, straight after loading. Constant nodes become
;    initializers; the opset-11 spelling of Squeeze/Unsqueeze (an axes
;    attribute) becomes the opset-13 spelling; every name a subgraph defines
;    that is already defined elsewhere is renamed, and every reference is
;    resolved innermost scope first, so afterwards each value has exactly one
;    definition; subgraph initializers move to the top-level graph; and every
;    value is checked to be used as the kind (tensor or sequence) it is.
; 2. PmdCall hands If, Loop and the sequence operators to PmcCall, which
;    returns a placeholder: their text depends on last-use information the
;    main schedule has not computed yet.
; 3. PmcFinish (onnx_control_emit.pbi), once last uses are known. Subgraphs
;    become generated procedures (PmModelGraph<k>), their values get tensor ids, the
;    placeholders become the inline control code, and a value is MOVED rather
;    than copied only when nothing can read it afterwards.
;
; The generated code calls the runtime in tensor_control_windows.pbi or
; tensor_control_portable.pmi, which are exported only for a model that uses
; these operators, so every other model's source closure is unchanged.
; ============================================================================

#PMC_MAX_DEPTH = 16
#PMC_KIND_TENSOR = 1
#PMC_KIND_SEQUENCE = 2

Structure PmcScope
  Map Names.s()
EndStructure

; One subgraph as emitted: its procedure, its value ids and which of those
; values belong to it (so the caller knows what it may move).
Structure PmcGraphInfo
  Proc.s
  NodeCount.i
  List InputIds.s()
  List OutputIds.s()
  Map Local.i()
  Map LastUse.i()
EndStructure

Global PmcError.s
Global PmcUsed.i
Global PmcOpset.i
Global PmcBodyNodeCount.i
Global PmcNextNode.i
Global PmcNextGraph.i
Global PmcRenames.i
Global NewList PmcBuffers.i()
Global NewList PmcGlobalLines.s()
Global NewList PmcProcedureLines.s()
Global NewMap PmcKind.i()
Global NewMap PmcSequenceElement.i()
; Nodes of other operators this file emits because of what they carry: an
; Identity whose input is a sequence.
Global NewMap PmcOwned.i()

Procedure.i PmcFail(Message.s)
  If PmcError = "" : PmcError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.s PmcLabel(*Node.PmoOnnxNode)
  If *Node\Name <> ""
    ProcedureReturn *Node\Operation + " node '" + *Node\Name + "'"
  EndIf
  ProcedureReturn *Node\Operation + " node (unnamed)"
EndProcedure

Procedure.i PmcIsControl(Operation.s)
  ProcedureReturn Bool(Operation = "If" Or Operation = "Loop")
EndProcedure

Procedure.i PmcIsSequenceOp(Operation.s)
  ProcedureReturn Bool(FindString("|SequenceEmpty|SequenceConstruct|SequenceInsert|SequenceAt|SequenceLength|SequenceErase|SplitToSequence|ConcatFromSequence|", "|" + Operation + "|"))
EndProcedure

; Control flow and sequences always run in the request stage: their result
; is decided per request, and bind-stage values must be plain tensors.
Procedure.i PmcRequestStage(Operation.s)
  ProcedureReturn Bool(PmcIsControl(Operation) Or PmcIsSequenceOp(Operation))
EndProcedure

Procedure.i PmcSequenceIdentity(*Node.PmoOnnxNode)
  ProcedureReturn Bool(FindMapElement(PmcOwned(), Str(*Node)) <> 0)
EndProcedure

; Every node this file emits: the control and sequence operators by name, and
; an Identity that carries a sequence.
Procedure.i PmcOwnsNode(*Node.PmoOnnxNode)
  ProcedureReturn Bool(PmcRequestStage(*Node\Operation) Or PmcSequenceIdentity(*Node))
EndProcedure

; OPSET FLOORS. The runtime-dimension path accepts ai.onnx 11 to 20, and each
; node must also meet its operator's floor. of the runtime-dimension operators this lane audited: the
; oldest ai.onnx opset whose definition is the one the kernel computes, when
; the older attribute spellings are refused by the attribute checks (the
; Squeeze/Unsqueeze axes attribute is rewritten above instead). Where a
; version differs only in the element types it admits, the older version is
; the floor. Operators this table does not name keep the opset 20 surface.
;   Add Sub Mul Div Pow Equal Greater Less And   7   (numpy broadcasting from 7)
;   GreaterOrEqual                               12  (does not exist before 12)
;   Exp Log Sqrt Abs Neg Floor Sigmoid Tanh      6
;   LeakyRelu                                    6
;   Sin Cos Atan                                 7
;   Cast                                         6   (saturate is refused)
;   Reshape                                      5
;   Shape Transpose MatMul                       1   (Shape start/end refused)
;   Expand                                       8
;   ConstantOfShape Where NonZero                9
;   Slice                                        10  (starts/ends as inputs)
;   Gather Concat Range Round CumSum ScatterND   11  (negative axes/indices)
;   Squeeze Unsqueeze Clip Resize                11  (Clip min/max inputs;
;                                                    Resize-10 has no roi)
;   Gemm                                         11  (optional C)
;   ReduceMean ReduceSum                         11  (an axes attribute is refused)
;   Softmax                                      13  (Softmax-11 flattens
;                                                    around the axis instead)
;   LSTM                                         7   (layout is refused)
;   LayerNormalization STFT                      17  (do not exist before 17)
;   If Loop and the sequence operators           11
Procedure.i PmcOpsetAccepted(Version.q)
  ProcedureReturn Bool(Version >= 11 And Version <= 20)
EndProcedure

Procedure.i PmcOperatorFloor(Operation.s)
  Select Operation
    Case "Shape", "Transpose", "MatMul" : ProcedureReturn 1
    Case "Reshape" : ProcedureReturn 5
    Case "Exp", "Log", "Sqrt", "Abs", "Neg", "Floor", "Sigmoid", "Tanh", "LeakyRelu", "Cast" : ProcedureReturn 6
    Case "Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "Less", "And", "Sin", "Cos", "Atan", "LSTM" : ProcedureReturn 7
    Case "Expand" : ProcedureReturn 8
    Case "Multinomial" : ProcedureReturn 7
    Case "ConstantOfShape", "Where", "NonZero" : ProcedureReturn 9
    Case "Slice" : ProcedureReturn 10
    Case "Gather", "Concat", "Range", "Round", "CumSum", "ScatterND", "Squeeze", "Unsqueeze", "Clip", "Resize", "Gemm",
         "ReduceMean", "ReduceSum", "If", "Loop", "SequenceEmpty", "SequenceConstruct", "SequenceInsert", "SequenceAt",
         "SequenceLength", "SequenceErase", "SplitToSequence", "ConcatFromSequence" : ProcedureReturn 11
    Case "GreaterOrEqual" : ProcedureReturn 12
    Case "Softmax" : ProcedureReturn 13
    Case "Bernoulli" : ProcedureReturn 15
    Case "LayerNormalization", "STFT" : ProcedureReturn 17
  EndSelect
  ProcedureReturn 20
EndProcedure

; The floor of any operator: the owning lane's table where one exists, else the
; table above.
Procedure.i PmcNodeFloor(Operation.s)
  CompilerIf Defined(PmoNsOwns, #PB_Procedure)
    If PmoNsOwns(Operation) : ProcedureReturn PmoNsFloor(Operation, #False) : EndIf
  CompilerEndIf
  CompilerIf Defined(PmoOpsOwns, #PB_Procedure)
    If PmoOpsOwns(Operation) : ProcedureReturn PmoOpsFloor(Operation) : EndIf
  CompilerEndIf
  ProcedureReturn PmcOperatorFloor(Operation)
EndProcedure

; Every node, subgraphs included, is checked against its floor.
Procedure.i PmcAuditNode(*Node.PmoOnnxNode)
  Protected Floor.i = PmcNodeFloor(*Node\Operation)
  If PmcOpset > 20
    ProcedureReturn PmcFail("The model imports ai.onnx opset " + Str(PmcOpset) + "; runtime-dimension emission implements operator definitions through opset 20.")
  EndIf
  If PmcOpset < Floor And Floor = 20
    ProcedureReturn PmcFail(PmcLabel(*Node) + ": the model imports ai.onnx opset " + Str(PmcOpset) + ", and runtime-dimension emission accepts " + *Node\Operation +
                            " only from a model importing opset 20, the definition it was checked against (where it implements " + *Node\Operation + " at all).")
  EndIf
  If PmcOpset < Floor
    ProcedureReturn PmcFail(PmcLabel(*Node) + ": the model imports ai.onnx opset " + Str(PmcOpset) + ", whose definition of " + *Node\Operation +
                            " differs from the one runtime-dimension emission implements, which is current from opset " + Str(Floor) + " through 20.")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmcAttribute(*Node.PmoOnnxNode, Name.s)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn @*Node\Attributes() : EndIf
  Next
  ProcedureReturn 0
EndProcedure

Procedure.i PmcAttributeGraph(*Node.PmoOnnxNode, Name.s)
  Protected *Attribute.PmoOnnxAttribute = PmcAttribute(*Node, Name)
  If *Attribute = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn *Attribute\Graph
EndProcedure

Procedure.s PmcInput(*Node.PmoOnnxNode, Index.i)
  If Index < 0 Or SelectElement(*Node\Inputs(), Index) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Inputs()
EndProcedure

Procedure.s PmcOutput(*Node.PmoOnnxNode, Index.i)
  If Index < 0 Or SelectElement(*Node\Outputs(), Index) = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn *Node\Outputs()
EndProcedure

Procedure PmcRelease()
  ForEach PmcBuffers()
    If PmcBuffers() : FreeMemory(PmcBuffers()) : EndIf
  Next
  ClearList(PmcBuffers())
  ClearList(PmcGlobalLines())
  ClearList(PmcProcedureLines())
  ClearMap(PmcKind())
  ClearMap(PmcSequenceElement())
  ClearMap(PmcOwned())
EndProcedure

; ---------------------------------------------------------------------------
; Stage 1a: Constant lowering and the opset-11 axes spelling
; ---------------------------------------------------------------------------

Procedure.i PmcLowerConstant(*Graph.PmoOnnxGraph, *Node.PmoOnnxNode)
  Protected Name.s
  Protected *Attribute.PmoOnnxAttribute
  Protected *Tensor.PmoOnnxTensor
  Protected Elements.q = 1
  Protected *Buffer
  Protected Index.i
  If ListSize(*Node\Inputs()) <> 0 Or ListSize(*Node\Outputs()) <> 1
    ProcedureReturn PmcFail(PmcLabel(*Node) + " must have no inputs and one output.")
  EndIf
  Name = PmcOutput(*Node, 0)
  If ListSize(*Node\Attributes()) <> 1
    ProcedureReturn PmcFail(PmcLabel(*Node) + " carries " + Str(ListSize(*Node\Attributes())) + " attributes; exactly one of value, value_float, value_floats, value_int or value_ints is required.")
  EndIf
  FirstElement(*Node\Attributes())
  *Attribute = @*Node\Attributes()
  LastElement(*Graph\Initializers())
  AddElement(*Graph\Initializers())
  *Tensor = @*Graph\Initializers()
  Select *Attribute\Name
    Case "value"
      If *Attribute\HasTensor = 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " attribute value holds no tensor.")
      EndIf
      CopyStructure(@*Attribute\Tensor, *Tensor, PmoOnnxTensor)
      *Tensor\Name = Name
      ; BOOL values arrive in int32_data unless raw_data is used; the
      ; initializer reader takes BOOL as raw bytes, so convert them here.
      If *Tensor\DataType = 9 And *Tensor\Raw\Bytes = 0
        ForEach *Tensor\Dims() : Elements * *Tensor\Dims() : Next
        If ListSize(*Tensor\Int32Data()) <> Elements Or Elements <= 0
          ProcedureReturn PmcFail(PmcLabel(*Node) + " holds a BOOL tensor whose data does not match its shape.")
        EndIf
        *Buffer = AllocateMemory(Elements)
        If *Buffer = 0 : ProcedureReturn PmcFail("Cannot allocate the BOOL value of " + PmcLabel(*Node) + ".") : EndIf
        LastElement(PmcBuffers()) : AddElement(PmcBuffers()) : PmcBuffers() = *Buffer
        Index = 0
        ForEach *Tensor\Int32Data()
          PokeA(*Buffer + Index, Bool(*Tensor\Int32Data() <> 0)) : Index + 1
        Next
        ClearList(*Tensor\Int32Data())
        *Tensor\Raw\Data = *Buffer : *Tensor\Raw\Bytes = Elements
      EndIf
    Case "value_float"
      *Tensor\Name = Name : *Tensor\DataType = 1
      AddElement(*Tensor\FloatData()) : *Tensor\FloatData() = *Attribute\FloatValue
    Case "value_floats"
      *Tensor\Name = Name : *Tensor\DataType = 1
      AddElement(*Tensor\Dims()) : *Tensor\Dims() = ListSize(*Attribute\Floats())
      ForEach *Attribute\Floats()
        AddElement(*Tensor\FloatData()) : *Tensor\FloatData() = *Attribute\Floats()
      Next
    Case "value_int"
      *Tensor\Name = Name : *Tensor\DataType = 7
      AddElement(*Tensor\Int64Data()) : *Tensor\Int64Data() = *Attribute\IntegerValue
    Case "value_ints"
      *Tensor\Name = Name : *Tensor\DataType = 7
      AddElement(*Tensor\Dims()) : *Tensor\Dims() = ListSize(*Attribute\Integers())
      ForEach *Attribute\Integers()
        AddElement(*Tensor\Int64Data()) : *Tensor\Int64Data() = *Attribute\Integers()
      Next
    Default
      ProcedureReturn PmcFail(PmcLabel(*Node) + " attribute " + *Attribute\Name + " is not implemented: string and sparse constants are not carried by this compiler.")
  EndSelect
  ProcedureReturn #True
EndProcedure

Procedure.s PmcUniqueName(Base.s, Map Taken.i())
  Protected Candidate.s
  Repeat
    PmcRenames + 1
    Candidate = Base + "#" + Str(PmcRenames)
  Until FindMapElement(Taken(), Candidate) = 0
  Taken(Candidate) = 1
  ProcedureReturn Candidate
EndProcedure

; Squeeze-11 and Unsqueeze-11 take axes as an attribute; from opset 13 it is
; the second input. The operation is the same, so the attribute becomes a
; constant input and the runtime kernel serves both.
Procedure.i PmcAdaptAxes(*Graph.PmoOnnxGraph, *Node.PmoOnnxNode, *Scope.PmcScope, Map Taken.i())
  Protected *Attribute.PmoOnnxAttribute
  Protected Name.s
  If PmcOpset >= 13 Or (*Node\Operation <> "Squeeze" And *Node\Operation <> "Unsqueeze") : ProcedureReturn #True : EndIf
  *Attribute = PmcAttribute(*Node, "axes")
  If *Attribute = 0 Or ListSize(*Node\Inputs()) <> 1 : ProcedureReturn #True : EndIf
  Name = PmcUniqueName("axes", Taken())
  *Scope\Names(Name) = Name
  LastElement(*Graph\Initializers())
  AddElement(*Graph\Initializers())
  *Graph\Initializers()\Name = Name
  *Graph\Initializers()\DataType = 7
  AddElement(*Graph\Initializers()\Dims()) : *Graph\Initializers()\Dims() = ListSize(*Attribute\Integers())
  ForEach *Attribute\Integers()
    AddElement(*Graph\Initializers()\Int64Data()) : *Graph\Initializers()\Int64Data() = *Attribute\Integers()
  Next
  LastElement(*Node\Inputs()) : AddElement(*Node\Inputs()) : *Node\Inputs() = Name
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = "axes" : DeleteElement(*Node\Attributes()) : Break : EndIf
  Next
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; Stage 1b: scope resolution by renaming
; ---------------------------------------------------------------------------

Procedure.s PmcLookup(List Chain.PmcScope(), Name.s, *Found.Integer)
  *Found\i = #False
  If LastElement(Chain()) = 0 : ProcedureReturn Name : EndIf
  Repeat
    If FindMapElement(Chain()\Names(), Name)
      *Found\i = #True
      ProcedureReturn Chain()\Names()
    EndIf
  Until PreviousElement(Chain()) = 0
  ProcedureReturn Name
EndProcedure

; A name a subgraph defines keeps its spelling unless the model already
; defines it somewhere else, in which case it gets a unique one. The inner
; definition then hides the outer one for every reference inside the
; subgraph, which is how ONNX scopes names.
Procedure.s PmcDefine(*Scope.PmcScope, Name.s, IsTop.i, Map Taken.i())
  Protected Unique.s = Name
  If Name = "" : ProcedureReturn "" : EndIf
  If IsTop = 0 And FindMapElement(Taken(), Name)
    Unique = PmcUniqueName(Name, Taken())
  EndIf
  Taken(Unique) = 1
  *Scope\Names(Name) = Unique
  ProcedureReturn Unique
EndProcedure

Procedure.i PmcResolveGraph(*Graph.PmoOnnxGraph, List Chain.PmcScope(), Map Taken.i(), Depth.i, IsTop.i)
  Protected *Scope.PmcScope
  Protected Found.Integer
  Protected Name.s
  Protected *Child.PmoOnnxGraph
  Protected *Node.PmoOnnxNode
  If Depth > #PMC_MAX_DEPTH
    ProcedureReturn PmcFail("Subgraphs are nested more than " + Str(#PMC_MAX_DEPTH) + " levels deep; this compiler bounds nesting so generated control flow stays within a fixed call depth.")
  EndIf
  LastElement(Chain())
  AddElement(Chain())
  *Scope = @Chain()

  ForEach *Graph\Nodes()
    If *Graph\Nodes()\Operation = "Constant" And (*Graph\Nodes()\Domain = "" Or *Graph\Nodes()\Domain = "ai.onnx")
      If PmcLowerConstant(*Graph, @*Graph\Nodes()) = 0 : ProcedureReturn #False : EndIf
      DeleteElement(*Graph\Nodes())
    ElseIf PmcAdaptAxes(*Graph, @*Graph\Nodes(), *Scope, Taken()) = 0
      ProcedureReturn #False
    EndIf
  Next

  ; The top-level graph never renames, and every top-level definition was
  ; registered before any subgraph was visited, so a subgraph name that
  ; repeats a later top-level output is renamed too.
  ForEach *Graph\Inputs()
    *Graph\Inputs()\Name = PmcDefine(*Scope, *Graph\Inputs()\Name, IsTop, Taken())
  Next
  ForEach *Graph\Initializers()
    ; An axes constant made by PmcAdaptAxes is already defined in this scope.
    If FindMapElement(*Scope\Names(), *Graph\Initializers()\Name) : Continue : EndIf
    *Graph\Initializers()\Name = PmcDefine(*Scope, *Graph\Initializers()\Name, IsTop, Taken())
  Next
  ForEach *Graph\Nodes()
    *Node = @*Graph\Nodes()
    ForEach *Node\Inputs()
      If *Node\Inputs() = "" : Continue : EndIf
      Name = PmcLookup(Chain(), *Node\Inputs(), @Found)
      If Found\i = 0
        If IsTop : Continue : EndIf
        ProcedureReturn PmcFail(PmcLabel(*Node) + " inside a subgraph reads '" + *Node\Inputs() + "', which no enclosing graph defines before it.")
      EndIf
      *Node\Inputs() = Name
    Next
    ForEach *Node\Attributes()
      If *Node\Attributes()\Graph
        *Child = *Node\Attributes()\Graph
        If PmcResolveGraph(*Child, Chain(), Taken(), Depth + 1, #False) = 0 : ProcedureReturn #False : EndIf
        LastElement(Chain())
      EndIf
    Next
    ForEach *Node\Outputs()
      *Node\Outputs() = PmcDefine(*Scope, *Node\Outputs(), IsTop, Taken())
    Next
  Next
  ForEach *Graph\Outputs()
    Name = PmcLookup(Chain(), *Graph\Outputs()\Name, @Found)
    If Found\i = 0 And IsTop = 0
      ProcedureReturn PmcFail("A subgraph output '" + *Graph\Outputs()\Name + "' is defined by no node, input or initializer the subgraph can see.")
    EndIf
    *Graph\Outputs()\Name = Name
  Next
  LastElement(Chain())
  DeleteElement(Chain())
  ProcedureReturn #True
EndProcedure

Procedure PmcHoist(*Top.PmoOnnxGraph, *Graph.PmoOnnxGraph)
  Protected *Child.PmoOnnxGraph
  ForEach *Graph\Nodes()
    ForEach *Graph\Nodes()\Attributes()
      If *Graph\Nodes()\Attributes()\Graph
        *Child = *Graph\Nodes()\Attributes()\Graph
        ForEach *Child\Initializers()
          LastElement(*Top\Initializers())
          AddElement(*Top\Initializers())
          CopyStructure(@*Child\Initializers(), @*Top\Initializers(), PmoOnnxTensor)
        Next
        ClearList(*Child\Initializers())
        PmcHoist(*Top, *Child)
      EndIf
    Next
  Next
EndProcedure

; ---------------------------------------------------------------------------
; Stage 1c: every value is used as the kind it is
; ---------------------------------------------------------------------------

Procedure.i PmcElementTypeCarried(ElementType.i)
  Select ElementType
    Case 1, 6, 7, 9 : ProcedureReturn #True
  EndSelect
  ProcedureReturn #False
EndProcedure

Procedure.i PmcRequireKind(*Node.PmoOnnxNode, Position.i, Wanted.i)
  Protected Name.s = PmcInput(*Node, Position)
  If Name = "" Or FindMapElement(PmcKind(), Name) = 0 : ProcedureReturn #True : EndIf
  If PmcKind() = Wanted : ProcedureReturn #True : EndIf
  If Wanted = #PMC_KIND_SEQUENCE
    ProcedureReturn PmcFail(PmcLabel(*Node) + " input " + Str(Position) + " ('" + Name + "') is a tensor where the operator requires a sequence.")
  EndIf
  ProcedureReturn PmcFail(PmcLabel(*Node) + " input " + Str(Position) + " ('" + Name + "') is a sequence where the operator requires a tensor.")
EndProcedure

Procedure.i PmcSetKind(Name.s, Kind.i, Element.i = 0)
  If Name = "" : ProcedureReturn #True : EndIf
  PmcKind(Name) = Kind
  If Kind = #PMC_KIND_SEQUENCE : PmcSequenceElement(Name) = Element : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmcAllowedAttributes(*Node.PmoOnnxNode, Allowed.s)
  ForEach *Node\Attributes()
    If FindString(Allowed, "|" + *Node\Attributes()\Name + "|") = 0
      ProcedureReturn PmcFail(PmcLabel(*Node) + " carries attribute " + *Node\Attributes()\Name + ", which this operator does not define or this compiler does not implement.")
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmcCountRange(*Node.PmoOnnxNode, MinimumInputs.i, MaximumInputs.i, Outputs.i)
  If ListSize(*Node\Inputs()) < MinimumInputs Or ListSize(*Node\Inputs()) > MaximumInputs
    ProcedureReturn PmcFail(PmcLabel(*Node) + " has " + Str(ListSize(*Node\Inputs())) + " inputs; the operator takes " + Str(MinimumInputs) + " to " + Str(MaximumInputs) + ".")
  EndIf
  If Outputs >= 0 And ListSize(*Node\Outputs()) <> Outputs
    ProcedureReturn PmcFail(PmcLabel(*Node) + " has " + Str(ListSize(*Node\Outputs())) + " outputs; the operator produces " + Str(Outputs) + ".")
  EndIf
  ProcedureReturn #True
EndProcedure

; Attribute, arity and value checks for the operators of this file. Used by
; PmdValidate for top-level nodes and by the body emitter for nested ones.
Procedure.i PmcValidateNode(*Node.PmoOnnxNode)
  Protected Value.q
  Select *Node\Operation
    Case "SequenceEmpty"
      If PmcAllowedAttributes(*Node, "|dtype|") = 0 Or PmcCountRange(*Node, 0, 0, 1) = 0 : ProcedureReturn #False : EndIf
      Value = PmoEmitAttrI(*Node, "dtype", 1)
      If PmcElementTypeCarried(Value) = 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " attribute dtype=" + Str(Value) + " is not FLOAT (1), INT32 (6), INT64 (7) or BOOL (9), the element types a sequence carries here.")
      EndIf
    Case "SequenceConstruct"
      If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 1, 2147483647, 1) = 0 : ProcedureReturn #False : EndIf
    Case "SequenceInsert"
      If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 2, 3, 1) = 0 : ProcedureReturn #False : EndIf
    Case "SequenceAt"
      If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 2, 2, 1) = 0 : ProcedureReturn #False : EndIf
    Case "SequenceErase"
      If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 1, 2, 1) = 0 : ProcedureReturn #False : EndIf
    Case "SequenceLength"
      If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 1, 1, 1) = 0 : ProcedureReturn #False : EndIf
    Case "SplitToSequence"
      If PmcAllowedAttributes(*Node, "|axis|keepdims|") = 0 Or PmcCountRange(*Node, 1, 2, 1) = 0 : ProcedureReturn #False : EndIf
      Value = PmoEmitAttrI(*Node, "keepdims", 1)
      If Value <> 0 And Value <> 1
        ProcedureReturn PmcFail(PmcLabel(*Node) + " attribute keepdims=" + Str(Value) + " is not 0 or 1.")
      EndIf
    Case "ConcatFromSequence"
      If PmcAllowedAttributes(*Node, "|axis|new_axis|") = 0 Or PmcCountRange(*Node, 1, 1, 1) = 0 : ProcedureReturn #False : EndIf
      If PmcAttribute(*Node, "axis") = 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has no axis attribute; the operator requires one.")
      EndIf
      Value = PmoEmitAttrI(*Node, "new_axis", 0)
      If Value <> 0 And Value <> 1
        ProcedureReturn PmcFail(PmcLabel(*Node) + " attribute new_axis=" + Str(Value) + " is not 0 or 1.")
      EndIf
    Case "If"
      If PmcAllowedAttributes(*Node, "|then_branch|else_branch|") = 0 Or PmcCountRange(*Node, 1, 1, -1) = 0 : ProcedureReturn #False : EndIf
      If PmcAttributeGraph(*Node, "then_branch") = 0 Or PmcAttributeGraph(*Node, "else_branch") = 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " needs both a then_branch and an else_branch graph attribute.")
      EndIf
      If PmcInput(*Node, 0) = ""
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has no condition input.")
      EndIf
    Case "Loop"
      If PmcAllowedAttributes(*Node, "|body|") = 0 Or PmcCountRange(*Node, 2, 2147483647, -1) = 0 : ProcedureReturn #False : EndIf
      If PmcAttributeGraph(*Node, "body") = 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has no body graph attribute.")
      EndIf
      If PmcInput(*Node, 0) = "" And PmcInput(*Node, 1) = ""
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has neither a trip count M nor a condition; the ONNX specification defines that form as a loop that never ends, so it is refused.")
      EndIf
  EndSelect
  ProcedureReturn #True
EndProcedure

Declare.i PmcInferGraph(*Graph.PmoOnnxGraph, IsTop.i, Depth.i)

Procedure.i PmcInferNode(*Node.PmoOnnxNode, Depth.i)
  Protected Operation.s = *Node\Operation
  Protected *Then.PmoOnnxGraph
  Protected *Else.PmoOnnxGraph
  Protected *Body.PmoOnnxGraph
  Protected Index.i
  Protected Carried.i
  Protected Scans.i
  Protected ThenKind.i
  Protected ElseKind.i
  Protected Name.s
  Protected Element.i
  If *Node\Domain <> "" And *Node\Domain <> "ai.onnx"
    ProcedureReturn PmcFail(PmcLabel(*Node) + " uses domain " + *Node\Domain + ", which the runtime-dimension path does not implement.")
  EndIf
  If PmcAuditNode(*Node) = 0 : ProcedureReturn #False : EndIf
  If PmcRequestStage(Operation)
    PmcUsed = #True
    If PmcValidateNode(*Node) = 0 : ProcedureReturn #False : EndIf
  EndIf
  Select Operation
    Case "SequenceEmpty"
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, PmoEmitAttrI(*Node, "dtype", 1))
    Case "SequenceConstruct"
      Element = 0
      For Index = 0 To ListSize(*Node\Inputs()) - 1
        If PmcRequireKind(*Node, Index, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      Next
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, 0)
    Case "SequenceInsert"
      If PmcRequireKind(*Node, 0, #PMC_KIND_SEQUENCE) = 0 Or PmcRequireKind(*Node, 1, #PMC_KIND_TENSOR) = 0 Or
         PmcRequireKind(*Node, 2, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      Element = 0
      If FindMapElement(PmcSequenceElement(), PmcInput(*Node, 0)) : Element = PmcSequenceElement() : EndIf
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, Element)
    Case "SequenceAt"
      If PmcRequireKind(*Node, 0, #PMC_KIND_SEQUENCE) = 0 Or PmcRequireKind(*Node, 1, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_TENSOR)
    Case "SequenceErase"
      If PmcRequireKind(*Node, 0, #PMC_KIND_SEQUENCE) = 0 Or PmcRequireKind(*Node, 1, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      Element = 0
      If FindMapElement(PmcSequenceElement(), PmcInput(*Node, 0)) : Element = PmcSequenceElement() : EndIf
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, Element)
    Case "SequenceLength", "ConcatFromSequence"
      If PmcRequireKind(*Node, 0, #PMC_KIND_SEQUENCE) = 0 : ProcedureReturn #False : EndIf
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_TENSOR)
    Case "SplitToSequence"
      If PmcRequireKind(*Node, 0, #PMC_KIND_TENSOR) = 0 Or PmcRequireKind(*Node, 1, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, 0)
    Case "If"
      If PmcRequireKind(*Node, 0, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      *Then = PmcAttributeGraph(*Node, "then_branch")
      *Else = PmcAttributeGraph(*Node, "else_branch")
      If ListSize(*Then\Inputs()) <> 0 Or ListSize(*Else\Inputs()) <> 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has a branch that declares inputs; If branches take none.")
      EndIf
      If ListSize(*Then\Outputs()) <> ListSize(*Node\Outputs()) Or ListSize(*Else\Outputs()) <> ListSize(*Node\Outputs())
        ProcedureReturn PmcFail(PmcLabel(*Node) + " has " + Str(ListSize(*Node\Outputs())) + " outputs but its branches return " + Str(ListSize(*Then\Outputs())) + " and " + Str(ListSize(*Else\Outputs())) + ".")
      EndIf
      If PmcInferGraph(*Then, #False, Depth + 1) = 0 Or PmcInferGraph(*Else, #False, Depth + 1) = 0 : ProcedureReturn #False : EndIf
      For Index = 0 To ListSize(*Node\Outputs()) - 1
        SelectElement(*Then\Outputs(), Index) : SelectElement(*Else\Outputs(), Index)
        ThenKind = PmcKind(*Then\Outputs()\Name) : ElseKind = PmcKind(*Else\Outputs()\Name)
        If ThenKind <> ElseKind
          ProcedureReturn PmcFail(PmcLabel(*Node) + " output " + Str(Index) + " is a tensor in one branch and a sequence in the other.")
        EndIf
        Element = 0
        If ThenKind = #PMC_KIND_SEQUENCE And FindMapElement(PmcSequenceElement(), *Then\Outputs()\Name) : Element = PmcSequenceElement() : EndIf
        PmcSetKind(PmcOutput(*Node, Index), ThenKind, Element)
      Next
    Case "Loop"
      If PmcRequireKind(*Node, 0, #PMC_KIND_TENSOR) = 0 Or PmcRequireKind(*Node, 1, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      *Body = PmcAttributeGraph(*Node, "body")
      Carried = ListSize(*Node\Inputs()) - 2
      If ListSize(*Body\Inputs()) <> Carried + 2
        ProcedureReturn PmcFail(PmcLabel(*Node) + " carries " + Str(Carried) + " values, so its body must declare " + Str(Carried + 2) + " inputs; it declares " + Str(ListSize(*Body\Inputs())) + ".")
      EndIf
      Scans = ListSize(*Body\Outputs()) - 1 - Carried
      If Scans < 0
        ProcedureReturn PmcFail(PmcLabel(*Node) + " body returns " + Str(ListSize(*Body\Outputs())) + " outputs; it must return the condition and the " + Str(Carried) + " carried values first.")
      EndIf
      If ListSize(*Node\Outputs()) <> Carried + Scans
        ProcedureReturn PmcFail(PmcLabel(*Node) + " declares " + Str(ListSize(*Node\Outputs())) + " outputs; its body produces " + Str(Carried) + " carried values and " + Str(Scans) + " scan outputs.")
      EndIf
      SelectElement(*Body\Inputs(), 0) : PmcSetKind(*Body\Inputs()\Name, #PMC_KIND_TENSOR)
      SelectElement(*Body\Inputs(), 1) : PmcSetKind(*Body\Inputs()\Name, #PMC_KIND_TENSOR)
      For Index = 0 To Carried - 1
        Name = PmcInput(*Node, Index + 2)
        If Name = ""
          ProcedureReturn PmcFail(PmcLabel(*Node) + " carried value " + Str(Index) + " has no initial value.")
        EndIf
        SelectElement(*Body\Inputs(), Index + 2)
        Element = 0
        If FindMapElement(PmcSequenceElement(), Name) : Element = PmcSequenceElement() : EndIf
        PmcSetKind(*Body\Inputs()\Name, PmcKind(Name), Element)
      Next
      If PmcInferGraph(*Body, #False, Depth + 1) = 0 : ProcedureReturn #False : EndIf
      SelectElement(*Body\Outputs(), 0)
      If PmcKind(*Body\Outputs()\Name) = #PMC_KIND_SEQUENCE
        ProcedureReturn PmcFail(PmcLabel(*Node) + " body returns a sequence as its condition.")
      EndIf
      For Index = 0 To Carried - 1
        SelectElement(*Body\Outputs(), Index + 1)
        Name = PmcInput(*Node, Index + 2)
        If PmcKind(*Body\Outputs()\Name) <> PmcKind(Name)
          ProcedureReturn PmcFail(PmcLabel(*Node) + " carried value " + Str(Index) + " changes between a tensor and a sequence inside the body.")
        EndIf
        Element = 0
        If FindMapElement(PmcSequenceElement(), Name) : Element = PmcSequenceElement() : EndIf
        PmcSetKind(PmcOutput(*Node, Index), PmcKind(Name), Element)
      Next
      For Index = 0 To Scans - 1
        SelectElement(*Body\Outputs(), Index + 1 + Carried)
        If PmcKind(*Body\Outputs()\Name) = #PMC_KIND_SEQUENCE
          ProcedureReturn PmcFail(PmcLabel(*Node) + " scan output " + Str(Index) + " is a sequence; Loop scan outputs are tensors stacked along a new first axis.")
        EndIf
        PmcSetKind(PmcOutput(*Node, Carried + Index), #PMC_KIND_TENSOR)
      Next
    Default
      Name = PmcInput(*Node, 0)
      If Operation = "Identity" And Name <> "" And FindMapElement(PmcKind(), Name) And PmcKind() = #PMC_KIND_SEQUENCE
        ; Identity-14 carries a sequence as well as a tensor. That form is a
        ; copy of the sequence's block (or a hand-over), emitted with the
        ; sequence operators.
        If PmcOpset < 14
          ProcedureReturn PmcFail(PmcLabel(*Node) + ": the model imports ai.onnx opset " + Str(PmcOpset) + ", and Identity takes a sequence only from opset 14.")
        EndIf
        If PmcAllowedAttributes(*Node, "|") = 0 Or PmcCountRange(*Node, 1, 1, 1) = 0 : ProcedureReturn #False : EndIf
        Element = 0
        If FindMapElement(PmcSequenceElement(), Name) : Element = PmcSequenceElement() : EndIf
        PmcOwned(Str(*Node)) = 1
        PmcUsed = #True
        PmcSetKind(PmcOutput(*Node, 0), #PMC_KIND_SEQUENCE, Element)
      Else
        For Index = 0 To ListSize(*Node\Inputs()) - 1
          If PmcRequireKind(*Node, Index, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
        Next
        ForEach *Node\Outputs()
          PmcSetKind(*Node\Outputs(), #PMC_KIND_TENSOR)
        Next
      EndIf
  EndSelect
  ProcedureReturn #True
EndProcedure

Procedure.i PmcInferGraph(*Graph.PmoOnnxGraph, IsTop.i, Depth.i)
  If IsTop
    ForEach *Graph\Inputs()
      Select *Graph\Inputs()\ValueKind
        Case #PMO_VALUE_TENSOR
          PmcSetKind(*Graph\Inputs()\Name, #PMC_KIND_TENSOR)
        Case #PMO_VALUE_SEQUENCE
          If PmcElementTypeCarried(*Graph\Inputs()\SequenceElementType) = 0
            ProcedureReturn PmcFail("Graph input '" + *Graph\Inputs()\Name + "' is a sequence of element type " + Str(*Graph\Inputs()\SequenceElementType) + "; a sequence input must declare FLOAT, INT32, INT64 or BOOL tensors.")
          EndIf
          PmcUsed = #True
          PmcSetKind(*Graph\Inputs()\Name, #PMC_KIND_SEQUENCE, *Graph\Inputs()\SequenceElementType)
        Case #PMO_VALUE_OPTIONAL
          ProcedureReturn PmcFail("Graph input '" + *Graph\Inputs()\Name + "' is an optional value; optional types (Optional, OptionalHasElement, OptionalGetElement) are not implemented.")
        Default
          ProcedureReturn PmcFail("Graph input '" + *Graph\Inputs()\Name + "' is a map, sparse tensor or nested sequence, which this compiler does not carry.")
      EndSelect
    Next
  EndIf
  ForEach *Graph\Initializers()
    PmcSetKind(*Graph\Initializers()\Name, #PMC_KIND_TENSOR)
  Next
  ForEach *Graph\Nodes()
    If PmcInferNode(@*Graph\Nodes(), Depth) = 0 : ProcedureReturn #False : EndIf
  Next
  ForEach *Graph\Outputs()
    If *Graph\Outputs()\ValueKind = #PMO_VALUE_OPTIONAL Or *Graph\Outputs()\ValueKind = #PMO_VALUE_OTHER
      ProcedureReturn PmcFail("Graph output '" + *Graph\Outputs()\Name + "' is declared as an optional, map or nested-sequence value, which this compiler does not carry.")
    EndIf
    If FindMapElement(PmcKind(), *Graph\Outputs()\Name)
      If PmcKind() = #PMC_KIND_SEQUENCE : PmcUsed = #True : EndIf
      If *Graph\Outputs()\ValueKind = #PMO_VALUE_SEQUENCE And PmcKind() <> #PMC_KIND_SEQUENCE
        ProcedureReturn PmcFail("Graph output '" + *Graph\Outputs()\Name + "' is declared as a sequence but a tensor is produced for it.")
      EndIf
      If *Graph\Outputs()\ValueKind = #PMO_VALUE_TENSOR And *Graph\Outputs()\HasTensorType And PmcKind() <> #PMC_KIND_TENSOR
        ProcedureReturn PmcFail("Graph output '" + *Graph\Outputs()\Name + "' is declared as a tensor but a sequence is produced for it.")
      EndIf
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

; Entry point of stage 1. Kokoro and every graph without subgraphs, Constant
; nodes or sequences leave this procedure unchanged.
Procedure.i PmcPrepareModel(*Model.PmoOnnxModel)
  Protected *Graph.PmoOnnxGraph = @*Model\Graph
  NewMap Taken.i()
  NewList Chain.PmcScope()
  PmcRelease()
  PmcError = "" : PmcUsed = #False : PmcRenames = 0 : PmcOpset = 0
  PmcBodyNodeCount = 0 : PmcNextGraph = 0
  ForEach *Model\Opsets()
    If *Model\Opsets()\Domain = "" Or *Model\Opsets()\Domain = "ai.onnx" : PmcOpset = *Model\Opsets()\Version : EndIf
  Next
  ForEach *Graph\Inputs() : Taken(*Graph\Inputs()\Name) = 1 : Next
  ForEach *Graph\Initializers() : Taken(*Graph\Initializers()\Name) = 1 : Next
  ForEach *Graph\Nodes()
    ForEach *Graph\Nodes()\Outputs()
      If *Graph\Nodes()\Outputs() <> "" : Taken(*Graph\Nodes()\Outputs()) = 1 : EndIf
    Next
  Next
  If PmcResolveGraph(*Graph, Chain(), Taken(), 0, #True) = 0 : ProcedureReturn #False : EndIf
  PmcHoist(*Graph, *Graph)
  If PmcInferGraph(*Graph, #True, 0) = 0 : ProcedureReturn #False : EndIf
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; Stage 2: the placeholder PmdCall returns
; ---------------------------------------------------------------------------

Procedure.s PmcCall(*Node.PmoOnnxNode)
  ProcedureReturn "; " + *Node\Operation + " is emitted once last uses are known"
EndProcedure
