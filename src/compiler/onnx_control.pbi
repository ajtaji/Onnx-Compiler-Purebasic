; ============================================================================
; onnx_control.pbi - If, Loop and the sequence operators on the
; runtime-dimension path
; ----------------------------------------------------------------------------
; Operator semantics are taken from the ONNX operator specification
; (https://onnx.ai/onnx/operators/): If-11/13/16/19, Loop-11/13/16/19,
; SequenceEmpty-11, SequenceConstruct-11, SequenceInsert-11, SequenceAt-11,
; SequenceLength-11, SequenceErase-11, SplitToSequence-11 and
; ConcatFromSequence-11; Scan-9 and SequenceMap-17, rewritten into Loop.
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
; Nodes the Scan and SequenceMap lowering added (by address).
Global NewMap PmcLowered.i()

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

; OPSET FLOORS. The runtime-dimension path accepts ai.onnx models through
; #PMO_OPSET_MAX (onnx_opsets.pbi, which also holds the ceilings), and each
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
  ProcedureReturn Bool(Version >= 11 And Version <= #PMO_OPSET_MAX)
EndProcedure

Procedure.i PmcOperatorFloor(Operation.s)
  Select Operation
    Case "Shape", "Transpose", "MatMul" : ProcedureReturn 1
    Case "Reshape" : ProcedureReturn 5
    Case "Exp", "Log", "Sqrt", "Abs", "Neg", "Floor", "Sigmoid", "Tanh", "LeakyRelu", "Cast" : ProcedureReturn 6
    Case "Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "Less", "And", "Sin", "Cos", "Atan", "LSTM" : ProcedureReturn 7
    Case "Expand" : ProcedureReturn 8
    Case "Scan" : ProcedureReturn 9
    Case "Multinomial" : ProcedureReturn 7
    Case "ConstantOfShape", "Where", "NonZero" : ProcedureReturn 9
    Case "Slice" : ProcedureReturn 10
    Case "Gather", "Concat", "Range", "Round", "CumSum", "ScatterND", "Squeeze", "Unsqueeze", "Clip", "Resize", "Gemm",
         "ReduceMean", "ReduceSum", "If", "Loop", "SequenceEmpty", "SequenceConstruct", "SequenceInsert", "SequenceAt",
         "SequenceLength", "SequenceErase", "SplitToSequence", "ConcatFromSequence" : ProcedureReturn 11
    Case "GreaterOrEqual" : ProcedureReturn 12
    Case "Softmax" : ProcedureReturn 13
    Case "Bernoulli" : ProcedureReturn 15
    Case "LayerNormalization", "STFT", "SequenceMap" : ProcedureReturn 17
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
  If PmcOpset > #PMO_OPSET_MAX
    ProcedureReturn PmcFail("The model imports ai.onnx opset " + Str(PmcOpset) + "; runtime-dimension emission implements operator definitions through opset " + Str(#PMO_OPSET_MAX) + ".")
  EndIf
  If FindMapElement(PmcLowered(), Str(*Node)) : ProcedureReturn #True : EndIf
  If PmoOpsetCeilingRefusal(*Node\Operation, PmcOpset) <> ""
    ProcedureReturn PmcFail(PmcLabel(*Node) + ": " + PmoOpsetCeilingRefusal(*Node\Operation, PmcOpset))
  EndIf
  If PmcOpset < Floor And Floor = 20
    ProcedureReturn PmcFail(PmcLabel(*Node) + ": the model imports ai.onnx opset " + Str(PmcOpset) + ", and runtime-dimension emission accepts " + *Node\Operation +
                            " only from a model importing opset 20 or later, the definition it was checked against (where it implements " + *Node\Operation + " at all).")
  EndIf
  If PmcOpset < Floor
    ProcedureReturn PmcFail(PmcLabel(*Node) + ": the model imports ai.onnx opset " + Str(PmcOpset) + ", whose definition of " + *Node\Operation +
                            " differs from the one runtime-dimension emission implements, which is current from opset " + Str(Floor) + " through " + Str(#PMO_OPSET_MAX) + ".")
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
  ClearMap(PmcLowered())
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
    Case "Scan"
      ; first pass only: Scan is rewritten into a Loop before the final pass
      *Body = PmcAttributeGraph(*Node, "body")
      If *Body = 0 : ProcedureReturn PmcFail(PmcLabel(*Node) + " has no body graph attribute.") : EndIf
      For Index = 0 To ListSize(*Node\Inputs()) - 1
        If PmcRequireKind(*Node, Index, #PMC_KIND_TENSOR) = 0 : ProcedureReturn #False : EndIf
      Next
      ForEach *Body\Inputs() : PmcSetKind(*Body\Inputs()\Name, #PMC_KIND_TENSOR) : Next
      If PmcInferGraph(*Body, #False, Depth + 1) = 0 : ProcedureReturn #False : EndIf
      ForEach *Node\Outputs() : PmcSetKind(*Node\Outputs(), #PMC_KIND_TENSOR) : Next
    Case "SequenceMap"
      ; first pass only, as Scan
      *Body = PmcAttributeGraph(*Node, "body")
      If *Body = 0 : ProcedureReturn PmcFail(PmcLabel(*Node) + " has no body graph attribute.") : EndIf
      If PmcRequireKind(*Node, 0, #PMC_KIND_SEQUENCE) = 0 : ProcedureReturn #False : EndIf
      ForEach *Body\Inputs() : PmcSetKind(*Body\Inputs()\Name, #PMC_KIND_TENSOR) : Next
      If PmcInferGraph(*Body, #False, Depth + 1) = 0 : ProcedureReturn #False : EndIf
      Index = 0
      ForEach *Node\Outputs()
        Element = 0
        If SelectElement(*Body\Outputs(), Index) : Element = *Body\Outputs()\ElementType : EndIf
        PmcSetKind(*Node\Outputs(), #PMC_KIND_SEQUENCE, Element)
        Index + 1
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

; ---------------------------------------------------------------------------
; Stage 1d: Scan and SequenceMap become Loop
; ---------------------------------------------------------------------------
; Scan-9 (the form current through opset 20) and SequenceMap-17 are both
; defined as a loop over one axis or one sequence. Rather than a second loop
; emitter they are rewritten here into the Loop this file already emits,
; with the nodes the specification's own definitions imply:
;
;   Scan         M = Shape(x0)[axis0] iterations; each scan input's slice is
;                Gather(x, i or M-1-i, axis); each scan output is the Loop's
;                stacked output, reversed with Slice(step -1) when its
;                direction is 1 and moved to its axis with Transpose.
;   SequenceMap  SequenceLength(s0) iterations; a sequence input's element is
;                SequenceAt(s, i), a tensor input is read as it is; each
;                output sequence starts as SequenceEmpty of the body output's
;                element type and grows by SequenceInsert (the operator's
;                ONNX function body, written out).
;
; The nodes added implement an operator whose own opset floor was audited, so
; their floors are not audited again (PmcLowered).

Procedure.i PmcIsLoweredOp(Operation.s)
  ProcedureReturn Bool(Operation = "Scan" Or Operation = "SequenceMap")
EndProcedure

; Adds a node before the current node of *Graph (Before = 1, the current node
; stays current) or after it (Before = 0, the new node becomes current).
; Inputs and Outputs are comma-separated names.
Procedure.i PmcAddNode(*Graph.PmoOnnxGraph, Operation.s, Inputs.s, Outputs.s, Before.i)
  Protected *Current = @*Graph\Nodes()
  Protected *Node.PmoOnnxNode
  Protected Index.i
  If Before And *Current
    InsertElement(*Graph\Nodes())
  Else
    AddElement(*Graph\Nodes())
  EndIf
  *Node = @*Graph\Nodes()
  *Node\Operation = Operation
  If Inputs <> ""
    For Index = 1 To CountString(Inputs, ",") + 1
      AddElement(*Node\Inputs()) : *Node\Inputs() = StringField(Inputs, Index, ",")
    Next
  EndIf
  For Index = 1 To CountString(Outputs, ",") + 1
    AddElement(*Node\Outputs()) : *Node\Outputs() = StringField(Outputs, Index, ",")
  Next
  PmcLowered(Str(*Node)) = 1
  If Before And *Current : ChangeCurrentElement(*Graph\Nodes(), *Current) : EndIf
  ProcedureReturn *Node
EndProcedure

Procedure PmcAddIntAttribute(*Node.PmoOnnxNode, Name.s, Value.q)
  AddElement(*Node\Attributes())
  *Node\Attributes()\Name = Name : *Node\Attributes()\AttributeType = 2 : *Node\Attributes()\IntegerValue = Value
EndProcedure

; An INT64 initializer of the top-level graph: a scalar, or a 1-D list.
Procedure.s PmcAddInt64(*Top.PmoOnnxGraph, Values.s, Scalar.i, Map Taken.i())
  Protected Name.s = PmcUniqueName("lowered", Taken())
  Protected Index.i
  Protected Count.i = CountString(Values, ",") + 1
  LastElement(*Top\Initializers())
  AddElement(*Top\Initializers())
  *Top\Initializers()\Name = Name
  *Top\Initializers()\DataType = 7
  If Scalar = 0
    AddElement(*Top\Initializers()\Dims()) : *Top\Initializers()\Dims() = Count
  EndIf
  For Index = 1 To Count
    AddElement(*Top\Initializers()\Int64Data()) : *Top\Initializers()\Int64Data() = Val(StringField(Values, Index, ","))
  Next
  ProcedureReturn Name
EndProcedure

; A BOOL true scalar initializer (BOOL initializers are read as raw bytes).
Procedure.s PmcAddTrue(*Top.PmoOnnxGraph, Map Taken.i())
  Protected Name.s = PmcUniqueName("lowered", Taken())
  Protected *Buffer = AllocateMemory(1)
  If *Buffer = 0 : PmcFail("Cannot allocate a lowered constant.") : ProcedureReturn "" : EndIf
  PokeA(*Buffer, 1)
  LastElement(PmcBuffers()) : AddElement(PmcBuffers()) : PmcBuffers() = *Buffer
  LastElement(*Top\Initializers())
  AddElement(*Top\Initializers())
  *Top\Initializers()\Name = Name
  *Top\Initializers()\DataType = 9
  *Top\Initializers()\Raw\Data = *Buffer
  *Top\Initializers()\Raw\Bytes = 1
  ProcedureReturn Name
EndProcedure

; A scalar value declaration for a Loop body's iteration number and condition.
Procedure PmcScalarValue(*Value.PmoOnnxValue, Name.s, ElementType.i)
  *Value\Name = Name : *Value\ElementType = ElementType
  *Value\HasTensorType = #True : *Value\HasShape = #True : *Value\ValueKind = #PMO_VALUE_TENSOR
  ClearList(*Value\Dims())
EndProcedure

; The declared rank of a value, from *Graph's and the top graph's
; declarations and initializers; -1 when neither says.
Procedure.i PmcDeclaredRank(*Top.PmoOnnxGraph, *Graph.PmoOnnxGraph, Name.s)
  Protected *G.PmoOnnxGraph
  Protected Pass.i
  For Pass = 0 To 1
    *G = *Graph : If Pass = 1 : *G = *Top : EndIf
    ForEach *G\Inputs()
      If *G\Inputs()\Name = Name And *G\Inputs()\HasShape : ProcedureReturn ListSize(*G\Inputs()\Dims()) : EndIf
    Next
    ForEach *G\Values()
      If *G\Values()\Name = Name And *G\Values()\HasShape : ProcedureReturn ListSize(*G\Values()\Dims()) : EndIf
    Next
    ForEach *G\Outputs()
      If *G\Outputs()\Name = Name And *G\Outputs()\HasShape : ProcedureReturn ListSize(*G\Outputs()\Dims()) : EndIf
    Next
    ForEach *G\Initializers()
      If *G\Initializers()\Name = Name : ProcedureReturn ListSize(*G\Initializers()\Dims()) : EndIf
    Next
  Next
  ProcedureReturn -1
EndProcedure

; Keeps only the body attribute of a node that becomes a Loop.
Procedure PmcKeepBody(*Node.PmoOnnxNode)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name <> "body" : DeleteElement(*Node\Attributes()) : EndIf
  Next
EndProcedure

; Rewrites the current node of *Graph, a Scan, into a Loop (see above).
Procedure.i PmcLowerScan(*Top.PmoOnnxGraph, *Graph.PmoOnnxGraph, Map Taken.i())
  Protected *Node.PmoOnnxNode = @*Graph\Nodes()
  Protected *Body.PmoOnnxGraph = PmcAttributeGraph(*Node, "body")
  Protected *Added.PmoOnnxNode
  Protected Label.s = PmcLabel(*Node)
  Protected M.i, N.i, K.i, j.i, a.i, r.i, d.i, Reverse.i
  Protected Shp.s, Trip.s, Last.s, One.s, Iter.s, CondIn.s, CondOut.s, Truth.s, Idx.s, Stacked.s, Final.s, Perm.s
  Protected Dim ScanIn.s(0)
  Protected Dim Elem.s(0)
  If PmcAllowedAttributes(*Node, "|body|num_scan_inputs|scan_input_axes|scan_input_directions|scan_output_axes|scan_output_directions|") = 0 : ProcedureReturn #False : EndIf
  If *Body = 0 : ProcedureReturn PmcFail(Label + " has no body graph attribute.") : EndIf
  If PmcAttribute(*Node, "num_scan_inputs") = 0 : ProcedureReturn PmcFail(Label + " has no num_scan_inputs attribute; the operator requires one.") : EndIf
  M = PmoEmitAttrI(*Node, "num_scan_inputs", 0)
  N = ListSize(*Node\Inputs()) - M
  If M < 1 Or N < 0
    ProcedureReturn PmcFail(Label + " attribute num_scan_inputs = " + Str(M) + " with " + Str(ListSize(*Node\Inputs())) + " inputs; at least one scan input and no more than the inputs are required.")
  EndIf
  If ListSize(*Body\Inputs()) <> N + M
    ProcedureReturn PmcFail(Label + " body declares " + Str(ListSize(*Body\Inputs())) + " inputs; it must take the " + Str(N) + " state values and the " + Str(M) + " scan slices.")
  EndIf
  K = ListSize(*Body\Outputs()) - N
  If K < 0 Or ListSize(*Node\Outputs()) <> N + K
    ProcedureReturn PmcFail(Label + " declares " + Str(ListSize(*Node\Outputs())) + " outputs; its body returns " + Str(ListSize(*Body\Outputs())) + ", of which " + Str(N) + " are state values.")
  EndIf
  If (PmcAttribute(*Node, "scan_input_axes") And PmoEmitAttrListCount(*Node, "scan_input_axes") <> M) Or
     (PmcAttribute(*Node, "scan_input_directions") And PmoEmitAttrListCount(*Node, "scan_input_directions") <> M) Or
     (PmcAttribute(*Node, "scan_output_axes") And PmoEmitAttrListCount(*Node, "scan_output_axes") <> K) Or
     (PmcAttribute(*Node, "scan_output_directions") And PmoEmitAttrListCount(*Node, "scan_output_directions") <> K)
    ProcedureReturn PmcFail(Label + " has an axes or directions attribute whose length is not the number of scan inputs (" + Str(M) + ") or scan outputs (" + Str(K) + ").")
  EndIf
  ReDim ScanIn(M) : ReDim Elem(M)
  For j = 0 To M - 1
    ScanIn(j) = PmcInput(*Node, N + j)
    If ScanIn(j) = "" : ProcedureReturn PmcFail(Label + " scan input " + Str(j) + " is missing.") : EndIf
    SelectElement(*Body\Inputs(), N + j) : Elem(j) = *Body\Inputs()\Name
    If PmoEmitAttrListI(*Node, "scan_input_directions", j, 0) <> 0 And PmoEmitAttrListI(*Node, "scan_input_directions", j, 0) <> 1
      ProcedureReturn PmcFail(Label + " scan_input_directions holds " + Str(PmoEmitAttrListI(*Node, "scan_input_directions", j, 0)) + "; 0 (forward) and 1 (reverse) are defined.")
    EndIf
    If PmoEmitAttrListI(*Node, "scan_input_directions", j, 0) = 1 : Reverse = #True : EndIf
  Next
  ; the trip count: the first scan input's extent along its axis
  a = PmoEmitAttrListI(*Node, "scan_input_axes", 0, 0)
  If a < 0
    r = PmcDeclaredRank(*Top, *Graph, ScanIn(0))
    If r < 0 : ProcedureReturn PmcFail(Label + " scan input 0 has a negative axis and an undeclared rank; declare its shape.") : EndIf
    a + r
  EndIf
  Shp = PmcUniqueName("lowered", Taken())
  Trip = PmcUniqueName("lowered", Taken())
  PmcAddNode(*Graph, "Shape", ScanIn(0), Shp, #True)
  *Added = PmcAddNode(*Graph, "Gather", Shp + "," + PmcAddInt64(*Top, Str(a), #True, Taken()), Trip, #True)
  PmcAddIntAttribute(*Added, "axis", 0)
  If Reverse
    Last = PmcUniqueName("lowered", Taken())
    One = PmcAddInt64(*Top, "1", #True, Taken())
    PmcAddNode(*Graph, "Sub", Trip + "," + One, Last, #True)
  EndIf
  Truth = PmcAddTrue(*Top, Taken())
  If Truth = "" : ProcedureReturn #False : EndIf
  ; the body: iteration number and condition first, the slices made inside
  Iter = PmcUniqueName("lowered", Taken())
  CondIn = PmcUniqueName("lowered", Taken())
  CondOut = PmcUniqueName("lowered", Taken())
  For j = 1 To M
    LastElement(*Body\Inputs()) : DeleteElement(*Body\Inputs())
  Next
  FirstElement(*Body\Inputs())
  InsertElement(*Body\Inputs()) : PmcScalarValue(@*Body\Inputs(), CondIn, 9)
  InsertElement(*Body\Inputs()) : PmcScalarValue(@*Body\Inputs(), Iter, 7)
  If FirstElement(*Body\Outputs())
    InsertElement(*Body\Outputs())
  Else
    AddElement(*Body\Outputs())
  EndIf
  PmcScalarValue(@*Body\Outputs(), CondOut, 9)
  FirstElement(*Body\Nodes())
  PmcAddNode(*Body, "Identity", CondIn, CondOut, #True)
  For j = 0 To M - 1
    a = PmoEmitAttrListI(*Node, "scan_input_axes", j, 0)
    If a < 0
      r = PmcDeclaredRank(*Top, *Graph, ScanIn(j))
      If r < 0 : ProcedureReturn PmcFail(Label + " scan input " + Str(j) + " has a negative axis and an undeclared rank; declare its shape.") : EndIf
      a + r
    EndIf
    Idx = Iter
    If PmoEmitAttrListI(*Node, "scan_input_directions", j, 0) = 1
      Idx = PmcUniqueName("lowered", Taken())
      PmcAddNode(*Body, "Sub", Last + "," + Iter, Idx, #True)
    EndIf
    *Added = PmcAddNode(*Body, "Gather", ScanIn(j) + "," + Idx, Elem(j), #True)
    PmcAddIntAttribute(*Added, "axis", a)
  Next
  ; the node itself becomes the Loop
  *Node\Operation = "Loop"
  PmcLowered(Str(*Node)) = 1
  For j = 1 To M
    LastElement(*Node\Inputs()) : DeleteElement(*Node\Inputs())
  Next
  If FirstElement(*Node\Inputs())
    InsertElement(*Node\Inputs())
  Else
    AddElement(*Node\Inputs())
  EndIf
  *Node\Inputs() = Truth
  InsertElement(*Node\Inputs()) : *Node\Inputs() = Trip
  ; scan outputs: reversed and moved to their axis after the Loop
  For j = 0 To K - 1
    d = PmoEmitAttrListI(*Node, "scan_output_directions", j, 0)
    a = PmoEmitAttrListI(*Node, "scan_output_axes", j, 0)
    If d <> 0 And d <> 1
      ProcedureReturn PmcFail(Label + " scan_output_directions holds " + Str(d) + "; 0 (forward) and 1 (reverse) are defined.")
    EndIf
    If a <> 0
      SelectElement(*Body\Outputs(), 1 + N + j)
      r = -1
      If *Body\Outputs()\HasShape : r = ListSize(*Body\Outputs()\Dims()) + 1 : EndIf
      If r < 0 : ProcedureReturn PmcFail(Label + " scan output " + Str(j) + " has axis " + Str(a) + " and a body output of undeclared rank; declare the body output's shape.") : EndIf
      If a < 0 : a + r : EndIf
      If a < 0 Or a >= r : ProcedureReturn PmcFail(Label + " scan output " + Str(j) + " axis is outside the rank " + Str(r) + " of the stacked output.") : EndIf
    EndIf
    If d = 0 And a = 0 : Continue : EndIf
    SelectElement(*Node\Outputs(), N + j)
    Final = *Node\Outputs()
    Stacked = PmcUniqueName("lowered", Taken())
    *Node\Outputs() = Stacked
    If d = 1
      If a = 0
        PmcAddNode(*Graph, "Slice", Stacked + "," + PmcAddInt64(*Top, "-1", #False, Taken()) + "," + PmcAddInt64(*Top, "-2147483648", #False, Taken()) + "," +
                                    PmcAddInt64(*Top, "0", #False, Taken()) + "," + PmcAddInt64(*Top, "-1", #False, Taken()), Final, #False)
        Continue
      EndIf
      Idx = PmcUniqueName("lowered", Taken())
      PmcAddNode(*Graph, "Slice", Stacked + "," + PmcAddInt64(*Top, "-1", #False, Taken()) + "," + PmcAddInt64(*Top, "-2147483648", #False, Taken()) + "," +
                                  PmcAddInt64(*Top, "0", #False, Taken()) + "," + PmcAddInt64(*Top, "-1", #False, Taken()), Idx, #False)
      Stacked = Idx
    EndIf
    *Added = PmcAddNode(*Graph, "Transpose", Stacked, Final, #False)
    AddElement(*Added\Attributes())
    *Added\Attributes()\Name = "perm" : *Added\Attributes()\AttributeType = 7
    For d = 0 To r - 1
      AddElement(*Added\Attributes()\Integers())
      If d < a
        *Added\Attributes()\Integers() = d + 1
      ElseIf d = a
        *Added\Attributes()\Integers() = 0
      Else
        *Added\Attributes()\Integers() = d
      EndIf
    Next
    ChangeCurrentElement(*Graph\Nodes(), *Node)
  Next
  PmcKeepBody(*Node)
  ChangeCurrentElement(*Graph\Nodes(), *Node)
  While NextElement(*Graph\Nodes())
    If FindMapElement(PmcLowered(), Str(@*Graph\Nodes())) = 0 : PreviousElement(*Graph\Nodes()) : Break : EndIf
  Wend
  ProcedureReturn #True
EndProcedure

; Rewrites the current node of *Graph, a SequenceMap, into a Loop (see above).
Procedure.i PmcLowerSequenceMap(*Top.PmoOnnxGraph, *Graph.PmoOnnxGraph, Map Taken.i())
  Protected *Node.PmoOnnxNode = @*Graph\Nodes()
  Protected *Body.PmoOnnxGraph = PmcAttributeGraph(*Node, "body")
  Protected *Added.PmoOnnxNode
  Protected Label.s = PmcLabel(*Node)
  Protected Count.i, K.i, j.i, Kind.i
  Protected Length.s, Truth.s, Iter.s, CondIn.s, CondOut.s, Name.s, Acc.s, AccOut.s
  Protected Dim Element.i(0)
  Protected Dim Outer.s(0)
  If PmcAllowedAttributes(*Node, "|body|") = 0 : ProcedureReturn #False : EndIf
  If *Body = 0 : ProcedureReturn PmcFail(Label + " has no body graph attribute.") : EndIf
  Count = ListSize(*Node\Inputs())
  K = ListSize(*Node\Outputs())
  If ListSize(*Body\Inputs()) <> Count
    ProcedureReturn PmcFail(Label + " has " + Str(Count) + " inputs but its body declares " + Str(ListSize(*Body\Inputs())) + ".")
  EndIf
  If ListSize(*Body\Outputs()) <> K Or K < 1
    ProcedureReturn PmcFail(Label + " has " + Str(K) + " outputs but its body returns " + Str(ListSize(*Body\Outputs())) + ".")
  EndIf
  ReDim Element(K) : ReDim Outer(Count)
  For j = 0 To K - 1
    SelectElement(*Body\Outputs(), j)
    Element(j) = *Body\Outputs()\ElementType
    If PmcElementTypeCarried(Element(j)) = 0
      ProcedureReturn PmcFail(Label + " body output " + Str(j) + " declares element type " + Str(Element(j)) + "; declare FLOAT, INT32, INT64 or BOOL, the element type of the output sequence.")
    EndIf
  Next
  For j = 0 To Count - 1
    Outer(j) = PmcInput(*Node, j)
    If Outer(j) = "" : ProcedureReturn PmcFail(Label + " input " + Str(j) + " is missing.") : EndIf
  Next
  Length = PmcUniqueName("lowered", Taken())
  PmcAddNode(*Graph, "SequenceLength", Outer(0), Length, #True)
  Truth = PmcAddTrue(*Top, Taken())
  If Truth = "" : ProcedureReturn #False : EndIf
  ; the node's inputs become the Loop's: count, condition, empty sequences
  ClearList(*Node\Inputs())
  AddElement(*Node\Inputs()) : *Node\Inputs() = Length
  AddElement(*Node\Inputs()) : *Node\Inputs() = Truth
  For j = 0 To K - 1
    Name = PmcUniqueName("lowered", Taken())
    *Added = PmcAddNode(*Graph, "SequenceEmpty", "", Name, #True)
    PmcAddIntAttribute(*Added, "dtype", Element(j))
    AddElement(*Node\Inputs()) : *Node\Inputs() = Name
  Next
  ; the body: element reads first, the inserts last
  Iter = PmcUniqueName("lowered", Taken())
  CondIn = PmcUniqueName("lowered", Taken())
  CondOut = PmcUniqueName("lowered", Taken())
  FirstElement(*Body\Nodes())
  PmcAddNode(*Body, "Identity", CondIn, CondOut, #True)
  For j = 0 To Count - 1
    SelectElement(*Body\Inputs(), j)
    Name = *Body\Inputs()\Name
    Kind = #PMC_KIND_TENSOR
    If FindMapElement(PmcKind(), Outer(j)) : Kind = PmcKind() : EndIf
    If Kind = #PMC_KIND_SEQUENCE
      PmcAddNode(*Body, "SequenceAt", Outer(j) + "," + Iter, Name, #True)
    ElseIf j = 0
      ProcedureReturn PmcFail(Label + " input 0 is a tensor; SequenceMap maps over a sequence.")
    Else
      PmcAddNode(*Body, "Identity", Outer(j), Name, #True)
    EndIf
  Next
  ClearList(*Body\Inputs())
  AddElement(*Body\Inputs()) : PmcScalarValue(@*Body\Inputs(), Iter, 7)
  AddElement(*Body\Inputs()) : PmcScalarValue(@*Body\Inputs(), CondIn, 9)
  For j = 0 To K - 1
    Acc = PmcUniqueName("lowered", Taken())
    AccOut = PmcUniqueName("lowered", Taken())
    AddElement(*Body\Inputs())
    *Body\Inputs()\Name = Acc : *Body\Inputs()\ValueKind = #PMO_VALUE_SEQUENCE : *Body\Inputs()\SequenceElementType = Element(j)
    SelectElement(*Body\Outputs(), j)
    Name = *Body\Outputs()\Name
    LastElement(*Body\Nodes())
    PmcAddNode(*Body, "SequenceInsert", Acc + "," + Name, AccOut, #False)
    *Body\Outputs()\Name = AccOut : *Body\Outputs()\ValueKind = #PMO_VALUE_SEQUENCE : *Body\Outputs()\SequenceElementType = Element(j)
    *Body\Outputs()\HasTensorType = #False
  Next
  FirstElement(*Body\Outputs())
  InsertElement(*Body\Outputs()) : PmcScalarValue(@*Body\Outputs(), CondOut, 9)
  *Node\Operation = "Loop"
  PmcLowered(Str(*Node)) = 1
  PmcKeepBody(*Node)
  ProcedureReturn #True
EndProcedure

Procedure.i PmcLowerGraph(*Top.PmoOnnxGraph, *Graph.PmoOnnxGraph, Map Taken.i())
  Protected *Child.PmoOnnxGraph
  Protected *Node.PmoOnnxNode
  ForEach *Graph\Nodes()
    *Node = @*Graph\Nodes()
    ForEach *Node\Attributes()
      If *Node\Attributes()\Graph
        *Child = *Node\Attributes()\Graph
        If PmcLowerGraph(*Top, *Child, Taken()) = 0 : ProcedureReturn #False : EndIf
      EndIf
    Next
    If *Node\Domain = "" Or *Node\Domain = "ai.onnx"
      If *Node\Operation = "Scan"
        If PmcLowerScan(*Top, *Graph, Taken()) = 0 : ProcedureReturn #False : EndIf
      ElseIf *Node\Operation = "SequenceMap"
        If PmcLowerSequenceMap(*Top, *Graph, Taken()) = 0 : ProcedureReturn #False : EndIf
      EndIf
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmcGraphLowers(*Graph.PmoOnnxGraph)
  Protected *Child.PmoOnnxGraph
  ForEach *Graph\Nodes()
    If PmcIsLoweredOp(*Graph\Nodes()\Operation) : ProcedureReturn #True : EndIf
    ForEach *Graph\Nodes()\Attributes()
      *Child = *Graph\Nodes()\Attributes()\Graph
      If *Child And PmcGraphLowers(*Child) : ProcedureReturn #True : EndIf
    Next
  Next
  ProcedureReturn #False
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
  ; Scan and SequenceMap become Loop, with the value kinds of the pass above;
  ; the lowered graph is then inferred afresh.
  If PmcGraphLowers(*Graph)
    If PmcLowerGraph(*Graph, *Graph, Taken()) = 0 : ProcedureReturn #False : EndIf
    ClearMap(PmcKind()) : ClearMap(PmcSequenceElement()) : ClearMap(PmcOwned())
    If PmcInferGraph(*Graph, #True, 0) = 0 : ProcedureReturn #False : EndIf
  EndIf
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; Stage 2: the placeholder PmdCall returns
; ---------------------------------------------------------------------------

Procedure.s PmcCall(*Node.PmoOnnxNode)
  ProcedureReturn "; " + *Node\Operation + " is emitted once last uses are known"
EndProcedure
