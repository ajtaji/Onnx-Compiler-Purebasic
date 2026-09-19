; ============================================================================
; onnx_control_emit.pbi - stage 3 of If, Loop and the sequence operators:
; generated procedures and control code
; ----------------------------------------------------------------------------
; Included by onnx_dynamic_emit.pbi just before PmoDynamicCommand, after
; PmdCall, because a subgraph's ordinary nodes are emitted through PmdCall.
; Stages 1 and 2 are in onnx_control.pbi.
; ============================================================================

Declare.s PmcNodeText(*Node.PmoOnnxNode, *Context.PmcGraphInfo, IsTop.i, Index.i, Map Ids.i(), Map Known.i(),
                      Map Persistent.i(), Map Last.i(), *IdCounter.Integer, *Ir.PmoIrModel)

; ---------------------------------------------------------------------------
; Stage 3: generated procedures and control code
; ---------------------------------------------------------------------------

Procedure.s PmcNoError()
  If PmdPortable : ProcedureReturn "DError=0 And DCancel=0" : EndIf
  ProcedureReturn "DError=" + Chr(34) + Chr(34) + " And DCancel=0"
EndProcedure

Procedure.s PmcIdOf(Map Ids.i(), Name.s)
  If Name = "" : ProcedureReturn "0" : EndIf
  If FindMapElement(Ids(), Name) = 0 Or Ids() <= 0
    PmcFail("Value '" + Name + "' has no producer when control flow is emitted.")
    ProcedureReturn "0"
  EndIf
  ProcedureReturn Str(Ids())
EndProcedure

Procedure.i PmcNewId(*IdCounter.Integer)
  *IdCounter\i + 1
  ProcedureReturn *IdCounter\i
EndProcedure

; Names a node's subgraphs read from outside them, recursively. After
; stage 1 every name has one definition, so "defined inside" is a set test.
Procedure PmcCollectDefinitions(*Graph.PmoOnnxGraph, Map Inside.i())
  Protected *Child.PmoOnnxGraph
  ForEach *Graph\Inputs() : Inside(*Graph\Inputs()\Name) = 1 : Next
  ForEach *Graph\Initializers() : Inside(*Graph\Initializers()\Name) = 1 : Next
  ForEach *Graph\Nodes()
    ForEach *Graph\Nodes()\Outputs() : Inside(*Graph\Nodes()\Outputs()) = 1 : Next
    ForEach *Graph\Nodes()\Attributes()
      If *Graph\Nodes()\Attributes()\Graph
        *Child = *Graph\Nodes()\Attributes()\Graph
        PmcCollectDefinitions(*Child, Inside())
      EndIf
    Next
  Next
EndProcedure

Procedure PmcCollectReferences(*Graph.PmoOnnxGraph, Map Inside.i(), Map Captured.i())
  Protected *Child.PmoOnnxGraph
  ForEach *Graph\Nodes()
    ForEach *Graph\Nodes()\Inputs()
      If *Graph\Nodes()\Inputs() <> "" And FindMapElement(Inside(), *Graph\Nodes()\Inputs()) = 0
        Captured(*Graph\Nodes()\Inputs()) = 1
      EndIf
    Next
    ForEach *Graph\Nodes()\Attributes()
      If *Graph\Nodes()\Attributes()\Graph
        *Child = *Graph\Nodes()\Attributes()\Graph
        PmcCollectReferences(*Child, Inside(), Captured())
      EndIf
    Next
  Next
  ForEach *Graph\Outputs()
    If *Graph\Outputs()\Name <> "" And FindMapElement(Inside(), *Graph\Outputs()\Name) = 0
      Captured(*Graph\Outputs()\Name) = 1
    EndIf
  Next
EndProcedure

Procedure PmcCaptures(*Node.PmoOnnxNode, Map Captured.i())
  Protected *Child.PmoOnnxGraph
  NewMap Inside.i()
  ForEach *Node\Attributes()
    If *Node\Attributes()\Graph
      *Child = *Node\Attributes()\Graph
      PmcCollectDefinitions(*Child, Inside())
    EndIf
  Next
  ForEach *Node\Attributes()
    If *Node\Attributes()\Graph
      *Child = *Node\Attributes()\Graph
      PmcCollectReferences(*Child, Inside(), Captured())
    EndIf
  Next
EndProcedure

; A value may be moved out of its slot only when nothing reads it afterwards.
Procedure.i PmcMovable(*Context.PmcGraphInfo, IsTop.i, IdText.s, Index.i, Map Known.i(), Map Persistent.i(), Map Last.i())
  If IdText = "0" : ProcedureReturn #False : EndIf
  If IsTop
    If FindMapElement(Last(), IdText) = 0 Or Last() <> Index : ProcedureReturn #False : EndIf
    If FindMapElement(Persistent(), IdText) And Persistent() : ProcedureReturn #False : EndIf
    If FindMapElement(Known(), IdText) And Known() : ProcedureReturn #False : EndIf
    ProcedureReturn #True
  EndIf
  If FindMapElement(*Context\Local(), IdText) = 0 : ProcedureReturn #False : EndIf
  If FindMapElement(*Context\LastUse(), IdText) = 0 Or *Context\LastUse() <> Index : ProcedureReturn #False : EndIf
  ProcedureReturn #True
EndProcedure

Procedure PmcAddLine(List Lines.s(), Text.s)
  LastElement(Lines())
  AddElement(Lines())
  Lines() = Text
EndProcedure

Procedure.s PmcJoin(List Lines.s())
  Protected Text.s
  ForEach Lines()
    If ListIndex(Lines()) > 0 : Text + Chr(10) : EndIf
    Text + Lines()
  Next
  ProcedureReturn Text
EndProcedure

; Emits one subgraph as a generated procedure. Its inputs and node outputs get
; fresh tensor ids; its nodes are appended to the IR so weight packing,
; quantization and storage reduction see what they read.
Procedure.i PmcEmitGraph(*Graph.PmoOnnxGraph, *Info.PmcGraphInfo, Map Ids.i(), Map Known.i(), Map Persistent.i(),
                         Map Last.i(), *IdCounter.Integer, *Ir.PmoIrModel)
  Protected Index.i
  Protected EndIndex.i
  Protected NodeNumber.i
  Protected Call.s
  Protected IdText.s
  Protected *Node.PmoOnnxNode
  Protected *Constant.PmoIrConstant
  NewList Body.s()
  NewMap Produced.i()
  NewMap Captured.i()

  PmcNextGraph + 1
  *Info\Proc = "PmModelGraph" + Str(PmcNextGraph)
  *Info\NodeCount = ListSize(*Graph\Nodes())
  ForEach *Graph\Inputs()
    Ids(*Graph\Inputs()\Name) = PmcNewId(*IdCounter)
    IdText = Str(Ids(*Graph\Inputs()\Name))
    PmcAddLine(*Info\InputIds(), IdText)
    *Info\Local(IdText) = 1
  Next
  ; Pass A: ids for node outputs and every last use inside this graph.
  Index = 0
  ForEach *Graph\Nodes()
    *Node = @*Graph\Nodes()
    ForEach *Node\Inputs()
      If *Node\Inputs() <> ""
        IdText = PmcIdOf(Ids(), *Node\Inputs())
        If PmcError <> "" : ProcedureReturn #False : EndIf
        *Info\LastUse(IdText) = Index
      EndIf
    Next
    If PmcIsControl(*Node\Operation)
      ClearMap(Captured())
      PmcCaptures(*Node, Captured())
      ForEach Captured()
        IdText = PmcIdOf(Ids(), MapKey(Captured()))
        If PmcError <> "" : ProcedureReturn #False : EndIf
        *Info\LastUse(IdText) = Index
      Next
    EndIf
    ForEach *Node\Outputs()
      If *Node\Outputs() <> ""
        Ids(*Node\Outputs()) = PmcNewId(*IdCounter)
        IdText = Str(Ids(*Node\Outputs()))
        *Info\Local(IdText) = 1
        Produced(IdText) = 1
      EndIf
    Next
    Index + 1
  Next
  EndIndex = Index
  ForEach *Graph\Outputs()
    IdText = PmcIdOf(Ids(), *Graph\Outputs()\Name)
    If PmcError <> "" : ProcedureReturn #False : EndIf
    PmcAddLine(*Info\OutputIds(), IdText)
    *Info\LastUse(IdText) = EndIndex
    ; A subgraph output that is a constant is live-out: pack it.
    If FindMapElement(*Ir\ConstantByName(), *Graph\Outputs()\Name)
      LastElement(*Ir\Outputs()) : AddElement(*Ir\Outputs()) : *Ir\Outputs() = *Graph\Outputs()\Name
    EndIf
  Next

  ; Pass B: the procedure text.
  PmcAddLine(Body(), "Procedure " + *Info\Proc + "()")
  PmcAddLine(Body(), "  If DRejectProgressReentry()")
  PmcAddLine(Body(), "    ProcedureReturn")
  PmcAddLine(Body(), "  EndIf")
  Index = 0
  ForEach *Graph\Nodes()
    *Node = @*Graph\Nodes()
    NodeNumber = PmcNextNode : PmcNextNode + 1 : PmcBodyNodeCount + 1
    LastElement(*Ir\Nodes()) : AddElement(*Ir\Nodes())
    *Ir\Nodes()\Index = NodeNumber : *Ir\Nodes()\Node = *Node
    If PmcOwnsNode(*Node)
      If PmcValidateNode(*Node) = 0 : ProcedureReturn #False : EndIf
      Call = PmcNodeText(*Node, *Info, #False, Index, Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir)
      If PmcError <> "" : ProcedureReturn #False : EndIf
    Else
      PmoDynamicError = ""
      Call = PmdCall(*Node, Ids())
      If PmoDynamicError <> "" : ProcedureReturn PmcFail(PmoDynamicError) : EndIf
    EndIf
    PmcAddLine(Body(), "  DNode=" + Str(NodeNumber))
    PmcAddLine(Body(), "  If DPoll(#PMD_PROGRESS_NODE_BEFORE," + Str(NodeNumber) + ")=0 : ProcedureReturn : EndIf")
    PmcAddLine(Body(), "  " + ReplaceString(Call, Chr(10), Chr(10) + "  "))
    PmcAddLine(Body(), "  If DPoll(#PMD_PROGRESS_NODE_AFTER," + Str(NodeNumber) + ")=0 : ProcedureReturn : EndIf")
    PmcAddLine(Body(), "  If DError<>" + Chr(34) + Chr(34) + " Or DCancel : ProcedureReturn : EndIf")
    ; Release what this graph owns at its last use; a node output nobody
    ; reads is released straight after the node that made it.
    ForEach *Info\Local()
      IdText = MapKey(*Info\Local())
      If FindMapElement(*Info\LastUse(), IdText)
        If *Info\LastUse() = Index : PmcAddLine(Body(), "  DRelease(" + IdText + ")") : EndIf
      ElseIf FindMapElement(Produced(), IdText)
        ForEach *Node\Outputs()
          If *Node\Outputs() <> "" And Str(Ids(*Node\Outputs())) = IdText
            PmcAddLine(Body(), "  DRelease(" + IdText + ")")
          EndIf
        Next
      EndIf
    Next
    Index + 1
  Next
  PmcAddLine(Body(), "EndProcedure")
  ; Children were appended while this text was built, so they precede it.
  ForEach Body()
    LastElement(PmcProcedureLines()) : AddElement(PmcProcedureLines()) : PmcProcedureLines() = Body()
  Next
  ProcedureReturn #True
EndProcedure

; When one source feeds several destinations, only its last use may move.
Procedure.i PmcLastOccurrence(List Sources.s(), Position.i)
  Protected Text.s
  If SelectElement(Sources(), Position) = 0 : ProcedureReturn #False : EndIf
  Text = Sources()
  While NextElement(Sources())
    If Sources() = Text : ProcedureReturn #False : EndIf
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.s PmcIfText(*Node.PmoOnnxNode, *Context.PmcGraphInfo, IsTop.i, Index.i, Map Ids.i(), Map Known.i(),
                      Map Persistent.i(), Map Last.i(), *IdCounter.Integer, *Ir.PmoIrModel)
  Protected State.s
  Protected Branch.i
  Protected Position.i
  Protected Output.s
  Protected Source.s
  Protected Move.i
  Protected *Graph.PmoOnnxGraph
  Protected Dim Info.PmcGraphInfo(1)
  NewList Lines.s()
  NewList Sources.s()
  PmcNextGraph + 1
  State = "PmIfGo" + Str(PmcNextGraph)
  PmcAddLine(PmcGlobalLines(), "Global " + State + ".i")
  For Branch = 0 To 1
    If Branch = 0 : *Graph = PmcAttributeGraph(*Node, "then_branch") : Else : *Graph = PmcAttributeGraph(*Node, "else_branch") : EndIf
    If PmcEmitGraph(*Graph, @Info(Branch), Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir) = 0 : ProcedureReturn "" : EndIf
  Next
  PmcAddLine(Lines(), State + "=DScalarTruth(" + PmcIdOf(Ids(), PmcInput(*Node, 0)) + ")")
  PmcAddLine(Lines(), "If " + PmcNoError())
  For Branch = 0 To 1
    If Branch = 0
      PmcAddLine(Lines(), "  If " + State + "<>0")
    Else
      PmcAddLine(Lines(), "  Else")
    EndIf
    PmcAddLine(Lines(), "    " + Info(Branch)\Proc + "()")
    PmcAddLine(Lines(), "    If " + PmcNoError())
    ClearList(Sources())
    ForEach Info(Branch)\OutputIds() : PmcAddLine(Sources(), Info(Branch)\OutputIds()) : Next
    For Position = 0 To ListSize(*Node\Outputs()) - 1
      Output = PmcOutput(*Node, Position)
      SelectElement(Sources(), Position) : Source = Sources()
      Move = Bool(FindMapElement(Info(Branch)\Local(), Source) And PmcLastOccurrence(Sources(), Position))
      If Output = ""
        If FindMapElement(Info(Branch)\Local(), Source) : PmcAddLine(Lines(), "      DRelease(" + Source + ")") : EndIf
        Continue
      EndIf
      PmcAddLine(Lines(), "      DValueTake(" + PmcIdOf(Ids(), Output) + "," + Source + "," + Str(Move) + ")")
    Next
    PmcAddLine(Lines(), "    EndIf")
  Next
  PmcAddLine(Lines(), "  EndIf")
  PmcAddLine(Lines(), "EndIf")
  ProcedureReturn PmcJoin(Lines())
EndProcedure

Procedure.s PmcScanDims(*Value.PmoOnnxValue, *Rank.Integer)
  Protected Text.s
  *Rank\i = -1
  If *Value\HasShape = 0 : ProcedureReturn "" : EndIf
  ForEach *Value\Dims()
    ; ONNX does not define the shape of a scan output after zero iterations;
    ; only a fully declared shape gives it one. Otherwise the runtime refuses.
    If *Value\Dims()\HasValue = 0 : ProcedureReturn "" : EndIf
    If Text <> "" : Text + "," : EndIf
    Text + Str(*Value\Dims()\Value)
  Next
  *Rank\i = ListSize(*Value\Dims())
  ProcedureReturn Text
EndProcedure

Procedure.s PmcLoopText(*Node.PmoOnnxNode, *Context.PmcGraphInfo, IsTop.i, Index.i, Map Ids.i(), Map Known.i(),
                        Map Persistent.i(), Map Last.i(), *IdCounter.Integer, *Ir.PmoIrModel)
  Protected *Body.PmoOnnxGraph = PmcAttributeGraph(*Node, "body")
  Protected Info.PmcGraphInfo
  Protected Tag.s
  Protected Carried.i
  Protected Scans.i
  Protected K.i
  Protected Initial.s
  Protected Move.i
  Protected Uses.i
  Protected Rank.Integer
  Protected Dims.s
  Protected Kind.i
  Protected Output.s
  NewList Lines.s()
  NewList InputIds.s()
  NewList OutputIds.s()
  NewList Temps.s()
  NewList Accumulators.s()
  NewList CarriedSources.s()
  NewMap Captured.i()
  NewMap Moved.i()

  Carried = ListSize(*Node\Inputs()) - 2
  Scans = ListSize(*Body\Outputs()) - 1 - Carried
  PmcNextGraph + 1
  Tag = Str(PmcNextGraph)
  PmcAddLine(PmcGlobalLines(), "Global PmLoopTrip" + Tag + ".i")
  PmcAddLine(PmcGlobalLines(), "Global PmLoopLimit" + Tag + ".i")
  PmcAddLine(PmcGlobalLines(), "Global PmLoopBounded" + Tag + ".i")
  PmcAddLine(PmcGlobalLines(), "Global PmLoopGo" + Tag + ".i")
  PmcCaptures(*Node, Captured())
  If PmcEmitGraph(*Body, @Info, Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir) = 0 : ProcedureReturn "" : EndIf
  ForEach Info\InputIds() : PmcAddLine(InputIds(), Info\InputIds()) : Next
  ForEach Info\OutputIds() : PmcAddLine(OutputIds(), Info\OutputIds()) : Next
  For K = 0 To Carried - 1
    PmcAddLine(Temps(), Str(PmcNewId(*IdCounter)))
    SelectElement(OutputIds(), K + 1) : PmcAddLine(CarriedSources(), OutputIds())
  Next
  For K = 0 To Scans - 1
    PmcAddLine(Accumulators(), Str(PmcNewId(*IdCounter)))
  Next

  PmcAddLine(Lines(), "PmLoopTrip" + Tag + "=0")
  If PmcInput(*Node, 0) <> ""
    PmcAddLine(Lines(), "PmLoopBounded" + Tag + "=1")
    PmcAddLine(Lines(), "PmLoopLimit" + Tag + "=DScalarCount(" + PmcIdOf(Ids(), PmcInput(*Node, 0)) + ")")
  Else
    PmcAddLine(Lines(), "PmLoopBounded" + Tag + "=0")
    PmcAddLine(Lines(), "PmLoopLimit" + Tag + "=0")
  EndIf
  If PmcInput(*Node, 1) <> ""
    PmcAddLine(Lines(), "PmLoopGo" + Tag + "=DScalarTruth(" + PmcIdOf(Ids(), PmcInput(*Node, 1)) + ")")
  Else
    PmcAddLine(Lines(), "PmLoopGo" + Tag + "=1")
  EndIf
  ; Initial carried values. The loop's own inputs are read once, here; a
  ; value nothing reads afterwards is moved into the body's slot.
  For K = 0 To Carried - 1
    Initial = PmcIdOf(Ids(), PmcInput(*Node, K + 2))
    Uses = 0
    ForEach *Node\Inputs()
      If *Node\Inputs() <> "" And Str(Ids(*Node\Inputs())) = Initial : Uses + 1 : EndIf
    Next
    Move = Bool(Uses = 1 And FindMapElement(Captured(), PmcInput(*Node, K + 2)) = 0 And
                PmcMovable(*Context, IsTop, Initial, Index, Known(), Persistent(), Last()))
    SelectElement(InputIds(), K + 2)
    PmcAddLine(Lines(), "DValueTake(" + InputIds() + "," + Initial + "," + Str(Move) + ")")
  Next
  ForEach Accumulators() : PmcAddLine(Lines(), "DRelease(" + Accumulators() + ")") : Next

  PmcAddLine(Lines(), "While PmLoopGo" + Tag + "<>0")
  PmcAddLine(Lines(), "  If PmLoopBounded" + Tag + "<>0")
  PmcAddLine(Lines(), "    If PmLoopTrip" + Tag + ">=PmLoopLimit" + Tag)
  PmcAddLine(Lines(), "      PmLoopGo" + Tag + "=0")
  PmcAddLine(Lines(), "    EndIf")
  PmcAddLine(Lines(), "  EndIf")
  If PmdPortable
    PmcAddLine(Lines(), "  If DError<>0 Or DCancel<>0")
  Else
    PmcAddLine(Lines(), "  If DError<>" + Chr(34) + Chr(34) + " Or DCancel<>0")
  EndIf
  PmcAddLine(Lines(), "    PmLoopGo" + Tag + "=0")
  PmcAddLine(Lines(), "  EndIf")
  PmcAddLine(Lines(), "  If PmLoopGo" + Tag + "<>0")
  SelectElement(InputIds(), 0) : PmcAddLine(Lines(), "    DScalarSet(" + InputIds() + ",7,PmLoopTrip" + Tag + ")")
  SelectElement(InputIds(), 1) : PmcAddLine(Lines(), "    DScalarSet(" + InputIds() + ",9,1)")
  PmcAddLine(Lines(), "    " + Info\Proc + "()")
  PmcAddLine(Lines(), "    If " + PmcNoError())
  If PmcInput(*Node, 1) <> ""
    SelectElement(OutputIds(), 0) : PmcAddLine(Lines(), "      PmLoopGo" + Tag + "=DScalarTruth(" + OutputIds() + ")")
  EndIf
  For K = 0 To Scans - 1
    SelectElement(OutputIds(), K + 1 + Carried) : SelectElement(Accumulators(), K)
    PmcAddLine(Lines(), "      DSeqAppendTensor(" + Accumulators() + "," + OutputIds() + ")")
  Next
  ; Two phases, so carried values that trade places are read before any
  ; slot is overwritten.
  For K = 0 To Carried - 1
    SelectElement(CarriedSources(), K) : SelectElement(InputIds(), K + 2) : SelectElement(Temps(), K)
    If CarriedSources() = InputIds() : Continue : EndIf
    Move = Bool(FindMapElement(Info\Local(), CarriedSources()) And PmcLastOccurrence(CarriedSources(), K))
    SelectElement(CarriedSources(), K)
    If Move : Moved(CarriedSources()) = 1 : EndIf
    PmcAddLine(Lines(), "      DValueTake(" + Temps() + "," + CarriedSources() + "," + Str(Move) + ")")
  Next
  For K = 0 To Carried - 1
    SelectElement(CarriedSources(), K) : SelectElement(InputIds(), K + 2) : SelectElement(Temps(), K)
    If CarriedSources() = InputIds() : Continue : EndIf
    PmcAddLine(Lines(), "      DValueMove(" + InputIds() + "," + Temps() + ")")
  Next
  ; Body outputs the body produced and nothing took over are released now.
  ForEach OutputIds()
    If FindMapElement(Info\Local(), OutputIds()) = 0 Or FindMapElement(Moved(), OutputIds()) : Continue : EndIf
    K = 0
    ForEach InputIds()
      If InputIds() = OutputIds() : K = 1 : EndIf
    Next
    If K = 0 : PmcAddLine(Lines(), "      DRelease(" + OutputIds() + ")") : EndIf
  Next
  PmcAddLine(Lines(), "    EndIf")
  PmcAddLine(Lines(), "    PmLoopTrip" + Tag + "=PmLoopTrip" + Tag + "+1")
  PmcAddLine(Lines(), "  EndIf")
  PmcAddLine(Lines(), "Wend")

  PmcAddLine(Lines(), "If " + PmcNoError())
  For K = 0 To Carried - 1
    Output = PmcOutput(*Node, K)
    SelectElement(InputIds(), K + 2)
    If Output = "" : Continue : EndIf
    PmcAddLine(Lines(), "  DValueMove(" + PmcIdOf(Ids(), Output) + "," + InputIds() + ")")
  Next
  For K = 0 To Scans - 1
    Output = PmcOutput(*Node, Carried + K)
    If Output = "" : Continue : EndIf
    SelectElement(*Body\Outputs(), K + 1 + Carried)
    Kind = *Body\Outputs()\ElementType
    Dims = PmcScanDims(@*Body\Outputs(), @Rank)
    SelectElement(Accumulators(), K)
    PmcAddLine(Lines(), "  DSeqStack(" + PmcIdOf(Ids(), Output) + "," + Accumulators() + "," + Str(Kind) + "," + Str(Rank\i) + "," + Chr(34) + Dims + Chr(34) + ")")
  Next
  PmcAddLine(Lines(), "EndIf")
  ForEach InputIds() : PmcAddLine(Lines(), "DRelease(" + InputIds() + ")") : Next
  ForEach Temps() : PmcAddLine(Lines(), "DRelease(" + Temps() + ")") : Next
  ForEach Accumulators() : PmcAddLine(Lines(), "DRelease(" + Accumulators() + ")") : Next
  ProcedureReturn PmcJoin(Lines())
EndProcedure

Procedure.s PmcNodeText(*Node.PmoOnnxNode, *Context.PmcGraphInfo, IsTop.i, Index.i, Map Ids.i(), Map Known.i(),
                        Map Persistent.i(), Map Last.i(), *IdCounter.Integer, *Ir.PmoIrModel)
  Protected y.s = PmcIdOf(Ids(), PmcOutput(*Node, 0))
  Protected a.s = PmcIdOf(Ids(), PmcInput(*Node, 0))
  Protected b.s = PmcIdOf(Ids(), PmcInput(*Node, 1))
  Protected c.s = PmcIdOf(Ids(), PmcInput(*Node, 2))
  Protected Joined.s
  Protected Move.i
  If PmcError <> "" : ProcedureReturn "" : EndIf
  Select *Node\Operation
    Case "SequenceEmpty"
      ProcedureReturn "DSeqEmpty(" + y + "," + Str(PmoEmitAttrI(*Node, "dtype", 1)) + ")"
    Case "SequenceConstruct"
      ForEach *Node\Inputs()
        If Joined <> "" : Joined + "," : EndIf
        Joined + PmcIdOf(Ids(), *Node\Inputs())
      Next
      ProcedureReturn "DSeqConstruct(" + y + "," + Chr(34) + Joined + Chr(34) + ")"
    Case "SequenceInsert"
      Move = Bool(a <> b And a <> c And PmcMovable(*Context, IsTop, a, Index, Known(), Persistent(), Last()))
      ProcedureReturn "DSeqInsert(" + y + "," + a + "," + b + "," + c + "," + Str(Move) + ")"
    Case "SequenceAt"
      ProcedureReturn "DSeqAt(" + y + "," + a + "," + b + ")"
    Case "SequenceLength"
      ProcedureReturn "DSeqLength(" + y + "," + a + ")"
    Case "SplitToSequence"
      ProcedureReturn "DSplitToSequence(" + y + "," + a + "," + b + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," + Str(PmoEmitAttrI(*Node, "keepdims", 1)) + ")"
    Case "Identity"
      Move = Bool(PmcMovable(*Context, IsTop, a, Index, Known(), Persistent(), Last()))
      ProcedureReturn "DValueTake(" + y + "," + a + "," + Str(Move) + ")"
    Case "ConcatFromSequence"
      ProcedureReturn "DConcatFromSequence(" + y + "," + a + "," + Str(PmoEmitAttrI(*Node, "axis", 0)) + "," + Str(PmoEmitAttrI(*Node, "new_axis", 0)) + ")"
    Case "If"
      ProcedureReturn PmcIfText(*Node, *Context, IsTop, Index, Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir)
    Case "Loop"
      ProcedureReturn PmcLoopText(*Node, *Context, IsTop, Index, Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir)
  EndSelect
  PmcFail("Internal: " + PmcLabel(*Node) + " reached the control-flow emitter.")
  ProcedureReturn ""
EndProcedure

; Stage 3 entry. Runs after the main schedule has computed last uses and
; before weights are quantized and packed.
Procedure.i PmcFinish(*Model.PmoOnnxModel, *Ir.PmoIrModel, Map Ids.i(), Map Known.i(), Map Persistent.i(),
                      Map Last.i(), List Calls.s(), *IdCounter.Integer, TopCount.i)
  Protected Index.i
  Protected IdText.s
  Protected Text.s
  Protected Top.PmcGraphInfo
  NewMap Captured.i()
  PmcError = ""
  PmcNextNode = TopCount
  PmcBodyNodeCount = 0
  If PmcUsed = 0 : ProcedureReturn #True : EndIf
  ; Captured outer values live until their If/Loop has run.
  Index = 0
  ForEach *Model\Graph\Nodes()
    If PmcIsControl(*Model\Graph\Nodes()\Operation)
      ClearMap(Captured())
      PmcCaptures(@*Model\Graph\Nodes(), Captured())
      ForEach Captured()
        IdText = PmcIdOf(Ids(), MapKey(Captured()))
        If PmcError <> "" : ProcedureReturn #False : EndIf
        If FindMapElement(Last(), IdText) = 0 Or Last() < Index : Last(IdText) = Index : EndIf
        If FindMapElement(Known(), IdText) And Known() : Persistent(IdText) = 1 : EndIf
      Next
    EndIf
    Index + 1
  Next
  Index = 0
  ForEach *Model\Graph\Nodes()
    If PmcOwnsNode(@*Model\Graph\Nodes())
      Text = PmcNodeText(@*Model\Graph\Nodes(), @Top, #True, Index, Ids(), Known(), Persistent(), Last(), *IdCounter, *Ir)
      If PmcError <> "" : ProcedureReturn #False : EndIf
      If SelectElement(Calls(), Index) = 0 : ProcedureReturn PmcFail("Internal: the schedule lost a control-flow node.") : EndIf
      ; The schedule indents the first line; indent the rest to match.
      Calls() = ReplaceString(Text, Chr(10), Chr(10) + "  ")
    EndIf
    Index + 1
  Next
  ProcedureReturn #True
EndProcedure

Procedure PmcEmitProcedures(File.i)
  If PmcUsed = 0 : ProcedureReturn : EndIf
  ForEach PmcGlobalLines() : PmdLine(File, PmcGlobalLines()) : Next
  ForEach PmcProcedureLines() : PmdLine(File, PmcProcedureLines()) : Next
EndProcedure

; The check PmModelExecute makes for a sequence graph input.
Procedure.s PmcInputCheck(Id.i, ElementType.i)
  Protected Text.s = "  If Dt(" + Str(Id) + ")\Kind<>#PMD_KIND_SEQUENCE"
  If ElementType : Text + " Or DSeqElementKind(" + Str(Id) + ")<>" + Str(ElementType) : EndIf
  ProcedureReturn Text + " : ProcedureReturn DFail(" + Chr(34) + "A sequence input was not supplied as a sequence of the declared element type; build it with DSeqReset and DSeqAppendShape." + Chr(34) + ") : EndIf"
EndProcedure

DataSection
  PmcWindowsRuntimeStart:
  IncludeBinary "../../runtime/tensor_control_windows.pbi"
  PmcWindowsRuntimeEnd:
  PmcPortableRuntimeStart:
  IncludeBinary "../../runtime/tensor_control_portable.pmi"
  PmcPortableRuntimeEnd:
EndDataSection

Procedure.s PmcRuntimeFile()
  If PmdPortable : ProcedureReturn "tensor_control_portable.pmi" : EndIf
  ProcedureReturn "tensor_control_windows.pbi"
EndProcedure

Procedure.i PmcExportRuntime(Folder.s)
  If PmcUsed = 0 : ProcedureReturn #True : EndIf
  If PmdPortable
    ProcedureReturn PmdWriteSupport(Folder + PmcRuntimeFile(), ?PmcPortableRuntimeStart, ?PmcPortableRuntimeEnd - ?PmcPortableRuntimeStart)
  EndIf
  ProcedureReturn PmdWriteSupport(Folder + PmcRuntimeFile(), ?PmcWindowsRuntimeStart, ?PmcWindowsRuntimeEnd - ?PmcWindowsRuntimeStart)
EndProcedure
