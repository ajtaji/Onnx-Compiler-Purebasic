; ============================================================================
; onnx_emit_norm_small.pbi - fixed-shape forms of InstanceNormalization, TopK,
; ScatterElements, ReduceMax, ReduceProd, Not and Pad
; ----------------------------------------------------------------------------
; Included by onnx_emit.pbi. For each node this writes one generated
; procedure that fills the kernel's argument block with the node's static
; shapes and calls the kernel in runtime/tensor_norm_small*.p* (Identity is
; an alias and needs none). Forms the ONNX specification
; (https://onnx.ai/onnx/operators/) allows up to opset 20 are either emitted
; or refused with one sentence naming the operator, the attribute or input and
; the value. The runtime-dimension path implements the same forms; see
; onnx_dynamic_norm_small.pbi.
; ============================================================================

Procedure.i PmoNsOwns(Operation.s)
  ProcedureReturn Bool(FindString("|InstanceNormalization|TopK|ScatterElements|Scatter|ReduceMax|ReduceProd|Not|Identity|Pad|", "|" + Operation + "|"))
EndProcedure

; The oldest ai.onnx opset whose definition of an operator is one these
; kernels compute. The fixed-shape path also reads the older attribute forms
; (TopK-1's k, Pad-2's pads and value); the runtime-dimension path does not.
;   InstanceNormalization 6   (InstanceNormalization-1 carried consumed_inputs)
;   TopK 1 fixed, 10 runtime  (TopK-10 has no largest/sorted and leaves ties
;                              unordered, so index order is one it allows)
;   ScatterElements 11        (reduction add/mul from 16, max/min from 18)
;   ReduceMax, ReduceProd 1   (axes attribute through 17, axes input from 18)
;   Not 1, Identity 1         (tensors)
;   Pad 2 fixed, 11 runtime   (Pad-1 named the attribute paddings; axes input
;                              from 18, mode wrap from 19)
; 0 for an operator this file does not own.
Procedure.i PmoNsFloor(Operation.s, FixedShape.i)
  Select Operation
    Case "InstanceNormalization" : ProcedureReturn 6
    Case "TopK" : If FixedShape : ProcedureReturn 1 : EndIf : ProcedureReturn 10
    Case "ScatterElements" : ProcedureReturn 11
    Case "Scatter" : ProcedureReturn 9
    Case "Pad" : If FixedShape : ProcedureReturn 2 : EndIf : ProcedureReturn 11
    Case "ReduceMax", "ReduceProd", "Not", "Identity" : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

; 1 when every node of a graph imported below the fixed-shape path's audited
; opset 7 is an operator above whose definition at that opset is the one
; implemented, so the older import changes nothing that is emitted.
Procedure.i PmoNsOpsetCovers(*Model.PmoOnnxModel, Opset.i)
  If Opset < 1 Or ListSize(*Model\Graph\Nodes()) = 0 : ProcedureReturn #False : EndIf
  ForEach *Model\Graph\Nodes()
    If PmoNsOwns(*Model\Graph\Nodes()\Operation) = 0 : ProcedureReturn #False : EndIf
    If Opset < PmoNsFloor(*Model\Graph\Nodes()\Operation, #True) : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.q PmoEmitNsMax(A.q, B.q)
  If A > B : ProcedureReturn A : EndIf
  ProcedureReturn B
EndProcedure

Procedure.i PmoEmitNsOpset(*Ir.PmoIrModel)
  Protected Version.i
  ForEach *Ir\Source\Opsets()
    If *Ir\Source\Opsets()\Domain = "" Or *Ir\Source\Opsets()\Domain = "ai.onnx"
      Version = *Ir\Source\Opsets()\Version
    EndIf
  Next
  ProcedureReturn Version
EndProcedure

Procedure.i PmoEmitNsFail(*Node.PmoOnnxNode, Reason.s)
  Protected Name.s = *Node\Name
  If Name = "" : Name = "(unnamed)" : EndIf
  ProcedureReturn PmoEmitFail(*Node\Operation + " node " + Name + ": " + Reason)
EndProcedure

Procedure.s PmoEmitNsTypeName(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "FLOAT"
    Case 6 : ProcedureReturn "INT32"
    Case 7 : ProcedureReturn "INT64"
    Case 9 : ProcedureReturn "BOOL"
  EndSelect
  ProcedureReturn "element type " + Str(ElementType)
EndProcedure

Procedure.i PmoEmitNsAttributePresent(*Node.PmoOnnxNode, Name.s)
  ForEach *Node\Attributes()
    If *Node\Attributes()\Name = Name : ProcedureReturn #True : EndIf
  Next
  ProcedureReturn #False
EndProcedure

; Refuses any attribute outside Allowed ("|a|b|").
Procedure.i PmoEmitNsAttributesAllowed(*Node.PmoOnnxNode, Allowed.s, Opset.i)
  ForEach *Node\Attributes()
    If FindString(Allowed, "|" + *Node\Attributes()\Name + "|") = 0
      If Trim(Allowed, "|") = ""
        ProcedureReturn PmoEmitNsFail(*Node, "attribute " + *Node\Attributes()\Name + " is not part of " + *Node\Operation + ", which takes no attribute.")
      EndIf
      ProcedureReturn PmoEmitNsFail(*Node, "attribute " + *Node\Attributes()\Name + " is not part of " + *Node\Operation + " at opset " + Str(Opset) +
                                           "; its attributes there are " + ReplaceString(Trim(Allowed, "|"), "|", ", ") + ".")
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

; Emits "  <array>(i) = <dim>" for every axis of a value.
Procedure PmoEmitNsDims(File.i, ArrayName.s, *Value.PmoIrValue)
  Protected Axis.i
  For Axis = 0 To PmoEmitRank(*Value) - 1
    PmoEmitLine(File, "  " + ArrayName + "(" + Str(Axis) + ") = " + Str(PmoEmitDim(*Value, Axis)))
  Next
EndProcedure

Procedure.i PmoEmitNsInstanceNorm(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Scale.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Bias.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  If Opset < 6 : ProcedureReturn PmoEmitNsFail(*Node, "the model imports ai.onnx opset " + Str(Opset) + ", whose InstanceNormalization differs from InstanceNormalization-6, the definition implemented.") : EndIf
  If PmoEmitNsAttributesAllowed(*Node, "|epsilon|", Opset) = 0 : ProcedureReturn #False : EndIf
  If *X = 0 Or *Scale = 0 Or *Bias = 0 : ProcedureReturn PmoEmitNsFail(*Node, "inputs input, scale and B are all required.") : EndIf
  If *X\ElementType <> 1 Or *Scale\ElementType <> 1 Or *Bias\ElementType <> 1
    ProcedureReturn PmoEmitNsFail(*Node, "input type " + PmoEmitNsTypeName(*X\ElementType) + " is not implemented; FLOAT is.")
  EndIf
  If PmoEmitRank(*X) < 3 : ProcedureReturn PmoEmitNsFail(*Node, "the input has rank " + Str(PmoEmitRank(*X)) + "; the specification's form is N x C x D1 ... Dn.") : EndIf
  If *Scale\Elements <> PmoEmitDim(*X, 1) Or *Bias\Elements <> PmoEmitDim(*X, 1)
    ProcedureReturn PmoEmitNsFail(*Node, "scale and B must each hold C = " + Str(PmoEmitDim(*X, 1)) + " elements.")
  EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*x, *scale, *bias, *dst)")
  PmoEmitLine(File, "  PmTensorInstanceNorm(*x, *scale, *bias, *dst, " + Str(PmoEmitDim(*X, 0)) + ", " + Str(PmoEmitDim(*X, 1)) + ", " +
                    Str(PmoEmitProduct(*X, 2, PmoEmitRank(*X) - 1)) + ", " + PmoEmitFloat(PmoEmitAttrF(*Node, "epsilon", 0.00001)) + ")")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitNsNot(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  If PmoEmitNsAttributesAllowed(*Node, "|", Opset) = 0 : ProcedureReturn #False : EndIf
  If *X = 0 Or *X\ElementType <> 9 : ProcedureReturn PmoEmitNsFail(*Node, "the input must be a BOOL tensor.") : EndIf
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitLine(File, "  PmTensorNot(*src, *dst, " + Str(*X\Elements) + ")")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitNsTopK(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *X.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Values.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Indices.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 1))
  Protected Ok.Integer, K.q, Axis.i, Rank.i, Width.q, Count.q, Index.i
  If Opset < 10
    If PmoEmitNsAttributesAllowed(*Node, "|axis|k|", Opset) = 0 : ProcedureReturn #False : EndIf
    If PmoEmitNsAttributePresent(*Node, "k") = 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute k is required before opset 10.") : EndIf
    K = PmoEmitAttrI(*Node, "k", 0)
  Else
    If Opset < 11
      If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
    ElseIf PmoEmitNsAttributesAllowed(*Node, "|axis|largest|sorted|", Opset) = 0
      ProcedureReturn #False
    EndIf
    If PmoEmitInput(*Node, 1) = "" : ProcedureReturn PmoEmitNsFail(*Node, "input K is required from opset 10.") : EndIf
    K = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), 0, @Ok)
    If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input K must be a constant INT64 tensor for fixed-shape emission.") : EndIf
  EndIf
  If *X = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input X is required.") : EndIf
  If *X\ElementType <> 1 And *X\ElementType <> 7
    ProcedureReturn PmoEmitNsFail(*Node, "input type " + PmoEmitNsTypeName(*X\ElementType) + " is not implemented by fixed-shape emission; FLOAT and INT64 are.")
  EndIf
  Rank = PmoEmitRank(*X)
  If Rank < 1 : ProcedureReturn PmoEmitNsFail(*Node, "the input must have rank one or more.") : EndIf
  Axis = PmoEmitAttrI(*Node, "axis", -1) : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", -1)) + " is outside the input rank " + Str(Rank) + ".") : EndIf
  Width = PmoEmitDim(*X, Axis)
  If K < 0 Or K > Width : ProcedureReturn PmoEmitNsFail(*Node, "K = " + Str(K) + " is outside 0.." + Str(Width) + ", the extent of axis " + Str(Axis) + ".") : EndIf
  Count = (*X\Elements / PmoEmitNsMax(Width, 1)) * K
  If Width = 0 : Count = 0 : EndIf
  For Index = 0 To 1
    If Index = 0 And *Values = 0 : Continue : EndIf
    If Index = 1 And *Indices = 0 : Continue : EndIf
    If Index = 0 And (*Values\Elements <> Count Or *Values\ElementType <> *X\ElementType) : ProcedureReturn PmoEmitNsFail(*Node, "output Values declares " + Str(*Values\Elements) + " elements; the specification gives " + Str(Count) + ".") : EndIf
    If Index = 1 And (*Indices\Elements <> Count Or *Indices\ElementType <> 7) : ProcedureReturn PmoEmitNsFail(*Node, "output Indices must be INT64 with " + Str(Count) + " elements.") : EndIf
  Next
  PmoEmitLine(File, "Global Dim " + ProcName + "Order.i(" + Str(2 * PmoEmitNsMax(Width, 1)) + ")")
  PmoEmitLine(File, "Global Dim " + ProcName + "SpareValues.a(" + Str(PmoEmitNsMax(Count * 8, 1)) + ")")
  PmoEmitLine(File, "Global Dim " + ProcName + "SpareIndices.a(" + Str(PmoEmitNsMax(Count * 8, 1)) + ")")
  PmoEmitLine(File, "Global " + ProcName + "Args.PmTensorTopKArgs")
  PmoEmitLine(File, "Procedure " + ProcName + "(*x, *values, *indices)")
  PmoEmitLine(File, "  " + ProcName + "Args\Src = *x")
  PmoEmitLine(File, "  " + ProcName + "Args\Values = *values")
  PmoEmitLine(File, "  If *values = 0 : " + ProcName + "Args\Values = @" + ProcName + "SpareValues(0) : EndIf")
  PmoEmitLine(File, "  " + ProcName + "Args\Indices = *indices")
  PmoEmitLine(File, "  If *indices = 0 : " + ProcName + "Args\Indices = @" + ProcName + "SpareIndices(0) : EndIf")
  PmoEmitLine(File, "  " + ProcName + "Args\Order = @" + ProcName + "Order(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\Outer = " + Str(PmoEmitProduct(*X, 0, Axis - 1)))
  PmoEmitLine(File, "  " + ProcName + "Args\Width = " + Str(Width))
  PmoEmitLine(File, "  " + ProcName + "Args\Inner = " + Str(PmoEmitProduct(*X, Axis + 1, Rank - 1)))
  PmoEmitLine(File, "  " + ProcName + "Args\K = " + Str(K))
  PmoEmitLine(File, "  " + ProcName + "Args\Largest = " + Str(Bool(PmoEmitAttrI(*Node, "largest", 1) <> 0)))
  PmoEmitLine(File, "  " + ProcName + "Args\Kind = " + Str(*X\ElementType))
  PmoEmitLine(File, "  PmTensorTopK(@" + ProcName + "Args)")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitNsScatterElements(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Data.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Indices.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 1))
  Protected *Updates.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 2))
  Protected Mode.s = PmoEmitAttrS(*Node, "reduction", "none"), Code.i, Rank.i, Axis.i, D.i
  If *Node\Operation = "Scatter"
    ; Scatter-9 is ScatterElements without reduction, deprecated for it from opset 11
    If Opset > 10 : ProcedureReturn PmoEmitNsFail(*Node, "Scatter is deprecated from opset 11 (use ScatterElements), and the model imports opset " + Str(Opset) + ".") : EndIf
  ElseIf Opset < 11
    ProcedureReturn PmoEmitNsFail(*Node, "ScatterElements does not exist before opset 11, and the model imports opset " + Str(Opset) + ".")
  EndIf
  If Opset < 16
    If PmoEmitNsAttributesAllowed(*Node, "|axis|", Opset) = 0 : ProcedureReturn #False : EndIf
  ElseIf PmoEmitNsAttributesAllowed(*Node, "|axis|reduction|", Opset) = 0
    ProcedureReturn #False
  EndIf
  Select Mode
    Case "none" : Code = 0
    Case "add" : Code = 1
    Case "mul" : Code = 2
    Case "max" : Code = 3
    Case "min" : Code = 4
    Default : Code = -1
  EndSelect
  If Code < 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute reduction = " + Mode + " is not a ScatterElements reduction; none, add, mul, max and min are.") : EndIf
  If Code >= 3 And Opset < 18 : ProcedureReturn PmoEmitNsFail(*Node, "attribute reduction = " + Mode + " is not part of ScatterElements before opset 18, and the model imports opset " + Str(Opset) + ".") : EndIf
  If *Data = 0 Or *Indices = 0 Or *Updates = 0 : ProcedureReturn PmoEmitNsFail(*Node, "inputs data, indices and updates are all required.") : EndIf
  If *Data\ElementType <> 1 And *Data\ElementType <> 7 And *Data\ElementType <> 9
    ProcedureReturn PmoEmitNsFail(*Node, "data type " + PmoEmitNsTypeName(*Data\ElementType) + " is not implemented by fixed-shape emission; FLOAT, INT64 and BOOL are.")
  EndIf
  If *Updates\ElementType <> *Data\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "updates must have the element type of data.") : EndIf
  If *Indices\ElementType <> 6 And *Indices\ElementType <> 7 : ProcedureReturn PmoEmitNsFail(*Node, "indices must be INT32 or INT64.") : EndIf
  If *Data\ElementType = 9 And (Code = 1 Or Code = 2) : ProcedureReturn PmoEmitNsFail(*Node, "attribute reduction = " + Mode + " is not defined for BOOL data.") : EndIf
  Rank = PmoEmitRank(*Data)
  If Rank < 1 Or PmoEmitRank(*Indices) <> Rank Or PmoEmitRank(*Updates) <> Rank
    ProcedureReturn PmoEmitNsFail(*Node, "data, indices and updates must share one rank of at least one.")
  EndIf
  Axis = PmoEmitAttrI(*Node, "axis", 0) : If Axis < 0 : Axis + Rank : EndIf
  If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute axis = " + Str(PmoEmitAttrI(*Node, "axis", 0)) + " is outside the data rank " + Str(Rank) + ".") : EndIf
  For D = 0 To Rank - 1
    If PmoEmitDim(*Updates, D) <> PmoEmitDim(*Indices, D) : ProcedureReturn PmoEmitNsFail(*Node, "updates must have the shape of indices.") : EndIf
    If D <> Axis And PmoEmitDim(*Indices, D) > PmoEmitDim(*Data, D) : ProcedureReturn PmoEmitNsFail(*Node, "indices extend past data on axis " + Str(D) + ".") : EndIf
  Next
  PmoEmitLine(File, "Global Dim " + ProcName + "DataDims.i(8)")
  PmoEmitLine(File, "Global Dim " + ProcName + "IndexDims.i(8)")
  PmoEmitLine(File, "Global " + ProcName + "Args.PmTensorScatterElementsArgs")
  PmoEmitLine(File, "Procedure " + ProcName + "(*data, *indices, *updates, *dst)")
  PmoEmitNsDims(File, ProcName + "DataDims", *Data)
  PmoEmitNsDims(File, ProcName + "IndexDims", *Indices)
  PmoEmitLine(File, "  " + ProcName + "Args\Data = *data")
  PmoEmitLine(File, "  " + ProcName + "Args\Indices = *indices")
  PmoEmitLine(File, "  " + ProcName + "Args\Updates = *updates")
  PmoEmitLine(File, "  " + ProcName + "Args\Dst = *dst")
  PmoEmitLine(File, "  " + ProcName + "Args\Rank = " + Str(Rank))
  PmoEmitLine(File, "  " + ProcName + "Args\Axis = " + Str(Axis))
  PmoEmitLine(File, "  " + ProcName + "Args\Reduction = " + Str(Code))
  PmoEmitLine(File, "  " + ProcName + "Args\Kind = " + Str(*Data\ElementType))
  PmoEmitLine(File, "  " + ProcName + "Args\IndexKind = " + Str(*Indices\ElementType))
  PmoEmitLine(File, "  " + ProcName + "Args\DataDims = @" + ProcName + "DataDims(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\IndexDims = @" + ProcName + "IndexDims(0)")
  PmoEmitLine(File, "  If PmTensorScatterElements(@" + ProcName + "Args) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitNsReduce(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Data.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Axes.PmoIrConstant
  Protected Op.s = *Node\Operation, Rank.i, Count.i, Index.i, Axis.q, Mask.i, Ok.Integer, Noop.i, Kept.q, D.i
  If *Data = 0 Or *Y = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input data and its output need concrete shapes.") : EndIf
  Rank = PmoEmitRank(*Data)
  If Rank > 8 : ProcedureReturn PmoEmitNsFail(*Node, "the input rank " + Str(Rank) + " exceeds eight.") : EndIf
  If Opset >= 18
    If PmoEmitNsAttributesAllowed(*Node, "|keepdims|noop_with_empty_axes|", Opset) = 0 : ProcedureReturn #False : EndIf
    If PmoEmitInput(*Node, 1) <> ""
      *Axes = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
      If *Axes = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input axes must be constant for fixed-shape emission.") : EndIf
      Count = *Axes\Elements
    EndIf
    Noop = Bool(PmoEmitAttrI(*Node, "noop_with_empty_axes", 0) <> 0)
  Else
    If PmoEmitNsAttributesAllowed(*Node, "|axes|keepdims|", Opset) = 0 : ProcedureReturn #False : EndIf
    If PmoEmitInput(*Node, 1) <> "" : ProcedureReturn PmoEmitNsFail(*Node, "input axes is not part of " + Op + " before opset 18; axes is an attribute there.") : EndIf
    Count = PmoEmitAttrListCount(*Node, "axes")
  EndIf
  If Op = "ReduceMax" And *Data\ElementType <> 1 And *Data\ElementType <> 7 And *Data\ElementType <> 9
    ProcedureReturn PmoEmitNsFail(*Node, "data type " + PmoEmitNsTypeName(*Data\ElementType) + " is not implemented by fixed-shape emission; FLOAT, INT64 and BOOL are.")
  EndIf
  If Op = "ReduceProd" And *Data\ElementType <> 1 And *Data\ElementType <> 7
    ProcedureReturn PmoEmitNsFail(*Node, "data type " + PmoEmitNsTypeName(*Data\ElementType) + " is not implemented by fixed-shape emission; FLOAT and INT64 are.")
  EndIf
  For Index = 0 To Count - 1
    If *Axes : Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), Index, @Ok) : Else : Axis = PmoEmitAttrListI(*Node, "axes", Index, 0) : EndIf
    If Axis < 0 : Axis + Rank : EndIf
    If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "axes names an axis outside the data rank " + Str(Rank) + ".") : EndIf
    If (Mask >> Axis) & 1 : ProcedureReturn PmoEmitNsFail(*Node, "axes names axis " + Str(Axis) + " twice.") : EndIf
    Mask | (1 << Axis)
  Next
  If Count = 0 And Noop = 0 : Mask = (1 << Rank) - 1 : EndIf
  Kept = 1
  For D = 0 To Rank - 1
    If ((Mask >> D) & 1) = 0 : Kept * PmoEmitDim(*Data, D) : EndIf
  Next
  If *Y\Elements <> Kept Or *Y\ElementType <> *Data\ElementType
    ProcedureReturn PmoEmitNsFail(*Node, "the declared output holds " + Str(*Y\Elements) + " elements; the reduction gives " + Str(Kept) + ".")
  EndIf
  PmoEmitLine(File, "Global Dim " + ProcName + "Dims.i(8)")
  PmoEmitLine(File, "Global " + ProcName + "Args.PmTensorReduceArgs")
  PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  PmoEmitNsDims(File, ProcName + "Dims", *Data)
  PmoEmitLine(File, "  " + ProcName + "Args\Src = *src")
  PmoEmitLine(File, "  " + ProcName + "Args\Dst = *dst")
  PmoEmitLine(File, "  " + ProcName + "Args\Rank = " + Str(Rank))
  PmoEmitLine(File, "  " + ProcName + "Args\Dims = @" + ProcName + "Dims(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\Mask = " + Str(Mask))
  PmoEmitLine(File, "  " + ProcName + "Args\Op = " + Str(Bool(Op = "ReduceProd")))
  PmoEmitLine(File, "  " + ProcName + "Args\Kind = " + Str(*Data\ElementType))
  PmoEmitLine(File, "  PmTensorReduce(@" + ProcName + "Args)")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoEmitNsPad(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, ProcName.s, Opset.i)
  Protected *Node.PmoOnnxNode = *Ref\Node
  Protected *Src.PmoIrValue = PmoEmitValue(*Ir, PmoEmitInput(*Node, 0))
  Protected *Y.PmoIrValue = PmoEmitValue(*Ir, PmoEmitOutput(*Node, 0))
  Protected *Pads.PmoIrConstant, *Axes.PmoIrConstant
  Protected Mode.s = PmoEmitAttrS(*Node, "mode", "constant"), Code.i, Rank.i, Count.i, Index.i, Axis.q, Seen.i, Ok.Integer
  Protected Before.q, After.q, Extent.q, HasValue.i, Width.q, Fill.f
  Dim Begins.q(8)
  If *Src = 0 Or *Y = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input data and its output need concrete shapes.") : EndIf
  Rank = PmoEmitRank(*Src)
  If Rank > 8 : ProcedureReturn PmoEmitNsFail(*Node, "the input rank " + Str(Rank) + " exceeds eight.") : EndIf
  Select Mode
    Case "constant" : Code = 0
    Case "reflect" : Code = 1
    Case "edge" : Code = 2
    Case "wrap" : Code = 3
    Default : Code = -1
  EndSelect
  If Code < 0 : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = " + Mode + " is not a Pad mode; constant, reflect, edge and wrap are.") : EndIf
  If Code = 3 And Opset < 19 : ProcedureReturn PmoEmitNsFail(*Node, "attribute mode = wrap is not part of Pad before opset 19, and the model imports opset " + Str(Opset) + ".") : EndIf
  If *Src\ElementType <> 1 And *Src\ElementType <> 7 And *Src\ElementType <> 9
    ProcedureReturn PmoEmitNsFail(*Node, "data type " + PmoEmitNsTypeName(*Src\ElementType) + " is not implemented by fixed-shape emission; FLOAT, INT64 and BOOL are.")
  EndIf
  If Opset < 11
    ; Pad-2: pads and value are attributes, and the data is floating point.
    If PmoEmitNsAttributesAllowed(*Node, "|mode|pads|value|", Opset) = 0 : ProcedureReturn #False : EndIf
    Count = PmoEmitAttrListCount(*Node, "pads")
    If Count <> 2 * Rank : ProcedureReturn PmoEmitNsFail(*Node, "attribute pads holds " + Str(Count) + " counts; the data rank " + Str(Rank) + " needs " + Str(2 * Rank) + ".") : EndIf
    For Index = 0 To Rank - 1
      Begins(Index) = PmoEmitAttrListI(*Node, "pads", Index, 0)
      Extent = PmoEmitDim(*Src, Index) + Begins(Index) + PmoEmitAttrListI(*Node, "pads", Index + Rank, 0)
      If Extent <> PmoEmitDim(*Y, Index) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(Index) + " is " + Str(PmoEmitDim(*Y, Index)) + "; the pads give " + Str(Extent) + ".") : EndIf
    Next
    Fill = PmoEmitAttrF(*Node, "value", 0.0)
    HasValue = Bool(Fill <> 0.0 And Code = 0)
  Else
    If PmoEmitNsAttributesAllowed(*Node, "|mode|", Opset) = 0 : ProcedureReturn #False : EndIf
    *Pads = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 1))
    If *Pads = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input pads must be constant for fixed-shape emission.") : EndIf
    If PmoEmitInput(*Node, 3) <> ""
      If Opset < 18 : ProcedureReturn PmoEmitNsFail(*Node, "input axes is not part of Pad before opset 18, and the model imports opset " + Str(Opset) + ".") : EndIf
      *Axes = PmoEmitConstant(*Ir, PmoEmitInput(*Node, 3))
      If *Axes = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input axes must be constant for fixed-shape emission.") : EndIf
      Count = *Axes\Elements
    Else
      Count = Rank
    EndIf
    If *Pads\Elements <> 2 * Count : ProcedureReturn PmoEmitNsFail(*Node, "input pads holds " + Str(*Pads\Elements) + " counts; " + Str(Count) + " padded axes need " + Str(2 * Count) + ".") : EndIf
    For Index = 0 To Count - 1
      Axis = Index
      If *Axes : Axis = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 3), Index, @Ok) : EndIf
      If Axis < 0 : Axis + Rank : EndIf
      If Axis < 0 Or Axis >= Rank : ProcedureReturn PmoEmitNsFail(*Node, "input axes names an axis outside the data rank " + Str(Rank) + ".") : EndIf
      If (Seen >> Axis) & 1 : ProcedureReturn PmoEmitNsFail(*Node, "input axes names axis " + Str(Axis) + " twice.") : EndIf
      Seen | (1 << Axis)
      Begins(Axis) = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), Index, @Ok)
      After = PmoEmitConstI(*Ir, PmoEmitInput(*Node, 1), Index + Count, @Ok)
      If Ok\i = 0 : ProcedureReturn PmoEmitNsFail(*Node, "input pads must be INT64.") : EndIf
      Extent = PmoEmitDim(*Src, Axis) + Begins(Axis) + After
      If Extent <> PmoEmitDim(*Y, Axis) : ProcedureReturn PmoEmitNsFail(*Node, "the declared output extent on axis " + Str(Axis) + " is " + Str(PmoEmitDim(*Y, Axis)) + "; the pads give " + Str(Extent) + ".") : EndIf
    Next
    HasValue = Bool(PmoEmitInput(*Node, 2) <> "")
  EndIf
  If PmoEmitRank(*Y) <> Rank Or *Y\ElementType <> *Src\ElementType : ProcedureReturn PmoEmitNsFail(*Node, "the declared output must have the data's rank and element type.") : EndIf
  Width = 1 : If Rank > 0 : Width = PmoEmitDim(*Y, Rank - 1) : EndIf
  PmoEmitLine(File, "Global Dim " + ProcName + "SrcDims.i(8)")
  PmoEmitLine(File, "Global Dim " + ProcName + "DstDims.i(8)")
  PmoEmitLine(File, "Global Dim " + ProcName + "Begins.i(8)")
  PmoEmitLine(File, "Global Dim " + ProcName + "Map.i(" + Str(PmoEmitNsMax(Width, 1)) + ")")
  If Opset < 11 And HasValue : PmoEmitLine(File, "Global Dim " + ProcName + "Value.a(8)") : EndIf
  PmoEmitLine(File, "Global " + ProcName + "Args.PmTensorPadArgs")
  If Opset >= 11 And HasValue
    PmoEmitLine(File, "Procedure " + ProcName + "(*src, *value, *dst)")
  Else
    PmoEmitLine(File, "Procedure " + ProcName + "(*src, *dst)")
  EndIf
  PmoEmitNsDims(File, ProcName + "SrcDims", *Src)
  PmoEmitNsDims(File, ProcName + "DstDims", *Y)
  For Index = 0 To Rank - 1
    PmoEmitLine(File, "  " + ProcName + "Begins(" + Str(Index) + ") = " + Str(Begins(Index)))
  Next
  PmoEmitLine(File, "  " + ProcName + "Args\Src = *src")
  PmoEmitLine(File, "  " + ProcName + "Args\Dst = *dst")
  PmoEmitLine(File, "  " + ProcName + "Args\Rank = " + Str(Rank))
  PmoEmitLine(File, "  " + ProcName + "Args\SrcDims = @" + ProcName + "SrcDims(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\DstDims = @" + ProcName + "DstDims(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\Begins = @" + ProcName + "Begins(0)")
  PmoEmitLine(File, "  " + ProcName + "Args\Mode = " + Str(Code))
  PmoEmitLine(File, "  " + ProcName + "Args\Kind = " + Str(*Src\ElementType))
  PmoEmitLine(File, "  " + ProcName + "Args\Map = @" + ProcName + "Map(0)")
  If HasValue And Opset >= 11 And Code = 0
    PmoEmitLine(File, "  " + ProcName + "Args\Value = *value")
  ElseIf HasValue And Opset < 11
    PmoEmitLine(File, "  PmTensorPut(@" + ProcName + "Value(0), 0, " + PmoEmitFloat(Fill) + ")")
    PmoEmitLine(File, "  " + ProcName + "Args\Value = @" + ProcName + "Value(0)")
  Else
    PmoEmitLine(File, "  " + ProcName + "Args\Value = 0")
  EndIf
  PmoEmitLine(File, "  If PmTensorPad(@" + ProcName + "Args) <> 0 : PmOnnxRuntimeOk = 0 : EndIf")
  PmoEmitLine(File, "EndProcedure") : PmoEmitLine(File)
  ProcedureReturn #True
EndProcedure

; One generated procedure for a node of these operators, registered in Calls.
Procedure.i PmoEmitNormSmallHelper(File.i, *Ir.PmoIrModel, *Ref.PmoIrNodeRef, Map Calls.s())
  Protected ProcName.s = "PmOnnxNode" + Str(*Ref\Index), Opset.i = PmoEmitNsOpset(*Ir), Done.i
  Select *Ref\Node\Operation
    Case "InstanceNormalization" : Done = PmoEmitNsInstanceNorm(File, *Ir, *Ref, ProcName, Opset)
    Case "Not" : Done = PmoEmitNsNot(File, *Ir, *Ref, ProcName, Opset)
    Case "TopK" : Done = PmoEmitNsTopK(File, *Ir, *Ref, ProcName, Opset)
    Case "ScatterElements", "Scatter" : Done = PmoEmitNsScatterElements(File, *Ir, *Ref, ProcName, Opset)
    Case "ReduceMax", "ReduceProd" : Done = PmoEmitNsReduce(File, *Ir, *Ref, ProcName, Opset)
    Case "Pad" : Done = PmoEmitNsPad(File, *Ir, *Ref, ProcName, Opset)
  EndSelect
  If Done = 0 Or PmoEmitError <> "" : ProcedureReturn #False : EndIf
  Calls(Str(*Ref\Index)) = ProcName
  ProcedureReturn #True
EndProcedure
