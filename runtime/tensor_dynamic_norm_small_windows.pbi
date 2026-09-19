; ======================================================================
; tensor_dynamic_norm_small_windows.pbi - runtime-dimension wrappers for
; InstanceNormalization, TopK, ScatterElements, ReduceMax, ReduceProd, Not,
; Identity and Pad on Windows.
; ----------------------------------------------------------------------
; Included at the end of tensor_dynamic_windows.pbi. Each wrapper checks the
; shapes and element types the ONNX specification requires, allocates its
; outputs and hands addresses to the kernels in tensor_norm_small_windows.pbi.
; A form this runtime cannot compute is refused with DFail naming the
; operator; nothing is approximated.
; ======================================================================
XIncludeFile "tensor_norm_small_windows.pbi"

; Temporary working memory counted against the request's limit.
Procedure.i DNsScratch(Bytes.i)
  Protected *memory
  If Bytes < 1 : Bytes = 1 : EndIf
  If Bytes > DLimit - DLive : DFail("Operator scratch exceeds the configured working-memory limit.") : ProcedureReturn 0 : EndIf
  *memory = AllocateMemory(Bytes)
  If *memory = 0 : DFail("Operator scratch allocation failed.") : ProcedureReturn 0 : EndIf
  DLive + Bytes : DPeak = DMax(DPeak, DLive)
  ProcedureReturn *memory
EndProcedure

Procedure DNsScratchFree(*Memory, Bytes.i)
  If *Memory = 0 : ProcedureReturn : EndIf
  If Bytes < 1 : Bytes = 1 : EndIf
  FreeMemory(*Memory) : DLive - Bytes
EndProcedure

; InstanceNormalization-6: y = scale * (x - mean) / sqrt(variance + epsilon) + B
; per instance per channel over an (N x C x D1 ... Dn) FLOAT input.
Procedure DInstanceNorm(Y.i,A.i,Scale.i,Bias.i,Epsilon.f)
  Protected channels.i
  If A = 0 Or Scale = 0 Or Bias = 0 : DFail("InstanceNormalization requires input, scale and B.") : ProcedureReturn : EndIf
  If Dt(A)\Kind<>1 Or Dt(Scale)\Kind<>1 Or Dt(Bias)\Kind<>1 : DFail("InstanceNormalization computes FLOAT input, scale and B only.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<3 : DFail("InstanceNormalization requires an input of rank three or more, N x C x D1 ... Dn.") : ProcedureReturn : EndIf
  channels=Dt(A)\D[1]
  If Dt(Scale)\Rank<>1 Or Dt(Scale)\Count<>channels Or Dt(Bias)\Rank<>1 Or Dt(Bias)\Count<>channels
    DFail("InstanceNormalization scale and B must each be one-dimensional with C elements.") : ProcedureReturn
  EndIf
  If DLike(Y,A)=0 : ProcedureReturn : EndIf
  PmFastInstanceNorm(Dt(A)\Data,Dt(Scale)\Data,Dt(Bias)\Data,Dt(Y)\Data,Dt(A)\D[0],channels,DProduct(A,2,Dt(A)\Rank-1),Epsilon)
EndProcedure

; TopK-11: the k largest (or smallest) elements along Axis and their INT64
; indices; equal elements keep index order.
Procedure DTopK(Values.i,Indices.i,A.i,K.i,Axis.i,Largest.i,Sorted.i)
  Protected Dim dims.i(7)
  Protected g.PmTensorTopKArgs, i.i, kept.i, width.i, rank.i, count.i, *values, *indices, *order, valueBytes.i, indexBytes.i
  If A = 0 Or K = 0 : DFail("TopK requires X and K.") : ProcedureReturn : EndIf
  rank=Dt(A)\Rank
  If rank<1 : DFail("TopK requires an input of rank one or more.") : ProcedureReturn : EndIf
  If Dt(A)\Kind<>1 And Dt(A)\Kind<>6 And Dt(A)\Kind<>7 : DFail("TopK computes FLOAT, INT32 and INT64 input only.") : ProcedureReturn : EndIf
  If Dt(K)\Kind<>7 Or Dt(K)\Count<>1 : DFail("TopK K must be a one-element INT64 tensor.") : ProcedureReturn : EndIf
  If Axis<0 : Axis+rank : EndIf
  If Axis<0 Or Axis>=rank : DFail("TopK axis is outside the input rank.") : ProcedureReturn : EndIf
  kept=DInt(K,0) : width=Dt(A)\D[Axis]
  If kept<0 Or kept>width : DFail("TopK K is negative or larger than the extent of its axis.") : ProcedureReturn : EndIf
  For i=0 To rank-1 : dims(i)=Dt(A)\D[i] : Next
  dims(Axis)=kept
  count=1 : For i=0 To rank-1 : count*dims(i) : Next
  valueBytes=count*DSize(Dt(A)\Kind) : indexBytes=count*8
  If Values
    If DAlloc(Values,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
    *values=Dt(Values)\Data
  Else
    *values=DNsScratch(valueBytes) : If *values=0 : ProcedureReturn : EndIf
  EndIf
  If Indices
    If DAlloc(Indices,7,rank,@dims(0))=0 : If Values=0 : DNsScratchFree(*values,valueBytes) : EndIf : ProcedureReturn : EndIf
    *indices=Dt(Indices)\Data
  Else
    *indices=DNsScratch(indexBytes) : If *indices=0 : If Values=0 : DNsScratchFree(*values,valueBytes) : EndIf : ProcedureReturn : EndIf
  EndIf
  If count>0
    *order=DNsScratch(2*width*8)
    If *order
      g\Src=Dt(A)\Data : g\Values=*values : g\Indices=*indices : g\Order=*order
      g\Outer=DProduct(A,0,Axis-1) : g\Width=width : g\Inner=DProduct(A,Axis+1,rank-1)
      g\K=kept : g\Largest=Bool(Largest<>0) : g\Kind=Dt(A)\Kind
      PmTensorTopK(@g)
      DNsScratchFree(*order,2*width*8)
    EndIf
  EndIf
  If Values=0 : DNsScratchFree(*values,valueBytes) : EndIf
  If Indices=0 : DNsScratchFree(*indices,indexBytes) : EndIf
EndProcedure

; ScatterElements-18: a copy of data with updates written (Reduction 0) or
; combined by 1 add, 2 mul, 3 max, 4 min at the positions indices name along
; Axis.
Procedure DScatterElements(Y.i,A.i,Indices.i,Updates.i,Axis.i,Reduction.i)
  Protected g.PmTensorScatterElementsArgs, rank.i, d.i, status.i
  If A = 0 Or Indices = 0 Or Updates = 0 : DFail("ScatterElements requires data, indices and updates.") : ProcedureReturn : EndIf
  rank=Dt(A)\Rank
  If rank<1 : DFail("ScatterElements requires data of rank one or more.") : ProcedureReturn : EndIf
  If PmNsElementBytes(Dt(A)\Kind)=0 : DFail("ScatterElements computes FLOAT, INT32, INT64 and BOOL data only.") : ProcedureReturn : EndIf
  If Dt(Updates)\Kind<>Dt(A)\Kind : DFail("ScatterElements updates must have the element type of data.") : ProcedureReturn : EndIf
  If Dt(Indices)\Kind<>6 And Dt(Indices)\Kind<>7 : DFail("ScatterElements indices must be INT32 or INT64.") : ProcedureReturn : EndIf
  If Dt(Indices)\Rank<>rank Or Dt(Updates)\Rank<>rank : DFail("ScatterElements requires data, indices and updates of the same rank.") : ProcedureReturn : EndIf
  If Axis<0 : Axis+rank : EndIf
  If Axis<0 Or Axis>=rank : DFail("ScatterElements axis is outside the data rank.") : ProcedureReturn : EndIf
  For d=0 To rank-1
    If Dt(Updates)\D[d]<>Dt(Indices)\D[d] : DFail("ScatterElements updates must have the shape of indices.") : ProcedureReturn : EndIf
    If d<>Axis And Dt(Indices)\D[d]>Dt(A)\D[d] : DFail("ScatterElements indices extend past data on an axis other than the scatter axis.") : ProcedureReturn : EndIf
  Next
  If DLike(Y,A)=0 : ProcedureReturn : EndIf
  g\Data=Dt(A)\Data : g\Indices=Dt(Indices)\Data : g\Updates=Dt(Updates)\Data : g\Dst=Dt(Y)\Data
  g\Rank=rank : g\Axis=Axis : g\Reduction=Reduction : g\Kind=Dt(A)\Kind : g\IndexKind=Dt(Indices)\Kind
  g\DataDims=@Dt(A)\D[0] : g\IndexDims=@Dt(Indices)\D[0]
  status=PmTensorScatterElements(@g)
  If status=1 : DFail("ScatterElements index is outside [-s, s-1] for an axis of extent s.") : EndIf
  If status=2 : DFail("ScatterElements reduction add and mul are not defined for BOOL data.") : EndIf
EndProcedure

; ReduceMax-20 (Op 0) and ReduceProd-18 (Op 1), with axes from the input
; tensor Axes (opset 18 and later) or from the attribute list AxesList
; (earlier opsets); an empty axes set reduces everything unless NoopEmpty.
Procedure DReduceSelect(Y.i,A.i,Axes.i,AxesList.s,Keep.i,NoopEmpty.i,Op.i)
  Protected Dim dims.i(7)
  Protected g.PmTensorReduceArgs, rank.i, count.i, i.i, axis.i, mask.i, j.i, name.s="ReduceMax"
  If Op=1 : name="ReduceProd" : EndIf
  If A = 0 : DFail(name+" requires data.") : ProcedureReturn : EndIf
  rank=Dt(A)\Rank
  If Op=0 And Dt(A)\Kind<>1 And Dt(A)\Kind<>6 And Dt(A)\Kind<>7 And Dt(A)\Kind<>9 : DFail("ReduceMax computes FLOAT, INT32, INT64 and BOOL data only.") : ProcedureReturn : EndIf
  If Op=1 And Dt(A)\Kind<>1 And Dt(A)\Kind<>6 And Dt(A)\Kind<>7 : DFail("ReduceProd computes FLOAT, INT32 and INT64 data only.") : ProcedureReturn : EndIf
  If Axes
    If Dt(Axes)\Kind<>7 Or Dt(Axes)\Rank>1 : DFail(name+" axes must be a one-dimensional INT64 tensor.") : ProcedureReturn : EndIf
    count=Dt(Axes)\Count
  ElseIf AxesList<>""
    count=CountString(AxesList,",")+1
  EndIf
  For i=0 To count-1
    If Axes : axis=DInt(Axes,i) : Else : axis=Val(StringField(AxesList,i+1,",")) : EndIf
    If axis<0 : axis+rank : EndIf
    If axis<0 Or axis>=rank : DFail(name+" axis is outside the data rank.") : ProcedureReturn : EndIf
    If (mask>>axis)&1 : DFail(name+" axes name the same axis twice.") : ProcedureReturn : EndIf
    mask|(1<<axis)
  Next
  If count=0
    If NoopEmpty
      If DLike(Y,A) : CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes) : EndIf
      ProcedureReturn
    EndIf
    mask=(1<<rank)-1
  EndIf
  For i=0 To rank-1
    If (mask>>i)&1
      If Keep : dims(j)=1 : j+1 : EndIf
    Else
      dims(j)=Dt(A)\D[i] : j+1
    EndIf
  Next
  If DAlloc(Y,Dt(A)\Kind,j,@dims(0))=0 : ProcedureReturn : EndIf
  g\Src=Dt(A)\Data : g\Dst=Dt(Y)\Data : g\Rank=rank : g\Dims=@Dt(A)\D[0]
  g\Mask=mask : g\Op=Op : g\Kind=Dt(A)\Kind
  PmTensorReduce(@g)
EndProcedure

; Not-1 over BOOL.
Procedure DNot(Y.i,A.i)
  If A = 0 Or Dt(A)\Kind<>9 : DFail("Not requires a BOOL tensor.") : ProcedureReturn : EndIf
  If DLike(Y,A) : PmTensorNot(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Count) : EndIf
EndProcedure

; Identity: a tensor copied into a new tensor of the same type and shape.
Procedure DIdentity(Y.i,A.i)
  If A = 0 Or Dt(A)\Data = 0 : DFail("Identity has no input tensor.") : ProcedureReturn : EndIf
  If DLike(Y,A) : CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes) : EndIf
EndProcedure

; Pad-19 with an optional Axes input (Pad-18 and later). Mode 0 constant,
; 1 reflect, 2 edge, 3 wrap.
Procedure DPadAxes(Y.i,A.i,Pads.i,Value.i,Axes.i,Mode.i)
  Protected Dim dims.i(7), Dim begins.i(7)
  Protected g.PmTensorPadArgs, rank.i, count.i, i.i, axis.i, seen.i, width.i, *map, status.i, ends.i
  If A = 0 Or Pads = 0 : DFail("Pad requires data and pads.") : ProcedureReturn : EndIf
  rank=Dt(A)\Rank
  If PmNsElementBytes(Dt(A)\Kind)=0 : DFail("Pad computes FLOAT, INT32, INT64 and BOOL data only.") : ProcedureReturn : EndIf
  If Dt(Pads)\Kind<>7 : DFail("Pad pads must be an INT64 tensor.") : ProcedureReturn : EndIf
  If Mode<0 Or Mode>3 : DFail("Pad mode must be constant, reflect, edge or wrap.") : ProcedureReturn : EndIf
  count=rank
  If Axes
    If Dt(Axes)\Kind<>6 And Dt(Axes)\Kind<>7 : DFail("Pad axes must be INT32 or INT64.") : ProcedureReturn : EndIf
    count=Dt(Axes)\Count
  EndIf
  If Dt(Pads)\Count<>2*count : DFail("Pad pads must hold two counts for every padded axis.") : ProcedureReturn : EndIf
  For i=0 To rank-1 : dims(i)=Dt(A)\D[i] : Next
  For i=0 To count-1
    axis=i : If Axes : axis=DInt(Axes,i) : EndIf
    If axis<0 : axis+rank : EndIf
    If axis<0 Or axis>=rank : DFail("Pad axis is outside the data rank.") : ProcedureReturn : EndIf
    If (seen>>axis)&1 : DFail("Pad axes name the same axis twice.") : ProcedureReturn : EndIf
    seen|(1<<axis)
    begins(axis)=DInt(Pads,i) : ends=DInt(Pads,i+count)
    dims(axis)=Dt(A)\D[axis]+begins(axis)+ends
    If dims(axis)<0 : DFail("Pad removes more elements than an axis holds.") : ProcedureReturn : EndIf
  Next
  If Value And Mode=0
    If Dt(Value)\Kind<>Dt(A)\Kind Or Dt(Value)\Count<1 : DFail("Pad constant_value must be one element of the data's element type.") : ProcedureReturn : EndIf
  EndIf
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  width=1 : If rank>0 : width=dims(rank-1) : EndIf
  *map=DNsScratch(DMax(width,1)*8) : If *map=0 : ProcedureReturn : EndIf
  g\Src=Dt(A)\Data : g\Dst=Dt(Y)\Data : g\Rank=rank : g\SrcDims=@Dt(A)\D[0] : g\DstDims=@dims(0)
  g\Begins=@begins(0) : g\Mode=Mode : g\Kind=Dt(A)\Kind : g\Map=*map
  If Value And Mode=0 : g\Value=Dt(Value)\Data : EndIf
  status=PmTensorPad(@g)
  DNsScratchFree(*map,DMax(width,1)*8)
  If status : DFail("Pad reflect, edge and wrap need at least one element left on every padded axis.") : EndIf
EndProcedure

Procedure DPad(Y.i,A.i,Pads.i,Value.i,Mode.i)
  DPadAxes(Y,A,Pads,Value,0,Mode)
EndProcedure
