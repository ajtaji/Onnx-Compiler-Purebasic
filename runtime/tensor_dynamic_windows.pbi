; Runtime dimensions for statically emitted tensor calls. No graph parser.
; The generated caller owns scheduling and releases each temporary at last use.
#PMO_USE_BATCHNORM = 1
#PMO_USE_STFT = 1
#PMO_USE_RESIZE = 1
#PMO_USE_CONVTRANSPOSE = 1
#PMO_USE_CONV = 1
#PMO_USE_LSTM = 1
#PMO_USE_INT8 = 1
XIncludeFile "tensor_fp32_windows.pbi"
XIncludeFile "tensor_simd_windows.pbi"

Structure PmDynamicTensor
  *Data
  Kind.i
  Rank.i
  Count.i
  Bytes.i
  Owned.i
  Scales.i
  D.i[8]
EndStructure
Global Dim Dt.PmDynamicTensor(#PMD_TENSOR_COUNT)
Global DError.s, DNode.i, DCancel.i
Global DLive.i, DPeak.i, DLimit.i = 1024 * 1024 * 1024
; Optional caller-owned synchronous boundary callback. It must not re-enter
; model/tensor code. Zero preserves the historical no-callback behaviour.
#PMD_PROGRESS_BIND_BEFORE=1
#PMD_PROGRESS_BIND_AFTER=2
#PMD_PROGRESS_NODE_BEFORE=3
#PMD_PROGRESS_NODE_AFTER=4
Prototype.i DProgressProcedure(Phase.i,Node.i)
Global DProgressCallback.DProgressProcedure
Global DProgressActive.i
CompilerIf Defined(PMD_PROFILE,#PB_Constant)=0
  #PMD_PROFILE=0
CompilerEndIf
CompilerIf #PMD_PROFILE
  Global Dim DTimes.q(#PMD_NODE_COUNT)
  Global DStarted.i
CompilerEndIf

Procedure.i DMax(A.i,B.i)
  If A>B : ProcedureReturn A : EndIf
  ProcedureReturn B
EndProcedure
Procedure.i DMin(A.i,B.i)
  If A<B : ProcedureReturn A : EndIf
  ProcedureReturn B
EndProcedure

Procedure.i DFail(Message.s)
  If DError = "" : DError = "Node " + Str(DNode) + ": " + Message : EndIf
  ProcedureReturn 0
EndProcedure

Procedure.i DRejectProgressReentry()
  If DProgressActive
    DCancel=1
    DFail("Progress callback re-entered the model runtime.")
    ProcedureReturn 1
  EndIf
  ProcedureReturn 0
EndProcedure

Procedure.i DSetProgressCallback(Callback.DProgressProcedure)
  If DProgressActive
    DCancel=1
    ProcedureReturn DFail("Progress callback cannot replace itself while running.")
  EndIf
  DProgressCallback=Callback
  ProcedureReturn 1
EndProcedure

Procedure.i DPoll(Phase.i,Node.i)
  Protected keepGoing.i
  If DCancel : ProcedureReturn 0 : EndIf
  If DProgressCallback=0 : ProcedureReturn 1 : EndIf
  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf
  DProgressActive=1
  keepGoing=DProgressCallback(Phase,Node)
  DProgressActive=0
  If keepGoing=0 Or DCancel : DCancel=1 : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i DSize(Kind.i)
  Select Kind
    Case 1,6 : ProcedureReturn 4
    Case 7 : ProcedureReturn 8
    Case 3,9 : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i DInt8Begin(*Input,Elements.i,Extra.i=0)
  Protected i.i,bytes.i=Elements+Extra
  If Elements<0 Or Extra<0 Or bytes<Elements Or bytes>DLimit-DLive : ProcedureReturn DFail("INT8 activation scratch exceeds available working memory.") : EndIf
  For i=0 To Elements-1
    If (PeekL(*Input+i*4) & $7F800000)=$7F800000 : ProcedureReturn DFail("INT8 activations contain NaN or infinity.") : EndIf
  Next
  PmTensorInt8Scratch=AllocateMemory(DMax(bytes,1))
  If PmTensorInt8Scratch=0 : ProcedureReturn DFail("INT8 activation scratch allocation failed.") : EndIf
  PmTensorInt8ScratchBytes=bytes : DLive+bytes : DPeak=DMax(DPeak,DLive)
  ProcedureReturn 1
EndProcedure

Procedure.i DFiniteTensor(Id.i)
  Protected i.i
  If Id=0 : ProcedureReturn 1 : EndIf
  For i=0 To Dt(Id)\Count-1
    If (PeekL(Dt(Id)\Data+i*4) & $7F800000)=$7F800000 : ProcedureReturn DFail("INT8 recurrent inputs contain NaN or infinity.") : EndIf
  Next
  ProcedureReturn 1
EndProcedure

Procedure DInt8End()
  If PmTensorInt8Scratch : FreeMemory(PmTensorInt8Scratch) : DLive-PmTensorInt8ScratchBytes : EndIf
  PmTensorInt8Scratch=0 : PmTensorInt8ScratchBytes=0
EndProcedure

Procedure DRelease(Id.i)
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : ProcedureReturn : EndIf
  If Dt(Id)\Owned
    FreeMemory(Dt(Id)\Data)
    DLive - Dt(Id)\Bytes
  EndIf
  ClearStructure(@Dt(Id), PmDynamicTensor)
EndProcedure

Procedure.i DAlloc(Id.i, Kind.i, Rank.i, *Dims)
  Protected i.i, count.i = 1, bytes.i, extent.i, size.i = DSize(Kind)
  If DError <> "" Or DCancel : ProcedureReturn DFail("Request cancelled or an earlier operation failed.") : EndIf
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT Or Rank < 0 Or Rank > 8 Or size = 0 : ProcedureReturn DFail("Unsupported tensor rank or type.") : EndIf
  For i = 0 To Rank - 1
    extent = PeekI(*Dims + i * 8)
    If extent < 0 Or extent > DLimit : ProcedureReturn DFail("Invalid tensor dimension.") : EndIf
    If extent > 0 And count > DLimit / extent : ProcedureReturn DFail("Tensor exceeds the configured memory limit.") : EndIf
    count * extent
  Next
  If count > DLimit / size : ProcedureReturn DFail("Tensor byte count exceeds the configured memory limit.") : EndIf
  bytes = count * size
  DRelease(Id)
  If bytes > DLimit - DLive : ProcedureReturn DFail("Request exceeds the configured working-memory limit.") : EndIf
  Dt(Id)\Data = AllocateMemory(DMax(bytes, 1))
  If Dt(Id)\Data = 0 : ProcedureReturn DFail("Working-memory allocation failed.") : EndIf
  Dt(Id)\Kind = Kind : Dt(Id)\Rank = Rank : Dt(Id)\Count = count
  Dt(Id)\Bytes = bytes : Dt(Id)\Owned = 1 : DLive + bytes
  DPeak = DMax(DPeak, DLive)
  For i = 0 To Rank - 1 : Dt(Id)\D[i] = PeekI(*Dims + i * 8) : Next
  ProcedureReturn 1
EndProcedure

Procedure.i DShape(Id.i, Kind.i, Rank.i, d0.i=0,d1.i=0,d2.i=0,d3.i=0,d4.i=0,d5.i=0,d6.i=0,d7.i=0)
  Protected Dim dims.i(7)
  dims(0)=d0 : dims(1)=d1 : dims(2)=d2 : dims(3)=d3
  dims(4)=d4 : dims(5)=d5 : dims(6)=d6 : dims(7)=d7
  ProcedureReturn DAlloc(Id,Kind,Rank,@dims(0))
EndProcedure

Procedure.d DGet(Id.i, Index.i=0)
  If Id <= 0 Or Index < 0 Or Index >= Dt(Id)\Count : DFail("Tensor read is out of bounds.") : ProcedureReturn 0 : EndIf
  Select Dt(Id)\Kind
    Case 1 : ProcedureReturn PeekF(Dt(Id)\Data+Index*4)
    Case 7 : ProcedureReturn PeekQ(Dt(Id)\Data+Index*8)
    Case 6 : ProcedureReturn PeekL(Dt(Id)\Data+Index*4)
    Case 9 : ProcedureReturn PeekA(Dt(Id)\Data+Index)
  EndSelect
EndProcedure

Procedure.i DInt(Id.i, Index.i=0)
  If Id <= 0 Or Index < 0 Or Index >= Dt(Id)\Count : DFail("Integer tensor read is out of bounds.") : ProcedureReturn 0 : EndIf
  If Dt(Id)\Kind=7 : ProcedureReturn PeekQ(Dt(Id)\Data+Index*8) : EndIf
  ProcedureReturn DGet(Id,Index)
EndProcedure

Procedure DPut(Id.i, Index.i, Value.d)
  Select Dt(Id)\Kind
    Case 1 : PokeF(Dt(Id)\Data+Index*4,Value)
    Case 7 : PokeQ(Dt(Id)\Data+Index*8,Value)
    Case 6 : PokeL(Dt(Id)\Data+Index*4,Value)
    Case 9 : PokeA(Dt(Id)\Data+Index,Bool(Value<>0))
  EndSelect
EndProcedure

Procedure.i DProduct(Id.i, First.i, Last.i)
  Protected i.i, n.i=1
  For i=First To Last : n*Dt(Id)\D[i] : Next
  ProcedureReturn n
EndProcedure

Procedure.i DLike(Y.i,A.i,Kind.i=0)
  If Kind=0 : Kind=Dt(A)\Kind : EndIf
  ProcedureReturn DAlloc(Y,Kind,Dt(A)\Rank,@Dt(A)\D[0])
EndProcedure

Procedure.i DBroadcastIndex(Index.i,Y.i,A.i)
  Protected ax.i, src.i, coord.i, stride.i=1, result.i
  For ax=Dt(Y)\Rank-1 To 0 Step -1
    coord=Index % Dt(Y)\D[ax] : Index / Dt(Y)\D[ax]
    src=ax-(Dt(Y)\Rank-Dt(A)\Rank)
    If src>=0
      If Dt(A)\D[src]<>1 : result+coord*stride : EndIf
      stride*Dt(A)\D[src]
    EndIf
  Next
  ProcedureReturn result
EndProcedure

Procedure.i DBroadcast(Y.i,A.i,B.i,Kind.i=0)
  Protected rank.i=DMax(Dt(A)\Rank,Dt(B)\Rank),i.i,ai.i,bi.i,ad.i,bd.i
  Protected Dim dims.i(7)
  If Kind=0 : Kind=Dt(A)\Kind : EndIf
  For i=0 To rank-1
    ai=i-rank+Dt(A)\Rank : bi=i-rank+Dt(B)\Rank : ad=1 : bd=1
    If ai>=0 : ad=Dt(A)\D[ai] : EndIf
    If bi>=0 : bd=Dt(B)\D[bi] : EndIf
    If ad<>bd And ad<>1 And bd<>1 : ProcedureReturn DFail("Incompatible broadcast dimensions.") : EndIf
    If ad=1 : dims(i)=bd : Else : dims(i)=ad : EndIf
  Next
  ProcedureReturn DAlloc(Y,Kind,rank,@dims(0))
EndProcedure

Procedure DBinary(Y.i,A.i,B.i,Op.i)
  Protected i.i, ai.i,bi.i,kind.i=Dt(A)\Kind
  Protected av.f,bv.f,v.f,ia.i,ib.i,iv.i
  Protected width.i,row.i,astep.i,bstep.i
  If Op=4 And Dt(A)\Kind=1 And Dt(B)\Kind=1 And Dt(B)\Count=1
    If PeekF(Dt(B)\Data)=2 : B=A : Op=2 : EndIf
  EndIf
  If Op>=5 : kind=9 : EndIf
  If DBroadcast(Y,A,B,kind)=0 : ProcedureReturn : EndIf
  If kind=1 And Dt(B)\Kind=1 And Op<=3
    width=1 : If Dt(Y)\Rank : width=Dt(Y)\D[Dt(Y)\Rank-1] : EndIf
    If width=0 : ProcedureReturn : EndIf
    If Dt(A)\Rank : astep=Bool(Dt(A)\D[Dt(A)\Rank-1]<>1) : EndIf
    If Dt(B)\Rank : bstep=Bool(Dt(B)\D[Dt(B)\Rank-1]<>1) : EndIf
    For row=0 To Dt(Y)\Count/width-1
      i=row*width : ai=DBroadcastIndex(i,Y,A) : bi=DBroadcastIndex(i,Y,B)
      PmFastBinary(Dt(A)\Data+ai*4,Dt(B)\Data+bi*4,Dt(Y)\Data+i*4,width,astep,bstep,Op)
    Next
    ProcedureReturn
  EndIf
  If Op=4 And Dt(A)\Kind=1 And Dt(B)\Kind=1 And Dt(B)\Count=1 And Dt(A)\Count=Dt(Y)\Count
    bv=PeekF(Dt(B)\Data)
    For i=0 To Dt(Y)\Count-1 : PokeF(Dt(Y)\Data+i*4,Pow(PeekF(Dt(A)\Data+i*4),bv)) : Next
    ProcedureReturn
  EndIf
  For i=0 To Dt(Y)\Count-1
    ai=DBroadcastIndex(i,Y,A) : bi=DBroadcastIndex(i,Y,B)
    If Dt(A)\Kind=1
      av=DGet(A,ai) : bv=DGet(B,bi)
      Select Op
        Case 0 : v=av+bv
        Case 1 : v=av-bv
        Case 2 : v=av*bv
        Case 3 : v=av/bv
        Case 4 : v=Pow(av,bv)
        Case 5 : v=Bool(av=bv)
        Case 6 : v=Bool(av>bv)
        Case 7 : v=Bool(av<bv)
        Case 8 : v=Bool(av>=bv)
        Case 9 : v=Bool(av<>0 And bv<>0)
      EndSelect
      DPut(Y,i,v)
    Else
      ia=DInt(A,ai) : ib=DInt(B,bi)
      Select Op
        Case 0 : iv=ia+ib
        Case 1 : iv=ia-ib
        Case 2 : iv=ia*ib
        Case 3
          If ib=0 : DFail("Integer division by zero.") : ProcedureReturn : EndIf
          iv=ia/ib
        Case 5 : iv=Bool(ia=ib)
        Case 6 : iv=Bool(ia>ib)
        Case 7 : iv=Bool(ia<ib)
        Case 8 : iv=Bool(ia>=ib)
        Case 9 : iv=Bool(ia<>0 And ib<>0)
        Default : DFail("Unsupported integer arithmetic.") : ProcedureReturn
      EndSelect
      If kind=7 : PokeQ(Dt(Y)\Data+i*8,iv) : Else : DPut(Y,i,iv) : EndIf
    EndIf
  Next
EndProcedure

Procedure DUnary(Y.i,A.i,Op.i,Alpha.f=0.01)
  If Dt(A)\Kind<>1 : DFail("This unary kernel requires FLOAT input.") : ProcedureReturn : EndIf
  If DLike(Y,A)=0 : ProcedureReturn : EndIf
  Select Op
    Case 0 To 4 : PmTensorUnaryMath(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Op)
    Case 5 To 7 : PmFastTrig(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Op-5)
    Case 8 : PmTensorSigmoid(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 9 : PmTensorTanh(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 10 : PmTensorFloor(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 11 : PmTensorRoundEven(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 12 : PmTensorLeakyRelu(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Alpha)
  EndSelect
EndProcedure

Procedure DCast(Y.i,A.i,Kind.i)
  Protected i.i
  If DLike(Y,A,Kind)=0 : ProcedureReturn : EndIf
  For i=0 To Dt(A)\Count-1 : DPut(Y,i,DGet(A,i)) : Next
EndProcedure

Procedure DShapeOf(Y.i,A.i)
  Protected i.i
  If DShape(Y,7,1,Dt(A)\Rank)=0 : ProcedureReturn : EndIf
  For i=0 To Dt(A)\Rank-1 : PokeQ(Dt(Y)\Data+i*8,Dt(A)\D[i]) : Next
EndProcedure

Procedure DReshape(Y.i,A.i,Shape.i,AllowZero.i=0)
  Protected Dim dims.i(7)
  Protected i.i,n.i=1,unknown.i=-1,rank.i=Dt(Shape)\Count
  If rank>8 : DFail("Reshape rank exceeds eight.") : ProcedureReturn : EndIf
  For i=0 To rank-1
    dims(i)=DInt(Shape,i)
    If dims(i)=0 And AllowZero=0
      If i>=Dt(A)\Rank : DFail("Reshape zero axis is missing.") : ProcedureReturn : EndIf
      dims(i)=Dt(A)\D[i]
    EndIf
    If dims(i)=-1
      If unknown>=0 : DFail("Reshape has multiple inferred axes.") : ProcedureReturn : EndIf
      unknown=i
    Else
      If dims(i)<0 Or (dims(i)>0 And n>DLimit/dims(i)) : DFail("Reshape extent overflow.") : ProcedureReturn : EndIf
      n*dims(i)
    EndIf
  Next
  If unknown>=0
    If n<=0 Or Dt(A)\Count % n : DFail("Reshape inferred extent is invalid.") : ProcedureReturn : EndIf
    dims(unknown)=Dt(A)\Count/n : n=Dt(A)\Count
  EndIf
  If n<>Dt(A)\Count : DFail("Reshape changes the element count.") : ProcedureReturn : EndIf
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0)) : CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes) : EndIf
EndProcedure

Procedure DViewAxes(Y.i,A.i,Axes.i,Unsqueeze.i)
  Protected Dim dims.i(7), Dim mask.i(7)
  Protected i.i,ax.i,j.i,rank.i=Dt(A)\Rank
  If Unsqueeze : rank+Dt(Axes)\Count : EndIf
  If rank>8 : DFail("View rank exceeds eight.") : ProcedureReturn : EndIf
  If Axes=0 And Unsqueeze=0
    For i=0 To rank-1 : mask(i)=Bool(Dt(A)\D[i]=1) : Next
  EndIf
  For i=0 To Dt(Axes)\Count-1
    ax=DInt(Axes,i) : If ax<0 : ax+rank : EndIf
    If ax<0 Or ax>=rank Or mask(ax) : DFail("View axis is invalid.") : ProcedureReturn : EndIf
    mask(ax)=1
  Next
  If Unsqueeze
    For i=0 To rank-1
      If mask(i) : dims(i)=1 : Else : dims(i)=Dt(A)\D[j] : j+1 : EndIf
    Next
  Else
    For i=0 To rank-1
      If mask(i)
        If Dt(A)\D[i]<>1 : DFail("Cannot squeeze a non-unit axis.") : ProcedureReturn : EndIf
      Else
        dims(j)=Dt(A)\D[i] : j+1
      EndIf
    Next
    rank=j
  EndIf
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0)) : CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes) : EndIf
EndProcedure

Procedure DTranspose(Y.i,A.i,Perm.s)
  Protected Dim dims.i(7), Dim axes.i(7), Dim strides.i(7), Dim seen.i(7)
  Protected i.i,j.i,n.i,coord.i,src.i,size.i=DSize(Dt(A)\Kind),rank.i=Dt(A)\Rank
  For i=0 To rank-1
    If Perm="" : axes(i)=rank-i-1 : Else : axes(i)=Val(StringField(Perm,i+1,",")) : EndIf
    If axes(i)<0 Or axes(i)>=rank : DFail("Invalid transpose permutation.") : ProcedureReturn : EndIf
    If seen(axes(i)) : DFail("Transpose permutation has a duplicate axis.") : ProcedureReturn : EndIf
    seen(axes(i))=1
    dims(i)=Dt(A)\D[axes(i)] : strides(i)=DProduct(A,axes(i)+1,rank-1)
  Next
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  For i=0 To Dt(Y)\Count-1
    n=i : src=0
    For j=rank-1 To 0 Step -1
      coord=n % dims(j) : n/dims(j) : src+coord*strides(j)
    Next
    CopyMemory(Dt(A)\Data+src*size,Dt(Y)\Data+i*size,size)
  Next
EndProcedure

Procedure DGather(Y.i,A.i,Indices.i,Axis.i)
  Protected Dim dims.i(7)
  Protected rank.i=Dt(A)\Rank+Dt(Indices)\Rank-1,i.i,j.i,k.i,p.i,idx.i,inner.i,outer.i,size.i=DSize(Dt(A)\Kind)
  If Axis<0 : Axis+Dt(A)\Rank : EndIf
  If rank>8 Or Axis<0 Or Axis>=Dt(A)\Rank : DFail("Invalid Gather rank or axis.") : ProcedureReturn : EndIf
  For i=0 To Axis-1 : dims(p)=Dt(A)\D[i] : p+1 : Next
  For i=0 To Dt(Indices)\Rank-1 : dims(p)=Dt(Indices)\D[i] : p+1 : Next
  For i=Axis+1 To Dt(A)\Rank-1 : dims(p)=Dt(A)\D[i] : p+1 : Next
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  inner=DProduct(A,Axis+1,Dt(A)\Rank-1) : outer=DProduct(A,0,Axis-1)
  For i=0 To outer-1
    For j=0 To Dt(Indices)\Count-1
      idx=DInt(Indices,j) : If idx<0 : idx+Dt(A)\D[Axis] : EndIf
      If idx<0 Or idx>=Dt(A)\D[Axis] : DFail("Gather index is out of bounds.") : ProcedureReturn : EndIf
      CopyMemory(Dt(A)\Data+(i*Dt(A)\D[Axis]+idx)*inner*size,Dt(Y)\Data+(i*Dt(Indices)\Count+j)*inner*size,inner*size)
    Next
  Next
EndProcedure

Procedure DConcat(Y.i,Inputs.s,Axis.i)
  Protected Dim dims.i(7)
  Protected count.i=CountString(Inputs,",")+1,a.i=Val(StringField(Inputs,1,",")),i.i,j.i,outer.i,inner.i,offset.i,span.i,size.i=DSize(Dt(a)\Kind)
  If Axis<0 : Axis+Dt(a)\Rank : EndIf
  If Axis<0 Or Axis>=Dt(a)\Rank : DFail("Invalid Concat axis.") : ProcedureReturn : EndIf
  For i=0 To Dt(a)\Rank-1 : dims(i)=Dt(a)\D[i] : Next
  dims(Axis)=0
  For i=1 To count
    a=Val(StringField(Inputs,i,","))
    If Dt(a)\Rank<>Dt(Val(StringField(Inputs,1,",")))\Rank Or DSize(Dt(a)\Kind)<>size : DFail("Concat rank/type mismatch.") : ProcedureReturn : EndIf
    For j=0 To Dt(a)\Rank-1
      If j<>Axis And Dt(a)\D[j]<>dims(j) : DFail("Concat non-axis extent mismatch.") : ProcedureReturn : EndIf
    Next
    dims(Axis)+Dt(a)\D[Axis]
  Next
  If DAlloc(Y,Dt(a)\Kind,Dt(a)\Rank,@dims(0))=0 : ProcedureReturn : EndIf
  outer=DProduct(Y,0,Axis-1) : inner=DProduct(Y,Axis+1,Dt(Y)\Rank-1)
  For j=0 To outer-1
    offset=j*dims(Axis)*inner
    For i=1 To count
      a=Val(StringField(Inputs,i,",")) : span=Dt(a)\D[Axis]*inner
      CopyMemory(Dt(a)\Data+j*span*size,Dt(Y)\Data+offset*size,span*size) : offset+span
    Next
  Next
EndProcedure

Procedure DSlice(Y.i,A.i,Starts.i,Ends.i,Axes.i,Steps.i)
  Protected Dim dims.i(7),Dim start.i(7),Dim stepv.i(7),Dim stride.i(7)
  Protected i.i,j.i,ax.i,s.i,e.i,st.i,n.i,src.i,c.i,size.i=DSize(Dt(A)\Kind)
  For i=0 To Dt(A)\Rank-1 : dims(i)=Dt(A)\D[i] : stepv(i)=1 : stride(i)=DProduct(A,i+1,Dt(A)\Rank-1) : Next
  For i=0 To Dt(Starts)\Count-1
    ax=i : If Axes : ax=DInt(Axes,i) : EndIf
    If ax<0 : ax+Dt(A)\Rank : EndIf
    If ax<0 Or ax>=Dt(A)\Rank : DFail("Invalid Slice axis.") : ProcedureReturn : EndIf
    n=Dt(A)\D[ax] : s=DInt(Starts,i) : e=DInt(Ends,i) : st=1
    If Steps : st=DInt(Steps,i) : EndIf
    If st=0 : DFail("Slice step is zero.") : ProcedureReturn : EndIf
    If s<0 : s+n : EndIf
    If e<0 : e+n : EndIf
    If st>0
      s=DMax(0,DMin(n,s)) : e=DMax(0,DMin(n,e)) : dims(ax)=DMax(0,(e-s+st-1)/st)
    Else
      s=DMax(-1,DMin(n-1,s)) : e=DMax(-1,DMin(n-1,e)) : dims(ax)=DMax(0,(s-e-st-1)/(-st))
    EndIf
    start(ax)=s : stepv(ax)=st
  Next
  If DAlloc(Y,Dt(A)\Kind,Dt(A)\Rank,@dims(0))=0 : ProcedureReturn : EndIf
  For i=0 To Dt(Y)\Count-1
    n=i : src=0
    For j=Dt(Y)\Rank-1 To 0 Step -1
      c=n % dims(j) : n/dims(j) : src+(start(j)+c*stepv(j))*stride(j)
    Next
    CopyMemory(Dt(A)\Data+src*size,Dt(Y)\Data+i*size,size)
  Next
EndProcedure

Procedure DExpand(Y.i,A.i,Shape.i)
  Protected Dim dims.i(7)
  Protected i.i,rank.i=Dt(Shape)\Count,size.i=DSize(Dt(A)\Kind),ax.i
  If rank>8 Or rank<Dt(A)\Rank : DFail("Invalid Expand rank.") : ProcedureReturn : EndIf
  For i=0 To rank-1
    dims(i)=DInt(Shape,i) : ax=i-rank+Dt(A)\Rank
    If ax>=0
      If dims(i)=1 : dims(i)=Dt(A)\D[ax] : EndIf
      If dims(i)<>Dt(A)\D[ax] And Dt(A)\D[ax]<>1 : DFail("Invalid Expand extent.") : ProcedureReturn : EndIf
    EndIf
  Next
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  For i=0 To Dt(Y)\Count-1 : CopyMemory(Dt(A)\Data+DBroadcastIndex(i,Y,A)*size,Dt(Y)\Data+i*size,size) : Next
EndProcedure

Procedure DConstantShape(Y.i,Shape.i,Kind.i,Value.d)
  Protected Dim dims.i(7)
  Protected i.i
  If Dt(Shape)\Count>8 : DFail("ConstantOfShape rank exceeds eight.") : ProcedureReturn : EndIf
  For i=0 To Dt(Shape)\Count-1 : dims(i)=DInt(Shape,i) : Next
  If DAlloc(Y,Kind,Dt(Shape)\Count,@dims(0))
    For i=0 To Dt(Y)\Count-1 : DPut(Y,i,Value) : Next
  EndIf
EndProcedure

Procedure DRange(Y.i,A.i,B.i,C.i)
  Protected start.d=DGet(A),stop.d=DGet(B),stepv.d=DGet(C),n.i,i.i,extent.d
  If stepv=0 : DFail("Range step is zero.") : ProcedureReturn : EndIf
  extent=(stop-start)/stepv
  If (PeekQ(@extent) & $7FF0000000000000)=$7FF0000000000000 Or extent>DLimit/DSize(Dt(A)\Kind) : DFail("Range exceeds finite memory bounds.") : ProcedureReturn : EndIf
  n=DMax(0,Round(extent,#PB_Round_Up))
  If DShape(Y,Dt(A)\Kind,1,n)
    For i=0 To n-1 : DPut(Y,i,start+i*stepv) : Next
  EndIf
EndProcedure

Procedure DWhere(Y.i,Cond.i,A.i,B.i)
  Protected i.i,src.i,index.i,size.i=DSize(Dt(A)\Kind),ax.i
  If DBroadcast(Y,A,B)=0 : ProcedureReturn : EndIf
  If Dt(Cond)\Rank>Dt(Y)\Rank : DFail("Where condition expands beyond the data shape.") : ProcedureReturn : EndIf
  For i=0 To Dt(Cond)\Rank-1
    ax=i+Dt(Y)\Rank-Dt(Cond)\Rank
    If Dt(Cond)\D[i]<>1 And Dt(Cond)\D[i]<>Dt(Y)\D[ax] : DFail("Where condition broadcast mismatch.") : ProcedureReturn : EndIf
  Next
  For i=0 To Dt(Y)\Count-1
    If DGet(Cond,DBroadcastIndex(i,Y,Cond)) : src=A : Else : src=B : EndIf
    index=DBroadcastIndex(i,Y,src)
    CopyMemory(Dt(src)\Data+index*size,Dt(Y)\Data+i*size,size)
  Next
EndProcedure

Procedure DNonZero(Y.i,A.i)
  Protected i.i,j.i,count.i,n.i,col.i,c.i
  For i=0 To Dt(A)\Count-1 : If DGet(A,i)<>0 : count+1 : EndIf : Next
  If DShape(Y,7,2,Dt(A)\Rank,count)=0 : ProcedureReturn : EndIf
  For i=0 To Dt(A)\Count-1
    If DGet(A,i)<>0
      n=i
      For j=Dt(A)\Rank-1 To 0 Step -1
        c=n % Dt(A)\D[j] : n/Dt(A)\D[j] : PokeQ(Dt(Y)\Data+(j*count+col)*8,c)
      Next
      col+1
    EndIf
  Next
EndProcedure

Procedure DScatter(Y.i,A.i,Indices.i,Updates.i)
  Protected k.i,i.i,j.i,offset.i,idx.i,block.i,size.i=DSize(Dt(A)\Kind),tuples.i
  If Dt(Indices)\Rank<1 : DFail("ScatterND indices need a tuple axis.") : ProcedureReturn : EndIf
  k=Dt(Indices)\D[Dt(Indices)\Rank-1]
  If k<=0 Or k>Dt(A)\Rank : DFail("Invalid ScatterND tuple width.") : ProcedureReturn : EndIf
  block=DProduct(A,k,Dt(A)\Rank-1) : tuples=Dt(Indices)\Count/k
  If Dt(Updates)\Count<>tuples*block : DFail("ScatterND update count mismatch.") : ProcedureReturn : EndIf
  If DLike(Y,A)=0 : ProcedureReturn : EndIf
  CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes)
  For i=0 To tuples-1
    offset=0
    For j=0 To k-1
      idx=DInt(Indices,i*k+j) : If idx<0 : idx+Dt(A)\D[j] : EndIf
      If idx<0 Or idx>=Dt(A)\D[j] : DFail("ScatterND index is out of bounds.") : ProcedureReturn : EndIf
      offset=offset*Dt(A)\D[j]+idx
    Next
    CopyMemory(Dt(Updates)\Data+i*block*size,Dt(Y)\Data+offset*block*size,block*size)
  Next
EndProcedure

Procedure DPad(Y.i,A.i,Pads.i,Value.i,Mode.i)
  Protected Dim dims.i(7),Dim starts.i(7),Dim strides.i(7)
  Protected i.i,j.i,n.i,c.i,src.i,outside.i,size.i=DSize(Dt(A)\Kind),v.d
  If Dt(Pads)\Count<>Dt(A)\Rank*2 : DFail("Pad extent count mismatch.") : ProcedureReturn : EndIf
  If Value : v=DGet(Value) : EndIf
  For i=0 To Dt(A)\Rank-1
    starts(i)=DInt(Pads,i) : dims(i)=Dt(A)\D[i]+starts(i)+DInt(Pads,i+Dt(A)\Rank)
    strides(i)=DProduct(A,i+1,Dt(A)\Rank-1)
  Next
  If DAlloc(Y,Dt(A)\Kind,Dt(A)\Rank,@dims(0))=0 : ProcedureReturn : EndIf
  For i=0 To Dt(Y)\Count-1
    n=i : src=0 : outside=0
    For j=Dt(A)\Rank-1 To 0 Step -1
      c=n % dims(j)-starts(j) : n/dims(j)
      If c<0 Or c>=Dt(A)\D[j]
        If Mode=1
          If Dt(A)\D[j]<2 : DFail("Reflect padding requires extent above one.") : ProcedureReturn : EndIf
          While c<0 Or c>=Dt(A)\D[j]
            If c<0 : c=-c : Else : c=2*Dt(A)\D[j]-2-c : EndIf
          Wend
        Else
          outside=1
        EndIf
      EndIf
      src+c*strides(j)
    Next
    If outside : DPut(Y,i,v) : Else : CopyMemory(Dt(A)\Data+src*size,Dt(Y)\Data+i*size,size) : EndIf
  Next
EndProcedure

Procedure DMatMul(Y.i,A.i,B.i)
  Protected Dim dims.i(7)
  Protected rank.i=DMax(Dt(A)\Rank,Dt(B)\Rank),i.i,ai.i,bi.i,ad.i,bd.i
  Protected m.i,k.i,n.i,batches.i,batch.i,aoff.i,boff.i,idx.i,c.i,astride.i,bstride.i
  If Dt(A)\Kind<>1 Or (Dt(B)\Kind<>1 And Dt(B)\Kind<>3) : DFail("MatMul requires FLOAT activations and FLOAT or INT8 weights.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<2 Or Dt(B)\Rank<2 Or rank>8 : DFail("MatMul requires rank two or above.") : ProcedureReturn : EndIf
  m=Dt(A)\D[Dt(A)\Rank-2] : k=Dt(A)\D[Dt(A)\Rank-1] : n=Dt(B)\D[Dt(B)\Rank-1]
  If k<>Dt(B)\D[Dt(B)\Rank-2] : DFail("MatMul contraction mismatch.") : ProcedureReturn : EndIf
  If Dt(B)\Kind=3 And (Dt(B)\Scales=0 Or k>133144) : DFail("INT8 MatMul scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
  batches=1
  For i=0 To rank-3
    ai=i-rank+Dt(A)\Rank : bi=i-rank+Dt(B)\Rank : ad=1 : bd=1
    If ai>=0 : ad=Dt(A)\D[ai] : EndIf
    If bi>=0 : bd=Dt(B)\D[bi] : EndIf
    If ad<>bd And ad<>1 And bd<>1 : DFail("MatMul batch mismatch.") : ProcedureReturn : EndIf
    dims(i)=DMax(ad,bd) : batches*dims(i)
  Next
  dims(rank-2)=m : dims(rank-1)=n
  If DAlloc(Y,1,rank,@dims(0))=0 : ProcedureReturn : EndIf
  For batch=0 To batches-1
    idx=batch : aoff=0 : boff=0 : astride=m*k : bstride=k*n
    For i=rank-3 To 0 Step -1
      c=idx % dims(i) : idx/dims(i)
      ai=i-rank+Dt(A)\Rank : bi=i-rank+Dt(B)\Rank
      If ai>=0
        If Dt(A)\D[ai]<>1 : aoff+c*astride : EndIf
        astride*Dt(A)\D[ai]
      EndIf
      If bi>=0
        If Dt(B)\D[bi]<>1 : boff+c*bstride : EndIf
        bstride*Dt(B)\D[bi]
      EndIf
    Next
    If Dt(B)\Kind=3
      If DInt8Begin(Dt(A)\Data+aoff*4,m*k)=0 : ProcedureReturn : EndIf
      PmTensorMatMul2Int8(Dt(A)\Data+aoff*4,Dt(B)\Data+boff,Dt(Y)\Data+batch*m*n*4,m,k,n,Dt(B)\Scales)
      DInt8End()
    Else
      PmFastGemm(Dt(A)\Data+aoff*4,Dt(B)\Data+boff*4,Dt(Y)\Data+batch*m*n*4,m,k,n)
    EndIf
  Next
EndProcedure

Procedure DGemm(Y.i,A.i,B.i,C.i,TransA.i,TransB.i,Alpha.f,Beta.f)
  Protected g.PmTensorGemmArgs
  If Dt(A)\Kind<>1 Or (Dt(B)\Kind<>1 And Dt(B)\Kind<>3) Or (C And Dt(C)\Kind<>1) Or TransA<0 Or TransA>1 Or TransB<0 Or TransB>1 : DFail("Gemm requires FLOAT activations, FLOAT/INT8 weights and valid transpose flags.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<>2 Or Dt(B)\Rank<>2 : DFail("Gemm requires matrices.") : ProcedureReturn : EndIf
  g\M=Dt(A)\D[TransA] : g\K=Dt(A)\D[1-TransA] : g\N=Dt(B)\D[1-TransB]
  If g\K<>Dt(B)\D[TransB] : DFail("Gemm contraction mismatch.") : ProcedureReturn : EndIf
  If DShape(Y,1,2,g\M,g\N)=0 : ProcedureReturn : EndIf
  g\A=Dt(A)\Data : g\B=Dt(B)\Data : g\C=Dt(C)\Data : g\Dst=Dt(Y)\Data
  g\CCount=Dt(C)\Count : g\TransA=TransA : g\TransB=TransB : g\Alpha=Alpha : g\Beta=Beta
  If Dt(B)\Kind=3
    If Dt(B)\Scales=0 Or g\K>133144 : DFail("INT8 Gemm scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
    If DInt8Begin(g\A,g\M*g\K)=0 : ProcedureReturn : EndIf
    g\WeightScales=Dt(B)\Scales
  EndIf
  PmTensorGemm(@g)
  If Dt(B)\Kind=3 : DInt8End() : EndIf
EndProcedure

Procedure DReduce(Y.i,A.i,Axes.i,Keep.i,Mean.i)
  Protected Dim dims.i(7)
  Protected i.i,axis.i=DInt(Axes),rank.i=Dt(A)\Rank,width.i
  If axis<0 : axis+rank : EndIf
  If Dt(Axes)\Count<>1 Or axis<>rank-1 : DFail("Reduction currently requires one final axis.") : ProcedureReturn : EndIf
  For i=0 To rank-1 : dims(i)=Dt(A)\D[i] : Next
  width=dims(rank-1)
  If Keep : dims(rank-1)=1 : Else : rank-1 : EndIf
  If width<=0 : DFail("Reduction width is empty.") : ProcedureReturn : EndIf
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  If Dt(A)\Kind<>1 : DFail("Reduction requires FLOAT.") : ProcedureReturn : EndIf
  PmFastReduce(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Count/width,width,Mean)
EndProcedure

Procedure DSoftmax(Y.i,A.i,Axis.i)
  Protected width.i
  If Dt(A)\Rank<1 Or Dt(A)\Kind<>1 : DFail("Softmax requires a FLOAT tensor with an axis.") : ProcedureReturn : EndIf
  width=Dt(A)\D[Dt(A)\Rank-1]
  If Axis<0 : Axis+Dt(A)\Rank : EndIf
  If Axis<>Dt(A)\Rank-1 Or width<=0 : DFail("Softmax requires a nonempty final axis.") : ProcedureReturn : EndIf
  If DLike(Y,A) : PmTensorSoftmaxLast(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Count/width,width) : EndIf
EndProcedure

Procedure DLayerNorm(Y.i,A.i,Scale.i,Bias.i,Epsilon.f)
  Protected width.i
  If Dt(A)\Rank<1 Or Dt(A)\Kind<>1 Or Dt(Scale)\Kind<>1 Or (Bias And Dt(Bias)\Kind<>1) : DFail("Layer normalization requires FLOAT tensors.") : ProcedureReturn : EndIf
  width=Dt(A)\D[Dt(A)\Rank-1]
  If width<=0 Or Dt(Scale)\Count<>width : DFail("Layer normalization width mismatch.") : ProcedureReturn : EndIf
  If Bias And Dt(Bias)\Count<>width : DFail("Layer normalization bias width mismatch.") : ProcedureReturn : EndIf
  If DLike(Y,A) : PmTensorLayerNormLast(Dt(A)\Data,Dt(Scale)\Data,Dt(Bias)\Data,Dt(Y)\Data,Dt(A)\Count/width,width,Epsilon) : EndIf
EndProcedure

Procedure DCumSum(Y.i,A.i,AxisTensor.i,Exclusive.i,Reverse.i)
  Protected axis.i=DInt(AxisTensor),outer.i,inner.i,width.i
  If Dt(A)\Kind<>1 And Dt(A)\Kind<>7 : DFail("CumSum requires FLOAT or INT64.") : ProcedureReturn : EndIf
  If axis<0 : axis+Dt(A)\Rank : EndIf
  If axis<0 Or axis>=Dt(A)\Rank : DFail("CumSum axis is invalid.") : ProcedureReturn : EndIf
  outer=DProduct(A,0,axis-1) : inner=DProduct(A,axis+1,Dt(A)\Rank-1) : width=Dt(A)\D[axis]
  If DLike(Y,A)=0 : ProcedureReturn : EndIf
  If Dt(A)\Kind=7
    PmTensorCumSumI64(Dt(A)\Data,Dt(Y)\Data,outer,width,inner,Exclusive,Reverse)
  Else
    PmTensorCumSumF32(Dt(A)\Data,Dt(Y)\Data,outer,width,inner,Exclusive,Reverse)
  EndIf
EndProcedure

Procedure DClip(Y.i,A.i,Lo.i,Hi.i)
  Protected low.f=-3.402823466e38,high.f=3.402823466e38
  If Dt(A)\Kind<>1 : DFail("Clip requires FLOAT input.") : ProcedureReturn : EndIf
  If Lo : low=DGet(Lo) : EndIf
  If Hi : high=DGet(Hi) : EndIf
  If DLike(Y,A) : PmTensorClip(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Count,low,high) : EndIf
EndProcedure

Procedure DConv(Y.i,A.i,W.i,B.i,PadLeft.i,PadRight.i,Stride.i,Dilation.i,Groups.i)
  Protected g.PmTensorConv1DArgs
  If Dt(A)\Kind<>1 Or (Dt(W)\Kind<>1 And Dt(W)\Kind<>3) Or (B And Dt(B)\Kind<>1) : DFail("Conv requires FLOAT activations and FLOAT/INT8 weights.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<>3 Or Dt(W)\Rank<>3 Or Stride<=0 Or Groups<=0 : DFail("Invalid Conv1D shape or attributes.") : ProcedureReturn : EndIf
  g\Batches=Dt(A)\D[0] : g\InChannels=Dt(A)\D[1] : g\InWidth=Dt(A)\D[2]
  g\OutChannels=Dt(W)\D[0] : g\Kernel=Dt(W)\D[2]
  g\OutWidth=(g\InWidth+PadLeft+PadRight-Dilation*(g\Kernel-1)-1)/Stride+1
  If g\InChannels<>Dt(W)\D[1]*Groups Or g\OutChannels % Groups : DFail("Conv group/channel mismatch.") : ProcedureReturn : EndIf
  If B And Dt(B)\Count<>g\OutChannels : DFail("Conv bias width mismatch.") : ProcedureReturn : EndIf
  If DShape(Y,1,3,g\Batches,g\OutChannels,g\OutWidth)=0 : ProcedureReturn : EndIf
  g\Src=Dt(A)\Data : g\Weight=Dt(W)\Data : g\Bias=Dt(B)\Data : g\Dst=Dt(Y)\Data
  g\PadLeft=PadLeft : g\Stride=Stride : g\Dilation=Dilation : g\Groups=Groups
  If Dt(W)\Kind=3
    If Dt(W)\Scales=0 Or g\InChannels/Groups*g\Kernel>133144 : DFail("INT8 Conv scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
    If DInt8Begin(g\Src,Dt(A)\Count)=0 : ProcedureReturn : EndIf
    g\WeightScales=Dt(W)\Scales : PmTensorConv1D(@g) : DInt8End()
  Else
    If PmFastConv(@g,DLimit-DLive)=0 : DFail("Convolution scratch exceeds available working memory.") : EndIf
  EndIf
EndProcedure

Procedure DConvTranspose(Y.i,A.i,W.i,B.i,PadLeft.i,PadRight.i,Stride.i,Dilation.i,Groups.i,OutputPad.i)
  Protected g.PmTensorConvTranspose1DArgs
  If Dt(A)\Kind<>1 Or Dt(W)\Kind<>1 Or (B And Dt(B)\Kind<>1) : DFail("Transposed Conv requires FLOAT tensors.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<>3 Or Dt(W)\Rank<>3 Or Stride<=0 Or Groups<=0 : DFail("Invalid transposed convolution.") : ProcedureReturn : EndIf
  g\Batches=Dt(A)\D[0] : g\InChannels=Dt(A)\D[1] : g\InWidth=Dt(A)\D[2]
  g\OutChannels=Dt(W)\D[1]*Groups : g\Kernel=Dt(W)\D[2]
  g\OutWidth=(g\InWidth-1)*Stride-PadLeft-PadRight+Dilation*(g\Kernel-1)+1+OutputPad
  If g\InChannels<>Dt(W)\D[0] Or g\InChannels % Groups : DFail("Transposed Conv channel mismatch.") : ProcedureReturn : EndIf
  If B And Dt(B)\Count<>g\OutChannels : DFail("Transposed Conv bias width mismatch.") : ProcedureReturn : EndIf
  If DShape(Y,1,3,g\Batches,g\OutChannels,g\OutWidth)=0 : ProcedureReturn : EndIf
  g\Src=Dt(A)\Data : g\Weight=Dt(W)\Data : g\Bias=Dt(B)\Data : g\Dst=Dt(Y)\Data
  g\PadLeft=PadLeft : g\Stride=Stride : g\Dilation=Dilation : g\Groups=Groups
  If PmFastConvTranspose(@g,DLimit-DLive)=0 : DFail("Transposed convolution scratch exceeds available working memory.") : EndIf
EndProcedure

Procedure DResize(Y.i,A.i,Scales.i,Sizes.i,Mode.i,Coordinate.i)
  Protected Dim dims.i(7)
  Protected i.i,last.i=Dt(A)\Rank-1,scale.f
  If last<0 Or Dt(A)\Kind<>1 : DFail("Resize requires a FLOAT tensor with an axis.") : ProcedureReturn : EndIf
  For i=0 To last
    If Sizes : dims(i)=DInt(Sizes,i) : Else : dims(i)=Int(Dt(A)\D[i]*DGet(Scales,i)) : EndIf
    If i<last And dims(i)<>Dt(A)\D[i] : DFail("Resize supports the final axis only.") : ProcedureReturn : EndIf
  Next
  If Dt(A)\D[last]<=0 Or dims(last)<=0 : DFail("Resize has an empty width.") : ProcedureReturn : EndIf
  scale=dims(last)/Dt(A)\D[last] : If Scales : scale=DGet(Scales,last) : EndIf
  If DAlloc(Y,1,Dt(A)\Rank,@dims(0))
    PmTensorResize1D(Dt(A)\Data,Dt(Y)\Data,DProduct(A,0,last-1),Dt(A)\D[last],dims(last),Mode,Coordinate,scale)
  EndIf
EndProcedure

Procedure DLstm(Y.i,YH.i,YC.i,X.i,W.i,R.i,B.i,Seq.i,IH.i,IC.i,Hidden.i,Directions.i)
  Protected g.PmTensorLstmArgs
  Protected seq64.i,i.i,valid.i,seqBytes.i
  If Dt(X)\Rank<>3 Or Dt(W)\Rank<>3 Or Dt(R)\Rank<>3 Or Dt(X)\Kind<>1 Or (Dt(W)\Kind<>1 And Dt(W)\Kind<>3) Or (Dt(R)\Kind<>1 And Dt(R)\Kind<>3) : DFail("LSTM requires FLOAT activations and rank-three FLOAT/INT8 weights.") : ProcedureReturn : EndIf
  g\Sequence=Dt(X)\D[0] : g\Batch=Dt(X)\D[1] : g\InputSize=Dt(X)\D[2]
  g\Hidden=Hidden : g\Directions=Directions
  If Hidden<=0 Or Hidden>4096 Or Directions<1 Or Directions>2 : DFail("LSTM dimensions are invalid.") : ProcedureReturn : EndIf
  If Dt(W)\D[0]<>Directions Or Dt(W)\D[1]<>4*Hidden Or Dt(W)\D[2]<>g\InputSize Or Dt(R)\D[0]<>Directions Or Dt(R)\D[1]<>4*Hidden Or Dt(R)\D[2]<>Hidden : DFail("LSTM weight dimensions do not match the input and hidden state.") : ProcedureReturn : EndIf
  If B And (Dt(B)\Kind<>1 Or Dt(B)\Count<>Directions*8*Hidden) : DFail("LSTM bias dimensions are invalid.") : ProcedureReturn : EndIf
  If Seq And (Dt(Seq)\Kind<>6 Or Dt(Seq)\Count<>g\Batch) : DFail("LSTM sequence lengths must be INT32 per batch.") : ProcedureReturn : EndIf
  If IH And (Dt(IH)\Kind<>1 Or Dt(IH)\Count<>Directions*g\Batch*Hidden) : DFail("LSTM initial hidden state is invalid.") : ProcedureReturn : EndIf
  If IC And (Dt(IC)\Kind<>1 Or Dt(IC)\Count<>Directions*g\Batch*Hidden) : DFail("LSTM initial cell state is invalid.") : ProcedureReturn : EndIf
  If DShape(Y,1,4,g\Sequence,Directions,g\Batch,Hidden)=0 : ProcedureReturn : EndIf
  If DShape(YH,1,3,Directions,g\Batch,Hidden)=0 : ProcedureReturn : EndIf
  If DShape(YC,1,3,Directions,g\Batch,Hidden)=0 : ProcedureReturn : EndIf
  g\X=Dt(X)\Data : g\W=Dt(W)\Data : g\R=Dt(R)\Data : g\B=Dt(B)\Data
  g\SeqLens=Dt(Seq)\Data : g\InitialH=Dt(IH)\Data : g\InitialC=Dt(IC)\Data
  g\Y=Dt(Y)\Data : g\YH=Dt(YH)\Data : g\YC=Dt(YC)\Data
  If Dt(W)\Kind=3 Or Dt(R)\Kind=3
    If DFiniteTensor(X)=0 Or DFiniteTensor(IH)=0 Or DFiniteTensor(IC)=0 : ProcedureReturn : EndIf
    If (Dt(W)\Kind=3 And Dt(W)\Scales=0) Or (Dt(R)\Kind=3 And Dt(R)\Scales=0) Or g\InputSize>133144 : DFail("INT8 LSTM scales are missing or reduction is too large.") : ProcedureReturn : EndIf
    If Seq
      seqBytes=g\Batch*8
      If seqBytes>DLimit-DLive : DFail("LSTM sequence metadata exceeds available memory.") : ProcedureReturn : EndIf
      seq64=AllocateMemory(seqBytes)
      If seq64=0 : DFail("LSTM sequence metadata allocation failed.") : ProcedureReturn : EndIf
      For i=0 To g\Batch-1
        valid=PeekL(Dt(Seq)\Data+i*4)
        If valid<0 Or valid>g\Sequence : FreeMemory(seq64) : DFail("LSTM sequence length is out of bounds.") : ProcedureReturn : EndIf
        PokeQ(seq64+i*8,valid)
      Next
      DLive+seqBytes : DPeak=DMax(DPeak,DLive) : g\SeqLens=seq64
    EndIf
    If DInt8Begin(g\X,g\InputSize,g\Hidden)
      g\WScales=Dt(W)\Scales : g\RScales=Dt(R)\Scales
      PmTensorLstm(@g) : DInt8End()
    EndIf
    If seq64 : FreeMemory(seq64) : DLive-seqBytes : EndIf
  Else
    If PmFastLstm(@g,DLimit-DLive)=0 : DFail("LSTM scratch or sequence lengths exceed the model bounds.") : EndIf
  EndIf
EndProcedure

Procedure DStft(Y.i,A.i,StepTensor.i,Window.i,LengthTensor.i,OneSided.i)
  Protected g.PmTensorStftArgs,complex.i=1
  If Dt(A)\Kind<>1 Or Dt(A)\Rank<2 Or Dt(A)\Rank>3 : DFail("STFT requires real FLOAT batches.") : ProcedureReturn : EndIf
  If Dt(A)\Rank=3 And Dt(A)\D[2]<>1 : DFail("Complex STFT input is not implemented.") : ProcedureReturn : EndIf
  g\Batches=Dt(A)\D[0] : g\SignalLength=Dt(A)\D[1] : g\FrameStep=DInt(StepTensor)
  If LengthTensor : g\FrameLength=DInt(LengthTensor) : Else : g\FrameLength=Dt(Window)\Count : EndIf
  If g\FrameStep<=0 Or g\FrameLength<=0 Or g\FrameLength>65536 : DFail("STFT frame parameters are invalid.") : ProcedureReturn : EndIf
  If Window And (Dt(Window)\Kind<>1 Or Dt(Window)\Count<>g\FrameLength) : DFail("STFT window length/type mismatch.") : ProcedureReturn : EndIf
  g\Frames=(g\SignalLength-g\FrameLength)/g\FrameStep+1
  g\Bins=g\FrameLength : If OneSided : g\Bins=g\FrameLength/2+1 : EndIf
  If DShape(Y,1,4,g\Batches,g\Frames,g\Bins,2)=0 : ProcedureReturn : EndIf
  While complex<2*g\FrameLength-1 : complex*2 : Wend
  If complex>(DLimit-DLive)/40 : DFail("STFT scratch exceeds the working-memory limit.") : ProcedureReturn : EndIf
  g\ScratchComplex=complex : g\Scratch=AllocateMemory(complex*40)
  If g\Scratch=0 : DFail("STFT scratch allocation failed.") : ProcedureReturn : EndIf
  g\Signal=Dt(A)\Data : g\Window=Dt(Window)\Data : g\Dst=Dt(Y)\Data
  If PmTensorStft(@g)=0 : DFail("STFT execution failed.") : EndIf
  FreeMemory(g\Scratch)
EndProcedure
