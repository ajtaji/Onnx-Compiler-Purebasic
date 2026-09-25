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
    Case 2,3,9 : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

; An INT8 operator scans its own activations for NaN and infinity while it
; finds the row scales, and allocates its own working buffers; it reports
; either failure through PmTensorInt8Fault (1 = not finite, 2 = no memory).
Procedure DInt8Begin()
  PmTensorInt8Fault=0
EndProcedure

Procedure.i DFiniteTensor(Id.i)
  Protected i.i
  If Id=0 : ProcedureReturn 1 : EndIf
  For i=0 To Dt(Id)\Count-1
    If (PeekL(Dt(Id)\Data+i*4) & $7F800000)=$7F800000 : ProcedureReturn DFail("INT8 recurrent inputs contain NaN or infinity.") : EndIf
  Next
  ProcedureReturn 1
EndProcedure

Procedure.i DInt8End()
  Select PmTensorInt8Fault
    Case 0 : ProcedureReturn 1
    Case 1 : ProcedureReturn DFail("INT8 activations contain NaN or infinity; the operator refuses to quantize them. Check the node that produced this input.")
  EndSelect
  ProcedureReturn DFail("INT8 working memory could not be allocated for this operator.")
EndProcedure

; ---- the block cache of a multi-threaded model --------------------------------
; A fresh block of a megabyte or more arrives as untouched pages, and the
; first write to each page is a page fault the system serves one at a time,
; however many workers write. With more than one thread, released blocks of
; that size are therefore kept (at most 64 blocks, 512 MB) and handed out
; again, zeroed by the pool, so DAlloc still returns zeroed memory. The cache
; is emptied when the pool stops (unbind, or a new thread count). With one
; thread nothing is kept and allocation is exactly the single-threaded one.
#PMD_CACHE_MIN = 1048576
#PMD_CACHE_SLOTS = 64
#PMD_CACHE_LIMIT = 536870912
Global Dim DCacheBlock.i(#PMD_CACHE_SLOTS - 1)
Global Dim DCacheBytes.i(#PMD_CACHE_SLOTS - 1)
Global DCacheCount.i, DCacheTotal.i
; Set by an operator around its own DAlloc when its kernel writes every
; element of the new tensor (Binary, Unary): a reused block is then handed
; out as it is, since the kernel overwrites all of it.
Global DAllocOverwrite.i

Procedure DCacheFlush()
  Protected i.i
  For i = 0 To DCacheCount - 1 : FreeMemory(DCacheBlock(i)) : Next
  DCacheCount = 0 : DCacheTotal = 0
EndProcedure
PmPoolStopHook = @DCacheFlush()

; The smallest kept block of at least `bytes` (and at most twice that), zeroed
; over `bytes`; 0 when there is none.
Procedure.i DCacheTake(bytes.i)
  Protected i.i, best.i = -1, *mem
  For i = 0 To DCacheCount - 1
    If DCacheBytes(i) >= bytes And DCacheBytes(i) / 2 <= bytes
      If best < 0 Or DCacheBytes(i) < DCacheBytes(best) : best = i : EndIf
    EndIf
  Next
  If best < 0 : ProcedureReturn 0 : EndIf
  *mem = DCacheBlock(best) : DCacheTotal - DCacheBytes(best)
  DCacheCount - 1
  DCacheBlock(best) = DCacheBlock(DCacheCount) : DCacheBytes(best) = DCacheBytes(DCacheCount)
  If DAllocOverwrite = 0 : PmPoolZeroFill(*mem, bytes) : EndIf
  ProcedureReturn *mem
EndProcedure

Procedure DCacheGive(*mem)
  Protected bytes.i = MemorySize(*mem), i.i
  If PmPoolThreads <= 1 Or bytes < #PMD_CACHE_MIN Or bytes > #PMD_CACHE_LIMIT
    FreeMemory(*mem) : ProcedureReturn
  EndIf
  ; Make room by returning the oldest kept blocks to the heap.
  While DCacheCount > 0 And (DCacheCount = #PMD_CACHE_SLOTS Or DCacheTotal + bytes > #PMD_CACHE_LIMIT)
    FreeMemory(DCacheBlock(0)) : DCacheTotal - DCacheBytes(0)
    For i = 1 To DCacheCount - 1 : DCacheBlock(i - 1) = DCacheBlock(i) : DCacheBytes(i - 1) = DCacheBytes(i) : Next
    DCacheCount - 1
  Wend
  DCacheBlock(DCacheCount) = *mem : DCacheBytes(DCacheCount) = bytes
  DCacheCount + 1 : DCacheTotal + bytes
EndProcedure

Procedure DRelease(Id.i)
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : ProcedureReturn : EndIf
  If Dt(Id)\Owned
    DCacheGive(Dt(Id)\Data)
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
  Dt(Id)\Data = 0
  If bytes >= #PMD_CACHE_MIN And PmPoolThreads > 1 : Dt(Id)\Data = DCacheTake(bytes) : EndIf
  If Dt(Id)\Data = 0 : Dt(Id)\Data = AllocateMemory(DMax(bytes, 1)) : EndIf
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
    Case 2 : ProcedureReturn PeekA(Dt(Id)\Data+Index)
    Case 3 : ProcedureReturn PeekB(Dt(Id)\Data+Index)
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
    ; Toward zero, as the ONNX reference and the bare-metal runtime convert; a
    ; plain assignment here would round to nearest (forum 860).
    Case 7 : PokeQ(Dt(Id)\Data+Index*8,IntQ(Value))
    Case 6 : PokeL(Dt(Id)\Data+Index*4,IntQ(Value))
    ; a NaN is true (forum 998): the host's <> answers false for it
    Case 9 : PokeA(Dt(Id)\Data+Index,Bool(Value<>0 Or (PeekQ(@Value) & $7FFFFFFFFFFFFFFF) > $7FF0000000000000))
    Case 2,3 : PokeA(Dt(Id)\Data+Index,IntQ(Value) & 255)
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

; ---- runtime-dimension loops on the pool -------------------------------------
; Each loop below that visits output elements (or output rows) in index order
; is cut into index ranges; a range runs the loop's own statements, so every
; element is computed exactly as the whole loop computed it. Loops that can
; fail part-way (integer division by zero, an index out of range) stay on
; the calling thread, where DFail may write DError.
Structure DIndexJob
  Kind.i
  Y.i
  A.i
  B.i
  C.i
  Count.i
  Chunk.i
  Size.i
  Rank.i
  Op.i
  Width.i
  AStep.i
  BStep.i
  RowSegments.i
  SegWidth.i
  Mask.i
  Inner.i
  Mean.i
  F1.f
  Dims.i[8]
  Strides.i[8]
  Start.i[8]
  Steps.i[8]
EndStructure

Declare DIndexTask(*j.DIndexJob,task.i,worker.i)

Procedure DIndexRun(*j.DIndexJob,grain.i,align.i=1)
  Protected tasks.i=PmPoolTasks(*j\Count,grain,align,@*j\Chunk)
  If tasks<=1
    *j\Chunk=*j\Count : DIndexTask(*j,0,0)
  Else
    PmPoolRun(@DIndexTask(),*j,tasks)
  EndIf
EndProcedure
Procedure DIndexTask(*j.DIndexJob,task.i,worker.i)
  Protected first.i=task* *j\Chunk,last.i=first+ *j\Chunk,i.i,n.i,src.i,c.i,d.i,index.i,base.i,at.i,coord.i,seg.i,row.i,c0.i,cols.i,ai.i,bi.i
  Protected Y.i=*j\Y,A.i=*j\A,B.i=*j\B,size.i=*j\Size,rank.i=*j\Rank
  Protected av.f,bv.f,v.f,acc.f,x.f,cf.f
  If last>*j\Count : last=*j\Count : EndIf
  If first>=last : ProcedureReturn : EndIf
  Select *j\Kind
    Case 0 ; DBinary, FLOAT fast path: row segments
      For seg=first To last-1
        row=seg/ *j\RowSegments : c0=(seg % *j\RowSegments)* *j\SegWidth
        cols=*j\Width-c0 : If cols>*j\SegWidth : cols=*j\SegWidth : EndIf
        i=row* *j\Width : ai=DBroadcastIndex(i,Y,A) : bi=DBroadcastIndex(i,Y,B)
        PmFastBinary(Dt(A)\Data+(ai+c0* *j\AStep)*4,Dt(B)\Data+(bi+c0* *j\BStep)*4,Dt(Y)\Data+(i+c0)*4,cols,*j\AStep,*j\BStep,*j\Op)
      Next
    Case 1 ; DBinary, Pow by one FLOAT scalar
      bv=*j\F1
      For i=first To last-1 : PokeF(Dt(Y)\Data+i*4,Pow(PeekF(Dt(A)\Data+i*4),bv)) : Next
    Case 2 ; DBinary, the general FLOAT broadcast loop
      For i=first To last-1
        ai=DBroadcastIndex(i,Y,A) : bi=DBroadcastIndex(i,Y,B)
        av=DGet(A,ai) : bv=DGet(B,bi)
        Select *j\Op
          Case 0 : v=av+bv
          Case 1 : v=av-bv
          Case 2 : v=av*bv
          Case 3 : v=av/bv
          Case 4 : v=Pow(av,bv)
          Case 5 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av=bv)
          Case 6 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av>bv)
          Case 7 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av<bv)
          Case 8 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av>=bv)
          Case 9 : v=Bool(av<>0 And bv<>0)
        EndSelect
        DPut(Y,i,v)
      Next
    Case 3 ; DTranspose
      For i=first To last-1
        n=i : src=0
        For d=rank-1 To 0 Step -1
          coord=n % *j\Dims[d] : n/ *j\Dims[d] : src+coord* *j\Strides[d]
        Next
        CopyMemory(Dt(A)\Data+src*size,Dt(Y)\Data+i*size,size)
      Next
    Case 4 ; DSlice
      For i=first To last-1
        n=i : src=0
        For d=rank-1 To 0 Step -1
          c=n % *j\Dims[d] : n/ *j\Dims[d] : src+(*j\Start[d]+c* *j\Steps[d])* *j\Strides[d]
        Next
        CopyMemory(Dt(A)\Data+src*size,Dt(Y)\Data+i*size,size)
      Next
    Case 5 ; DExpand
      For i=first To last-1 : CopyMemory(Dt(A)\Data+DBroadcastIndex(i,Y,A)*size,Dt(Y)\Data+i*size,size) : Next
    Case 6 ; DWhere: C is the condition
      For i=first To last-1
        If DGet(*j\C,DBroadcastIndex(i,Y,*j\C)) : src=A : Else : src=B : EndIf
        index=DBroadcastIndex(i,Y,src)
        CopyMemory(Dt(src)\Data+index*size,Dt(Y)\Data+i*size,size)
      Next
    Case 7 ; DCast
      For i=first To last-1 : DPut(Y,i,DGet(A,i)) : Next
    Case 8 ; DReduceAxes: one output per index, its elements in row-major order
      cf=*j\Inner
      For i=first To last-1
        base=0 : n=i
        For d=rank-1 To 0 Step -1
          If ((*j\Mask>>d)&1)=0 : coord=n%Dt(A)\D[d] : n/Dt(A)\D[d] : base+coord* *j\Strides[d] : EndIf
        Next
        acc=0.0
        For c=0 To *j\Inner-1
          at=base : n=c
          For d=rank-1 To 0 Step -1
            If (*j\Mask>>d)&1 : coord=n%Dt(A)\D[d] : n/Dt(A)\D[d] : at+coord* *j\Strides[d] : EndIf
          Next
          x=PeekF(Dt(A)\Data+at*4) : acc=acc+x
        Next
        If *j\Mean : acc=acc/cf : EndIf
        PokeF(Dt(Y)\Data+i*4,acc)
      Next
  EndSelect
EndProcedure
Procedure DBinary(Y.i,A.i,B.i,Op.i)
  Protected j.DIndexJob
  Protected i.i, ai.i,bi.i,kind.i=Dt(A)\Kind
  Protected av.f,bv.f,v.f,ia.i,ib.i,iv.i
  Protected width.i,row.i,astep.i,bstep.i
  If Op=4 And Dt(A)\Kind=1 And Dt(B)\Kind=1 And Dt(B)\Count=1
    ; the exponent's bits are exactly 2.0 (a NaN exponent is not; forum 998)
    If PeekL(Dt(B)\Data)=$40000000 : B=A : Op=2 : EndIf
  EndIf
  If Op>=5 : kind=9 : EndIf
  ; Every path below writes every element of Y (or fails the request).
  DAllocOverwrite=1
  If DBroadcast(Y,A,B,kind)=0 : DAllocOverwrite=0 : ProcedureReturn : EndIf
  DAllocOverwrite=0
  If kind=1 And Dt(B)\Kind=1 And Op<=3
    width=1 : If Dt(Y)\Rank : width=Dt(Y)\D[Dt(Y)\Rank-1] : EndIf
    If width=0 : ProcedureReturn : EndIf
    If Dt(A)\Rank : astep=Bool(Dt(A)\D[Dt(A)\Rank-1]<>1) : EndIf
    If Dt(B)\Rank : bstep=Bool(Dt(B)\D[Dt(B)\Rank-1]<>1) : EndIf
    ; Segments: whole rows, or, for rows wider than two grains, pieces of a
    ; row starting at multiples of 4 so every element keeps its place in the
    ; four-wide loop or its scalar tail.
    j\Kind=0 : j\Y=Y : j\A=A : j\B=B : j\Op=Op : j\Width=width : j\AStep=astep : j\BStep=bstep
    If width>=2*#PMELEM_CHEAP And PmPoolThreads>1
      j\SegWidth=(#PMELEM_CHEAP+3)&~3 : If PmPoolForceSplit : j\SegWidth=4 : EndIf
      j\RowSegments=(width+j\SegWidth-1)/j\SegWidth
    Else
      j\SegWidth=width : j\RowSegments=1
    EndIf
    j\Count=(Dt(Y)\Count/width)*j\RowSegments
    DIndexRun(@j,PmPoolMax(1,#PMELEM_CHEAP/j\SegWidth))
    ProcedureReturn
  EndIf
  If Op=4 And Dt(A)\Kind=1 And Dt(B)\Kind=1 And Dt(B)\Count=1 And Dt(A)\Count=Dt(Y)\Count
    j\Kind=1 : j\Y=Y : j\A=A : j\F1=PeekF(Dt(B)\Data) : j\Count=Dt(Y)\Count
    DIndexRun(@j,#PMELEM_DEAR)
    ProcedureReturn
  EndIf
  If Dt(A)\Kind=1 And Op>=0 And Op<=9
    j\Kind=2 : j\Y=Y : j\A=A : j\B=B : j\Op=Op : j\Count=Dt(Y)\Count
    DIndexRun(@j,#PMELEM_DEAR)
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
        Case 5 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av=bv)
        Case 6 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av>bv)
        Case 7 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av<bv)
        Case 8 : v=Bool(PmTensorIsNan(av)=0 And PmTensorIsNan(bv)=0 And av>=bv)
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

; The two kernels with closed op sets report by clearing a flag - they sit under
; this file and cannot name DError - so this dispatcher is what turns a refusal
; into one. Both flags are armed per DUnary call so a refusal on one node can
; never be read as a refusal on the next. The Select itself had no Default, so
; a code above 12 returned with the destination untouched and nothing said.
Procedure DUnary(Y.i,A.i,Op.i,Alpha.f=0.01)
  If Dt(A)\Kind<>1 : DFail("This unary kernel requires FLOAT input.") : ProcedureReturn : EndIf
  ; Every kernel below writes every element of Y (or the op is refused).
  DAllocOverwrite=1
  If DLike(Y,A)=0 : DAllocOverwrite=0 : ProcedureReturn : EndIf
  DAllocOverwrite=0
  PmTensorUnaryMathOk=1 : PmTensorTrigOk=1
  Select Op
    Case 0 To 4 : PmTensorUnaryMath(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Op)
    Case 5 To 7 : PmFastTrig(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Op-5)
    Case 8 : PmTensorSigmoid(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 9 : PmTensorTanh(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 10 : PmTensorFloor(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 11 : PmTensorRoundEven(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count)
    Case 12 : PmTensorLeakyRelu(Dt(A)\Data,Dt(Y)\Data,Dt(Y)\Count,Alpha)
    Default : DFail("Unsupported unary operation code: this dispatcher routes 0 Exp, 1 Log, 2 Sqrt, 3 Abs, 4 Neg, 5 Sin, 6 Cos, 7 Atan, 8 Sigmoid, 9 Tanh, 10 Floor, 11 Round and 12 LeakyRelu only. Check the operator the model emitted for this node.") : ProcedureReturn
  EndSelect
  If PmTensorUnaryMathOk=0 : DFail("Unsupported unary math operation code: this kernel computes 0 Exp, 1 Log, 2 Sqrt, 3 Abs and 4 Neg only. Check the operator the model emitted for this node.") : EndIf
  If PmTensorTrigOk=0 : DFail("Unsupported trigonometric operation code: this kernel computes 0 Sin, 1 Cos and 2 Atan only. Check the operator the model emitted for this node.") : EndIf
EndProcedure

Procedure DCast(Y.i,A.i,Kind.i)
  Protected job.DIndexJob
  If DLike(Y,A,Kind)=0 : ProcedureReturn : EndIf
  job\Kind=7 : job\Y=Y : job\A=A : job\Count=Dt(A)\Count
  DIndexRun(@job,#PMELEM_CHEAP/8)
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
  Protected job.DIndexJob
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
  job\Kind=3 : job\Y=Y : job\A=A : job\Size=size : job\Rank=rank : job\Count=Dt(Y)\Count
  For i=0 To rank-1 : job\Dims[i]=dims(i) : job\Strides[i]=strides(i) : Next
  DIndexRun(@job,#PMELEM_CHEAP/4)
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
  Protected job.DIndexJob
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
  job\Kind=4 : job\Y=Y : job\A=A : job\Size=size : job\Rank=Dt(Y)\Rank : job\Count=Dt(Y)\Count
  For i=0 To 7 : job\Dims[i]=dims(i) : job\Start[i]=start(i) : job\Steps[i]=stepv(i) : job\Strides[i]=stride(i) : Next
  DIndexRun(@job,#PMELEM_CHEAP/4)
EndProcedure

Procedure DExpand(Y.i,A.i,Shape.i)
  Protected job.DIndexJob
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
  job\Kind=5 : job\Y=Y : job\A=A : job\Size=size : job\Count=Dt(Y)\Count
  DIndexRun(@job,#PMELEM_CHEAP/4)
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
  If (PeekQ(@start) & $7FFFFFFFFFFFFFFF) > $7FF0000000000000 Or (PeekQ(@stop) & $7FFFFFFFFFFFFFFF) > $7FF0000000000000 Or (PeekQ(@stepv) & $7FFFFFFFFFFFFFFF) > $7FF0000000000000
    DFail("Range start, limit and delta must be numbers; a NaN gives no element count.") : ProcedureReturn
  EndIf
  If stepv=0 : DFail("Range step is zero.") : ProcedureReturn : EndIf
  extent=(stop-start)/stepv
  If (PeekQ(@extent) & $7FF0000000000000)=$7FF0000000000000 Or extent>DLimit/DSize(Dt(A)\Kind) : DFail("Range exceeds finite memory bounds.") : ProcedureReturn : EndIf
  n=DMax(0,Round(extent,#PB_Round_Up))
  If DShape(Y,Dt(A)\Kind,1,n)
    For i=0 To n-1 : DPut(Y,i,start+i*stepv) : Next
  EndIf
EndProcedure

Procedure DWhere(Y.i,Cond.i,A.i,B.i)
  Protected job.DIndexJob
  Protected i.i,size.i=DSize(Dt(A)\Kind),rank.i=DMax(DMax(Dt(A)\Rank,Dt(B)\Rank),Dt(Cond)\Rank),k.i,t.i,d.i,e.i
  Protected Dim dims.i(7)
  ; The condition, X and Y broadcast together (Where-16).
  For i=0 To rank-1
    e=1
    For k=0 To 2
      t=B : If k=0 : t=Cond : ElseIf k=1 : t=A : EndIf
      d=1 : If i-rank+Dt(t)\Rank>=0 : d=Dt(t)\D[i-rank+Dt(t)\Rank] : EndIf
      If d<>1
        If e<>1 And e<>d : DFail("Incompatible broadcast dimensions.") : ProcedureReturn : EndIf
        e=d
      EndIf
    Next
    dims(i)=e
  Next
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  job\Kind=6 : job\Y=Y : job\A=A : job\B=B : job\C=Cond : job\Size=size : job\Count=Dt(Y)\Count
  DIndexRun(@job,#PMELEM_CHEAP/8)
EndProcedure

; An element other than zero; a NaN is one (forum 998).
Procedure.i DNonZeroAt(A.i,i.i)
  If Dt(A)\Kind=1 : ProcedureReturn Bool((PeekL(Dt(A)\Data+i*4) & $7FFFFFFF)<>0) : EndIf
  ProcedureReturn Bool(DGet(A,i)<>0)
EndProcedure

Procedure DNonZero(Y.i,A.i)
  Protected i.i,j.i,count.i,n.i,col.i,c.i
  For i=0 To Dt(A)\Count-1 : If DNonZeroAt(A,i) : count+1 : EndIf : Next
  If DShape(Y,7,2,Dt(A)\Rank,count)=0 : ProcedureReturn : EndIf
  For i=0 To Dt(A)\Count-1
    If DNonZeroAt(A,i)
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

Structure DMatMulBatches
  A.i : B.i : Y.i : M.i : K.i : N.i : Batches.i : Chunk.i : Offsets.i
EndStructure

Procedure DMatMulBatchTask(*j.DMatMulBatches,task.i,worker.i)
  Protected batch.i=task* *j\Chunk,last.i=batch+ *j\Chunk,g.PmFastGemmJob
  If last>*j\Batches : last=*j\Batches : EndIf
  While batch<last
    g\A=*j\A+PeekI(*j\Offsets+batch*16)*4 : g\B=*j\B+PeekI(*j\Offsets+batch*16+8)*4
    g\Dst=*j\Y+batch* *j\M* *j\N*4 : g\Bias=0 : g\K=*j\K : g\N=*j\N : g\M=*j\M
    g\First=0 : g\Last=*j\M : g\ColFirst=0 : g\ColLast=*j\N
    PmFastGemmWorker(@g)
    batch+1
  Wend
EndProcedure

Procedure DMatMul(Y.i,A.i,B.i)
  Protected bj.DMatMulBatches,macs.q,tasks.i
  Protected Dim dims.i(7)
  Protected rank.i=DMax(Dt(A)\Rank,Dt(B)\Rank),i.i,ai.i,bi.i,ad.i,bd.i
  Protected m.i,k.i,n.i,batches.i,batch.i,aoff.i,boff.i,idx.i,c.i,astride.i,bstride.i
  If Dt(A)\Kind<>1 Or (Dt(B)\Kind<>1 And Dt(B)\Kind<>3 And Dt(B)\Kind<>33) : DFail("MatMul requires FLOAT activations and FLOAT or INT8 weights.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<2 Or Dt(B)\Rank<2 Or rank>8 : DFail("MatMul requires rank two or above.") : ProcedureReturn : EndIf
  m=Dt(A)\D[Dt(A)\Rank-2] : k=Dt(A)\D[Dt(A)\Rank-1] : n=Dt(B)\D[Dt(B)\Rank-1]
  If k<>Dt(B)\D[Dt(B)\Rank-2] : DFail("MatMul contraction mismatch.") : ProcedureReturn : EndIf
  If Dt(B)\Kind=3 And (Dt(B)\Scales=0 Or k>133144) : DFail("INT8 MatMul scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
  If Dt(B)\Kind=33 And Dt(B)\Scales=0 : DFail("Wide INT8 MatMul scales are missing.") : ProcedureReturn : EndIf
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
  ; Many small FLOAT products (attention heads): the batches themselves are
  ; split across the pool, each product whole in one task. A product big
  ; enough to split on its own is split inside PmFastGemm instead.
  macs=m : macs*PmPoolMax(k,1) : macs*n
  If Dt(B)\Kind=1 And batches>1 And macs<2*#PMFAST_GEMM_GRAIN And PmPoolThreads>1 And m>0 And n>0
    bj\Offsets=AllocateMemory(batches*16+16)
  EndIf
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
    If Dt(B)\Kind=3 Or Dt(B)\Kind=33
      DInt8Begin()
      If PmI8MatMul(Dt(A)\Data+aoff*4,Dt(B)\Data+boff,Dt(Y)\Data+batch*m*n*4,m,k,n,Dt(B)\Scales,Bool(Dt(B)\Kind=33),Dt(B)\Count)=0 And PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
      If DInt8End()=0 : ProcedureReturn : EndIf
    ElseIf bj\Offsets
      PokeI(bj\Offsets+batch*16,aoff) : PokeI(bj\Offsets+batch*16+8,boff)
    Else
      PmFastGemm(Dt(A)\Data+aoff*4,Dt(B)\Data+boff*4,Dt(Y)\Data+batch*m*n*4,m,k,n)
    EndIf
  Next
  If bj\Offsets
    bj\A=Dt(A)\Data : bj\B=Dt(B)\Data : bj\Y=Dt(Y)\Data : bj\M=m : bj\K=k : bj\N=n : bj\Batches=batches
    tasks=PmPoolTasks(batches,PmPoolMax(1,#PMFAST_GEMM_GRAIN/PmPoolMax(macs,1)),1,@bj\Chunk)
    If tasks<=1
      bj\Chunk=batches : DMatMulBatchTask(@bj,0,0)
    Else
      PmPoolRun(@DMatMulBatchTask(),@bj,tasks)
    EndIf
    FreeMemory(bj\Offsets)
  EndIf
EndProcedure

Procedure DGemm(Y.i,A.i,B.i,C.i,TransA.i,TransB.i,Alpha.f,Beta.f)
  Protected g.PmTensorGemmArgs
  If Dt(A)\Kind<>1 Or (Dt(B)\Kind<>1 And Dt(B)\Kind<>3 And Dt(B)\Kind<>33) Or (C And Dt(C)\Kind<>1) Or TransA<0 Or TransA>1 Or TransB<0 Or TransB>1 : DFail("Gemm requires FLOAT activations, FLOAT/INT8 weights and valid transpose flags.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<>2 Or Dt(B)\Rank<>2 : DFail("Gemm requires matrices.") : ProcedureReturn : EndIf
  g\M=Dt(A)\D[TransA] : g\K=Dt(A)\D[1-TransA] : g\N=Dt(B)\D[1-TransB]
  If g\K<>Dt(B)\D[TransB] : DFail("Gemm contraction mismatch.") : ProcedureReturn : EndIf
  If DShape(Y,1,2,g\M,g\N)=0 : ProcedureReturn : EndIf
  g\A=Dt(A)\Data : g\B=Dt(B)\Data : g\C=Dt(C)\Data : g\Dst=Dt(Y)\Data
  g\CCount=Dt(C)\Count : g\TransA=TransA : g\TransB=TransB : g\Alpha=Alpha : g\Beta=Beta
  If Dt(B)\Kind=3 Or Dt(B)\Kind=33
    If Dt(B)\Scales=0 Or (Dt(B)\Kind=3 And g\K>133144) : DFail("INT8 Gemm scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
    DInt8Begin()
    g\WeightScales=Dt(B)\Scales : g\Wide=Bool(Dt(B)\Kind=33)
  EndIf
  PmTensorGemm(@g)
  If Dt(B)\Kind=3 Or Dt(B)\Kind=33 : DInt8End() : EndIf
EndProcedure

; ReduceSum and ReduceMean over any set of axes (absent or empty axes: all
; of them): every output sums its elements in row-major order, the mean
; divides that sum once by the count. The one-final-axis form keeps the fast
; kernel below.
Procedure DReduceAxes(Y.i,A.i,Axes.i,Keep.i,Mean.i)
  Protected job.DIndexJob
  Protected Dim dims.i(7)
  Protected Dim stride.i(7)
  Protected i.i,j.i,d.i,rank.i=Dt(A)\Rank,axis.i,mask.i,count.i,outCount.i,inner.i,n.i,base.i,at.i,coord.i
  Protected acc.f,x.f,cf.f
  If Dt(A)\Kind<>1 : DFail("Reduction requires FLOAT.") : ProcedureReturn : EndIf
  If Axes : count=Dt(Axes)\Count : EndIf
  For i=0 To count-1
    axis=DInt(Axes,i) : If axis<0 : axis+rank : EndIf
    If axis<0 Or axis>=rank : DFail("A reduction axis is outside the data rank.") : ProcedureReturn : EndIf
    If (mask>>axis)&1 : DFail("Reduction axes name the same axis twice.") : ProcedureReturn : EndIf
    mask|(1<<axis)
  Next
  If count=0 : mask=(1<<rank)-1 : EndIf
  j=0 : outCount=1 : inner=1 : n=1
  For d=rank-1 To 0 Step -1 : stride(d)=n : n*Dt(A)\D[d] : Next
  For d=0 To rank-1
    If (mask>>d)&1
      inner*Dt(A)\D[d]
      If Keep : dims(j)=1 : j+1 : EndIf
    Else
      dims(j)=Dt(A)\D[d] : j+1 : outCount*Dt(A)\D[d]
    EndIf
  Next
  If DAlloc(Y,1,j,@dims(0))=0 : ProcedureReturn : EndIf
  ; Outputs split across the pool; each output's sum runs whole in one task.
  job\Kind=8 : job\Y=Y : job\A=A : job\Rank=rank : job\Mask=mask : job\Inner=inner : job\Mean=Mean : job\Count=outCount
  For d=0 To 7 : job\Strides[d]=stride(d) : Next
  DIndexRun(@job,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(inner,1)))
EndProcedure

Procedure DReduce(Y.i,A.i,Axes.i,Keep.i,Mean.i)
  Protected Dim dims.i(7)
  Protected i.i,axis.i,rank.i=Dt(A)\Rank,width.i
  If Axes=0 : DReduceAxes(Y,A,Axes,Keep,Mean) : ProcedureReturn : EndIf
  If Dt(Axes)\Count<>1 : DReduceAxes(Y,A,Axes,Keep,Mean) : ProcedureReturn : EndIf
  axis=DInt(Axes)
  If axis<0 : axis+rank : EndIf
  If axis<>rank-1 : DReduceAxes(Y,A,Axes,Keep,Mean) : ProcedureReturn : EndIf
  For i=0 To rank-1 : dims(i)=Dt(A)\D[i] : Next
  width=dims(rank-1)
  If Keep : dims(rank-1)=1 : Else : rank-1 : EndIf
  If width<=0 : DFail("Reduction width is empty.") : ProcedureReturn : EndIf
  If DAlloc(Y,Dt(A)\Kind,rank,@dims(0))=0 : ProcedureReturn : EndIf
  If Dt(A)\Kind<>1 : DFail("Reduction requires FLOAT.") : ProcedureReturn : EndIf
  PmFastReduce(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Count/width,width,Mean)
EndProcedure

; noop_with_empty_axes = 1: absent or empty axes copy the data unchanged.
Procedure DReduceNoopEmpty(Y.i,A.i,Axes.i,Keep.i,Mean.i)
  Protected empty.i
  If Axes=0 : empty=1 : ElseIf Dt(Axes)\Count=0 : empty=1 : EndIf
  If empty
    If DLike(Y,A)=0 : ProcedureReturn : EndIf
    CopyMemory(Dt(A)\Data,Dt(Y)\Data,Dt(A)\Bytes)
    ProcedureReturn
  EndIf
  DReduce(Y,A,Axes,Keep,Mean)
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
  If Dt(A)\Kind<>1 Or (Dt(W)\Kind<>1 And Dt(W)\Kind<>3 And Dt(W)\Kind<>33) Or (B And Dt(B)\Kind<>1) : DFail("Conv requires FLOAT activations and FLOAT/INT8 weights.") : ProcedureReturn : EndIf
  If Dt(A)\Rank<>3 Or Dt(W)\Rank<>3 Or Stride<=0 Or Groups<=0 : DFail("Invalid Conv1D shape or attributes.") : ProcedureReturn : EndIf
  g\Batches=Dt(A)\D[0] : g\InChannels=Dt(A)\D[1] : g\InWidth=Dt(A)\D[2]
  g\OutChannels=Dt(W)\D[0] : g\Kernel=Dt(W)\D[2]
  g\OutWidth=(g\InWidth+PadLeft+PadRight-Dilation*(g\Kernel-1)-1)/Stride+1
  If g\InChannels<>Dt(W)\D[1]*Groups Or g\OutChannels % Groups : DFail("Conv group/channel mismatch.") : ProcedureReturn : EndIf
  If B And Dt(B)\Count<>g\OutChannels : DFail("Conv bias width mismatch.") : ProcedureReturn : EndIf
  If DShape(Y,1,3,g\Batches,g\OutChannels,g\OutWidth)=0 : ProcedureReturn : EndIf
  g\Src=Dt(A)\Data : g\Weight=Dt(W)\Data : g\Bias=Dt(B)\Data : g\Dst=Dt(Y)\Data
  g\PadLeft=PadLeft : g\Stride=Stride : g\Dilation=Dilation : g\Groups=Groups
  If Dt(W)\Kind=3 Or Dt(W)\Kind=33
    If Dt(W)\Scales=0 Or (Dt(W)\Kind=3 And g\InChannels/Groups*g\Kernel>133144) : DFail("INT8 Conv scales are missing or its reduction exceeds the INT32 bound.") : ProcedureReturn : EndIf
    DInt8Begin()
    g\WeightScales=Dt(W)\Scales : g\Wide=Bool(Dt(W)\Kind=33) : PmTensorConv1D(@g) : DInt8End()
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
  If Dt(X)\Rank<>3 Or Dt(W)\Rank<>3 Or Dt(R)\Rank<>3 Or Dt(X)\Kind<>1 Or (Dt(W)\Kind<>1 And Dt(W)\Kind<>3 And Dt(W)\Kind<>33) Or (Dt(R)\Kind<>1 And Dt(R)\Kind<>3 And Dt(R)\Kind<>33) : DFail("LSTM requires FLOAT activations and rank-three FLOAT/INT8 weights.") : ProcedureReturn : EndIf
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
  If Dt(W)\Kind=3 Or Dt(R)\Kind=3 Or Dt(W)\Kind=33 Or Dt(R)\Kind=33
    If DFiniteTensor(X)=0 Or DFiniteTensor(IH)=0 Or DFiniteTensor(IC)=0 : ProcedureReturn : EndIf
    If (Dt(W)\Kind<>1 And Dt(W)\Scales=0) Or (Dt(R)\Kind<>1 And Dt(R)\Scales=0) Or g\InputSize>133144 Or g\Hidden>133144 : DFail("INT8 LSTM scales are missing or reduction is too large.") : ProcedureReturn : EndIf
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
    DInt8Begin()
    If Dt(W)\Kind<>1 : g\WScales=Dt(W)\Scales : g\WWide=Bool(Dt(W)\Kind=33) : EndIf
    If Dt(R)\Kind<>1 : g\RScales=Dt(R)\Scales : g\RWide=Bool(Dt(R)\Kind=33) : EndIf
    If g\WScales And g\RScales
      PmI8Lstm(@g)
    Else
      PmTensorLstm(@g)
    EndIf
    DInt8End()
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
; Shape start/end, Conv and ConvTranspose auto_pad SAME, ScatterND reductions.
XIncludeFile "tensor_dynamic_forms_windows.pbi"

; InstanceNormalization, TopK, ScatterElements, ReduceMax, ReduceProd, Not,
; Identity and Pad live beside their kernels.
XIncludeFile "tensor_dynamic_norm_small_windows.pbi"
