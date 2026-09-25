; Runtime-dimension entries for attribute forms whose effect needs a run-time
; extent: Shape with start/end, Conv and ConvTranspose with auto_pad SAME, and
; ScatterND with a reduction. Each computes what the attribute selects and
; hands the rest to the kernel the default form already uses, so the default
; forms emit and run exactly as before.
; Specification: https://onnx.ai/onnx/operators/ - Shape-15, Conv-11,
; ConvTranspose-11, ScatterND-18.

; Shape-15: dimensions start..end-1 of the input's shape. A negative value
; counts back from the rank; both are then clamped to 0..rank, and an end at or
; before the start gives an empty shape.
Procedure DShapeRange(Y.i,A.i,Start.i,Finish.i)
  Protected i.i,rank.i=Dt(A)\Rank
  If Start<0 : Start+rank : EndIf
  If Finish<0 : Finish+rank : EndIf
  If Start<0 : Start=0 : EndIf
  If Start>rank : Start=rank : EndIf
  If Finish<0 : Finish=0 : EndIf
  If Finish>rank : Finish=rank : EndIf
  If Finish<Start : Finish=Start : EndIf
  If DShape(Y,7,1,Finish-Start)=0 : ProcedureReturn : EndIf
  For i=Start To Finish-1 : PokeQ(Dt(Y)\Data+(i-Start)*8,Dt(A)\D[i]) : Next
EndProcedure

; Conv-11 auto_pad SAME_UPPER/SAME_LOWER: out = ceil(in / stride), total
; padding max(0, (out-1)*stride + (kernel-1)*dilation + 1 - in), the odd unit
; at the end for UPPER and at the start for LOWER.
Procedure DConvSame(Y.i,A.i,W.i,B.i,Lower.i,Stride.i,Dilation.i,Groups.i)
  Protected width.i,kernel.i,outWidth.i,total.i,begin.i
  If Dt(A)\Rank<>3 Or Dt(W)\Rank<>3 Or Stride<=0 Or Dilation<=0 : DFail("Invalid Conv1D shape or attributes.") : ProcedureReturn : EndIf
  width=Dt(A)\D[2] : kernel=Dt(W)\D[2]
  outWidth=(width+Stride-1)/Stride
  total=(outWidth-1)*Stride+(kernel-1)*Dilation+1-width
  If total<0 : total=0 : EndIf
  If Lower : begin=total-total/2 : Else : begin=total/2 : EndIf
  DConv(Y,A,W,B,begin,total-begin,Stride,Dilation,Groups)
EndProcedure

; ConvTranspose-11 auto_pad SAME_UPPER/SAME_LOWER: out = in * stride, total
; padding stride*(in-1) + output_padding + (kernel-1)*dilation + 1 - out;
; SAME_UPPER puts total/2 at the start and the rest at the end, SAME_LOWER the
; reverse.
Procedure DConvTransposeSame(Y.i,A.i,W.i,B.i,Lower.i,Stride.i,Dilation.i,Groups.i,OutputPad.i)
  Protected width.i,kernel.i,total.i,begin.i
  If Dt(A)\Rank<>3 Or Dt(W)\Rank<>3 Or Stride<=0 Or Dilation<=0 : DFail("Invalid transposed convolution.") : ProcedureReturn : EndIf
  width=Dt(A)\D[2] : kernel=Dt(W)\D[2]
  total=Stride*(width-1)+OutputPad+(kernel-1)*Dilation+1-width*Stride
  If total<0 : DFail("ConvTranspose auto_pad SAME needs a negative total padding for this input width, kernel and stride; the specification's output width in*stride cannot be reached.") : ProcedureReturn : EndIf
  If Lower : begin=total-total/2 : Else : begin=total/2 : EndIf
  DConvTranspose(Y,A,W,B,begin,total-begin,Stride,Dilation,Groups,OutputPad)
EndProcedure

; ScatterND-18 with reduction 1 add, 2 mul, 3 max, 4 min: each update is
; combined with the element already at its destination, in index order, so a
; repeated index accumulates. FLOAT, INT32 and INT64 data.
Procedure DScatterReduce(Y.i,A.i,Indices.i,Updates.i,Reduction.i)
  Protected k.i,i.i,j.i,e.i,offset.i,idx.i,block.i,tuples.i,dst.i,src.i
  Protected f.f,g.f,q.q,r.q
  If Dt(A)\Kind<>1 And Dt(A)\Kind<>6 And Dt(A)\Kind<>7 : DFail("ScatterND with a reduction requires FLOAT, INT32 or INT64 data.") : ProcedureReturn : EndIf
  If Reduction<1 Or Reduction>4 : DFail("ScatterND reduction code is outside 1 add, 2 mul, 3 max, 4 min.") : ProcedureReturn : EndIf
  If Dt(Updates)\Kind<>Dt(A)\Kind : DFail("ScatterND updates must have the data's element type.") : ProcedureReturn : EndIf
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
    For e=0 To block-1
      dst=offset*block+e : src=i*block+e
      If Dt(A)\Kind=1
        f=PeekF(Dt(Y)\Data+dst*4) : g=PeekF(Dt(Updates)\Data+src*4)
        Select Reduction
          Case 1 : f=f+g
          Case 2 : f=f*g
          ; numpy.maximum and numpy.minimum: a NaN on either side wins (forum 998)
          Case 3 : If PmTensorIsNan(g)<>0 Or (PmTensorIsNan(f)=0 And g>f) : f=g : EndIf
          Case 4 : If PmTensorIsNan(g)<>0 Or (PmTensorIsNan(f)=0 And g<f) : f=g : EndIf
        EndSelect
        PokeF(Dt(Y)\Data+dst*4,f)
      Else
        If Dt(A)\Kind=7
          q=PeekQ(Dt(Y)\Data+dst*8) : r=PeekQ(Dt(Updates)\Data+src*8)
        Else
          q=PeekL(Dt(Y)\Data+dst*4) : r=PeekL(Dt(Updates)\Data+src*4)
        EndIf
        Select Reduction
          Case 1 : q=q+r
          Case 2 : q=q*r
          Case 3 : If r>q : q=r : EndIf
          Case 4 : If r<q : q=r : EndIf
        EndSelect
        If Dt(A)\Kind=7 : PokeQ(Dt(Y)\Data+dst*8,q) : Else : PokeL(Dt(Y)\Data+dst*4,q) : EndIf
      EndIf
    Next
  Next
EndProcedure
