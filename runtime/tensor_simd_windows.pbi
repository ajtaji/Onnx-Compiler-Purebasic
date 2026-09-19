; Generated-code kernels in the host language with local x64 SIMD loops.
; No DLL, alternate inference engine, graph interpreter, or build process.
; Native-ASM backend ABI: only the documented volatile registers are used.
; See the host toolchain reference, the inline-assembly chapter (inlinedasm).
; Windows reports CPU+OS AVX availability (39), with SSE2 as x64 fallback:
; https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
Global PmFastAvx.i=IsProcessorFeaturePresent_(39)
Declare PmFastGemm(*A,*B,*Dst,M.i,K.i,N.i,*Bias=0)

Procedure PmFastReduce(*Src,*Dst,Outer.i,Width.i,Mean.i)
  Protected row.i,j.i,ps.i=*Src,length.i=Width,total.f,divisor.f=Width
  For row=0 To Outer-1
    CompilerIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
      !mov rax,[p.v_ps]
      !mov rdx,[p.v_length]
      !xorps xmm0,xmm0
      !test rdx,rdx
      !jz pmfastreduce_store
      !pmfastreduce_loop:
      !addss xmm0,[rax]
      !add rax,4
      !dec rdx
      !jnz pmfastreduce_loop
      !pmfastreduce_store:
      !movss [p.v_total],xmm0
    CompilerElse
      total=0 : For j=0 To Width-1 : total+PeekF(ps+j*4) : Next
    CompilerEndIf
    If Mean : total/divisor : EndIf
    PokeF(*Dst+row*4,total) : ps+Width*4
  Next
EndProcedure

; This Select used to have no Default, so a code it did not recognise returned
; with the destination untouched and nothing said - silent, and indistinguishable
; from a successful call.  PmTensorTrig in tensor_fp32_windows.pbi owns the op
; set, so every code this kernel does not compute itself goes there and is
; refused by clearing PmTensorTrigOk.  Armed here, per call, because this file
; is included before DError exists and so cannot raise the error itself; DUnary
; reads the flag back and turns it into a DFail.
Procedure PmFastTrig(*Src,*Dst,Count.i,Op.i)
  Protected i.i
  PmTensorTrigOk=1
  Select Op
    Case 0
      For i=0 To Count-1 : PokeF(*Dst+i*4,Sin(PeekF(*Src+i*4))) : Next
    Case 1
      For i=0 To Count-1 : PokeF(*Dst+i*4,Cos(PeekF(*Src+i*4))) : Next
    Case 2
      For i=0 To Count-1 : PokeF(*Dst+i*4,ATan(PeekF(*Src+i*4))) : Next
    Default
      PmTensorTrig(*Src,*Dst,Count,Op)
  EndSelect
EndProcedure

Procedure PmFastBinary(*A,*B,*Dst,Count.i,AStride.i,BStride.i,Op.i)
  Protected i.i,pa.i=*A,pb.i=*B,pd.i=*Dst,astep.i=AStride,bstep.i=BStride,operation.i=Op
  Protected av.f,bv.f,value.f
  CompilerIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
    While i+4<=Count
      !mov rax,[p.v_pa]
      !mov rcx,[p.v_pb]
      !mov rdx,[p.v_pd]
      !cmp qword [p.v_astep],0
      !jne pmfastbinary_a4
      !movss xmm0,[rax]
      !shufps xmm0,xmm0,0
      !jmp pmfastbinary_b
      !pmfastbinary_a4:
      !movups xmm0,[rax]
      !pmfastbinary_b:
      !cmp qword [p.v_bstep],0
      !jne pmfastbinary_b4
      !movss xmm1,[rcx]
      !shufps xmm1,xmm1,0
      !jmp pmfastbinary_op
      !pmfastbinary_b4:
      !movups xmm1,[rcx]
      !pmfastbinary_op:
      !cmp qword [p.v_operation],0
      !jne pmfastbinary_sub
      !addps xmm0,xmm1
      !jmp pmfastbinary_store
      !pmfastbinary_sub:
      !cmp qword [p.v_operation],1
      !jne pmfastbinary_mul
      !subps xmm0,xmm1
      !jmp pmfastbinary_store
      !pmfastbinary_mul:
      !cmp qword [p.v_operation],2
      !jne pmfastbinary_div
      !mulps xmm0,xmm1
      !jmp pmfastbinary_store
      !pmfastbinary_div:
      !divps xmm0,xmm1
      !pmfastbinary_store:
      !movups [rdx],xmm0
      pa+astep*16 : pb+bstep*16 : pd+16 : i+4
    Wend
  CompilerEndIf
  While i<Count
    av=PeekF(pa) : bv=PeekF(pb)
    Select Op
      Case 0 : value=av+bv
      Case 1 : value=av-bv
      Case 2 : value=av*bv
      Case 3 : value=av/bv
    EndSelect
    PokeF(pd,value) : pa+astep*4 : pb+bstep*4 : pd+4 : i+1
  Wend
EndProcedure

Procedure PmFastRow(*A,*B,*Dst,K.i,N.i,Bias.f,BWidth.i=0)
  Protected col.i,pa.i,pb.i,pd.i,stride.i=N*4,inner.i,sum.f
  Protected length.i=K,biasvalue.f=Bias
  If BWidth : stride=BWidth*4 : EndIf
  CompilerIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
    If PmFastAvx
      While col+16<=N
        pa=*A : pb=*B+col*4 : pd=*Dst+col*4
        !mov rax,[p.v_pa]
        !mov rcx,[p.v_pb]
        !mov r9,[p.v_pd]
        !mov r8,[p.v_stride]
        !mov rdx,[p.v_length]
        !vbroadcastss ymm0,[p.v_biasvalue]
        !vmovaps ymm1,ymm0
        !test rdx,rdx
        !jz pmfastrow_store16
        !pmfastrow_loop16:
        !vbroadcastss ymm2,[rax]
        !vmulps ymm3,ymm2,[rcx]
        !vaddps ymm0,ymm0,ymm3
        !vmulps ymm3,ymm2,[rcx+32]
        !vaddps ymm1,ymm1,ymm3
        !add rax,4
        !add rcx,r8
        !dec rdx
        !jnz pmfastrow_loop16
        !pmfastrow_store16:
        !vmovups [r9],ymm0
        !vmovups [r9+32],ymm1
        !vzeroupper
        col+16
      Wend
    EndIf
    While col+4<=N
      pa=*A : pb=*B+col*4 : pd=*Dst+col*4
      !mov rax,[p.v_pa]
      !mov rcx,[p.v_pb]
      !mov r9,[p.v_pd]
      !mov r8,[p.v_stride]
      !mov rdx,[p.v_length]
      !movss xmm0,[p.v_biasvalue]
      !shufps xmm0,xmm0,0
      !test rdx,rdx
      !jz pmfastrow_store4
      !pmfastrow_loop4:
      !movss xmm2,[rax]
      !shufps xmm2,xmm2,0
      !movups xmm3,[rcx]
      !mulps xmm2,xmm3
      !addps xmm0,xmm2
      !add rax,4
      !add rcx,r8
      !dec rdx
      !jnz pmfastrow_loop4
      !pmfastrow_store4:
      !movups [r9],xmm0
      col+4
    Wend
  CompilerEndIf
  While col<N
    sum=Bias : pa=*A : pb=*B+col*4
    For inner=0 To K-1
      sum+PeekF(pa)*PeekF(pb) : pa+4 : pb+stride
    Next
    PokeF(*Dst+col*4,sum) : col+1
  Wend
EndProcedure

Procedure.i PmFastConvTranspose(*g.PmTensorConvTranspose1DArgs,Available.i)
  Protected inGroup.i=*g\InChannels/ *g\Groups,outGroup.i=*g\OutChannels/ *g\Groups
  Protected rows.i=outGroup* *g\Kernel,weightBytes.i,outputBytes.i,weights.i,columns.i
  Protected bn.i,group.i,ic.i,oc.i,kx.i,ix.i,ox.i,row.i,src.i,dst.i,i.i,bias.f
  If rows<=0 Or inGroup<=0 Or *g\InWidth<=0 Or rows>Available/4/inGroup : ProcedureReturn 0 : EndIf
  weightBytes=rows*inGroup*4
  If rows>(Available-weightBytes)/4/ *g\InWidth : ProcedureReturn 0 : EndIf
  outputBytes=rows* *g\InWidth*4
  weights=AllocateMemory(weightBytes) : columns=AllocateMemory(outputBytes)
  If weights=0 Or columns=0
    If weights : FreeMemory(weights) : EndIf
    If columns : FreeMemory(columns) : EndIf
    ProcedureReturn 0
  EndIf
  For group=0 To *g\Groups-1
    For oc=0 To outGroup-1
      For kx=0 To *g\Kernel-1
        row=oc* *g\Kernel+kx
        For ic=0 To inGroup-1
          src=((group*inGroup+ic)*outGroup+oc)* *g\Kernel+kx
          PokeF(weights+(row*inGroup+ic)*4,PeekF(*g\Weight+src*4))
        Next
      Next
    Next
    For bn=0 To *g\Batches-1
      PmFastGemm(weights,*g\Src+(bn* *g\InChannels+group*inGroup)* *g\InWidth*4,columns,rows,inGroup,*g\InWidth)
      For oc=0 To outGroup-1
        dst=*g\Dst+(bn* *g\OutChannels+group*outGroup+oc)* *g\OutWidth*4
        bias=0 : If *g\Bias : bias=PeekF(*g\Bias+(group*outGroup+oc)*4) : EndIf
        For ox=0 To *g\OutWidth-1 : PokeF(dst+ox*4,bias) : Next
        For kx=0 To *g\Kernel-1
          src=columns+(oc* *g\Kernel+kx)* *g\InWidth*4
          For ix=0 To *g\InWidth-1
            ox=ix* *g\Stride- *g\PadLeft+kx* *g\Dilation
            If ox>=0 And ox< *g\OutWidth : PokeF(dst+ox*4,PeekF(dst+ox*4)+PeekF(src+ix*4)) : EndIf
          Next
        Next
      Next
    Next
  Next
  FreeMemory(weights) : FreeMemory(columns)
  ProcedureReturn 1
EndProcedure

Structure PmFastGemmJob
  A.i : B.i : Dst.i : Bias.i
  K.i : N.i : First.i : Last.i
EndStructure

; Four output rows share each input tile. Save every extra host-side nonvolatile
; register before using it; never reference stack locals while RSP is moved.
Procedure PmFastFourRows(*A,*B,*Dst,K.i,N.i,*Bias)
  Protected col.i,pa.i,pb.i,pd.i,length.i=K,stride.i=N*4
  Protected bias0.f,bias1.f,bias2.f,bias3.f,row.i
  If *Bias
    bias0=PeekF(*Bias) : bias1=PeekF(*Bias+4) : bias2=PeekF(*Bias+8) : bias3=PeekF(*Bias+12)
  EndIf
  CompilerIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
    While col+16<=N
      pa=*A : pb=*B+col*4 : pd=*Dst+col*4
      !mov rax,[p.v_pa]
      !mov rcx,[p.v_pb]
      !mov r9,[p.v_pd]
      !mov r8,[p.v_stride]
      !mov rdx,[p.v_length]
      !vbroadcastss ymm0,[p.v_bias0]
      !vbroadcastss ymm1,[p.v_bias1]
      !vbroadcastss ymm2,[p.v_bias2]
      !vbroadcastss ymm3,[p.v_bias3]
      !push r10
      !push r11
      !push r12
      !sub rsp,128
      !movdqu [rsp],xmm4
      !movdqu [rsp+16],xmm5
      !movdqu [rsp+32],xmm6
      !movdqu [rsp+48],xmm7
      !movdqu [rsp+64],xmm8
      !movdqu [rsp+80],xmm9
      !movdqu [rsp+96],xmm10
      !movdqu [rsp+112],xmm11
      !lea r10,[rax+rdx*4]
      !lea r11,[r10+rdx*4]
      !lea r12,[r11+rdx*4]
      !vmovaps ymm4,ymm0
      !vmovaps ymm5,ymm1
      !vmovaps ymm6,ymm2
      !vmovaps ymm7,ymm3
      !test rdx,rdx
      !jz pmfastfour_store
      !pmfastfour_loop:
      !vmovups ymm10,[rcx]
      !vmovups ymm11,[rcx+32]
      !vbroadcastss ymm8,[rax]
      !vmulps ymm9,ymm8,ymm10
      !vaddps ymm0,ymm0,ymm9
      !vmulps ymm9,ymm8,ymm11
      !vaddps ymm4,ymm4,ymm9
      !vbroadcastss ymm8,[r10]
      !vmulps ymm9,ymm8,ymm10
      !vaddps ymm1,ymm1,ymm9
      !vmulps ymm9,ymm8,ymm11
      !vaddps ymm5,ymm5,ymm9
      !vbroadcastss ymm8,[r11]
      !vmulps ymm9,ymm8,ymm10
      !vaddps ymm2,ymm2,ymm9
      !vmulps ymm9,ymm8,ymm11
      !vaddps ymm6,ymm6,ymm9
      !vbroadcastss ymm8,[r12]
      !vmulps ymm9,ymm8,ymm10
      !vaddps ymm3,ymm3,ymm9
      !vmulps ymm9,ymm8,ymm11
      !vaddps ymm7,ymm7,ymm9
      !add rax,4
      !add r10,4
      !add r11,4
      !add r12,4
      !add rcx,r8
      !dec rdx
      !jnz pmfastfour_loop
      !pmfastfour_store:
      !vmovups [r9],ymm0
      !vmovups [r9+32],ymm4
      !add r9,r8
      !vmovups [r9],ymm1
      !vmovups [r9+32],ymm5
      !add r9,r8
      !vmovups [r9],ymm2
      !vmovups [r9+32],ymm6
      !add r9,r8
      !vmovups [r9],ymm3
      !vmovups [r9+32],ymm7
      !vzeroupper
      !movdqu xmm4,[rsp]
      !movdqu xmm5,[rsp+16]
      !movdqu xmm6,[rsp+32]
      !movdqu xmm7,[rsp+48]
      !movdqu xmm8,[rsp+64]
      !movdqu xmm9,[rsp+80]
      !movdqu xmm10,[rsp+96]
      !movdqu xmm11,[rsp+112]
      !add rsp,128
      !pop r12
      !pop r11
      !pop r10
      col+16
    Wend
  CompilerEndIf
  ; Tail B rows retain the original row stride N.
  For row=0 To 3
    bias0=0 : If *Bias : bias0=PeekF(*Bias+row*4) : EndIf
    PmFastRow(*A+row*K*4,*B+col*4,*Dst+(row*N+col)*4,K,N-col,bias0,N)
  Next
EndProcedure

Procedure.i PmFastLstm(*g.PmTensorLstmArgs,Available.i)
  Protected width.i=4* *g\Hidden,wtBytes.i,rtBytes.i,xBytes.i,rBytes.i,total.i
  Protected scratch.i,wt.i,rt.i,xproj.i,rproj.i,dir.i,i.i,j.i,t.i,bn.i,unit.i,stepIndex.i,valid.i,state.i,inputRow.i,outRow.i,bbase.i
  Protected iv.f,ov.f,fv.f,cv.f,previous.f
  If width<=0 Or *g\InputSize<=0 Or *g\Batch<=0 Or *g\Sequence<=0 : ProcedureReturn 0 : EndIf
  total=*g\InputSize+ *g\Hidden+ *g\Sequence* *g\Batch+ *g\Batch
  If total<=0 Or width>Available/4/total : ProcedureReturn 0 : EndIf
  wtBytes=width* *g\InputSize*4 : rtBytes=width* *g\Hidden*4
  xBytes=width* *g\Sequence* *g\Batch*4 : rBytes=width* *g\Batch*4
  scratch=AllocateMemory(wtBytes+rtBytes+xBytes+rBytes)
  If scratch=0 : ProcedureReturn 0 : EndIf
  wt=scratch : rt=wt+wtBytes : xproj=rt+rtBytes : rproj=xproj+xBytes
  For i=0 To *g\Directions* *g\Batch* *g\Hidden-1
    iv=0 : cv=0
    If *g\InitialH : iv=PeekF(*g\InitialH+i*4) : EndIf
    If *g\InitialC : cv=PeekF(*g\InitialC+i*4) : EndIf
    PokeF(*g\YH+i*4,iv) : PokeF(*g\YC+i*4,cv)
  Next
  For dir=0 To *g\Directions-1
    For i=0 To width-1
      For j=0 To *g\InputSize-1
        PokeF(wt+(j*width+i)*4,PeekF(*g\W+((dir*width+i)* *g\InputSize+j)*4))
      Next
      For j=0 To *g\Hidden-1
        PokeF(rt+(j*width+i)*4,PeekF(*g\R+((dir*width+i)* *g\Hidden+j)*4))
      Next
    Next
    PmFastGemm(*g\X,wt,xproj,*g\Sequence* *g\Batch,*g\InputSize,width)
    For stepIndex=0 To *g\Sequence-1
      PmFastGemm(*g\YH+dir* *g\Batch* *g\Hidden*4,rt,rproj,*g\Batch,*g\Hidden,width)
      For bn=0 To *g\Batch-1
        valid=*g\Sequence
        If *g\SeqLens : valid=PeekL(*g\SeqLens+bn*4) : EndIf
        If valid<0 Or valid> *g\Sequence : FreeMemory(scratch) : ProcedureReturn 0 : EndIf
        If stepIndex>=valid : Continue : EndIf
        t=stepIndex : If dir=1 : t=valid-1-stepIndex : EndIf
        inputRow=(t* *g\Batch+bn)*width
        state=(dir* *g\Batch+bn)* *g\Hidden
        outRow=((t* *g\Directions+dir)* *g\Batch+bn)* *g\Hidden
        bbase=dir*8* *g\Hidden
        For unit=0 To *g\Hidden-1
          iv=0 : ov=0 : fv=0 : cv=0
          If *g\B
            iv=PeekF(*g\B+(bbase+unit)*4)+PeekF(*g\B+(bbase+4* *g\Hidden+unit)*4)
            ov=PeekF(*g\B+(bbase+ *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+5* *g\Hidden+unit)*4)
            fv=PeekF(*g\B+(bbase+2* *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+6* *g\Hidden+unit)*4)
            cv=PeekF(*g\B+(bbase+3* *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+7* *g\Hidden+unit)*4)
          EndIf
          iv+PeekF(xproj+(inputRow+unit)*4) : iv+PeekF(rproj+(bn*width+unit)*4)
          ov+PeekF(xproj+(inputRow+ *g\Hidden+unit)*4) : ov+PeekF(rproj+(bn*width+ *g\Hidden+unit)*4)
          fv+PeekF(xproj+(inputRow+2* *g\Hidden+unit)*4) : fv+PeekF(rproj+(bn*width+2* *g\Hidden+unit)*4)
          cv+PeekF(xproj+(inputRow+3* *g\Hidden+unit)*4) : cv+PeekF(rproj+(bn*width+3* *g\Hidden+unit)*4)
          previous=PeekF(*g\YC+(state+unit)*4)
          cv=PmTensorSigmoidValue(fv)*previous+PmTensorSigmoidValue(iv)*PmTensorTanhValue(cv)
          ov=PmTensorSigmoidValue(ov)*PmTensorTanhValue(cv)
          PokeF(*g\YC+(state+unit)*4,cv) : PokeF(*g\Y+(outRow+unit)*4,ov)
        Next
        CopyMemory(*g\Y+outRow*4,*g\YH+state*4,*g\Hidden*4)
      Next
    Next
  Next
  FreeMemory(scratch)
  ProcedureReturn 1
EndProcedure

Procedure PmFastGemmWorker(*g.PmFastGemmJob)
  Protected row.i=*g\First,bias.f,biasptr.i
  If PmFastAvx And *g\N>=16
    While row+4<=*g\Last
      biasptr=0 : If *g\Bias : biasptr=*g\Bias+row*4 : EndIf
      PmFastFourRows(*g\A+row* *g\K*4,*g\B,*g\Dst+row* *g\N*4,*g\K,*g\N,biasptr)
      row+4
    Wend
  EndIf
  While row<*g\Last
    bias=0 : If *g\Bias : bias=PeekF(*g\Bias+row*4) : EndIf
    PmFastRow(*g\A+row* *g\K*4,*g\B,*g\Dst+row* *g\N*4,*g\K,*g\N,bias)
    row+1
  Wend
EndProcedure

Procedure PmFastGemm(*A,*B,*Dst,M.i,K.i,N.i,*Bias=0)
  Protected workers.i=1,i.i,chunk.i,macs.q=M
  Protected Dim jobs.PmFastGemmJob(7),Dim threads.i(7)
  macs*K : macs*N
  If macs>=4000000 And M>=8 : workers=8 : EndIf
  If workers>CountCPUs(#PB_System_ProcessCPUs) : workers=CountCPUs(#PB_System_ProcessCPUs) : EndIf
  If workers<1 : workers=1 : EndIf
  chunk=(M+workers-1)/workers
  For i=0 To workers-1
    jobs(i)\A=*A : jobs(i)\B=*B : jobs(i)\Dst=*Dst : jobs(i)\Bias=*Bias
    jobs(i)\K=K : jobs(i)\N=N : jobs(i)\First=i*chunk : jobs(i)\Last=(i+1)*chunk
    If jobs(i)\Last>M : jobs(i)\Last=M : EndIf
    If i=workers-1
      PmFastGemmWorker(@jobs(i))
    Else
      threads(i)=CreateThread(@PmFastGemmWorker(),@jobs(i))
      If threads(i)=0 : PmFastGemmWorker(@jobs(i)) : EndIf
    EndIf
  Next
  For i=0 To workers-1 : If threads(i) : WaitThread(threads(i)) : EndIf : Next
EndProcedure

Structure PmFastConvTileJob
  Args.i : First.i : Last.i : Error.i
EndStructure

; Clip a strided source window once, not once per output element. Contiguous
; interiors use a byte copy (including signed zero and NaN payload bits).
Procedure PmFastCopyWindow(*src,*dst,inWidth.i,start.i,count.i,stride.i)
  Protected first.i,last.i,ix.i,ox.i
  If count<=0 : ProcedureReturn : EndIf
  If start<0 : first=(-start+stride-1)/stride : EndIf
  If first>count : first=count : EndIf
  If start>=inWidth
    last=0
  Else
    last=(inWidth-1-start)/stride+1
    If last>count : last=count : EndIf
  EndIf
  If last<first : last=first : EndIf
  If first : FillMemory(*dst,first*4,0) : EndIf
  If last<count : FillMemory(*dst+last*4,(count-last)*4,0) : EndIf
  If last>first
    ix=start+first*stride
    If stride=1
      CopyMemory(*src+ix*4,*dst+first*4,(last-first)*4)
    Else
      For ox=first To last-1
        PokeL(*dst+ox*4,PeekL(*src+ix*4)) : ix+stride
      Next
    EndIf
  EndIf
EndProcedure

Procedure PmFastConvTileWorker(*job.PmFastConvTileJob)
  Protected *g.PmTensorConv1DArgs=*job\Args
  Protected block.i=128,blocks.i=(*g\OutWidth+127)/128,k.i=*g\InChannels* *g\Kernel
  Protected columns.i=AllocateMemory(k*block*4),output.i=AllocateMemory(*g\OutChannels*block*4)
  Protected task.i,bn.i,ox0.i,width.i,ic.i,kx.i,row.i,src.i,dst.i,oc.i
  Protected gemm.PmFastGemmJob
  If columns=0 Or output=0
    If columns : FreeMemory(columns) : EndIf
    If output : FreeMemory(output) : EndIf
    *job\Error=1 : ProcedureReturn
  EndIf
  For task=*job\First To *job\Last-1
    bn=task/blocks : ox0=(task % blocks)*block : width=block
    If ox0+width> *g\OutWidth : width=*g\OutWidth-ox0 : EndIf
    row=0
    For ic=0 To *g\InChannels-1
      src=*g\Src+(bn* *g\InChannels+ic)* *g\InWidth*4
      For kx=0 To *g\Kernel-1
        dst=columns+row*width*4
        PmFastCopyWindow(src,dst,*g\InWidth,ox0* *g\Stride- *g\PadLeft+kx* *g\Dilation,width,*g\Stride)
        row+1
      Next
    Next
    gemm\A=*g\Weight : gemm\B=columns : gemm\Dst=output : gemm\Bias=*g\Bias
    gemm\K=k : gemm\N=width : gemm\First=0 : gemm\Last=*g\OutChannels
    PmFastGemmWorker(@gemm)
    For oc=0 To *g\OutChannels-1
      CopyMemory(output+oc*width*4,*g\Dst+((bn* *g\OutChannels+oc)* *g\OutWidth+ox0)*4,width*4)
    Next
  Next
  FreeMemory(columns) : FreeMemory(output)
EndProcedure

Procedure.i PmFastConvTiles(*g.PmTensorConv1DArgs,Available.i)
  Protected workers.i=8,tasks.i=(*g\OutWidth+127)/128* *g\Batches,bytes.i,i.i,chunk.i,failed.i
  Protected Dim jobs.PmFastConvTileJob(7),Dim threads.i(7)
  bytes=(*g\InChannels* *g\Kernel+ *g\OutChannels)*128*4
  If workers>tasks : workers=tasks : EndIf
  If workers>CountCPUs(#PB_System_ProcessCPUs) : workers=CountCPUs(#PB_System_ProcessCPUs) : EndIf
  While workers>1 And bytes>Available/workers : workers-1 : Wend
  If workers<1 Or bytes>Available : ProcedureReturn 0 : EndIf
  chunk=(tasks+workers-1)/workers
  For i=0 To workers-1
    jobs(i)\Args=*g : jobs(i)\First=i*chunk : jobs(i)\Last=(i+1)*chunk
    If jobs(i)\Last>tasks : jobs(i)\Last=tasks : EndIf
    If i=workers-1
      PmFastConvTileWorker(@jobs(i))
    Else
      threads(i)=CreateThread(@PmFastConvTileWorker(),@jobs(i))
      If threads(i)=0 : PmFastConvTileWorker(@jobs(i)) : EndIf
    EndIf
  Next
  For i=0 To workers-1
    If threads(i) : WaitThread(threads(i)) : EndIf
    If jobs(i)\Error : failed=1 : EndIf
  Next
  ProcedureReturn 1-failed
EndProcedure

Procedure.i PmFastConv(*g.PmTensorConv1DArgs,Available.i)
  Protected inGroup.i=*g\InChannels / *g\Groups,outGroup.i=*g\OutChannels / *g\Groups
  Protected k.i=inGroup* *g\Kernel,bytes.i,columns.i,bn.i,group.i,ic.i,kx.i,row.i,src.i,dst.i,bias.i
  If *g\OutWidth>256 And *g\Groups=1 : ProcedureReturn PmFastConvTiles(*g,Available) : EndIf
  If k<=0 Or *g\OutWidth<=0 Or k>Available/4/ *g\OutWidth : ProcedureReturn 0 : EndIf
  bytes=k* *g\OutWidth*4 : columns=AllocateMemory(bytes)
  If columns=0 : ProcedureReturn 0 : EndIf
  For bn=0 To *g\Batches-1
    For group=0 To *g\Groups-1
      row=0
      For ic=0 To inGroup-1
        src=*g\Src+((bn* *g\InChannels+group*inGroup+ic)* *g\InWidth)*4
        For kx=0 To *g\Kernel-1
          dst=columns+row* *g\OutWidth*4
          PmFastCopyWindow(src,dst,*g\InWidth,- *g\PadLeft+kx* *g\Dilation,*g\OutWidth,*g\Stride)
          row+1
        Next
      Next
      bias=0 : If *g\Bias : bias=*g\Bias+group*outGroup*4 : EndIf
      PmFastGemm(*g\Weight+group*outGroup*k*4,columns,*g\Dst+(bn* *g\OutChannels+group*outGroup)* *g\OutWidth*4,outGroup,k,*g\OutWidth,bias)
    Next
  Next
  FreeMemory(columns)
  ProcedureReturn 1
EndProcedure
