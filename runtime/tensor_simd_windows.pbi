; Generated-code kernels in the host language with local x64 SIMD loops.
; No DLL, alternate inference engine, graph interpreter, or build process.
; Native-ASM backend ABI: only the documented volatile registers are used.
; See the host toolchain reference, the inline-assembly chapter (inlinedasm).
; Windows reports CPU+OS AVX availability (39), with SSE2 as x64 fallback:
; https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
Global PmFastAvx.i=IsProcessorFeaturePresent_(39)
Declare PmFastGemm(*A,*B,*Dst,M.i,K.i,N.i,*Bias=0,Grain.i=0)
Declare PmFastCopyWindow(*src,*dst,inWidth.i,start.i,count.i,stride.i)

Procedure PmFastReduceSerial(*Src,*Dst,Outer.i,Width.i,Mean.i)
  Protected row.i,j.i,ps.i=*Src,length.i=Width,total.f,divisor.f=Width
  For row=0 To Outer-1
    CompilerIf #PB_Compiler_Backend=#PB_Backend_C
      ; one binary32 sum from +0 in index order, as the FASM's addss chain
      !{ typedef float pm_v4sf __attribute__((vector_size(16))); const float *pm_p = (const float *)v_ps; pm_v4sf pm_t = {0.0f}; for (long long pm_j = 0; pm_j < v_length; pm_j++) pm_t = __builtin_ia32_addss(pm_t, (pm_v4sf){pm_p[pm_j]}); v_total = pm_t[0]; }
    CompilerElseIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
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
;
; Sin, Cos and Atan run on the pool through PmFastTrigRange (in
; tensor_fp32_windows.pbi, the same loops), split by element ranges.
Procedure PmFastTrig(*Src,*Dst,Count.i,Op.i)
  PmTensorTrigOk=1
  Select Op
    Case 0 To 2
      PmElemRun(18,*Src,0,*Dst,Count,Op,0,0.0,0.0,#PMELEM_DEAR)
    Default
      PmTensorTrig(*Src,*Dst,Count,Op)
  EndSelect
EndProcedure

; Rows split across the pool; a row's sum is never divided between tasks.
Structure PmFastReduceJob
  Src.i : Dst.i : Rows.i : Width.i : Mean.i : Chunk.i
EndStructure

Procedure PmFastReduceTask(*j.PmFastReduceJob,task.i,worker.i)
  Protected first.i=task* *j\Chunk,n.i=*j\Rows-first
  If n>*j\Chunk : n=*j\Chunk : EndIf
  If n>0 : PmFastReduceSerial(*j\Src+first* *j\Width*4,*j\Dst+first*4,n,*j\Width,*j\Mean) : EndIf
EndProcedure

Procedure PmFastReduce(*Src,*Dst,Outer.i,Width.i,Mean.i)
  Protected j.PmFastReduceJob,tasks.i
  j\Src=*Src : j\Dst=*Dst : j\Rows=Outer : j\Width=Width : j\Mean=Mean
  tasks=PmPoolTasks(Outer,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(Width,1)),1,@j\Chunk)
  If tasks<=1
    PmFastReduceSerial(*Src,*Dst,Outer,Width,Mean)
  Else
    PmPoolRun(@PmFastReduceTask(),@j,tasks)
  EndIf
EndProcedure

Procedure PmFastBinary(*A,*B,*Dst,Count.i,AStride.i,BStride.i,Op.i)
  Protected i.i,pa.i=*A,pb.i=*B,pd.i=*Dst,astep.i=AStride,bstep.i=BStride,operation.i=Op
  Protected av.f,bv.f,value.f
  CompilerIf #PB_Compiler_Backend=#PB_Backend_C
    ; four lanes a turn, a scalar operand broadcast: the FASM's loop. A NaN
    ; lane of a sum or product is computed again in the FASM's operand order
    ; (THE C BACK END, tensor_fp32_windows.pbi)
    !{
    !  typedef float pm_v4sf __attribute__((vector_size(16)));
    !  typedef float pm_v4sf_u __attribute__((vector_size(16), aligned(1)));
    !  __attribute__((target("tune=generic"))) long long pm_kernel(const char *pa, const char *pb, char *pd, long long n, long long as, long long bs, long long op) {
    !    long long i = 0;
    !    for (; i + 4 <= n; i += 4) {
    !      pm_v4sf a, b, r;
    !      if (as != 0) a = *(const pm_v4sf_u *)pa; else { float x = *(const float *)pa; a = (pm_v4sf){x, x, x, x}; }
    !      if (bs != 0) b = *(const pm_v4sf_u *)pb; else { float x = *(const float *)pb; b = (pm_v4sf){x, x, x, x}; }
    !      if (op == 0) r = a + b;
    !      else if (op == 1) r = a - b;
    !      else if (op == 2) r = a * b;
    !      else r = a / b;
    !      if ((op == 0 || op == 2) && __builtin_ia32_movmskps(__builtin_ia32_cmpunordps(r, r))) {
    !        for (int l = 0; l < 4; l++) {
    !          pm_v4sf x = {a[l]}, y = {b[l]};
    !          r[l] = (op == 0 ? __builtin_ia32_addss(x, y) : __builtin_ia32_mulss(x, y))[0];
    !        }
    !      }
    !      *(pm_v4sf_u *)pd = r;
    !      pa += as * 16; pb += bs * 16; pd += 16;
    !    }
    !    return i;
    !  }
    !  long long pm_m = pm_kernel((const char *)v_pa, (const char *)v_pb, (char *)v_pd, v_count, v_astep, v_bstep, v_operation);
    !  v_pa += v_astep * 4 * pm_m; v_pb += v_bstep * 4 * pm_m; v_pd += 4 * pm_m; v_i = pm_m;
    !}
  CompilerElseIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
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
  CompilerIf #PB_Compiler_Backend=#PB_Backend_C
    ; sixteen columns a turn with AVX, then four with SSE: each lane its own
    ; column, from the bias, product then sum, both binary32. A column that
    ; comes out NaN is computed again one step at a time in the FASM's operand
    ; order (THE C BACK END, tensor_fp32_windows.pbi)
    If PmFastAvx
      !{
      !  typedef float pm_v8sf __attribute__((vector_size(32)));
      !  typedef float pm_v8sf_u __attribute__((vector_size(32), aligned(1)));
      !  __attribute__((target("avx"))) long long pm_kernel(const float *a, const char *b, char *d, long long stride, long long length, long long n, long long col, float bias) {
      !    for (; col + 16 <= n; col += 16) {
      !      const char *pb = b + col * 4;
      !      pm_v8sf acc0 = {bias, bias, bias, bias, bias, bias, bias, bias}, acc1 = acc0;
      !      for (long long k = 0; k < length; k++) {
      !        float x = a[k];
      !        pm_v8sf av = {x, x, x, x, x, x, x, x};
      !        pm_v8sf p0 = av * *(const pm_v8sf_u *)pb;
      !        acc0 = acc0 + p0;
      !        pm_v8sf p1 = av * *(const pm_v8sf_u *)(pb + 32);
      !        acc1 = acc1 + p1;
      !        pb += stride;
      !      }
      !      *(pm_v8sf_u *)(d + col * 4) = acc0;
      !      *(pm_v8sf_u *)(d + col * 4 + 32) = acc1;
      !    }
      !    return col;
      !  }
      !  v_col = pm_kernel((const float *)p_a, (const char *)p_b, (char *)p_dst, v_stride, v_length, v_n, v_col, v_biasvalue);
      !}
    EndIf
    !{
    !  typedef float pm_v4sf __attribute__((vector_size(16)));
    !  typedef float pm_v4sf_u __attribute__((vector_size(16), aligned(1)));
    !  __attribute__((target("tune=generic"))) long long pm_sse(const float *a, const char *b, char *d, long long stride, long long length, long long n, long long col, float bias) {
    !    for (; col + 4 <= n; col += 4) {
    !      const char *pb = b + col * 4;
    !      pm_v4sf acc = {bias, bias, bias, bias};
    !      for (long long k = 0; k < length; k++) {
    !        float x = a[k];
    !        pm_v4sf av = {x, x, x, x};
    !        av = av * *(const pm_v4sf_u *)pb;
    !        acc = acc + av;
    !        pb += stride;
    !      }
    !      *(pm_v4sf_u *)(d + col * 4) = acc;
    !    }
    !    return col;
    !  }
    !  const float *pm_a = (const float *)p_a;
    !  v_col = pm_sse(pm_a, (const char *)p_b, (char *)p_dst, v_stride, v_length, v_n, v_col, v_biasvalue);
    !  for (long long pm_c = 0; pm_c < v_col; pm_c++) {
    !    if (!__builtin_isnan(((const float *)p_dst)[pm_c])) continue;
    !    const char *pm_b = (const char *)p_b + pm_c * 4;
    !    pm_v4sf pm_s = {v_biasvalue};
    !    for (long long pm_k = 0; pm_k < v_length; pm_k++) {
    !      pm_s = __builtin_ia32_addss(pm_s, __builtin_ia32_mulss((pm_v4sf){pm_a[pm_k]}, (pm_v4sf){*(const float *)pm_b}));
    !      pm_b += v_stride;
    !    }
    !    ((float *)p_dst)[pm_c] = pm_s[0];
    !  }
    !}
  CompilerElseIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
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
  ; the last N mod 4 columns: one lane each, every product and sum rounded
  ; to binary32 as in the vector lanes (the host compiler's own expression
  ; would keep the product wider)
  CompilerIf #PB_Compiler_Backend=#PB_Backend_C
    !{
    !  typedef float pm_v4sf __attribute__((vector_size(16)));
    !  const float *pm_a = (const float *)p_a;
    !  for (; v_col < v_n; v_col++) {
    !    const char *pm_b = (const char *)p_b + v_col * 4;
    !    pm_v4sf pm_acc = {v_biasvalue};
    !    for (long long pm_k = 0; pm_k < v_length; pm_k++) {
    !      pm_v4sf pm_p = __builtin_ia32_mulss((pm_v4sf){pm_a[pm_k]}, (pm_v4sf){*(const float *)pm_b});
    !      pm_acc = __builtin_ia32_addss(pm_acc, pm_p);
    !      pm_b += v_stride;
    !    }
    !    *(float *)((char *)p_dst + v_col * 4) = pm_acc[0];
    !  }
    !}
  CompilerElseIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
    While col<N
      pa=*A : pb=*B+col*4 : pd=*Dst+col*4
      !mov rax,[p.v_pa]
      !mov rcx,[p.v_pb]
      !mov r8,[p.v_stride]
      !mov rdx,[p.v_length]
      !movss xmm0,[p.v_biasvalue]
      !test rdx,rdx
      !jz pmfastrow_store1
      !pmfastrow_loop1:
      !movss xmm2,[rax]
      !mulss xmm2,[rcx]
      !addss xmm0,xmm2
      !add rax,4
      !add rcx,r8
      !dec rdx
      !jnz pmfastrow_loop1
      !pmfastrow_store1:
      !mov r9,[p.v_pd]
      !movss [r9],xmm0
      col+1
    Wend
  CompilerElse
    While col<N
      sum=Bias : pa=*A : pb=*B+col*4
      For inner=0 To K-1
        sum+PeekF(pa)*PeekF(pb) : pa+4 : pb+stride
      Next
      PokeF(*Dst+col*4,sum) : col+1
    Wend
  CompilerEndIf
EndProcedure

; ConvTranspose on the runtime-dimension path, in the definition's order:
; each output starts at its bias and takes, input channel by input channel of
; its group, its taps in ascending input position, each product and each sum
; rounded to binary32 (as every target and the fixed-shape path). The outputs
; of one residue r = (ox + PadLeft) mod Stride share their taps (kx with
; kx * Dilation mod Stride = r), so for each residue and group they are one
; product: A[ocg][(icg, t)] = W[ic][ocg][kx_t], taps t in ascending input
; position; B[(icg, t)][j] = X[ic][j + c_t], the input row gathered (zeros
; past its ends); K = inGroup * taps, summed in that order by PmFastGemm from
; the bias. PmConvTFixEdges then corrects what the gathered zeros can change.
Structure PmFastConvTGather
  G.i : B.i : Cols.i : Bn.i : Group.i : InGroup.i : Taps.i : Offsets.i : J0.i : Chunk.i
EndStructure

Procedure PmFastConvTGatherTask(*c.PmFastConvTGather,task.i,worker.i)
  Protected *g.PmTensorConvTranspose1DArgs=*c\G,row.i=task* *c\Chunk,last.i=row+ *c\Chunk,icg.i,t.i,src.i
  If last>*c\InGroup* *c\Taps : last=*c\InGroup* *c\Taps : EndIf
  While row<last
    icg=row/ *c\Taps : t=row % *c\Taps
    src=*g\Src+((*c\Bn* *g\InChannels+*c\Group* *c\InGroup+icg)* *g\InWidth)*4
    PmFastCopyWindow(src,*c\B+row* *c\Cols*4,*g\InWidth,*c\J0+PeekI(*c\Offsets+t*8),*c\Cols,1)
    row+1
  Wend
EndProcedure

; The product's columns to their outputs, Stride apart, for output channels
; [first, last) of the chunk.
Structure PmFastConvTPlace
  C.i : Dst.i : Cols.i : Stride.i : RowStride.i : Rows.i : Chunk.i : Bias.i : Taps.i
EndStructure

Procedure PmFastConvTPlaceTask(*c.PmFastConvTPlace,task.i,worker.i)
  Protected ocg.i=task* *c\Chunk,last.i=ocg+ *c\Chunk,j.i,dst.i,src.i,bits.l,stride.i=*c\Stride*4
  If last>*c\Rows : last=*c\Rows : EndIf
  While ocg<last
    dst=*c\Dst+ocg* *c\RowStride
    If *c\Taps>0
      src=*c\C+ocg* *c\Cols*4
      For j=0 To *c\Cols-1 : PokeL(dst,PeekL(src)) : dst+stride : src+4 : Next
    Else
      bits=0 : If *c\Bias : bits=PeekL(*c\Bias+ocg*4) : EndIf
      For j=0 To *c\Cols-1 : PokeL(dst,bits) : dst+stride : Next
    EndIf
    ocg+1
  Wend
EndProcedure

; A[ocg][(icg, t)] = W[group * inGroup + icg][ocg][kx_t], rows [first, last)
Structure PmFastConvTWeights
  G.i : A.i : Group.i : InGroup.i : OutGroup.i : Taps.i : TapList.i : Chunk.i
EndStructure

Procedure PmFastConvTWeightsTask(*c.PmFastConvTWeights,task.i,worker.i)
  Protected *g.PmTensorConvTranspose1DArgs=*c\G,ocg.i=task* *c\Chunk,last.i=ocg+ *c\Chunk,icg.i,t.i,dst.i,src.i,kn.i=*g\Kernel,m.i=*c\Taps
  If last>*c\OutGroup : last=*c\OutGroup : EndIf
  While ocg<last
    dst=*c\A+ocg* *c\InGroup*m*4
    For icg=0 To *c\InGroup-1
      src=*g\Weight+((*c\Group* *c\InGroup+icg)* *c\OutGroup+ocg)*kn*4
      For t=0 To m-1
        PokeL(dst,PeekL(src+PeekI(*c\TapList+t*8)*4)) : dst+4
      Next
    Next
    ocg+1
  Wend
EndProcedure

#PMFAST_CONVT_COLUMNS = 1024

Procedure.i PmFastConvTranspose(*g.PmTensorConvTranspose1DArgs,Available.i)
  Protected inGroup.i=*g\InChannels/ *g\Groups,outGroup.i=*g\OutChannels/ *g\Groups
  Protected st.i=*g\Stride,dl.i=*g\Dilation,pl.i=*g\PadLeft,kn.i=*g\Kernel
  Protected rho.i,m.i,kx.i,t.i,ox0.i,n.i,group.i,bn.i,j0.i,cols.i,ocg.i,icg.i,tasks.i,bias.i
  Protected *taps,*offsets,*a,*b,*c,aBytes.i,bBytes.i,cBytes.i
  Protected ga.PmFastConvTGather,pc.PmFastConvTPlace,wa.PmFastConvTWeights
  If inGroup<=0 Or outGroup<=0 Or st<=0 Or kn<=0 Or dl<=0 Or *g\OutWidth<=0 Or *g\InWidth<=0 : ProcedureReturn 0 : EndIf
  aBytes=outGroup*inGroup*kn*4 : bBytes=inGroup*kn*#PMFAST_CONVT_COLUMNS*4 : cBytes=outGroup*#PMFAST_CONVT_COLUMNS*4
  If aBytes+bBytes+cBytes>Available : ProcedureReturn 0 : EndIf
  *taps=AllocateMemory(kn*8+8) : *offsets=AllocateMemory(kn*8+8)
  *a=AllocateMemory(aBytes+16) : *b=AllocateMemory(bBytes+16) : *c=AllocateMemory(cBytes+16)
  If *taps=0 Or *offsets=0 Or *a=0 Or *b=0 Or *c=0
    If *taps : FreeMemory(*taps) : EndIf
    If *offsets : FreeMemory(*offsets) : EndIf
    If *a : FreeMemory(*a) : EndIf
    If *b : FreeMemory(*b) : EndIf
    If *c : FreeMemory(*c) : EndIf
    ProcedureReturn 0
  EndIf
  For rho=0 To st-1
    ; this residue's taps in ascending input position: descending kx
    m=0
    For kx=kn-1 To 0 Step -1
      If (kx*dl) % st=rho : PokeI(*taps+m*8,kx) : m+1 : EndIf
    Next
    ox0=((rho-pl) % st+st) % st
    If ox0>=*g\OutWidth : Continue : EndIf
    n=(*g\OutWidth-1-ox0)/st+1
    For t=0 To m-1 : PokeI(*offsets+t*8,(ox0+pl-PeekI(*taps+t*8)*dl)/st) : Next
    For group=0 To *g\Groups-1
      bias=0 : If *g\Bias : bias=*g\Bias+group*outGroup*4 : EndIf
      If m>0
        wa\G=*g : wa\A=*a : wa\Group=group : wa\InGroup=inGroup : wa\OutGroup=outGroup : wa\Taps=m : wa\TapList=*taps
        tasks=PmPoolTasks(outGroup,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(inGroup*m,1)),1,@wa\Chunk)
        If tasks<=1
          wa\Chunk=outGroup : PmFastConvTWeightsTask(@wa,0,0)
        Else
          PmPoolRun(@PmFastConvTWeightsTask(),@wa,tasks)
        EndIf
      EndIf
      For bn=0 To *g\Batches-1
        j0=0
        While j0<n
          cols=n-j0 : If cols>#PMFAST_CONVT_COLUMNS : cols=#PMFAST_CONVT_COLUMNS : EndIf
          If m>0
            ga\G=*g : ga\B=*b : ga\Cols=cols : ga\Bn=bn : ga\Group=group : ga\InGroup=inGroup : ga\Taps=m : ga\Offsets=*offsets : ga\J0=j0
            tasks=PmPoolTasks(inGroup*m,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(cols,1)),1,@ga\Chunk)
            If tasks<=1
              ga\Chunk=inGroup*m : PmFastConvTGatherTask(@ga,0,0)
            Else
              PmPoolRun(@PmFastConvTGatherTask(),@ga,tasks)
            EndIf
            PmFastGemm(*a,*b,*c,outGroup,inGroup*m,cols,bias)
          EndIf
          pc\C=*c : pc\Cols=cols : pc\Stride=st : pc\Rows=outGroup : pc\Bias=bias : pc\Taps=m
          pc\RowStride=*g\OutWidth*4 : pc\Dst=*g\Dst+((bn* *g\OutChannels+group*outGroup)* *g\OutWidth+ox0+j0*st)*4
          tasks=PmPoolTasks(outGroup,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(cols,1)),1,@pc\Chunk)
          If tasks<=1
            pc\Chunk=outGroup : PmFastConvTPlaceTask(@pc,0,0)
          Else
            PmPoolRun(@PmFastConvTPlaceTask(),@pc,tasks)
          EndIf
          j0+cols
        Wend
      Next
    Next
  Next
  FreeMemory(*taps) : FreeMemory(*offsets) : FreeMemory(*a) : FreeMemory(*b) : FreeMemory(*c)
  PmConvTFixEdges(*g)
  ProcedureReturn 1
EndProcedure

Structure PmFastGemmJob
  A.i : B.i : Dst.i : Bias.i
  K.i : N.i : First.i : Last.i
  ColFirst.i : ColLast.i
  M.i : RowBlock.i : ColBlock.i : ColTasks.i
EndStructure

; Four output rows share each input tile. Save every extra host-side nonvolatile
; register before using it; never reference stack locals while RSP is moved.
; Columns [0, Width) of rows N apart (Width = N when omitted): *B and *Dst
; may start at any column, and every column is computed exactly as it is in
; the full-width call (lanes are independent; the last N mod 4 columns of a
; row are the scalar tail either way).
Procedure PmFastFourRows(*A,*B,*Dst,K.i,N.i,*Bias,Width.i=-1)
  Protected col.i,pa.i,pb.i,pd.i,length.i=K,stride.i=N*4
  Protected bias0.f,bias1.f,bias2.f,bias3.f,row.i
  If Width<0 : Width=N : EndIf
  If *Bias
    bias0=PeekF(*Bias) : bias1=PeekF(*Bias+4) : bias2=PeekF(*Bias+8) : bias3=PeekF(*Bias+12)
  EndIf
  CompilerIf #PB_Compiler_Backend=#PB_Backend_C
    ; the FASM's tile: four rows of A, sixteen columns of B, eight
    ; accumulators, product then sum, both binary32 (AVX; the caller checks).
    ; An output that comes out NaN is computed again as PmFastRow does.
    !{
    !  typedef float pm_v8sf __attribute__((vector_size(32)));
    !  typedef float pm_v8sf_u __attribute__((vector_size(32), aligned(1)));
    !  __attribute__((target("avx"))) long long pm_kernel(const float *a, const char *b, char *d, long long stride, long long length, long long width, long long col, const float *bias) {
    !    for (; col + 16 <= width; col += 16) {
    !      const char *pb = b + col * 4;
    !      const float *a1 = a + length, *a2 = a1 + length, *a3 = a2 + length;
    !      pm_v8sf c0 = {bias[0], bias[0], bias[0], bias[0], bias[0], bias[0], bias[0], bias[0]};
    !      pm_v8sf c1 = {bias[1], bias[1], bias[1], bias[1], bias[1], bias[1], bias[1], bias[1]};
    !      pm_v8sf c2 = {bias[2], bias[2], bias[2], bias[2], bias[2], bias[2], bias[2], bias[2]};
    !      pm_v8sf c3 = {bias[3], bias[3], bias[3], bias[3], bias[3], bias[3], bias[3], bias[3]};
    !      pm_v8sf c4 = c0, c5 = c1, c6 = c2, c7 = c3;
    !      for (long long k = 0; k < length; k++) {
    !        pm_v8sf b0 = *(const pm_v8sf_u *)pb, b1 = *(const pm_v8sf_u *)(pb + 32), x, p;
    !        x = (pm_v8sf){a[k], a[k], a[k], a[k], a[k], a[k], a[k], a[k]};
    !        p = x * b0; c0 = c0 + p; p = x * b1; c4 = c4 + p;
    !        x = (pm_v8sf){a1[k], a1[k], a1[k], a1[k], a1[k], a1[k], a1[k], a1[k]};
    !        p = x * b0; c1 = c1 + p; p = x * b1; c5 = c5 + p;
    !        x = (pm_v8sf){a2[k], a2[k], a2[k], a2[k], a2[k], a2[k], a2[k], a2[k]};
    !        p = x * b0; c2 = c2 + p; p = x * b1; c6 = c6 + p;
    !        x = (pm_v8sf){a3[k], a3[k], a3[k], a3[k], a3[k], a3[k], a3[k], a3[k]};
    !        p = x * b0; c3 = c3 + p; p = x * b1; c7 = c7 + p;
    !        pb += stride;
    !      }
    !      char *pd = d + col * 4;
    !      *(pm_v8sf_u *)pd = c0; *(pm_v8sf_u *)(pd + 32) = c4; pd += stride;
    !      *(pm_v8sf_u *)pd = c1; *(pm_v8sf_u *)(pd + 32) = c5; pd += stride;
    !      *(pm_v8sf_u *)pd = c2; *(pm_v8sf_u *)(pd + 32) = c6; pd += stride;
    !      *(pm_v8sf_u *)pd = c3; *(pm_v8sf_u *)(pd + 32) = c7;
    !    }
    !    return col;
    !  }
    !  typedef float pm_v4sf __attribute__((vector_size(16)));
    !  float pm_bias[4] = {v_bias0, v_bias1, v_bias2, v_bias3};
    !  v_col = pm_kernel((const float *)p_a, (const char *)p_b, (char *)p_dst, v_stride, v_length, v_width, v_col, pm_bias);
    !  for (int pm_r = 0; pm_r < 4; pm_r++) {
    !    const float *pm_a = (const float *)p_a + pm_r * v_length;
    !    float *pm_d = (float *)((char *)p_dst + pm_r * v_stride);
    !    for (long long pm_c = 0; pm_c < v_col; pm_c++) {
    !      if (!__builtin_isnan(pm_d[pm_c])) continue;
    !      const char *pm_b = (const char *)p_b + pm_c * 4;
    !      pm_v4sf pm_s = {pm_bias[pm_r]};
    !      for (long long pm_k = 0; pm_k < v_length; pm_k++) {
    !        pm_s = __builtin_ia32_addss(pm_s, __builtin_ia32_mulss((pm_v4sf){pm_a[pm_k]}, (pm_v4sf){*(const float *)pm_b}));
    !        pm_b += v_stride;
    !      }
    !      pm_d[pm_c] = pm_s[0];
    !    }
    !  }
    !}
  CompilerElseIf #PB_Compiler_Backend=#PB_Backend_Asm And #PB_Compiler_Processor=#PB_Processor_x64
    While col+16<=Width
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
    PmFastRow(*A+row*K*4,*B+col*4,*Dst+(row*N+col)*4,K,Width-col,bias0,N)
  Next
EndProcedure

; The gates of one recurrent step for units [u0, u1), exactly as the whole
; loop computed them. A unit reads its own four products and C(t-1) and
; writes its own C(t) and Y.
Structure PmFastLstmStep
  G.i : XProj.i : RProj.i : Bn.i : InputRow.i : State.i : OutRow.i : BBase.i : Chunk.i
  Rt.i : HRow.i : Fused.i
EndStructure

; W and R transposed into wt [InputSize][4H] and rt [Hidden][4H]: rows
; [first, last) of the two stacked, row j < InputSize of wt, else of rt.
Structure PmFastLstmTranspose
  G.i : Wt.i : Rt.i : Dir.i : Rows.i : Chunk.i
EndStructure

Procedure PmFastLstmTransposeTask(*t.PmFastLstmTranspose,task.i,worker.i)
  Protected *g.PmTensorLstmArgs=*t\G,width.i=4* *g\Hidden,i.i,j.i=task* *t\Chunk,last.i=j+ *t\Chunk,dir.i=*t\Dir
  If last>*t\Rows : last=*t\Rows : EndIf
  While j<last
    If j<*g\InputSize
      For i=0 To width-1 : PokeF(*t\Wt+(j*width+i)*4,PeekF(*g\W+((dir*width+i)* *g\InputSize+j)*4)) : Next
    Else
      For i=0 To width-1 : PokeF(*t\Rt+((j-*g\InputSize)*width+i)*4,PeekF(*g\R+((dir*width+i)* *g\Hidden+j-*g\InputSize)*4)) : Next
    EndIf
    j+1
  Wend
EndProcedure

Procedure PmFastLstmUnits(*s.PmFastLstmStep,u0.i,u1.i)
  Protected *g.PmTensorLstmArgs=*s\G,width.i=4* *g\Hidden,unit.i,bbase.i=*s\BBase,bn.i=*s\Bn,gate.i,col.i
  Protected xproj.i=*s\XProj,rproj.i=*s\RProj,inputRow.i=*s\InputRow,state.i=*s\State,outRow.i=*s\OutRow
  Protected iv.f,ov.f,fv.f,cv.f,previous.f
  Protected lb.l,la.l,ln.f,le.f,ld.f
  ; Fused step (Hidden a multiple of 4, u0 and u1 multiples of 4): this
  ; task's four column ranges of the recurrent product for batch row bn.
  ; Every column is a vector lane, as in the whole-row product (4H has no
  ; scalar tail), so each value is the one PmFastGemm would store.
  If *s\Fused
    For gate=0 To 3
      col=gate* *g\Hidden+u0
      PmFastRow(*s\HRow,*s\Rt+col*4,rproj+(bn*width+col)*4,*g\Hidden,u1-u0,0.0,width)
    Next
  EndIf
  For unit=u0 To u1-1
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
    PmLstmSigmoidInPlace(iv)
    PmLstmSigmoidInPlace(ov)
    PmLstmSigmoidInPlace(fv)
    PmLstmTanhInPlace(cv)
    PmLstmCellInPlace(cv, fv, previous, iv, cv)
    previous=cv
    PmLstmTanhInPlace(previous)
    ov=ov*previous
    PokeF(*g\YC+(state+unit)*4,cv) : PokeF(*g\Y+(outRow+unit)*4,ov)
  Next
EndProcedure

Procedure PmFastLstmUnitsTask(*s.PmFastLstmStep,task.i,worker.i)
  Protected u0.i=task* *s\Chunk,u1.i=u0+ *s\Chunk,*g.PmTensorLstmArgs=*s\G
  If u1>*g\Hidden : u1=*g\Hidden : EndIf
  If u0<u1 : PmFastLstmUnits(*s,u0,u1) : EndIf
EndProcedure

; A recurrent step's product is small and strictly serial between steps, so
; it is split at a finer grain than a free-standing product.
#PMFAST_STEP_GRAIN = 16384
#PMFAST_STEP_UNITS = 32
Procedure.i PmFastLstm(*g.PmTensorLstmArgs,Available.i)
  Protected st.PmFastLstmStep,stepTasks.i,tj.PmFastLstmTranspose,tasks.i,fused.i
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
  stepTasks=PmPoolTasks(*g\Hidden,#PMFAST_STEP_UNITS,4,@st\Chunk)
  ; One hand-off per step when the product can be cut at unit boundaries.
  fused=Bool(stepTasks>1 And *g\Hidden % 4=0)
  If *g\SeqLens : FillMemory(*g\Y,*g\Sequence* *g\Directions* *g\Batch* *g\Hidden*4,0) : EndIf
  For i=0 To *g\Directions* *g\Batch* *g\Hidden-1
    iv=0 : cv=0
    If *g\InitialH : iv=PeekF(*g\InitialH+i*4) : EndIf
    If *g\InitialC : cv=PeekF(*g\InitialC+i*4) : EndIf
    PokeF(*g\YH+i*4,iv) : PokeF(*g\YC+i*4,cv)
  Next
  For dir=0 To *g\Directions-1
    ; The two transposes, split across the pool by destination row.
    tj\G=*g : tj\Wt=wt : tj\Rt=rt : tj\Dir=dir : tj\Rows=*g\InputSize+ *g\Hidden
    tasks=PmPoolTasks(tj\Rows,PmPoolMax(1,#PMELEM_CHEAP/width),1,@tj\Chunk)
    If tasks<=1
      tj\Chunk=tj\Rows : PmFastLstmTransposeTask(@tj,0,0)
    Else
      PmPoolRun(@PmFastLstmTransposeTask(),@tj,tasks)
    EndIf
    PmFastGemm(*g\X,wt,xproj,*g\Sequence* *g\Batch,*g\InputSize,width)
    ; Each step: the recurrent product split by columns (one hand-off), then
    ; the gates split by units (another); H(t-1) is read by both and written
    ; only after both, by the copy below.
    For stepIndex=0 To *g\Sequence-1
      If fused=0 : PmFastGemm(*g\YH+dir* *g\Batch* *g\Hidden*4,rt,rproj,*g\Batch,*g\Hidden,width,0,#PMFAST_STEP_GRAIN) : EndIf
      For bn=0 To *g\Batch-1
        valid=*g\Sequence
        If *g\SeqLens : valid=PeekL(*g\SeqLens+bn*4) : EndIf
        If valid<0 Or valid> *g\Sequence : FreeMemory(scratch) : ProcedureReturn 0 : EndIf
        If stepIndex>=valid : Continue : EndIf
        t=stepIndex : If dir=1 : t=valid-1-stepIndex : EndIf
        st\G=*g : st\XProj=xproj : st\RProj=rproj : st\Bn=bn : st\Rt=rt : st\Fused=fused
        st\HRow=*g\YH+(dir* *g\Batch+bn)* *g\Hidden*4
        st\InputRow=(t* *g\Batch+bn)*width
        st\State=(dir* *g\Batch+bn)* *g\Hidden
        st\OutRow=((t* *g\Directions+dir)* *g\Batch+bn)* *g\Hidden
        st\BBase=dir*8* *g\Hidden
        If stepTasks<=1
          PmFastLstmUnits(@st,0,*g\Hidden)
        Else
          PmPoolRun(@PmFastLstmUnitsTask(),@st,stepTasks)
        EndIf
        CopyMemory(*g\Y+st\OutRow*4,*g\YH+st\State*4,*g\Hidden*4)
      Next
    Next
  Next
  FreeMemory(scratch)
  ProcedureReturn 1
EndProcedure

; Rows [First, Last) and columns [ColFirst, ColLast) of the product, rows N
; apart. ColLast = 0 means every column.
Procedure PmFastGemmWorker(*g.PmFastGemmJob)
  Protected row.i=*g\First,bias.f,biasptr.i,c0.i=*g\ColFirst,c1.i=*g\ColLast,width.i
  If c1<=0 : c0=0 : c1=*g\N : EndIf
  width=c1-c0
  If width<=0 : ProcedureReturn : EndIf
  If PmFastAvx And width>=16
    While row+4<=*g\Last
      biasptr=0 : If *g\Bias : biasptr=*g\Bias+row*4 : EndIf
      PmFastFourRows(*g\A+row* *g\K*4,*g\B+c0*4,*g\Dst+(row* *g\N+c0)*4,*g\K,*g\N,biasptr,width)
      row+4
    Wend
  EndIf
  While row<*g\Last
    bias=0 : If *g\Bias : bias=PeekF(*g\Bias+row*4) : EndIf
    PmFastRow(*g\A+row* *g\K*4,*g\B+c0*4,*g\Dst+(row* *g\N+c0)*4,*g\K,width,bias,*g\N)
    row+1
  Wend
EndProcedure

Procedure PmFastGemmTask(*g.PmFastGemmJob,task.i,worker.i)
  Protected j.PmFastGemmJob,rt.i=task/ *g\ColTasks,ct.i=task % *g\ColTasks
  CopyStructure(*g,@j,PmFastGemmJob)
  j\First=rt* *g\RowBlock : j\Last=j\First+ *g\RowBlock : If j\Last>*g\M : j\Last=*g\M : EndIf
  j\ColFirst=ct* *g\ColBlock : j\ColLast=j\ColFirst+ *g\ColBlock : If j\ColLast>*g\N : j\ColLast=*g\N : EndIf
  If j\First<j\Last And j\ColFirst<j\ColLast : PmFastGemmWorker(@j) : EndIf
EndProcedure

; The product on the pool: blocks of four rows first, then blocks of sixteen
; columns when there are too few rows to keep every thread busy. A task owns
; its block of outputs; every output's sum runs over the whole of K in one
; task. Grain is the smallest task in multiply-accumulates (0: the measured
; default); a recurrent step passes a smaller one.
#PMFAST_GEMM_GRAIN = 262144
Procedure PmFastGemm(*A,*B,*Dst,M.i,K.i,N.i,*Bias=0,Grain.i=0)
  Protected j.PmFastGemmJob,macs.q=M,want.i,rowTasks.i,colTasks.i
  j\A=*A : j\B=*B : j\Dst=*Dst : j\Bias=*Bias : j\K=K : j\N=N : j\M=M
  j\First=0 : j\Last=M : j\ColFirst=0 : j\ColLast=N
  If M<=0 Or N<=0 : ProcedureReturn : EndIf
  If Grain<=0 : Grain=#PMFAST_GEMM_GRAIN : EndIf
  macs*PmPoolMax(K,1) : macs*N
  If PmPoolForceSplit
    want=4*PmPoolThreads+3
  Else
    want=macs/Grain
    If want>8*PmPoolThreads : want=8*PmPoolThreads : EndIf
  EndIf
  If PmPoolThreads<=1 Or want<2
    PmFastGemmWorker(@j) : ProcedureReturn
  EndIf
  ; Blocks of 128 columns keep a task's panel of B (K x 128) in cache while
  ; its rows run over it; rows are then shared out to reach `want` tasks.
  ; Narrow products (N <= 256) keep all their columns in one block unless
  ; there are too few rows to go round.
  If N>256 : j\ColBlock=128 : Else : j\ColBlock=(N+15)&~15 : EndIf
  If PmPoolForceSplit : j\ColBlock=16 : EndIf
  colTasks=(N+j\ColBlock-1)/j\ColBlock
  rowTasks=(want+colTasks-1)/colTasks
  If rowTasks>(M+3)/4 : rowTasks=(M+3)/4 : EndIf
  If rowTasks<1 : rowTasks=1 : EndIf
  j\RowBlock=((M+rowTasks-1)/rowTasks+3)&~3
  rowTasks=(M+j\RowBlock-1)/j\RowBlock
  If rowTasks*colTasks<want And colTasks<(N+15)/16
    colTasks=(want+rowTasks-1)/rowTasks
    If colTasks>(N+15)/16 : colTasks=(N+15)/16 : EndIf
    j\ColBlock=((N+colTasks-1)/colTasks+15)&~15
    colTasks=(N+j\ColBlock-1)/j\ColBlock
  EndIf
  j\ColTasks=colTasks
  If rowTasks*colTasks<=1
    PmFastGemmWorker(@j)
  Else
    PmPoolRun(@PmFastGemmTask(),@j,rowTasks*colTasks)
  EndIf
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

; One 128-position block of one batch: columns gathered into this worker's
; own buffer, one product, then the block copied to its place in Dst.
; With fewer blocks than two per thread, each block's output channels are
; also cut into row groups (at least 32 rows, multiples of 4); a task then
; gathers its block itself and computes its group's rows only. The gathering
; is repeated per group; the product is not, and each output's sum is the
; same whichever group computes it.
Structure PmFastConvTilesCtx
  Args.i : Blocks.i : Columns.i : Outputs.i : Failed.i : RowGroups.i : RowBlock.i
EndStructure

Procedure PmFastConvTileRun(*g.PmTensorConv1DArgs,task.i,blocks.i,columns.i,output.i,rowGroups.i=1,rowBlock.i=0)
  Protected block.i=128,k.i=*g\InChannels* *g\Kernel
  Protected bn.i,ox0.i,width.i,ic.i,kx.i,row.i,src.i,dst.i,oc.i,r0.i=0,r1.i=*g\OutChannels
  Protected gemm.PmFastGemmJob
  If rowGroups>1
    r0=(task % rowGroups)*rowBlock : r1=r0+rowBlock : If r1>*g\OutChannels : r1=*g\OutChannels : EndIf
    task=task/rowGroups
    If r0>=r1 : ProcedureReturn : EndIf
  EndIf
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
  gemm\K=k : gemm\N=width : gemm\First=r0 : gemm\Last=r1
  PmFastGemmWorker(@gemm)
  For oc=r0 To r1-1
    CopyMemory(output+oc*width*4,*g\Dst+((bn* *g\OutChannels+oc)* *g\OutWidth+ox0)*4,width*4)
  Next
EndProcedure

; The gathered block and the product go in the worker's own buffer
; (PmPoolScratch), which outlives the call: every column row is written by
; the gather and every output by the product before either is read.
Procedure PmFastConvTileTask(*c.PmFastConvTilesCtx,task.i,worker.i)
  Protected *g.PmTensorConv1DArgs=*c\Args,columns.i,cb.i=*g\InChannels* *g\Kernel*128*4
  columns=PmPoolScratch(worker,cb+ *g\OutChannels*128*4)
  If columns=0 : PokeI(*c\Failed+task*8,1) : ProcedureReturn : EndIf
  PmFastConvTileRun(*g,task,*c\Blocks,columns,columns+cb,*c\RowGroups,*c\RowBlock)
EndProcedure

; Blocks split across the pool, each worker in its own buffer. The
; working-memory budget covers every thread's buffer or the blocks run on
; the calling thread alone.
Procedure.i PmFastConvTiles(*g.PmTensorConv1DArgs,Available.i)
  Protected c.PmFastConvTilesCtx,tasks.i,bytes.i,i.i,failed.i,threads.i=PmPoolThreads
  c\Blocks=(*g\OutWidth+127)/128 : tasks=c\Blocks* *g\Batches : c\Args=*g
  bytes=(*g\InChannels* *g\Kernel+ *g\OutChannels)*128*4
  If bytes>Available Or tasks<=0 : ProcedureReturn 0 : EndIf
  If bytes>Available/threads : threads=1 : EndIf
  c\RowGroups=1 : c\RowBlock=*g\OutChannels
  If threads>1 And (tasks<2*threads Or PmPoolForceSplit)
    c\RowGroups=(2*threads+tasks-1)/tasks
    If PmPoolForceSplit : c\RowGroups=(*g\OutChannels+3)/4 : EndIf
    If c\RowGroups>*g\OutChannels/32 And PmPoolForceSplit=0 : c\RowGroups=*g\OutChannels/32 : EndIf
    If c\RowGroups<1 : c\RowGroups=1 : EndIf
    c\RowBlock=((*g\OutChannels+c\RowGroups-1)/c\RowGroups+3)&~3
    c\RowGroups=(*g\OutChannels+c\RowBlock-1)/c\RowBlock
    tasks*c\RowGroups
  EndIf
  If tasks<2 : threads=1 : EndIf
  c\Failed=AllocateMemory(tasks*8+8)
  If c\Failed=0 : ProcedureReturn 0 : EndIf
  If threads=1
    For i=0 To tasks-1 : PmFastConvTileTask(@c,i,0) : Next
  Else
    PmPoolRun(@PmFastConvTileTask(),@c,tasks)
  EndIf
  ; A block whose worker could not get its buffer runs again on worker 0's.
  For i=0 To tasks-1
    If PeekI(c\Failed+i*8)
      PokeI(c\Failed+i*8,0) : PmFastConvTileTask(@c,i,0)
      If PeekI(c\Failed+i*8) : failed=1 : Break : EndIf
    EndIf
  Next
  FreeMemory(c\Failed)
  ProcedureReturn 1-failed
EndProcedure
Structure PmFastIm2Col
  G.i : Columns.i : Bn.i : Group.i : InGroup.i : Rows.i : Chunk.i
EndStructure

; Rows [first, first + chunk) of the gathered input, row = ic * Kernel + kx.
Procedure PmFastIm2ColTask(*m.PmFastIm2Col,task.i,worker.i)
  Protected *g.PmTensorConv1DArgs=*m\G,row.i=task* *m\Chunk,last.i=row+ *m\Chunk,ic.i,kx.i,src.i
  If last>*m\Rows : last=*m\Rows : EndIf
  While row<last
    ic=row/ *g\Kernel : kx=row % *g\Kernel
    src=*g\Src+((*m\Bn* *g\InChannels+*m\Group* *m\InGroup+ic)* *g\InWidth)*4
    PmFastCopyWindow(src,*m\Columns+row* *g\OutWidth*4,*g\InWidth,- *g\PadLeft+kx* *g\Dilation,*g\OutWidth,*g\Stride)
    row+1
  Wend
EndProcedure

Procedure.i PmFastConv(*g.PmTensorConv1DArgs,Available.i)
  Protected im.PmFastIm2Col,tasks.i
  Protected inGroup.i=*g\InChannels / *g\Groups,outGroup.i=*g\OutChannels / *g\Groups
  Protected k.i=inGroup* *g\Kernel,bytes.i,columns.i,bn.i,group.i,ic.i,kx.i,row.i,src.i,dst.i,bias.i
  If *g\OutWidth>256 And *g\Groups=1 : ProcedureReturn PmFastConvTiles(*g,Available) : EndIf
  If k<=0 Or *g\OutWidth<=0 Or k>Available/4/ *g\OutWidth : ProcedureReturn 0 : EndIf
  bytes=k* *g\OutWidth*4 : columns=AllocateMemory(bytes)
  If columns=0 : ProcedureReturn 0 : EndIf
  For bn=0 To *g\Batches-1
    For group=0 To *g\Groups-1
      ; The gathered rows (ic, kx) are independent copies: split across the pool.
      im\G=*g : im\Columns=columns : im\Bn=bn : im\Group=group : im\InGroup=inGroup : im\Rows=inGroup* *g\Kernel
      tasks=PmPoolTasks(im\Rows,PmPoolMax(1,#PMELEM_CHEAP/PmPoolMax(*g\OutWidth,1)),1,@im\Chunk)
      If tasks<=1
        im\Chunk=im\Rows : PmFastIm2ColTask(@im,0,0)
      Else
        PmPoolRun(@PmFastIm2ColTask(),@im,tasks)
      EndIf
      bias=0 : If *g\Bias : bias=*g\Bias+group*outGroup*4 : EndIf
      PmFastGemm(*g\Weight+group*outGroup*k*4,columns,*g\Dst+(bn* *g\OutChannels+group*outGroup)* *g\OutWidth*4,outGroup,k,*g\OutWidth,bias)
    Next
  Next
  FreeMemory(columns)
  ProcedureReturn 1
EndProcedure
