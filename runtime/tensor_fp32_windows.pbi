; Windows counterpart of tensor_fp32.pmi; keep portable tensor semantics synchronized.
; Native x64 SIMD and worker implementations live here and are emitted with the model.
; The Windows x64 build uses PeekF/PokeF and PeekI/PokeI instead of Forge typed raw dereferences.
; ======================================================================
; tensor_fp32.pmi - portable FP32 tensor kernels used by generated ONNX programs.
; ----------------------------------------------------------------------
; The MAIN file must write EnableFloatingPoint before including this file.
; Procedures take byte addresses. Tensor elements are contiguous IEEE-754
; binary32 values unless a procedure's name says otherwise.
;
; This is deliberately an execution library, not a graph interpreter.
; The host compiler has already checked shapes and operator revisions,
; folded constants, selected kernels and planned memory before any of these
; procedures run. There are no operator names or dynamic tensor descriptors
; on the board.
;
; This file includes no other library. Generated programs that use Exp,
; Log, Pow or Sqr list RaspberryPi4/Lib/math.pi4 before this file.
; ======================================================================

Global Dim PmTensorCrcTable.i(256)
Global PmTensorCrcReady.i

Procedure PmTensorCrcInit()
  Protected i.i
  Protected bit.i
  Protected value.i
  If PmTensorCrcReady <> 0
    ProcedureReturn
  EndIf
  i = 0
  While i < 256
    value = i
    bit = 0
    While bit < 8
      If (value & 1) <> 0
        value = ((value >> 1) & $7FFFFFFF) ! $EDB88320
      Else
        value = (value >> 1) & $7FFFFFFF
      EndIf
      bit = bit + 1
    Wend
    PmTensorCrcTable(i) = value & $FFFFFFFF
    i = i + 1
  Wend
  PmTensorCrcReady = 1
EndProcedure

Procedure.i PmTensorCrc32(*data, bytes.i)
  Protected i.i
  Protected index.i
  Protected crc.i
  PmTensorCrcInit()
  crc = $FFFFFFFF
  i = 0
  While i < bytes
    index = (crc ! (PeekA(*data + i) & 255)) & 255
    crc = ((crc >> 8) & $00FFFFFF) ! PmTensorCrcTable(index)
    i = i + 1
  Wend
  ProcedureReturn (crc ! $FFFFFFFF) & $FFFFFFFF
EndProcedure

; Inline host loads/stores: no procedure call per scalar tensor access. PeekF
; and PokeF retain the same binary32 load/store boundaries as the procedures.
Macro PmTensorGet(base, index)
  PeekF((base) + (index) * 4)
EndMacro

Macro PmTensorPut(base, index, value)
  PokeF((base) + (index) * 4, (value))
EndMacro

; Runtime shape/index tensors use signed ONNX INT64 on 64-bit backends.
; A future 32-bit target profile must provide an equivalent pair or reject
; models whose data path actually contains INT64 values.
Procedure.i PmTensorGetI64(*base, index.i)
  Protected p.i
  p = *base + index * 8
  ProcedureReturn PeekI(p)
EndProcedure

Procedure PmTensorPutI64(*base, index.i, value.i)
  Protected p.i
  p = *base + index * 8
  PokeI(p, value)
EndProcedure

Procedure.i PmTensorGetBool(*base, index.i)
  ProcedureReturn PeekA(*base + index) & 255
EndProcedure

Procedure PmTensorPutBool(*base, index.i, value.i)
  If value <> 0 : value = 1 : EndIf
  PokeA(*base + index, value)
EndProcedure

Procedure PmTensorZero(*dst, count.i)
  Protected i.i
  Protected p.i
  i = 0
  p = *dst
  While i < count
    PokeF(p, 0.0)
    p = p + 4
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorCopy(*src, *dst, count.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  i = 0
  ps = *src
  pd = *dst
  While i < count
    PokeF(pd, PeekF(ps))
    ps = ps + 4
    pd = pd + 4
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorCopyI64(*src, *dst, count.i)
  Protected i.i
  i = 0
  While i < count
    PmTensorPutI64(*dst, i, PmTensorGetI64(*src, i))
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorCopyBool(*src, *dst, count.i)
  Protected i.i
  i = 0
  While i < count
    PmTensorPutBool(*dst, i, PmTensorGetBool(*src, i))
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorAdd(*a, *b, *dst, count.i)
  Protected i.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  i = 0 : pa = *a : pb = *b : pd = *dst
  While i < count
    PokeF(pd, PeekF(pa) + PeekF(pb))
    pa = pa + 4 : pb = pb + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorSub(*a, *b, *dst, count.i)
  Protected i.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  i = 0 : pa = *a : pb = *b : pd = *dst
  While i < count
    PokeF(pd, PeekF(pa) - PeekF(pb))
    pa = pa + 4 : pb = pb + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorMul(*a, *b, *dst, count.i)
  Protected i.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  i = 0 : pa = *a : pb = *b : pd = *dst
  While i < count
    PokeF(pd, PeekF(pa) * PeekF(pb))
    pa = pa + 4 : pb = pb + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorDiv(*a, *b, *dst, count.i)
  Protected i.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  i = 0 : pa = *a : pb = *b : pd = *dst
  While i < count
    PokeF(pd, PeekF(pa) / PeekF(pb))
    pa = pa + 4 : pb = pb + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

; op: 0 add, 1 subtract, 2 multiply, 3 divide. scalarSide is 0 when
; the scalar is the left operand and 1 when it is the right operand.
Procedure PmTensorBinaryScalar(*vec, scalar.f, *dst, count.i, op.i, scalarSide.i)
  Protected i.i
  Protected pv.i
  Protected pd.i
  Protected v.f
  i = 0 : pv = *vec : pd = *dst
  While i < count
    v = PeekF(pv)
    If op = 0
      PokeF(pd, v + scalar)
    ElseIf op = 1
      If scalarSide = 0 : PokeF(pd, scalar - v) : Else : PokeF(pd, v - scalar) : EndIf
    ElseIf op = 2
      PokeF(pd, v * scalar)
    Else
      If scalarSide = 0 : PokeF(pd, scalar / v) : Else : PokeF(pd, v / scalar) : EndIf
    EndIf
    pv = pv + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorRelu(*src, *dst, count.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    If v < 0.0 : v = 0.0 : EndIf
    PokeF(pd, v)
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorLeakyRelu(*src, *dst, count.i, alpha.f)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    If v < 0.0 : v = v * alpha : EndIf
    PokeF(pd, v)
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure.f PmTensorSigmoidValue(value.f)
  Protected e.f
  If value >= 0.0
    e = Exp(0.0 - value)
    ProcedureReturn 1.0 / (1.0 + e)
  EndIf
  e = Exp(value)
  ProcedureReturn e / (1.0 + e)
EndProcedure

Procedure.f PmTensorTanhValue(value.f)
  Protected e.f
  If value > 10.0 : ProcedureReturn 1.0 : EndIf
  If value < -10.0 : ProcedureReturn -1.0 : EndIf
  e = Exp(value + value)
  ProcedureReturn (e - 1.0) / (e + 1.0)
EndProcedure

Procedure PmTensorSigmoid(*src, *dst, count.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    PokeF(pd, PmTensorSigmoidValue(v))
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorTanh(*src, *dst, count.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    PokeF(pd, PmTensorTanhValue(v))
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

; Refusal signals for the two elementwise kernels whose op sets are closed.
; Same convention as the portable twin: this file sits UNDER both of its
; consumers - the dynamic runtime, which has DError, and a statically emitted
; program, which has PmOnnxRuntimeOk - and can name neither, so a kernel here
; only ever CLEARS a flag and whoever ran it reads it back and raises that
; layer's error.  Nothing here ever sets one; the caller arms it per call.
;
; These are MIRRORED from tensor_fp32.pmi rather than shared with it.  The two
; files are never in one build: a Windows program includes this one and a
; Pi 4 / UNO Q / Pico program includes the portable one, and each is copied
; whole beside the model it was generated for.  A third file holding the two
; declarations would have to be added to both targets' copy sets to remove two
; lines from each, and a generated Windows program would then need a file the
; portable one also needs but under a different name.  The whole file is
; already a mirror of tensor_fp32.pmi by design - see the header - so these
; keep that arrangement rather than inventing a second one for two globals.
Global PmTensorUnaryMathOk.i
Global PmTensorTrigOk.i

; Elementwise unary math.
;
; THE OP SET IS CLOSED AND EXPLICIT: 0 Exp, 1 Log, 2 Sqrt, 3 Abs, 4 Neg.  Those
; are the only codes any emitter produces - onnx_emit.pbi writes 0..4 from a
; Select over Exp/Log/Sqrt/Abs/Neg, and onnx_dynamic_emit.pbi writes the same
; 0..4 as DUnary codes, which DUnary passes through unchanged.
;
; This procedure used to end its chain with a bare Else that NEGATED, so op 5,
; op 99 or op -1 quietly returned -x and the graph ran on with wrong numbers in
; it.  Anything outside the set is now REFUSED: the destination is left exactly
; as the caller left it and PmTensorUnaryMathOk is cleared.  The op is decided
; once, before the loop, because a per-element test can only ever be a slower
; way to get the same answer.
Procedure PmTensorUnaryMath(*src, *dst, count.i, op.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  If op < 0 Or op > 4
    PmTensorUnaryMathOk = 0
    ProcedureReturn
  EndIf
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    If op = 0
      PokeF(pd, Exp(v))
    ElseIf op = 1
      PokeF(pd, Log(v))
    ElseIf op = 2
      PokeF(pd, Sqr(v))
    ElseIf op = 3
      If v < 0.0 : v = 0.0 - v : EndIf
      PokeF(pd, v)
    Else
      PokeF(pd, 0.0 - v)
    EndIf
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

; Transcendentals used by synthesis and signal-processing graphs.
;
; THE OP SET IS CLOSED AND EXPLICIT: 0 Sin, 1 Cos, 2 Atan.  Those are the only
; codes any emitter produces - onnx_emit.pbi writes them directly, and
; onnx_dynamic_emit.pbi writes DUnary 5/6/7 which PmFastTrig turns into 0/1/2.
; The including math library supplies these three.
;
; This procedure used to end its chain with a bare Else that computed ATan, so
; op 3, op 7 or op -1 quietly returned an arctangent and the graph ran on with
; wrong numbers in it.  Anything outside the set is now REFUSED: the
; destination is left exactly as the caller left it and PmTensorTrigOk is
; cleared.  PmFastTrig hands every code it does not own down to here, and
; DUnary turns a cleared flag into a DError, so the model stops on the node
; that did it.  The op is decided once, before the loop.
Procedure PmTensorTrig(*src, *dst, count.i, op.i)
  Protected i.i
  Protected v.f
  If op < 0 Or op > 2
    PmTensorTrigOk = 0
    ProcedureReturn
  EndIf
  i = 0
  While i < count
    v = PmTensorGet(*src, i)
    If op = 0
      v = Sin(v)
    ElseIf op = 1
      v = Cos(v)
    Else
      v = ATan(v)
    EndIf
    PmTensorPut(*dst, i, v)
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorFloor(*src, *dst, count.i)
  Protected i.i
  Protected n.i
  Protected whole.f
  Protected value.f
  i = 0
  While i < count
    value = PmTensorGet(*src, i)
    n = Int(value)
    whole = n
    If value < 0.0 And whole <> value : n = n - 1 : EndIf
    whole = n
    PmTensorPut(*dst, i, whole)
    i = i + 1
  Wend
EndProcedure

; ONNX Round uses nearest integer with ties to even.
Procedure PmTensorRoundEven(*src, *dst, count.i)
  Protected i.i
  Protected base.i
  Protected result.i
  Protected fraction.f
  Protected whole.f
  Protected value.f
  i = 0
  While i < count
    value = PmTensorGet(*src, i)
    base = Int(value)
    whole = base
    If value < 0.0 And whole <> value : base = base - 1 : EndIf
    whole = base
    fraction = value - whole
    result = base
    If fraction > 0.5
      result = base + 1
    ElseIf fraction = 0.5 And (base & 1) <> 0
      result = base + 1
    EndIf
    whole = result
    PmTensorPut(*dst, i, whole)
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorPow(*a, *b, *dst, count.i)
  Protected i.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  i = 0 : pa = *a : pb = *b : pd = *dst
  While i < count
    PokeF(pd, Pow(PeekF(pa), PeekF(pb)))
    pa = pa + 4 : pb = pb + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorPowScalar(*vec, scalar.f, *dst, count.i, scalarSide.i)
  Protected i.i
  Protected value.f
  i = 0
  While i < count
    value = PmTensorGet(*vec, i)
    If scalarSide = 0
      value = Pow(scalar, value)
    Else
      value = Pow(value, scalar)
    EndIf
    PmTensorPut(*dst, i, value)
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorClip(*src, *dst, count.i, lo.f, hi.f)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    If v < lo : v = lo : EndIf
    If v > hi : v = hi : EndIf
    PokeF(pd, v)
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Structure PmTensorInt8RowArgs
  A.i
  B.i
  Dst.i
  K.i
  N.i
  Scales.i
  AScale.f
EndStructure

Structure PmTensorInt8DotArgs
  A.i
  B.i
  Count.i
  StepA.i
  StepB.i
EndStructure

Structure PmTensorDotInt8Args
  A.i
  B.i
  Count.i
  AScale.f
  WeightScale.f
EndStructure

CompilerIf #PMO_USE_INT8 = 1
Global PmTensorInt8Scratch.i
Global PmTensorInt8ScratchBytes.i

; Reuse one activation for eight adjacent columns of row-major INT8 weights.
Procedure PmTensorInt8Row8(*g.PmTensorInt8RowArgs)
  Protected pa.i=*g\A, pb.i=*g\B, pd.i=*g\Dst, count.i=*g\K, stride.i=*g\N, scales.i=*g\Scales
  Protected fa.f=*g\AScale
  CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
    !mov rax,[p.v_pa]
    !mov rdx,[p.v_pb]
    !mov rcx,[p.v_count]
    !mov r8,[p.v_stride]
    !pxor xmm0,xmm0
    !pxor xmm1,xmm1
    !test rcx,rcx
    !jz pmint8row_store
    !pmint8row_inner:
    !movq xmm2,[rdx]
    !punpcklbw xmm2,xmm2
    !psraw xmm2,8
    !movzx r9d,byte [rax]
    !movd xmm3,r9d
    !punpcklbw xmm3,xmm3
    !psraw xmm3,8
    !pshuflw xmm3,xmm3,0
    !pshufd xmm3,xmm3,0
    !pmullw xmm2,xmm3
    !movdqa xmm4,xmm2
    !psraw xmm4,15
    !movdqa xmm5,xmm2
    !punpcklwd xmm2,xmm4
    !punpckhwd xmm5,xmm4
    !paddd xmm0,xmm2
    !paddd xmm1,xmm5
    !inc rax
    !add rdx,r8
    !dec rcx
    !jnz pmint8row_inner
    !pmint8row_store:
    ; Match scalar host intermediates: rounded FP32 sum, then FP64 scale/bias.
    !cvtdq2ps xmm0,xmm0
    !cvtdq2ps xmm1,xmm1
    !cvtss2sd xmm3,[p.v_fa]
    !shufpd xmm3,xmm3,0
    !mov rax,[p.v_pd]
    !mov rdx,[p.v_scales]
    !cvtps2pd xmm2,xmm0
    !mulpd xmm2,xmm3
    !cvtps2pd xmm4,[rdx+0]
    !mulpd xmm2,xmm4
    !cvtpd2ps xmm2,xmm2
    !movq [rax+0],xmm2
    !psrldq xmm0,8
    !cvtps2pd xmm2,xmm0
    !mulpd xmm2,xmm3
    !cvtps2pd xmm4,[rdx+8]
    !mulpd xmm2,xmm4
    !cvtpd2ps xmm2,xmm2
    !movq [rax+8],xmm2
    !cvtps2pd xmm2,xmm1
    !mulpd xmm2,xmm3
    !cvtps2pd xmm4,[rdx+16]
    !mulpd xmm2,xmm4
    !cvtpd2ps xmm2,xmm2
    !movq [rax+16],xmm2
    !psrldq xmm1,8
    !cvtps2pd xmm2,xmm1
    !mulpd xmm2,xmm3
    !cvtps2pd xmm4,[rdx+24]
    !mulpd xmm2,xmm4
    !cvtpd2ps xmm2,xmm2
    !movq [rax+24],xmm2
  CompilerEndIf
EndProcedure

; Exact signed-byte dot, with strides selected outside the reduction loop.
Procedure.i PmTensorInt8Dot(*g.PmTensorInt8DotArgs)
  Protected pa.i
  Protected pb.i
  Protected left.i
  Protected sa.i
  Protected sb.i
  Protected total.i
  pa = *g\A : pb = *g\B : left = *g\Count
  sa = *g\StepA : sb = *g\StepB : total = 0
  CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
    !mov rax,[p.v_pa]
    !mov rdx,[p.v_pb]
    !mov rcx,[p.v_left]
    !mov r8,[p.v_sa]
    !mov r9,[p.v_sb]
    !xor r10d,r10d
    !cmp r8,1
    !jne pmint8dot_tail
    !cmp r9,1
    !jne pmint8dot_tail
    !pxor xmm0,xmm0
    !pmint8dot_vector:
    !cmp rcx,8
    !jl pmint8dot_reduce
    !movq xmm1,[rax]
    !movq xmm2,[rdx]
    !punpcklbw xmm1,xmm1
    !punpcklbw xmm2,xmm2
    !psraw xmm1,8
    !psraw xmm2,8
    !pmaddwd xmm1,xmm2
    !paddd xmm0,xmm1
    !add rax,8
    !add rdx,8
    !sub rcx,8
    !jmp pmint8dot_vector
    !pmint8dot_reduce:
    !pshufd xmm1,xmm0,78
    !paddd xmm0,xmm1
    !pshufd xmm1,xmm0,177
    !paddd xmm0,xmm1
    !movd r10d,xmm0
    !pmint8dot_tail:
    !test rcx,rcx
    !jz pmint8dot_done
    !pmint8dot_scalar:
    !movsx r11d,byte [rax]
    !movsx r8d,byte [rdx]
    !imul r11d,r8d
    !add r10d,r11d
    !mov r8,[p.v_sa]
    !add rax,r8
    !add rdx,r9
    !dec rcx
    !jnz pmint8dot_scalar
    !pmint8dot_done:
    !movsxd r10,r10d
    !mov [p.v_total],r10
    ProcedureReturn total
  CompilerElse
  While left >= 4
    total = total + PeekB(pa) * PeekB(pb)
    total = total + PeekB(pa + sa) * PeekB(pb + sb)
    total = total + PeekB(pa + sa * 2) * PeekB(pb + sb * 2)
    total = total + PeekB(pa + sa * 3) * PeekB(pb + sb * 3)
    pa = pa + sa * 4 : pb = pb + sb * 4 : left = left - 4
  Wend
  While left > 0
    total = total + PeekB(pa) * PeekB(pb)
    pa = pa + sa : pb = pb + sb : left = left - 1
  Wend
  ProcedureReturn total
  CompilerEndIf
EndProcedure

Procedure.i PmTensorGetInt8(*base, index.i)
  ProcedureReturn PeekB(*base + index)
EndProcedure

Procedure.f PmTensorDynamicScale(*base, count.i)
  Protected i.i
  Protected value.f
  Protected magnitude.f
  Protected maximum.f
  i = 0 : maximum = 0.0
  While i < count
    value = PmTensorGet(*base, i)
    magnitude = value
    If magnitude < 0.0 : magnitude = 0.0 - magnitude : EndIf
    If magnitude > maximum : maximum = magnitude : EndIf
    i = i + 1
  Wend
  If maximum = 0.0 : ProcedureReturn 1.0 : EndIf
  ProcedureReturn maximum / 127.0
EndProcedure

Procedure.i PmTensorQuantizeInt8(value.f, scale.f)
  Protected quantized.i
  Protected bit.i
  Protected negative.i
  Protected threshold.f
  quantized = 0 : negative = 0
  If value < 0.0 : negative = 1 : value = 0.0 - value : EndIf
  bit = 64 : threshold = scale * 64.0
  While bit > 0
    If value >= threshold : quantized = quantized + bit : value = value - threshold : EndIf
    threshold = threshold * 0.5 : bit = bit / 2
  Wend
  If value >= scale * 0.5 : quantized = quantized + 1 : EndIf
  If quantized > 127 : quantized = 127 : EndIf
  If negative <> 0 : quantized = 0 - quantized : EndIf
  ProcedureReturn quantized
EndProcedure

Procedure PmTensorQuantizeBuffer(*src, *dst, count.i, scale.f)
  Protected index.i
  Protected pa.i=*src, pd.i=*dst, remaining.i=count
  Protected threshold.f=scale*64.0, half.f=0.5
  index = 0
  CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
    ; Below normal range the scalar final half-scale comparison uses a
    ; nonzero FP64 intermediate that a rounded FP32 threshold can lose.
    If scale>=1.17549435e-38
    ; Same seven rounded subtraction steps as the scalar quantizer, four lanes.
    ; No reciprocal approximation or change to halfway rounding.
    !mov rax,[p.v_pa]
    !mov rdx,[p.v_pd]
    !mov rcx,[p.v_remaining]
    !pmint8quant_four:
    !cmp rcx,4
    !jl pmint8quant_done
    !movups xmm1,[rax]
    !movdqa xmm0,xmm1
    !pxor xmm5,xmm5
    !cmpltps xmm0,xmm5
    !pcmpeqd xmm5,xmm5
    !psrld xmm5,1
    !andps xmm1,xmm5
    !pxor xmm2,xmm2
    !movss xmm3,[p.v_threshold]
    !shufps xmm3,xmm3,0
    !mov r9d,64
    !movd xmm4,r9d
    !pshufd xmm4,xmm4,0
    !mov r8d,7
    !pmint8quant_bit:
    !movaps xmm5,xmm3
    !cmpleps xmm5,xmm1
    !pand xmm5,xmm4
    !paddd xmm2,xmm5
    !movaps xmm5,xmm3
    !cmpleps xmm5,xmm1
    !andps xmm5,xmm3
    !subps xmm1,xmm5
    !movss xmm5,[p.v_half]
    !shufps xmm5,xmm5,0
    !mulps xmm3,xmm5
    !psrld xmm4,1
    !dec r8d
    !jnz pmint8quant_bit
    !cmpleps xmm3,xmm1
    !pcmpeqd xmm4,xmm4
    !psrld xmm4,31
    !pand xmm3,xmm4
    !paddd xmm2,xmm3
    !packssdw xmm2,xmm2
    !packsswb xmm2,xmm2
    !punpcklbw xmm2,xmm2
    !psraw xmm2,8
    !punpcklwd xmm2,xmm2
    !psrad xmm2,16
    !pxor xmm2,xmm0
    !psubd xmm2,xmm0
    !packssdw xmm2,xmm2
    !packsswb xmm2,xmm2
    !movd [rdx],xmm2
    !add rax,16
    !add rdx,4
    !sub rcx,4
    !jmp pmint8quant_four
    !pmint8quant_done:
    !mov [p.v_remaining],rcx
    index=count-remaining
    EndIf
  CompilerEndIf
  While index < count
    PokeA(*dst + index, PmTensorQuantizeInt8(PmTensorGet(*src, index), scale) & 255)
    index = index + 1
  Wend
EndProcedure

Procedure.f PmTensorDotInt8Scaled(*g.PmTensorDotInt8Args)
  Protected dot.PmTensorInt8DotArgs
  Protected converted.f
  dot\A=*g\A : dot\B=*g\B : dot\Count=*g\Count : dot\StepA=1 : dot\StepB=1
  converted=PmTensorInt8Dot(@dot)
  ProcedureReturn converted * *g\AScale * *g\WeightScale
EndProcedure
CompilerEndIf

; A is MxK, B is KxN and dst is MxN, all row-major.
Procedure PmTensorMatMul2(*a, *b, *dst, m.i, k.i, n.i)
  Protected row.i
  Protected col.i
  Protected inner.i
  Protected pa.i
  Protected pb.i
  Protected pd.i
  Protected sum.f
  row = 0
  While row < m
    col = 0
    While col < n
      sum = 0.0
      inner = 0
      pa = *a + (row * k) * 4
      pb = *b + col * 4
      While inner < k
        sum = sum + PeekF(pa) * PeekF(pb)
        pa = pa + 4
        pb = pb + n * 4
        inner = inner + 1
      Wend
      pd = *dst + (row * n + col) * 4
      PokeF(pd, sum)
      col = col + 1
    Wend
    row = row + 1
  Wend
EndProcedure

CompilerIf #PMO_USE_INT8 = 1
Procedure PmTensorMatMul2Int8(*a, *b, *dst, m.i, k.i, n.i, *weightScales)
  Protected row.i
  Protected col.i
  Protected aScale.f
  Protected converted.f
  Protected dot.PmTensorInt8DotArgs
  aScale = PmTensorDynamicScale(*a, m * k)
  Protected eight.PmTensorInt8RowArgs
  PmTensorQuantizeBuffer(*a, PmTensorInt8Scratch, m * k, aScale)
  dot\Count=k : dot\StepA=1 : dot\StepB=n
  row=0
  While row<m
    dot\A=PmTensorInt8Scratch+row*k : col=0
    CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
    eight\A=dot\A : eight\K=k : eight\N=n : eight\AScale=aScale
    While col+7<n
      eight\B=*b+col : eight\Dst=*dst+(row*n+col)*4 : eight\Scales=*weightScales+col*4
      PmTensorInt8Row8(@eight)
      col=col+8
    Wend
    CompilerEndIf
    While col<n
      dot\B=*b+col
      converted=PmTensorInt8Dot(@dot)
      PmTensorPut(*dst,row*n+col,converted*aScale*PmTensorGet(*weightScales,col))
      col=col+1
    Wend
    row=row+1
  Wend
EndProcedure
CompilerEndIf

Structure PmTensorGemmArgs
  A.i
  B.i
  C.i
  Dst.i
  M.i
  K.i
  N.i
  TransA.i
  TransB.i
  Alpha.f
  Beta.f
  CCount.i
  WeightScales.i
EndStructure

; ONNX Gemm: Y = alpha * A' * B' + beta * C. C may be absent (0), a
; scalar (cCount=1), one value per column (cCount=n), or a full MxN matrix.
; One argument block is used because the A64 calling convention implemented
; by Forge currently has eight register arguments and no stack arguments.
Procedure PmTensorGemm(*g.PmTensorGemmArgs)
  Protected dot.PmTensorInt8DotArgs
  Protected row.i
  Protected col.i
  Protected inner.i
  Protected ia.i
  Protected ib.i
  Protected ic.i
  Protected accumulator.i
  Protected aScale.f
  Protected converted.f
  Protected sum.f
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    aScale = PmTensorDynamicScale(*g\A, *g\M * *g\K)
    PmTensorQuantizeBuffer(*g\A, PmTensorInt8Scratch, *g\M * *g\K, aScale)
    dot\Count=*g\K : dot\StepA=1 : dot\StepB=*g\N
    If *g\TransA : dot\StepA=*g\M : EndIf
    If *g\TransB : dot\StepB=1 : EndIf
    row=0
    While row<*g\M
      ia=row* *g\K : If *g\TransA : ia=row : EndIf
      dot\A=PmTensorInt8Scratch+ia : col=0
      While col<*g\N
        ib=col : If *g\TransB : ib=col* *g\K : EndIf
        dot\B=*g\B+ib
        converted=PmTensorInt8Dot(@dot)
        sum=converted*aScale*PmTensorGet(*g\WeightScales,col)
        sum=sum* *g\Alpha
        If *g\C<>0 And *g\CCount>0
          ic=row* *g\N+col
          If *g\CCount=1 : ic=0 : ElseIf *g\CCount=*g\N : ic=col : EndIf
          sum=sum+*g\Beta*PmTensorGet(*g\C,ic)
        EndIf
        PmTensorPut(*g\Dst,row* *g\N+col,sum)
        col=col+1
      Wend
      row=row+1
    Wend
    ProcedureReturn
  EndIf
  CompilerEndIf
  row = 0
  While row < *g\M
    col = 0
    While col < *g\N
      sum = 0.0 : accumulator = 0
      inner = 0
      While inner < *g\K
        If *g\TransA = 0 : ia = row * *g\K + inner : Else : ia = inner * *g\M + row : EndIf
        If *g\TransB = 0 : ib = inner * *g\N + col : Else : ib = col * *g\K + inner : EndIf
          sum = sum + PmTensorGet(*g\A, ia) * PmTensorGet(*g\B, ib)
        inner = inner + 1
      Wend
      sum = sum * *g\Alpha
      If *g\C <> 0 And *g\CCount > 0
        If *g\CCount = 1
          ic = 0
        ElseIf *g\CCount = *g\N
          ic = col
        Else
          ic = row * *g\N + col
        EndIf
        sum = sum + *g\Beta * PmTensorGet(*g\C, ic)
      EndIf
      PmTensorPut(*g\Dst, row * *g\N + col, sum)
      col = col + 1
    Wend
    row = row + 1
  Wend
EndProcedure

; Softmax over the final dimension. outer is the product of every earlier
; dimension and width is the final dimension.
Procedure PmTensorSoftmaxLast(*src, *dst, outer.i, width.i)
  Protected o.i
  Protected j.i
  Protected base.i
  Protected v.f
  Protected mx.f
  Protected sum.f
  o = 0
  While o < outer
    base = o * width
    mx = PmTensorGet(*src, base)
    j = 1
    While j < width
      v = PmTensorGet(*src, base + j)
      If v > mx : mx = v : EndIf
      j = j + 1
    Wend
    sum = 0.0
    j = 0
    While j < width
      v = Exp(PmTensorGet(*src, base + j) - mx)
      PmTensorPut(*dst, base + j, v)
      sum = sum + v
      j = j + 1
    Wend
    j = 0
    While j < width
      PmTensorPut(*dst, base + j, PmTensorGet(*dst, base + j) / sum)
      j = j + 1
    Wend
    o = o + 1
  Wend
EndProcedure

Procedure PmTensorReduceMeanLast(*src, *dst, outer.i, width.i)
  Protected o.i
  Protected j.i
  Protected base.i
  Protected sum.f
  Protected divisor.f
  divisor = width
  o = 0
  While o < outer
    base = o * width
    sum = 0.0
    j = 0
    While j < width
      sum = sum + PmTensorGet(*src, base + j)
      j = j + 1
    Wend
    PmTensorPut(*dst, o, sum / divisor)
    o = o + 1
  Wend
EndProcedure

Procedure PmTensorReduceSumLast(*src, *dst, outer.i, width.i)
  Protected row.i
  Protected col.i
  Protected sum.f
  row = 0
  While row < outer
    sum = 0.0
    col = 0
    While col < width
      sum = sum + PmTensorGet(*src, row * width + col)
      col = col + 1
    Wend
    PmTensorPut(*dst, row, sum)
    row = row + 1
  Wend
EndProcedure

; Prefix sum along one contiguous logical axis. Data before/after that axis
; is represented as outer and inner products, so this handles every rank.
Procedure PmTensorCumSumF32(*src, *dst, outer.i, width.i, inner.i, exclusive.i, reverse.i)
  Protected o.i
  Protected j.i
  Protected k.i
  Protected at.i
  Protected sum.f
  o = 0
  While o < outer
    k = 0
    While k < inner
      sum = 0.0
      j = 0
      While j < width
        If reverse <> 0 : at = width - 1 - j : Else : at = j : EndIf
        If exclusive <> 0
          PmTensorPut(*dst, (o * width + at) * inner + k, sum)
          sum = sum + PmTensorGet(*src, (o * width + at) * inner + k)
        Else
          sum = sum + PmTensorGet(*src, (o * width + at) * inner + k)
          PmTensorPut(*dst, (o * width + at) * inner + k, sum)
        EndIf
        j = j + 1
      Wend
      k = k + 1
    Wend
    o = o + 1
  Wend
EndProcedure

Procedure PmTensorCumSumI64(*src, *dst, outer.i, width.i, inner.i, exclusive.i, reverse.i)
  Protected o.i
  Protected j.i
  Protected k.i
  Protected at.i
  Protected sum.i
  o = 0
  While o < outer
    k = 0
    While k < inner
      sum = 0
      j = 0
      While j < width
        If reverse <> 0 : at = width - 1 - j : Else : at = j : EndIf
        If exclusive <> 0
          PmTensorPutI64(*dst, (o * width + at) * inner + k, sum)
          sum = sum + PmTensorGetI64(*src, (o * width + at) * inner + k)
        Else
          sum = sum + PmTensorGetI64(*src, (o * width + at) * inner + k)
          PmTensorPutI64(*dst, (o * width + at) * inner + k, sum)
        EndIf
        j = j + 1
      Wend
      k = k + 1
    Wend
    o = o + 1
  Wend
EndProcedure

; LayerNormalization over the final axis. scale and bias each contain width
; elements; bias may be 0. The saved mean and inverse deviation outputs are
; intentionally not part of this inference-only kernel.
Procedure PmTensorLayerNormLast(*src, *scale, *bias, *dst, outer.i, width.i, epsilon.f)
  Protected o.i
  Protected j.i
  Protected base.i
  Protected v.f
  Protected mean.f
  Protected variance.f
  Protected inv.f
  Protected divisor.f
  divisor = width
  o = 0
  While o < outer
    base = o * width
    mean = 0.0
    j = 0
    While j < width
      mean = mean + PmTensorGet(*src, base + j)
      j = j + 1
    Wend
    mean = mean / divisor
    variance = 0.0
    j = 0
    While j < width
      v = PmTensorGet(*src, base + j) - mean
      variance = variance + v * v
      j = j + 1
    Wend
    inv = 1.0 / Sqr(variance / divisor + epsilon)
    j = 0
    While j < width
      v = (PmTensorGet(*src, base + j) - mean) * inv
      v = v * PmTensorGet(*scale, j)
      If *bias <> 0 : v = v + PmTensorGet(*bias, j) : EndIf
      PmTensorPut(*dst, base + j, v)
      j = j + 1
    Wend
    o = o + 1
  Wend
EndProcedure

Structure PmTensorBatchNormArgs
  Src.i
  Scale.i
  Bias.i
  Mean.i
  Variance.i
  Dst.i
  N.i
  C.i
  Spatial.i
  Epsilon.f
EndStructure

CompilerIf #PMO_USE_BATCHNORM = 1
Procedure PmTensorBatchNorm(*g.PmTensorBatchNormArgs)
  Protected bn.i
  Protected ch.i
  Protected s.i
  Protected idx.i
  Protected mul.f
  Protected add.f
  bn = 0
  While bn < *g\N
    ch = 0
    While ch < *g\C
      mul = PmTensorGet(*g\Scale, ch) / Sqr(PmTensorGet(*g\Variance, ch) + *g\Epsilon)
      add = PmTensorGet(*g\Bias, ch) - PmTensorGet(*g\Mean, ch) * mul
      s = 0
      While s < *g\Spatial
        idx = (bn * *g\C + ch) * *g\Spatial + s
        PmTensorPut(*g\Dst, idx, PmTensorGet(*g\Src, idx) * mul + add)
        s = s + 1
      Wend
      ch = ch + 1
    Wend
    bn = bn + 1
  Wend
EndProcedure
CompilerEndIf

; ONNX STFT for a real-valued [batch,signal] input. Output is contiguous
; [batch,frames,bins,2] with real then imaginary components.
Structure PmTensorStftArgs
  Signal.i
  Window.i
  Dst.i
  Scratch.i
  Batches.i
  SignalLength.i
  FrameStep.i
  FrameLength.i
  Frames.i
  Bins.i
  ScratchComplex.i
EndStructure

CompilerIf #PMO_USE_STFT = 1
Procedure.i PmTensorStftBitReverse(Value.i, Bits.i)
  Protected Result.i
  Protected Bit.i
  While Bit < Bits
    Result = (Result << 1) | (Value & 1)
    Value = Value >> 1
    Bit = Bit + 1
  Wend
  ProcedureReturn Result
EndProcedure

; This is the same bit-reversed radix-2 butterfly order used by the ONNX
; Runtime CPU DFT.  Inverse calls deliberately reuse the forward roots and
; the Bluestein caller reverses nonzero bins when it extracts the result.
Procedure PmTensorStftFft(*srcR, *srcI, *dstR, *dstI, *rootsR, *rootsI, count.i, inverse.i)
  Protected bits.i
  Protected stageBits.i
  Protected span.i
  Protected midpoint.i
  Protected i.i
  Protected k.i
  Protected j.i
  Protected firstIndex.i
  Protected secondIndex.i
  Protected evenIndex.i
  Protected oddIndex.i
  Protected reversed.i
  Protected evenR.f
  Protected evenI.f
  Protected oddR.f
  Protected oddI.f
  Protected rootR.f
  Protected rootI.f
  Protected firstR.f
  Protected firstI.f
  Protected secondR.f
  Protected secondI.f
  Protected scale.f
  i = count
  While i > 1 : bits = bits + 1 : i = i >> 1 : Wend
  i = 0
  While i < count
    reversed = PmTensorStftBitReverse(i, bits)
    PmTensorPut(*dstR, i, PmTensorGet(*srcR, reversed))
    PmTensorPut(*dstI, i, PmTensorGet(*srcI, reversed))
    i = i + 1
  Wend
  span = 2
  While span <= count
    midpoint = span >> 1
    stageBits = stageBits + 1
    k = 0
    While k < midpoint
      firstIndex = PmTensorStftBitReverse(k, stageBits)
      secondIndex = PmTensorStftBitReverse(midpoint + k, stageBits)
      j = 0
      While j < count
        evenIndex = k + j
        oddIndex = evenIndex + midpoint
        evenR = PmTensorGet(*dstR, evenIndex)
        evenI = PmTensorGet(*dstI, evenIndex)
        oddR = PmTensorGet(*dstR, oddIndex)
        oddI = PmTensorGet(*dstI, oddIndex)
        rootR = PmTensorGet(*rootsR, firstIndex)
        rootI = PmTensorGet(*rootsI, firstIndex)
        firstR = evenR + (rootR * oddR - rootI * oddI)
        firstI = evenI + (rootR * oddI + rootI * oddR)
        rootR = PmTensorGet(*rootsR, secondIndex)
        rootI = PmTensorGet(*rootsI, secondIndex)
        secondR = evenR + (rootR * oddR - rootI * oddI)
        secondI = evenI + (rootR * oddI + rootI * oddR)
        PmTensorPut(*dstR, evenIndex, firstR)
        PmTensorPut(*dstI, evenIndex, firstI)
        PmTensorPut(*dstR, oddIndex, secondR)
        PmTensorPut(*dstI, oddIndex, secondI)
        j = j + span
      Wend
      k = k + 1
    Wend
    span = span << 1
  Wend
  If inverse
    scale = count
    i = 0
    While i < count
      PmTensorPut(*dstR, i, PmTensorGet(*dstR, i) / scale)
      PmTensorPut(*dstI, i, PmTensorGet(*dstI, i) / scale)
      i = i + 1
    Wend
  EndIf
EndProcedure

Procedure.i PmTensorStft(*g.PmTensorStftArgs)
  Protected bn.i
  Protected frame.i
  Protected bin.i
  Protected sample.i
  Protected signalIndex.i
  Protected outputIndex.i
  Protected complexCount.i
  Protected bits.i
  Protected reversed.i
  Protected mirror.i
  Protected *work1R
  Protected *work1I
  Protected *work2R
  Protected *work2I
  Protected *bR
  Protected *bI
  Protected *chirpR
  Protected *chirpI
  Protected *rootsR
  Protected *rootsI
  Protected value.f
  Protected sourceR.f
  Protected sourceI.f
  Protected weightR.f
  Protected weightI.f
  Protected resultR.f
  Protected resultI.f
  Protected pi.f
  Protected tau.f
  Protected angular.f
  Protected exponent.f
  Protected angle.f
  Protected complexFloat.f
  Protected sampleFloat.f
  Protected frameLengthFloat.f
  If *g = 0 Or *g\Signal = 0 Or *g\Dst = 0 Or *g\Scratch = 0
    ProcedureReturn 0
  EndIf
  If *g\Batches <= 0 Or *g\SignalLength < *g\FrameLength Or *g\FrameStep <= 0
    ProcedureReturn 0
  EndIf
  If *g\FrameLength <= 0 Or *g\Frames <= 0 Or *g\Bins <> (*g\FrameLength >> 1) + 1
    ProcedureReturn 0
  EndIf
  If *g\Frames > ((*g\SignalLength - *g\FrameLength) / *g\FrameStep) + 1
    ProcedureReturn 0
  EndIf
  complexCount = 1
  While complexCount < *g\FrameLength * 2 - 1 : complexCount = complexCount << 1 : Wend
  If complexCount > *g\ScratchComplex : ProcedureReturn 0 : EndIf
  *work1R = *g\Scratch
  *work1I = *work1R + complexCount * 4
  *work2R = *work1I + complexCount * 4
  *work2I = *work2R + complexCount * 4
  *bR = *work2I + complexCount * 4
  *bI = *bR + complexCount * 4
  *chirpR = *bI + complexCount * 4
  *chirpI = *chirpR + complexCount * 4
  *rootsR = *chirpI + complexCount * 4
  *rootsI = *rootsR + complexCount * 4

  pi = 3.141592653589793
  tau = 6.283185307179586
  complexFloat = complexCount
  frameLengthFloat = *g\FrameLength
  angular = (tau * -1.0) / complexFloat
  sample = complexCount
  While sample > 1 : bits = bits + 1 : sample = sample >> 1 : Wend
  sample = 0
  While sample < complexCount
    reversed = PmTensorStftBitReverse(sample, bits)
    sampleFloat = sample
    angle = sampleFloat * angular
    PmTensorPut(*rootsR, reversed, Cos(angle))
    PmTensorPut(*rootsI, reversed, Sin(angle))
    sample = sample + 1
  Wend

  PmTensorZero(*work1R, complexCount)
  PmTensorZero(*work1I, complexCount)
  PmTensorZero(*chirpR, complexCount)
  PmTensorZero(*chirpI, complexCount)
  sample = 0
  While sample < *g\FrameLength
    exponent = pi * -1.0
    sampleFloat = sample
    exponent = exponent * sampleFloat
    exponent = exponent * sampleFloat
    exponent = exponent / frameLengthFloat
    weightR = Cos(exponent)
    weightI = Sin(exponent)
    PmTensorPut(*chirpR, sample, weightR)
    PmTensorPut(*chirpI, sample, weightI)
    PmTensorPut(*work1R, sample, weightR)
    PmTensorPut(*work1I, sample, weightI * -1.0)
    sample = sample + 1
  Wend
  sample = complexCount - *g\FrameLength + 1
  While sample < complexCount
    mirror = complexCount - sample
    PmTensorPut(*work1R, sample, PmTensorGet(*work1R, mirror))
    PmTensorPut(*work1I, sample, PmTensorGet(*work1I, mirror))
    sample = sample + 1
  Wend
  PmTensorStftFft(*work1R, *work1I, *bR, *bI, *rootsR, *rootsI, complexCount, 0)

  bn = 0
  While bn < *g\Batches
    frame = 0
    While frame < *g\Frames
      PmTensorZero(*work1R, complexCount)
      PmTensorZero(*work1I, complexCount)
      sample = 0
      While sample < *g\FrameLength
        signalIndex = bn * *g\SignalLength + frame * *g\FrameStep + sample
        value = PmTensorGet(*g\Signal, signalIndex)
        If *g\Window <> 0 : value = value * PmTensorGet(*g\Window, sample) : EndIf
        weightR = PmTensorGet(*chirpR, sample)
        weightI = PmTensorGet(*chirpI, sample)
        PmTensorPut(*work1R, sample, value * weightR)
        PmTensorPut(*work1I, sample, value * weightI)
        sample = sample + 1
      Wend
      PmTensorStftFft(*work1R, *work1I, *work2R, *work2I, *rootsR, *rootsI, complexCount, 0)
      sample = 0
      While sample < complexCount
        sourceR = PmTensorGet(*work2R, sample)
        sourceI = PmTensorGet(*work2I, sample)
        weightR = PmTensorGet(*bR, sample)
        weightI = PmTensorGet(*bI, sample)
        resultR = sourceR * weightR - sourceI * weightI
        resultI = sourceR * weightI + sourceI * weightR
        PmTensorPut(*work2R, sample, resultR)
        PmTensorPut(*work2I, sample, resultI)
        sample = sample + 1
      Wend
      PmTensorStftFft(*work2R, *work2I, *work1R, *work1I, *rootsR, *rootsI, complexCount, 1)
      bin = 0
      While bin < *g\Bins
        mirror = bin
        If mirror > 0 : mirror = complexCount - mirror : EndIf
        sourceR = PmTensorGet(*work1R, mirror)
        sourceI = PmTensorGet(*work1I, mirror)
        weightR = PmTensorGet(*chirpR, bin)
        weightI = PmTensorGet(*chirpI, bin)
        resultR = sourceR * weightR - sourceI * weightI
        resultI = sourceR * weightI + sourceI * weightR
        outputIndex = ((bn * *g\Frames + frame) * *g\Bins + bin) * 2
        PmTensorPut(*g\Dst, outputIndex, resultR)
        PmTensorPut(*g\Dst, outputIndex + 1, resultI)
        bin = bin + 1
      Wend
      frame = frame + 1
    Wend
    bn = bn + 1
  Wend
  ProcedureReturn 1
EndProcedure
CompilerEndIf

; Resize the final axis of a contiguous FLOAT tensor. mode 0 is nearest/floor,
; mode 1 is linear. coordinateMode 0 is asymmetric, 1 is half_pixel.
CompilerIf #PMO_USE_RESIZE = 1
Procedure PmTensorResize1D(*src, *dst, outer.i, inWidth.i, outWidth.i, mode.i, coordinateMode.i, scale.f)
  Protected row.i
  Protected ox.i
  Protected ix0.i
  Protected ix1.i
  Protected source.f
  Protected base.i
  Protected low.f
  Protected high.f
  Protected fraction.f
  Protected whole.f
  row = 0
  While row < outer
    base = row * inWidth
    ox = 0
    While ox < outWidth
      If coordinateMode = 1
        source = ox
        source = (source + 0.5) / scale - 0.5
      Else
        source = ox
        source = source / scale
      EndIf
      ix0 = Int(source)
      whole = ix0
      If source < 0.0 And whole <> source : ix0 = ix0 - 1 : EndIf
      If mode = 0
        If ix0 < 0 : ix0 = 0 : EndIf
        If ix0 >= inWidth : ix0 = inWidth - 1 : EndIf
        PmTensorPut(*dst, row * outWidth + ox, PmTensorGet(*src, base + ix0))
      Else
        ix1 = ix0 + 1
        whole = ix0
        fraction = source - whole
        If ix0 < 0 : ix0 = 0 : EndIf
        If ix1 < 0 : ix1 = 0 : EndIf
        If ix0 >= inWidth : ix0 = inWidth - 1 : EndIf
        If ix1 >= inWidth : ix1 = inWidth - 1 : EndIf
        low = PmTensorGet(*src, base + ix0)
        high = PmTensorGet(*src, base + ix1)
        PmTensorPut(*dst, row * outWidth + ox, low + (high - low) * fraction)
      EndIf
      ox = ox + 1
    Wend
    row = row + 1
  Wend
EndProcedure
CompilerEndIf

; ONNX 1-D transposed convolution. W is [inChannels,
; outChannels/group, kernel]. The output is initialized with bias before the
; input contributions are scattered into it.
Structure PmTensorConvTranspose1DArgs
  Src.i
  Weight.i
  Bias.i
  Dst.i
  Batches.i
  InChannels.i
  InWidth.i
  OutChannels.i
  OutWidth.i
  Kernel.i
  PadLeft.i
  Stride.i
  Dilation.i
  Groups.i
EndStructure

CompilerIf #PMO_USE_CONVTRANSPOSE = 1
Procedure PmTensorConvTranspose1D(*g.PmTensorConvTranspose1DArgs)
  Protected bn.i
  Protected ic.i
  Protected ix.i
  Protected ocg.i
  Protected oc.i
  Protected ox.i
  Protected kx.i
  Protected group.i
  Protected inPerGroup.i
  Protected outPerGroup.i
  Protected srcIndex.i
  Protected weightIndex.i
  Protected dstIndex.i
  Protected value.f
  inPerGroup = *g\InChannels / *g\Groups
  outPerGroup = *g\OutChannels / *g\Groups
  bn = 0
  While bn < *g\Batches
    oc = 0
    While oc < *g\OutChannels
      ox = 0
      While ox < *g\OutWidth
        value = 0.0
        If *g\Bias <> 0 : value = PmTensorGet(*g\Bias, oc) : EndIf
        PmTensorPut(*g\Dst, (bn * *g\OutChannels + oc) * *g\OutWidth + ox, value)
        ox = ox + 1
      Wend
      oc = oc + 1
    Wend
    ic = 0
    While ic < *g\InChannels
      group = ic / inPerGroup
      ix = 0
      While ix < *g\InWidth
        srcIndex = (bn * *g\InChannels + ic) * *g\InWidth + ix
        value = PmTensorGet(*g\Src, srcIndex)
        ocg = 0
        While ocg < outPerGroup
          oc = group * outPerGroup + ocg
          kx = 0
          While kx < *g\Kernel
            ox = ix * *g\Stride - *g\PadLeft + kx * *g\Dilation
            If ox >= 0 And ox < *g\OutWidth
              weightIndex = (ic * outPerGroup + ocg) * *g\Kernel + kx
              dstIndex = (bn * *g\OutChannels + oc) * *g\OutWidth + ox
              PmTensorPut(*g\Dst, dstIndex, PmTensorGet(*g\Dst, dstIndex) + value * PmTensorGet(*g\Weight, weightIndex))
            EndIf
            kx = kx + 1
          Wend
          ocg = ocg + 1
        Wend
        ix = ix + 1
      Wend
      ic = ic + 1
    Wend
    bn = bn + 1
  Wend
EndProcedure
CompilerEndIf

; NCW 1-D convolution. W is [outChannels, inChannels/group, kernel].
Structure PmTensorConv1DArgs
  Src.i
  Weight.i
  Bias.i
  Dst.i
  Batches.i
  InChannels.i
  InWidth.i
  OutChannels.i
  OutWidth.i
  Kernel.i
  PadLeft.i
  Stride.i
  Dilation.i
  Groups.i
  WeightScales.i
EndStructure

CompilerIf #PMO_USE_CONV = 1
CompilerIf #PMO_USE_INT8 = 1
; Eight neighboring outputs, exact signed products, no additional scratch.
Procedure PmTensorConv1DInt8Eight(src.i, weights.i, dst.i, channels.i, width.i, kernel.i, dilation.i, aScale.f, wScale.f, bias.f)
  Protected fa.f=aScale, fw.f=wScale, fb.f=bias
  CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
    !mov r8,[p.v_src]
    !mov rdx,[p.v_weights]
    !mov r9,[p.v_width]
    !mov r10,[p.v_dilation]
    !pxor xmm0,xmm0
    !pxor xmm1,xmm1
    !pmint8conv_channel:
    !mov rax,r8
    !mov rcx,[p.v_kernel]
    !pmint8conv_kernel:
    !movq xmm2,[rax]
    !punpcklbw xmm2,xmm2
    !psraw xmm2,8
    !movzx r11d,byte [rdx]
    !movd xmm3,r11d
    !punpcklbw xmm3,xmm3
    !psraw xmm3,8
    !pshuflw xmm3,xmm3,0
    !pshufd xmm3,xmm3,0
    !pmullw xmm2,xmm3
    !movdqa xmm4,xmm2
    !psraw xmm4,15
    !movdqa xmm5,xmm2
    !punpcklwd xmm2,xmm4
    !punpckhwd xmm5,xmm4
    !paddd xmm0,xmm2
    !paddd xmm1,xmm5
    !add rax,r10
    !inc rdx
    !dec rcx
    !jnz pmint8conv_kernel
    !add r8,r9
    !dec qword [p.v_channels]
    !jnz pmint8conv_channel
    ; Match scalar host intermediates: rounded FP32 sum, then FP64 scale/bias.
    !cvtdq2ps xmm0,xmm0
    !cvtdq2ps xmm1,xmm1
    !cvtss2sd xmm3,[p.v_fa]
    !shufpd xmm3,xmm3,0
    !mov rax,[p.v_dst]
    !cvtss2sd xmm4,[p.v_fw]
    !shufpd xmm4,xmm4,0
    !cvtss2sd xmm5,[p.v_fb]
    !shufpd xmm5,xmm5,0
    !cvtps2pd xmm2,xmm0
    !mulpd xmm2,xmm3
    !mulpd xmm2,xmm4
    !addpd xmm2,xmm5
    !cvtpd2ps xmm2,xmm2
    !movq [rax+0],xmm2
    !psrldq xmm0,8
    !cvtps2pd xmm2,xmm0
    !mulpd xmm2,xmm3
    !mulpd xmm2,xmm4
    !addpd xmm2,xmm5
    !cvtpd2ps xmm2,xmm2
    !movq [rax+8],xmm2
    !cvtps2pd xmm2,xmm1
    !mulpd xmm2,xmm3
    !mulpd xmm2,xmm4
    !addpd xmm2,xmm5
    !cvtpd2ps xmm2,xmm2
    !movq [rax+16],xmm2
    !psrldq xmm1,8
    !cvtps2pd xmm2,xmm1
    !mulpd xmm2,xmm3
    !mulpd xmm2,xmm4
    !addpd xmm2,xmm5
    !cvtpd2ps xmm2,xmm2
    !movq [rax+24],xmm2
  CompilerEndIf
EndProcedure

; Padding is clipped once per output, not checked for every multiply.
Procedure PmTensorConv1DInt8Range(*g.PmTensorConv1DArgs, first.i, last.i, aScale.f)
  Protected work.i
  Protected bn.i
  Protected oc.i
  Protected ox.i
  Protected start.i
  Protected beginK.i
  Protected endK.i
  Protected ic.i
  Protected channels.i
  Protected groupBase.i
  Protected srcBase.i
  Protected weightBase.i
  Protected total.i
  Protected value.f
  Protected bias.f
  Protected wScale.f
  Protected dot.PmTensorInt8DotArgs
  channels=*g\InChannels / *g\Groups
  dot\StepA=*g\Dilation : dot\StepB=1
  work=first
  While work<last
    bn=work / *g\OutChannels : oc=work % *g\OutChannels
    groupBase=(oc / (*g\OutChannels / *g\Groups))*channels
    srcBase=PmTensorInt8Scratch+(bn* *g\InChannels+groupBase)* *g\InWidth
    weightBase=*g\Weight+oc*channels* *g\Kernel
    bias=0.0 : If *g\Bias : bias=PmTensorGet(*g\Bias,oc) : EndIf
    wScale=PmTensorGet(*g\WeightScales,oc)
    ox=0
    While ox<*g\OutWidth
      start=ox* *g\Stride-*g\PadLeft
      CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64
      If *g\Stride=1 And start>=0 And ox+7<*g\OutWidth And start+7+(*g\Kernel-1)* *g\Dilation<*g\InWidth
        PmTensorConv1DInt8Eight(srcBase+start,weightBase,*g\Dst+(work* *g\OutWidth+ox)*4,channels,*g\InWidth,*g\Kernel,*g\Dilation,aScale,wScale,bias)
        ox=ox+8
        Continue
      EndIf
      CompilerEndIf
      beginK=0
      If start<0 : beginK=(0-start+*g\Dilation-1) / *g\Dilation : EndIf
      endK=*g\Kernel
      If start+(*g\Kernel-1)* *g\Dilation>=*g\InWidth
        endK=(*g\InWidth-1-start) / *g\Dilation+1
        If start>=*g\InWidth : endK=0 : EndIf
      EndIf
      dot\Count=endK-beginK : total=0 : ic=0
      If dot\Count>0
        dot\A=srcBase+start+beginK* *g\Dilation
        dot\B=weightBase+beginK
        While ic<channels
          total=total+PmTensorInt8Dot(@dot)
          dot\A=dot\A+*g\InWidth : dot\B=dot\B+*g\Kernel
          ic=ic+1
        Wend
      EndIf
      value=total : value=bias+value*aScale*wScale
      PmTensorPut(*g\Dst,work* *g\OutWidth+ox,value)
      ox=ox+1
    Wend
    work=work+1
  Wend
EndProcedure
CompilerEndIf
Structure PmTensorConv1DWorker
  Args.i
  WorkStart.i
  WorkEnd.i
  AScale.f
EndStructure

Procedure PmTensorConv1DWorker(*worker.PmTensorConv1DWorker)
  Protected *g.PmTensorConv1DArgs = *worker\Args
  Protected work.i
  Protected bn.i
  Protected oc.i
  Protected ox.i
  Protected icg.i
  Protected kx.i
  Protected ix.i
  Protected group.i
  Protected inPerGroup.i
  Protected outPerGroup.i
  Protected ic.i
  Protected srcIndex.i
  Protected weightIndex.i
  Protected accumulator.i
  Protected converted.f
  Protected sum.f
  inPerGroup = *g\InChannels / *g\Groups
  outPerGroup = *g\OutChannels / *g\Groups
  work = *worker\WorkStart
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales
    PmTensorConv1DInt8Range(*g,*worker\WorkStart,*worker\WorkEnd,*worker\AScale)
    ProcedureReturn
  EndIf
  CompilerEndIf
  While work < *worker\WorkEnd
    bn = work / *g\OutChannels
    oc = work % *g\OutChannels
    group = oc / outPerGroup
    ox = 0
    While ox < *g\OutWidth
      sum = 0.0 : accumulator = 0
      If *g\Bias <> 0 : sum = PmTensorGet(*g\Bias, oc) : EndIf
      icg = 0
      While icg < inPerGroup
        ic = group * inPerGroup + icg
        kx = 0
        While kx < *g\Kernel
          ix = ox * *g\Stride - *g\PadLeft + kx * *g\Dilation
          If ix >= 0 And ix < *g\InWidth
            srcIndex = (bn * *g\InChannels + ic) * *g\InWidth + ix
            weightIndex = (oc * inPerGroup + icg) * *g\Kernel + kx
              sum = sum + PmTensorGet(*g\Src, srcIndex) * PmTensorGet(*g\Weight, weightIndex)
          EndIf
          kx = kx + 1
        Wend
        icg = icg + 1
      Wend
      PmTensorPut(*g\Dst, (bn * *g\OutChannels + oc) * *g\OutWidth + ox, sum)
      ox = ox + 1
    Wend
    work = work + 1
  Wend
EndProcedure

Procedure PmTensorConv1D(*g.PmTensorConv1DArgs)
  Protected workerCount.i = CountCPUs(#PB_System_ProcessCPUs)
  Protected workCount.i = *g\Batches * *g\OutChannels
  Protected chunk.i
  Protected first.i
  Protected index.i
  Protected totalMacs.q
  Protected aScale.f
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    aScale = PmTensorDynamicScale(*g\Src, *g\Batches * *g\InChannels * *g\InWidth)
    PmTensorQuantizeBuffer(*g\Src, PmTensorInt8Scratch, *g\Batches * *g\InChannels * *g\InWidth, aScale)
  EndIf
  CompilerEndIf
  totalMacs = workCount : totalMacs * *g\OutWidth : totalMacs * (*g\InChannels / *g\Groups) : totalMacs * *g\Kernel
  If workerCount > workCount : workerCount = workCount : EndIf
  If totalMacs < 1000000 Or workerCount < 2 : workerCount = 1 : EndIf
  Dim workers.PmTensorConv1DWorker(workerCount - 1)
  Dim threads.i(workerCount - 1)
  chunk = (workCount + workerCount - 1) / workerCount
  For index = 0 To workerCount - 1
    first = index * chunk
    workers(index)\Args = *g : workers(index)\WorkStart = first
    workers(index)\WorkEnd = first + chunk : workers(index)\AScale = aScale
    If workers(index)\WorkEnd > workCount : workers(index)\WorkEnd = workCount : EndIf
    If workerCount = 1 Or workers(index)\WorkStart >= workers(index)\WorkEnd
      PmTensorConv1DWorker(@workers(index))
    Else
      threads(index) = CreateThread(@PmTensorConv1DWorker(), @workers(index))
      If threads(index) = 0 : PmTensorConv1DWorker(@workers(index)) : EndIf
    EndIf
  Next
  For index = 0 To workerCount - 1
    If threads(index) : WaitThread(threads(index)) : EndIf
  Next
EndProcedure
CompilerEndIf

; NCHW 2-D convolution. W is [outChannels, inChannels/group, kH, kW].
Structure PmTensorConv2DArgs
  Src.i
  Weight.i
  Bias.i
  Dst.i
  Batches.i
  InChannels.i
  InH.i
  InW.i
  OutChannels.i
  OutH.i
  OutW.i
  KernelH.i
  KernelW.i
  PadTop.i
  PadLeft.i
  StrideH.i
  StrideW.i
  DilationH.i
  DilationW.i
  Groups.i
  WeightScales.i
EndStructure

CompilerIf #PMO_USE_CONV = 1
CompilerIf #PMO_USE_INT8 = 1
Procedure PmTensorConv2DInt8(*g.PmTensorConv2DArgs, aScale.f)
  Protected bn.i
  Protected oc.i
  Protected oy.i
  Protected ox.i
  Protected ic.i
  Protected ky.i
  Protected sx.i
  Protected sy.i
  Protected bx.i
  Protected ex.i
  Protected by.i
  Protected ey.i
  Protected channels.i
  Protected srcBase.i
  Protected weightBase.i
  Protected sourceRow.i
  Protected weightRow.i
  Protected total.i
  Protected value.f
  Protected bias.f
  Protected wScale.f
  Protected dot.PmTensorInt8DotArgs
  channels=*g\InChannels / *g\Groups
  dot\StepA=*g\DilationW : dot\StepB=1
  bn=0
  While bn<*g\Batches
    oc=0
    While oc<*g\OutChannels
      srcBase=PmTensorInt8Scratch+(bn* *g\InChannels+(oc / (*g\OutChannels / *g\Groups))*channels)* *g\InH* *g\InW
      weightBase=*g\Weight+oc*channels* *g\KernelH* *g\KernelW
      bias=0.0 : If *g\Bias : bias=PmTensorGet(*g\Bias,oc) : EndIf
      wScale=PmTensorGet(*g\WeightScales,oc)
      oy=0
      While oy<*g\OutH
        sy=oy* *g\StrideH-*g\PadTop : by=0 : ey=*g\KernelH
        If sy<0 : by=(0-sy+*g\DilationH-1) / *g\DilationH : EndIf
        If sy+(*g\KernelH-1)* *g\DilationH>=*g\InH
          ey=(*g\InH-1-sy) / *g\DilationH+1
          If sy>=*g\InH : ey=0 : EndIf
        EndIf
        ox=0
        While ox<*g\OutW
          sx=ox* *g\StrideW-*g\PadLeft : bx=0 : ex=*g\KernelW
          If sx<0 : bx=(0-sx+*g\DilationW-1) / *g\DilationW : EndIf
          If sx+(*g\KernelW-1)* *g\DilationW>=*g\InW
            ex=(*g\InW-1-sx) / *g\DilationW+1
            If sx>=*g\InW : ex=0 : EndIf
          EndIf
          dot\Count=ex-bx : total=0 : ic=0
          If dot\Count>0 And ey>by
            sourceRow=srcBase+(sy+by* *g\DilationH)* *g\InW+sx+bx* *g\DilationW
            weightRow=weightBase+by* *g\KernelW+bx
            While ic<channels
              dot\A=sourceRow : dot\B=weightRow : ky=by
              While ky<ey
                total=total+PmTensorInt8Dot(@dot)
                dot\A=dot\A+*g\DilationH* *g\InW : dot\B=dot\B+*g\KernelW
                ky=ky+1
              Wend
              sourceRow=sourceRow+*g\InH* *g\InW
              weightRow=weightRow+*g\KernelH* *g\KernelW
              ic=ic+1
            Wend
          EndIf
          value=total : value=bias+value*aScale*wScale
          PmTensorPut(*g\Dst,((bn* *g\OutChannels+oc)* *g\OutH+oy)* *g\OutW+ox,value)
          ox=ox+1
        Wend
        oy=oy+1
      Wend
      oc=oc+1
    Wend
    bn=bn+1
  Wend
EndProcedure
CompilerEndIf

Procedure PmTensorConv2D(*g.PmTensorConv2DArgs)
  Protected bn.i
  Protected oc.i
  Protected oy.i
  Protected ox.i
  Protected icg.i
  Protected ky.i
  Protected kx.i
  Protected iy.i
  Protected ix.i
  Protected group.i
  Protected inPerGroup.i
  Protected outPerGroup.i
  Protected ic.i
  Protected srcIndex.i
  Protected weightIndex.i
  Protected dstIndex.i
  Protected accumulator.i
  Protected aScale.f
  Protected converted.f
  Protected sum.f
  inPerGroup = *g\InChannels / *g\Groups
  outPerGroup = *g\OutChannels / *g\Groups
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    aScale = PmTensorDynamicScale(*g\Src, *g\Batches * *g\InChannels * *g\InH * *g\InW)
    PmTensorQuantizeBuffer(*g\Src, PmTensorInt8Scratch, *g\Batches * *g\InChannels * *g\InH * *g\InW, aScale)
    PmTensorConv2DInt8(*g,aScale)
    ProcedureReturn
  EndIf
  CompilerEndIf
  bn = 0
  While bn < *g\Batches
    oc = 0
    While oc < *g\OutChannels
      group = oc / outPerGroup
      oy = 0
      While oy < *g\OutH
        ox = 0
        While ox < *g\OutW
          sum = 0.0 : accumulator = 0
          If *g\Bias <> 0 : sum = PmTensorGet(*g\Bias, oc) : EndIf
          icg = 0
          While icg < inPerGroup
            ic = group * inPerGroup + icg
            ky = 0
            While ky < *g\KernelH
              iy = oy * *g\StrideH - *g\PadTop + ky * *g\DilationH
              If iy >= 0 And iy < *g\InH
                kx = 0
                While kx < *g\KernelW
                  ix = ox * *g\StrideW - *g\PadLeft + kx * *g\DilationW
                  If ix >= 0 And ix < *g\InW
                    srcIndex = ((bn * *g\InChannels + ic) * *g\InH + iy) * *g\InW + ix
                    weightIndex = ((oc * inPerGroup + icg) * *g\KernelH + ky) * *g\KernelW + kx
                      sum = sum + PmTensorGet(*g\Src, srcIndex) * PmTensorGet(*g\Weight, weightIndex)
                  EndIf
                  kx = kx + 1
                Wend
              EndIf
              ky = ky + 1
            Wend
            icg = icg + 1
          Wend
          dstIndex = ((bn * *g\OutChannels + oc) * *g\OutH + oy) * *g\OutW + ox
          PmTensorPut(*g\Dst, dstIndex, sum)
          ox = ox + 1
        Wend
        oy = oy + 1
      Wend
      oc = oc + 1
    Wend
    bn = bn + 1
  Wend
EndProcedure
CompilerEndIf

; ONNX LSTM, layout=0, default activation triplet, without peepholes.
; Gate order in ONNX weights is input, output, forget, cell (iofc).
Structure PmTensorLstmArgs
  X.i
  W.i
  R.i
  B.i
  SeqLens.i
  InitialH.i
  InitialC.i
  Y.i
  YH.i
  YC.i
  Sequence.i
  Batch.i
  InputSize.i
  Hidden.i
  Directions.i
  WScales.i
  RScales.i
EndStructure

CompilerIf #PMO_USE_LSTM = 1
Structure PmTensorLstmFloatFourArgs
  A.i
  B.i
  Count.i
  Hidden.i
  Unit.i
  IValue.f
  OValue.f
  FValue.f
  CValue.f
EndStructure

Procedure PmTensorLstmFloatFour(*g.PmTensorLstmFloatFourArgs)
  Protected gate.i
  Protected index.i
  Protected offset.i
  Protected left.f
  Protected right.f
  Protected sum.f
  gate = 0
  While gate < 4
    offset = gate : offset = offset * *g\Hidden
    offset = offset + *g\Unit : offset = offset * *g\Count
    index = 0 : sum = 0.0
    While index < *g\Count
      left = PmTensorGet(*g\A, index)
      right = PmTensorGet(*g\B, offset + index)
      sum = sum + left * right
      index = index + 1
    Wend
    Select gate
      Case 0 : *g\IValue = sum
      Case 1 : *g\OValue = sum
      Case 2 : *g\FValue = sum
      Case 3 : *g\CValue = sum
    EndSelect
    gate = gate + 1
  Wend
EndProcedure

Procedure PmTensorLstm(*g.PmTensorLstmArgs)
  Protected dir.i
  Protected timeStep.i
  Protected t.i
  Protected bn.i
  Protected unit.i
  Protected k.i
  Protected stateIndex.i
  Protected xIndex.i
  Protected wBase.i
  Protected rBase.i
  Protected bBase.i
  Protected yIndex.i
  Protected valid.i
  Protected iv.f
  Protected ov.f
  Protected fv.f
  Protected cv.f
  Protected previousC.f
  Protected newC.f
  Protected xScale.f
  Protected hScale.f
  Protected leftPointer.i
  Protected rightPointer.i
  Protected dotValue.f
  Protected dot.PmTensorDotInt8Args
  Protected floatFour.PmTensorLstmFloatFourArgs

  ; Initial state becomes the mutable final-state buffers.
  dir = 0
  While dir < *g\Directions
    bn = 0
    While bn < *g\Batch
      unit = 0
      While unit < *g\Hidden
        stateIndex = (dir * *g\Batch + bn) * *g\Hidden + unit
        If *g\InitialH <> 0
          PmTensorPut(*g\YH, stateIndex, PmTensorGet(*g\InitialH, stateIndex))
        Else
          PmTensorPut(*g\YH, stateIndex, 0.0)
        EndIf
        If *g\InitialC <> 0
          PmTensorPut(*g\YC, stateIndex, PmTensorGet(*g\InitialC, stateIndex))
        Else
          PmTensorPut(*g\YC, stateIndex, 0.0)
        EndIf
        unit = unit + 1
      Wend
      bn = bn + 1
    Wend
    dir = dir + 1
  Wend

  dir = 0
  While dir < *g\Directions
    timeStep = 0
    While timeStep < *g\Sequence
      bn = 0
      While bn < *g\Batch
        valid = *g\Sequence
        If *g\SeqLens <> 0 : valid = PmTensorGetI64(*g\SeqLens, bn) : EndIf
        If dir = 0
          t = timeStep
        Else
          t = valid - 1 - timeStep
        EndIf
        If timeStep < valid
          xIndex = (t * *g\Batch + bn) * *g\InputSize
          stateIndex = (dir * *g\Batch + bn) * *g\Hidden
          CompilerIf #PMO_USE_INT8 = 1
          If *g\WScales <> 0
            leftPointer = *g\X : leftPointer = leftPointer + xIndex * 4
            xScale = PmTensorDynamicScale(leftPointer, *g\InputSize)
            PmTensorQuantizeBuffer(leftPointer, PmTensorInt8Scratch, *g\InputSize, xScale)
          EndIf
          If *g\RScales <> 0
            leftPointer = *g\YH : leftPointer = leftPointer + stateIndex * 4
            hScale = PmTensorDynamicScale(leftPointer, *g\Hidden)
            PmTensorQuantizeBuffer(leftPointer, PmTensorInt8Scratch + *g\InputSize, *g\Hidden, hScale)
          EndIf
          CompilerEndIf
          unit = 0
          While unit < *g\Hidden
            iv = 0.0 : ov = 0.0 : fv = 0.0 : cv = 0.0
            bBase = dir * 8 * *g\Hidden
            If *g\B <> 0
              iv = PmTensorGet(*g\B, bBase + unit) + PmTensorGet(*g\B, bBase + 4 * *g\Hidden + unit)
              ov = PmTensorGet(*g\B, bBase + *g\Hidden + unit) + PmTensorGet(*g\B, bBase + 5 * *g\Hidden + unit)
              fv = PmTensorGet(*g\B, bBase + 2 * *g\Hidden + unit) + PmTensorGet(*g\B, bBase + 6 * *g\Hidden + unit)
              cv = PmTensorGet(*g\B, bBase + 3 * *g\Hidden + unit) + PmTensorGet(*g\B, bBase + 7 * *g\Hidden + unit)
            EndIf
            wBase = (dir * 4 * *g\Hidden) * *g\InputSize
            CompilerIf #PMO_USE_INT8 = 1
            If *g\WScales <> 0
              dot\A = PmTensorInt8Scratch : dot\Count = *g\InputSize : dot\AScale = xScale
              rightPointer = *g\W : rightPointer = rightPointer + wBase + unit * *g\InputSize
              dot\WeightScale = PmTensorGet(*g\WScales, dir * 4 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : iv = iv + dotValue
              rightPointer = *g\W : rightPointer = rightPointer + wBase + (*g\Hidden + unit) * *g\InputSize
              dot\WeightScale = PmTensorGet(*g\WScales, dir * 4 * *g\Hidden + *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : ov = ov + dotValue
              rightPointer = *g\W : rightPointer = rightPointer + wBase + (2 * *g\Hidden + unit) * *g\InputSize
              dot\WeightScale = PmTensorGet(*g\WScales, dir * 4 * *g\Hidden + 2 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : fv = fv + dotValue
              rightPointer = *g\W : rightPointer = rightPointer + wBase + (3 * *g\Hidden + unit) * *g\InputSize
              dot\WeightScale = PmTensorGet(*g\WScales, dir * 4 * *g\Hidden + 3 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : cv = cv + dotValue
            Else
              leftPointer = *g\X : k = xIndex : k = k * 4 : leftPointer = leftPointer + k
              rightPointer = *g\W : k = wBase : k = k * 4 : rightPointer = rightPointer + k
              floatFour\A = leftPointer : floatFour\B = rightPointer
              floatFour\Count = *g\InputSize : floatFour\Hidden = *g\Hidden : floatFour\Unit = unit
              PmTensorLstmFloatFour(@floatFour)
              iv = iv + floatFour\IValue : ov = ov + floatFour\OValue
              fv = fv + floatFour\FValue : cv = cv + floatFour\CValue
            EndIf
            CompilerElse
              leftPointer = *g\X : k = xIndex : k = k * 4 : leftPointer = leftPointer + k
              rightPointer = *g\W : k = wBase : k = k * 4 : rightPointer = rightPointer + k
              floatFour\A = leftPointer : floatFour\B = rightPointer
              floatFour\Count = *g\InputSize : floatFour\Hidden = *g\Hidden : floatFour\Unit = unit
              PmTensorLstmFloatFour(@floatFour)
              iv = iv + floatFour\IValue : ov = ov + floatFour\OValue
              fv = fv + floatFour\FValue : cv = cv + floatFour\CValue
            CompilerEndIf
            stateIndex = (dir * *g\Batch + bn) * *g\Hidden
            rBase = (dir * 4 * *g\Hidden) * *g\Hidden
            CompilerIf #PMO_USE_INT8 = 1
            If *g\RScales <> 0
              dot\A = PmTensorInt8Scratch + *g\InputSize : dot\Count = *g\Hidden : dot\AScale = hScale
              rightPointer = *g\R : rightPointer = rightPointer + rBase + unit * *g\Hidden
              dot\WeightScale = PmTensorGet(*g\RScales, dir * 4 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : iv = iv + dotValue
              rightPointer = *g\R : rightPointer = rightPointer + rBase + (*g\Hidden + unit) * *g\Hidden
              dot\WeightScale = PmTensorGet(*g\RScales, dir * 4 * *g\Hidden + *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : ov = ov + dotValue
              rightPointer = *g\R : rightPointer = rightPointer + rBase + (2 * *g\Hidden + unit) * *g\Hidden
              dot\WeightScale = PmTensorGet(*g\RScales, dir * 4 * *g\Hidden + 2 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : fv = fv + dotValue
              rightPointer = *g\R : rightPointer = rightPointer + rBase + (3 * *g\Hidden + unit) * *g\Hidden
              dot\WeightScale = PmTensorGet(*g\RScales, dir * 4 * *g\Hidden + 3 * *g\Hidden + unit)
              dot\B = rightPointer : dotValue = PmTensorDotInt8Scaled(@dot) : cv = cv + dotValue
            Else
              leftPointer = *g\YH : k = stateIndex : k = k * 4 : leftPointer = leftPointer + k
              rightPointer = *g\R : k = rBase : k = k * 4 : rightPointer = rightPointer + k
              floatFour\A = leftPointer : floatFour\B = rightPointer
              floatFour\Count = *g\Hidden : floatFour\Hidden = *g\Hidden : floatFour\Unit = unit
              PmTensorLstmFloatFour(@floatFour)
              iv = iv + floatFour\IValue : ov = ov + floatFour\OValue
              fv = fv + floatFour\FValue : cv = cv + floatFour\CValue
            EndIf
            CompilerElse
              leftPointer = *g\YH : k = stateIndex : k = k * 4 : leftPointer = leftPointer + k
              rightPointer = *g\R : k = rBase : k = k * 4 : rightPointer = rightPointer + k
              floatFour\A = leftPointer : floatFour\B = rightPointer
              floatFour\Count = *g\Hidden : floatFour\Hidden = *g\Hidden : floatFour\Unit = unit
              PmTensorLstmFloatFour(@floatFour)
              iv = iv + floatFour\IValue : ov = ov + floatFour\OValue
              fv = fv + floatFour\FValue : cv = cv + floatFour\CValue
            CompilerEndIf
            stateIndex = (dir * *g\Batch + bn) * *g\Hidden + unit
            previousC = PmTensorGet(*g\YC, stateIndex)
            iv = PmTensorSigmoidValue(iv)
            ov = PmTensorSigmoidValue(ov)
            fv = PmTensorSigmoidValue(fv)
            cv = PmTensorTanhValue(cv)
            newC = fv * previousC + iv * cv
            PmTensorPut(*g\YC, stateIndex, newC)
            yIndex = ((t * *g\Directions + dir) * *g\Batch + bn) * *g\Hidden + unit
            PmTensorPut(*g\Y, yIndex, ov * PmTensorTanhValue(newC))
            unit = unit + 1
          Wend
          ; Copy the completed timestep only after all recurrent reads used H(t-1).
          unit = 0
          While unit < *g\Hidden
            stateIndex = (dir * *g\Batch + bn) * *g\Hidden + unit
            yIndex = ((t * *g\Directions + dir) * *g\Batch + bn) * *g\Hidden + unit
            PmTensorPut(*g\YH, stateIndex, PmTensorGet(*g\Y, yIndex))
            unit = unit + 1
          Wend
        EndIf
        bn = bn + 1
      Wend
      timeStep = timeStep + 1
    Wend
    dir = dir + 1
  Wend
EndProcedure
CompilerEndIf
