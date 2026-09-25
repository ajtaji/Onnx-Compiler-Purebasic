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

; The worker pool every parallel operator below runs on.
XIncludeFile "tensor_pool_windows.pbi"

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

; NaN and the host compiler (forums 995, 997, 998). PureBasic before 6.41
; answers a float comparison with a NaN operand by the operand order and by
; whether a side is a literal (v < 0.0 and v <= 0.0 true, a > b, a >= b and
; a = b true, a <> b false); 6.41 answers as IEEE 754 does. Where a kernel's
; answer for a NaN rests on a comparison, #PMO_HOST_NAN_BUG = 1 compiles a test
; on the float's bits in front of it; with a correct compiler the plain
; comparison is compiled and costs nothing more. A program may define the
; constant itself first, to force either branch (the gates force 1).
CompilerIf Defined(PMO_HOST_NAN_BUG, #PB_Constant) = 0
  CompilerIf #PB_Compiler_Version < 641
    #PMO_HOST_NAN_BUG = 1
  CompilerElse
    #PMO_HOST_NAN_BUG = 0
  CompilerEndIf
CompilerEndIf

; Square root, correctly rounded (C6c-1): binary32 sqrt computed through
; binary64 and rounded once is correctly rounded, so Sqr() assigned to a
; binary32 variable is. Its NaN is x86's (a NaN argument quieted, FFC00000
; for a negative one), the definition every target reproduces.
Procedure.f PmTensorSqrtF(x.f)
  ; Correctly rounded (C6c-1): binary32 sqrt computed through binary64 and
  ; rounded once is. Its NaN is x86's - the argument quieted, or FFC00000 for
  ; a negative one - which is the definition every target reproduces.
  ProcedureReturn Sqr(x)
EndProcedure

Procedure.i PmTensorIsNan(v.f)
  ProcedureReturn Bool((PeekL(@v) & $7FFFFFFF) > $7F800000)
EndProcedure

; The element at index of a FLOAT tensor is a NaN.
Macro PmTensorNanAt(base, index)
  Bool((PeekL((base) + (index) * 4) & $7FFFFFFF) > $7F800000)
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

Procedure PmTensorZeroSerial(*dst, count.i)
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

Procedure PmTensorCopySerial(*src, *dst, count.i)
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

Procedure PmTensorAddSerial(*a, *b, *dst, count.i)
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

Procedure PmTensorSubSerial(*a, *b, *dst, count.i)
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

Procedure PmTensorMulSerial(*a, *b, *dst, count.i)
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

Procedure PmTensorDivSerial(*a, *b, *dst, count.i)
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
Procedure PmTensorBinaryScalarSerial(*vec, scalar.f, *dst, count.i, op.i, scalarSide.i)
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

Procedure PmTensorReluSerial(*src, *dst, count.i)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    CompilerIf #PMO_HOST_NAN_BUG = 1
      If (PeekL(ps) & $7FFFFFFF) <= $7F800000
        If v < 0.0 : v = 0.0 : EndIf
      EndIf
    CompilerElse
      If v < 0.0 : v = 0.0 : EndIf
    CompilerEndIf
    PokeF(pd, v)
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

Procedure PmTensorLeakyReluSerial(*src, *dst, count.i, alpha.f)
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
  CompilerIf #PMO_HOST_NAN_BUG = 1
    If PmTensorIsNan(value) <> 0 : ProcedureReturn value : EndIf
  CompilerEndIf
  If value > 10.0 : ProcedureReturn 1.0 : EndIf
  If value < -10.0 : ProcedureReturn -1.0 : EndIf
  e = Exp(value + value)
  ProcedureReturn (e - 1.0) / (e + 1.0)
EndProcedure

Procedure PmTensorSigmoidSerial(*src, *dst, count.i)
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

Procedure PmTensorTanhSerial(*src, *dst, count.i)
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
Procedure PmTensorUnaryMathSerial(*src, *dst, count.i, op.i)
  ; the op is decided once, outside the loop
  Protected i.i
  Protected v.f
  Protected mask.q = $7FFFFFFF7FFFFFFF
  Protected flip.q = $8000000080000000
  If op < 0 Or op > 4
    PmTensorUnaryMathOk = 0
    ProcedureReturn
  EndIf
  i = 0
  Select op
    Case 0
      While i < count : PokeF(*dst + i * 4, Exp(PeekF(*src + i * 4))) : i = i + 1 : Wend
    Case 1
      While i < count : PokeF(*dst + i * 4, Log(PeekF(*src + i * 4))) : i = i + 1 : Wend
    Case 2
      ; correctly rounded (C6c-1); the NaN of an invalid argument is x86's,
      ; which is the definition every target reproduces
      While i < count : PokeF(*dst + i * 4, Sqr(PeekF(*src + i * 4))) : i = i + 1 : Wend
    Case 3, 4
      ; IEEE 754 abs and negate: the sign bit alone (-0 has abs +0, a NaN
      ; keeps its payload), as numpy computes them; eight bytes at a time
      If op = 3
        While i + 2 <= count : PokeQ(*dst + i * 4, PeekQ(*src + i * 4) & mask) : i = i + 2 : Wend
        If i < count : PokeL(*dst + i * 4, PeekL(*src + i * 4) & $7FFFFFFF) : EndIf
      Else
        While i + 2 <= count : PokeQ(*dst + i * 4, PeekQ(*src + i * 4) ! flip) : i = i + 2 : Wend
        If i < count : PokeL(*dst + i * 4, PeekL(*src + i * 4) ! $80000000) : EndIf
      EndIf
  EndSelect
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
Procedure PmTensorTrigSerial(*src, *dst, count.i, op.i)
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

Procedure PmTensorFloorSerial(*src, *dst, count.i)
  Protected i.i
  Protected n.i
  Protected whole.f
  Protected value.f
  i = 0
  While i < count
    value = PmTensorGet(*src, i)
    If (PeekL(*src + i * 4) & $7FFFFFFF) >= $4B000000
      ; |value| >= 2^23, an infinity or a NaN: already its own floor
      PmTensorPut(*dst, i, value)
    Else
      n = Int(value)
      whole = n
      If value < 0.0 And whole <> value : n = n - 1 : EndIf
      whole = n
      PmTensorPut(*dst, i, whole)
    EndIf
    i = i + 1
  Wend
EndProcedure

; ONNX Round uses nearest integer with ties to even.
Procedure PmTensorRoundEvenSerial(*src, *dst, count.i)
  Protected i.i
  Protected base.i
  Protected result.i
  Protected fraction.f
  Protected whole.f
  Protected value.f
  i = 0
  While i < count
    value = PmTensorGet(*src, i)
    If (PeekL(*src + i * 4) & $7FFFFFFF) >= $4B000000
      ; |value| >= 2^23, an infinity or a NaN: already an integer
      PmTensorPut(*dst, i, value)
    Else
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
    EndIf
    i = i + 1
  Wend
EndProcedure

Procedure PmTensorPowSerial(*a, *b, *dst, count.i)
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

Procedure PmTensorPowScalarSerial(*vec, scalar.f, *dst, count.i, scalarSide.i)
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

Procedure PmTensorClipSerial(*src, *dst, count.i, lo.f, hi.f)
  Protected i.i
  Protected ps.i
  Protected pd.i
  Protected v.f
  i = 0 : ps = *src : pd = *dst
  While i < count
    v = PeekF(ps)
    ; a NaN passes through (forum 997)
    CompilerIf #PMO_HOST_NAN_BUG = 1
      If (PeekL(ps) & $7FFFFFFF) <= $7F800000
        If v < lo : v = lo : EndIf
        If v > hi : v = hi : EndIf
      EndIf
    CompilerElse
      If v < lo : v = lo : EndIf
      If v > hi : v = hi : EndIf
    CompilerEndIf
    PokeF(pd, v)
    ps = ps + 4 : pd = pd + 4 : i = i + 1
  Wend
EndProcedure

; ---- the element-wise operators on the pool ----------------------------------
; Each public name below cuts its elements into contiguous ranges and runs
; the serial kernel above on each range. Every element is computed by the
; same statement whichever range holds it, so any split gives the same bits.
; Grains (elements per task) were measured on the reference machine: a
; cheap element is one add, compare or copy; a dear one calls a library
; transcendental.
#PMELEM_CHEAP = 32768
#PMELEM_DEAR = 4096

Structure PmElemJob
  Kind.i
  A.i
  B.i
  Dst.i
  Count.i
  Chunk.i
  Op.i
  Side.i
  F1.f
  F2.f
EndStructure

; The loops of the runtime-dimension path's PmFastTrig (tensor_simd_windows.pbi),
; exactly as they were written there, for one range of elements.
Procedure PmFastTrigRange(*Src, *Dst, Count.i, Op.i)
  Protected i.i
  Select Op
    Case 0
      For i=0 To Count-1 : PokeF(*Dst+i*4,Sin(PeekF(*Src+i*4))) : Next
    Case 1
      For i=0 To Count-1 : PokeF(*Dst+i*4,Cos(PeekF(*Src+i*4))) : Next
    Case 2
      For i=0 To Count-1 : PokeF(*Dst+i*4,ATan(PeekF(*Src+i*4))) : Next
  EndSelect
EndProcedure

Procedure PmElemTask(*j.PmElemJob, task.i, worker.i)
  Protected first.i = task * *j\Chunk, n.i = *j\Count - first, a.i, b.i, d.i
  If n > *j\Chunk : n = *j\Chunk : EndIf
  If n <= 0 : ProcedureReturn : EndIf
  a = *j\A + first * 4 : b = *j\B + first * 4 : d = *j\Dst + first * 4
  Select *j\Kind
    Case 0 : PmTensorAddSerial(a, b, d, n)
    Case 1 : PmTensorSubSerial(a, b, d, n)
    Case 2 : PmTensorMulSerial(a, b, d, n)
    Case 3 : PmTensorDivSerial(a, b, d, n)
    Case 4 : PmTensorPowSerial(a, b, d, n)
    Case 5 : PmTensorBinaryScalarSerial(a, *j\F1, d, n, *j\Op, *j\Side)
    Case 6 : PmTensorPowScalarSerial(a, *j\F1, d, n, *j\Side)
    Case 7 : PmTensorReluSerial(a, d, n)
    Case 8 : PmTensorLeakyReluSerial(a, d, n, *j\F1)
    Case 9 : PmTensorSigmoidSerial(a, d, n)
    Case 10 : PmTensorTanhSerial(a, d, n)
    Case 11 : PmTensorUnaryMathSerial(a, d, n, *j\Op)
    Case 12 : PmTensorTrigSerial(a, d, n, *j\Op)
    Case 13 : PmTensorFloorSerial(a, d, n)
    Case 14 : PmTensorRoundEvenSerial(a, d, n)
    Case 15 : PmTensorClipSerial(a, d, n, *j\F1, *j\F2)
    Case 16 : PmTensorCopySerial(a, d, n)
    Case 17 : PmTensorZeroSerial(d, n)
    Case 18 : PmFastTrigRange(a, d, n, *j\Op)
  EndSelect
EndProcedure

Procedure PmElemRun(kind.i, *a, *b, *dst, count.i, op.i, side.i, f1.f, f2.f, grain.i)
  Protected j.PmElemJob, tasks.i
  j\Kind = kind : j\A = *a : j\B = *b : j\Dst = *dst : j\Count = count
  j\Op = op : j\Side = side : j\F1 = f1 : j\F2 = f2
  tasks = PmPoolTasks(count, grain, 16, @j\Chunk)
  If tasks <= 1
    j\Chunk = count : PmElemTask(@j, 0, 0)
  Else
    PmPoolRun(@PmElemTask(), @j, tasks)
  EndIf
EndProcedure

Procedure PmTensorZero(*dst, count.i)
  PmElemRun(17, 0, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorCopy(*src, *dst, count.i)
  PmElemRun(16, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorAdd(*a, *b, *dst, count.i)
  PmElemRun(0, *a, *b, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorSub(*a, *b, *dst, count.i)
  PmElemRun(1, *a, *b, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorMul(*a, *b, *dst, count.i)
  PmElemRun(2, *a, *b, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorDiv(*a, *b, *dst, count.i)
  PmElemRun(3, *a, *b, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorPow(*a, *b, *dst, count.i)
  PmElemRun(4, *a, *b, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_DEAR)
EndProcedure
Procedure PmTensorBinaryScalar(*vec, scalar.f, *dst, count.i, op.i, scalarSide.i)
  PmElemRun(5, *vec, 0, *dst, count, op, scalarSide, scalar, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorPowScalar(*vec, scalar.f, *dst, count.i, scalarSide.i)
  PmElemRun(6, *vec, 0, *dst, count, 0, scalarSide, scalar, 0.0, #PMELEM_DEAR)
EndProcedure
Procedure PmTensorRelu(*src, *dst, count.i)
  PmElemRun(7, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorLeakyRelu(*src, *dst, count.i, alpha.f)
  PmElemRun(8, *src, 0, *dst, count, 0, 0, alpha, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorSigmoid(*src, *dst, count.i)
  PmElemRun(9, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_DEAR)
EndProcedure
Procedure PmTensorTanh(*src, *dst, count.i)
  PmElemRun(10, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_DEAR)
EndProcedure
; An op outside the closed set goes to the serial kernel, which refuses it
; by clearing its flag; only the calling thread ever writes the flag.
Procedure PmTensorUnaryMath(*src, *dst, count.i, op.i)
  If op < 0 Or op > 4 : PmTensorUnaryMathSerial(*src, *dst, count, op) : ProcedureReturn : EndIf
  If op <= 2
    PmElemRun(11, *src, 0, *dst, count, op, 0, 0.0, 0.0, #PMELEM_DEAR)
  Else
    PmElemRun(11, *src, 0, *dst, count, op, 0, 0.0, 0.0, #PMELEM_CHEAP)
  EndIf
EndProcedure
Procedure PmTensorTrig(*src, *dst, count.i, op.i)
  If op < 0 Or op > 2 : PmTensorTrigSerial(*src, *dst, count, op) : ProcedureReturn : EndIf
  PmElemRun(12, *src, 0, *dst, count, op, 0, 0.0, 0.0, #PMELEM_DEAR)
EndProcedure
Procedure PmTensorFloor(*src, *dst, count.i)
  PmElemRun(13, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorRoundEven(*src, *dst, count.i)
  PmElemRun(14, *src, 0, *dst, count, 0, 0, 0.0, 0.0, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorClip(*src, *dst, count.i, lo.f, hi.f)
  PmElemRun(15, *src, 0, *dst, count, 0, 0, lo, hi, #PMELEM_CHEAP)
EndProcedure

; ---- row-wise helpers --------------------------------------------------------
; A row job runs a serial kernel over rows [first, first + n) by offsetting
; its base addresses; every row is computed whole by one task.
Structure PmRowJob
  Kind.i
  Src.i
  Dst.i
  Rows.i
  Width.i
  Chunk.i
  SrcRowBytes.i
  DstRowBytes.i
  Args.i
  I1.i
  I2.i
  I3.i
  F1.f
EndStructure

Declare PmRowTask(*j.PmRowJob, task.i, worker.i)

; Rows per task so each task holds at least `grain` elements.
Procedure.i PmRowTasks(*j.PmRowJob, grain.i)
  Protected w.i = *j\Width, tasks.i
  If w < 1 : w = 1 : EndIf
  tasks = PmPoolTasks(*j\Rows, (grain + w - 1) / w, 1, @*j\Chunk)
  If tasks < 1 : tasks = 1 : *j\Chunk = *j\Rows : EndIf
  ProcedureReturn tasks
EndProcedure

Procedure PmRowRun(*j.PmRowJob, grain.i)
  Protected tasks.i = PmRowTasks(*j, grain)
  If tasks <= 1
    *j\Chunk = *j\Rows : PmRowTask(*j, 0, 0)
  Else
    PmPoolRun(@PmRowTask(), *j, tasks)
  EndIf
EndProcedure

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

; ---------------------------------------------------------------------------
; INT8 execution (forum 671). One scheme on every target:
;
;  * weights: symmetric INT8, one FP32 scale per output channel. A WIDE weight
;    (the compiler's precision plan) carries a second INT8 plane after the
;    first: w ~= scale * (Q1 + Q2 / 254);
;  * activations: quantized at run time with ONE SCALE PER ROW of the
;    reduction - per token (MatMul, Gemm), per time position (Conv), per step
;    (LSTM). Narrow rows use +-127, wide rows +-32767. For a row with maximum
;    magnitude m: m < 1e-30 gives scale 0 and an all-zero row; otherwise
;    scale = m / qmax, and q = round-half-even(x * (qmax / m));
;  * products are exact integers accumulated in INT32. A wide accumulator is
;    converted to FP32 at least every 512 reduction elements (16-bit x 8-bit
;    products; 512 of them stay inside INT32). For each output the FP32 sum
;    F = sum over taps and chunks of float(acc) * rowScale, then
;    y = bias + wScale * F, or bias + wScale * (F1 + F2 / 254) when wide.
;
; A NaN or an infinity in an INT8 operator's activations is refused by the
; caller before any integer is formed.
;
; On this host the kernels hold activations as INT16 PAIRS along the
; reduction (two neighbouring reduction elements in one 32-bit word) so a
; single vpmaddwd forms two products. Weights are laid out the same way once,
; on first use, and kept until PmTensorInt8Release (the model's close).
; AVX2 is used when the processor has it; otherwise the same arithmetic runs
; in plain code and gives the same integers and the same FP32 sums.
; ---------------------------------------------------------------------------
CompilerIf #PMO_USE_INT8 = 1
Global PmTensorInt8Scratch.i
Global PmTensorInt8ScratchBytes.i
Global PmI8Avx2.i = IsProcessorFeaturePresent_(40)
Global PmTensorInt8Fault.i
Global NewMap PmI8Prepared.i()
Global PmI8PrepareMutex.i = CreateMutex()
#PMI8_FLUSH_PAIRS = 256
#PMI8_WIDE_RESIDUAL = 254.0
#PMI8_TINY = 1.0e-30

Procedure PmTensorInt8Release()
  LockMutex(PmI8PrepareMutex)
  ForEach PmI8Prepared()
    If PmI8Prepared() : FreeMemory(PmI8Prepared()) : EndIf
  Next
  ClearMap(PmI8Prepared())
  UnlockMutex(PmI8PrepareMutex)
EndProcedure

; Largest |x| as its bit pattern (bits of a finite magnitude order like the
; magnitude; any pattern >= $7F800000 is a NaN or an infinity).
Procedure.i PmI8MaxBitsContig(*src, count.i)
  Protected p.i=*src, n.i=count, best.i=0, v.i, blocks.i
  If PmI8Avx2 And n>=8
    blocks=n/8
    !mov rax,[p.v_p]
    !mov rcx,[p.v_blocks]
    !vpcmpeqd ymm1,ymm1,ymm1
    !vpsrld ymm1,ymm1,1
    !vpxor ymm0,ymm0,ymm0
    !pmi8mbc_loop:
    !vpand ymm2,ymm1,[rax]
    !vpmaxud ymm0,ymm0,ymm2
    !add rax,32
    !dec rcx
    !jnz pmi8mbc_loop
    !vextracti128 xmm2,ymm0,1
    !vpmaxud xmm0,xmm0,xmm2
    !vpshufd xmm2,xmm0,78
    !vpmaxud xmm0,xmm0,xmm2
    !vpshufd xmm2,xmm0,177
    !vpmaxud xmm0,xmm0,xmm2
    !vmovd eax,xmm0
    !mov [p.v_best],rax
    !vzeroupper
    p+blocks*32 : n-blocks*8
  EndIf
  While n>0
    v=PeekL(p) & $7FFFFFFF
    If v>best : best=v : EndIf
    p+4 : n-1
  Wend
  ProcedureReturn best
EndProcedure

; out[c] = largest |src[r*rowbytes + c*4]| over rows, for c < cols.
Procedure PmI8MaxBitsColumns(*src, rows.i, rowbytes.i, cols.i, *out)
  Protected c.i=0, r.i, v.i, best.i, base.i, blocks.i=cols/8, ps.i=*src, po.i=*out
  If PmI8Avx2 And blocks>0 And rows>0
    !mov r8,[p.v_ps]
    !mov r9,[p.v_po]
    !mov r10,[p.v_blocks]
    !mov r11,[p.v_rowbytes]
    !mov rdx,[p.v_rows]
    !vpcmpeqd ymm1,ymm1,ymm1
    !vpsrld ymm1,ymm1,1
    !pmi8mcol_block:
    !vpxor ymm0,ymm0,ymm0
    !mov rax,r8
    !mov rcx,rdx
    !pmi8mcol_row:
    !vpand ymm2,ymm1,[rax]
    !vpmaxud ymm0,ymm0,ymm2
    !add rax,r11
    !dec rcx
    !jnz pmi8mcol_row
    !vmovdqu [r9],ymm0
    !add r8,32
    !add r9,32
    !dec r10
    !jnz pmi8mcol_block
    !vzeroupper
    c=blocks*8
  EndIf
  While c<cols
    best=0 : base=*src+c*4
    For r=0 To rows-1
      v=PeekL(base+r*rowbytes) & $7FFFFFFF
      If v>best : best=v : EndIf
    Next
    PokeL(*out+c*4,best) : c+1
  Wend
EndProcedure

; Scale and reciprocal for each row maximum. Returns 0 for a NaN or infinity.
Procedure.i PmI8ScalesFromBits(*bits, count.i, qmax.f, *scale, *inv)
  Protected i.i, b.i, m.f
  For i=0 To count-1
    b=PeekL(*bits+i*4)
    If b>=$7F800000 Or b<0 : ProcedureReturn 0 : EndIf
    m=PeekF(*bits+i*4)
    If m<#PMI8_TINY
      PokeF(*scale+i*4,0.0) : PokeF(*inv+i*4,0.0)
    Else
      PokeF(*scale+i*4,m/qmax) : PokeF(*inv+i*4,qmax/m)
    EndIf
  Next
  ProcedureReturn 1
EndProcedure

; Round-half-even of one FP32 product (the MXCSR default), for the plain path.
Procedure.i PmI8RoundProduct(x.f, inv.f)
  Protected v.f=x*inv, q.i
  !cvtss2si eax,[p.v_v]
  !movsxd rax,eax
  !mov [p.v_q],rax
  ProcedureReturn q
EndProcedure

; Two neighbouring reduction rows (src1 = 0: a zero row) at `cols` positions,
; each with its own reciprocal, into INT16 pairs.
Procedure PmI8QuantColumnPair(*src0, *src1, cols.i, *inv, qmax.i, *dst)
  Protected c.i=0, blocks.i=cols/8, a.i, b.i, p0.i=*src0, p1.i=*src1, pi.i=*inv, pd.i=*dst, lim.i=qmax
  If PmI8Avx2 And blocks>0
    !mov r8,[p.v_p0]
    !mov r9,[p.v_p1]
    !mov r10,[p.v_pi]
    !mov r11,[p.v_pd]
    !mov rcx,[p.v_blocks]
    !vmovd xmm4,dword [p.v_lim]
    !vpbroadcastd ymm4,xmm4
    !vpxor ymm5,ymm5,ymm5
    !vpsubd ymm5,ymm5,ymm4
    !vpcmpeqd ymm3,ymm3,ymm3
    !vpsrld ymm3,ymm3,16
    !pmi8qcp_loop:
    !vmovups ymm2,[r10]
    !vmulps ymm0,ymm2,[r8]
    !vcvtps2dq ymm0,ymm0
    !vpminsd ymm0,ymm0,ymm4
    !vpmaxsd ymm0,ymm0,ymm5
    !vpand ymm0,ymm0,ymm3
    !test r9,r9
    !jz pmi8qcp_store
    !vmulps ymm1,ymm2,[r9]
    !vcvtps2dq ymm1,ymm1
    !vpminsd ymm1,ymm1,ymm4
    !vpmaxsd ymm1,ymm1,ymm5
    !vpslld ymm1,ymm1,16
    !vpor ymm0,ymm0,ymm1
    !add r9,32
    !pmi8qcp_store:
    !vmovdqu [r11],ymm0
    !add r8,32
    !add r10,32
    !add r11,32
    !dec rcx
    !jnz pmi8qcp_loop
    !vzeroupper
    c=blocks*8
  EndIf
  While c<cols
    a=PmI8RoundProduct(PeekF(*src0+c*4),PeekF(*inv+c*4))
    If a>qmax : a=qmax : ElseIf a<-qmax : a=-qmax : EndIf
    b=0
    If *src1
      b=PmI8RoundProduct(PeekF(*src1+c*4),PeekF(*inv+c*4))
      If b>qmax : b=qmax : ElseIf b<-qmax : b=-qmax : EndIf
    EndIf
    PokeW(*dst+c*4,a) : PokeW(*dst+c*4+2,b)
    c+1
  Wend
EndProcedure

; One row along the reduction: element k goes to half (k & 1) of the pair at
; dst + (k >> 1) * dststride. An odd count leaves the last high half as is.
Procedure PmI8QuantRowPairs(*src, count.i, inv.f, qmax.i, *dst, dststride.i)
  Protected k.i=0, a.i, blocks.i=count/4, ps.i=*src, pd.i=*dst, fi.f=inv
  If blocks>0
    !mov r8,[p.v_ps]
    !mov r9,[p.v_pd]
    !mov r10,[p.v_dststride]
    !mov rcx,[p.v_blocks]
    !movss xmm3,[p.v_fi]
    !shufps xmm3,xmm3,0
    !pmi8qrp_loop:
    !movups xmm0,[r8]
    !mulps xmm0,xmm3
    !cvtps2dq xmm0,xmm0
    !packssdw xmm0,xmm0
    !movd [r9],xmm0
    !psrlq xmm0,32
    !movd [r9+r10],xmm0
    !add r8,16
    !lea r9,[r9+r10*2]
    !dec rcx
    !jnz pmi8qrp_loop
    k=blocks*4
  EndIf
  While k<count
    a=PmI8RoundProduct(PeekF(*src+k*4),inv)
    If a>qmax : a=qmax : ElseIf a<-qmax : a=-qmax : EndIf
    PokeW(*dst+(k>>1)*dststride+(k & 1)*2,a)
    k+1
  Wend
EndProcedure

; Four prepared weight rows x sixteen positions, every tap: for each tap the
; activation pairs start at X + XTap(tap) and the position scales at
; S + STap(tap); the tap's pairs run in chunks of at most Flush steps, each
; chunk's exact INT32 sums added into F[r][p] as float(acc) * S[p].
Structure PmI8TileArgs
  W.i
  WStride.i
  X.i
  XStride.i
  XTap.i
  S.i
  STap.i
  Taps.i
  Pairs.i
  Flush.i
  F.i
EndStructure

; The reference form of one chunk, and of the flush, for processors without
; AVX2: the same integers, then the same single-precision convert, multiply
; and add the vector flush performs.
Procedure PmI8ChunkPlain(pw.i, ws.i, px.i, xs.i, n.i, psc.i, pf.i)
  Protected r.i, p.i, i.i, acc.l, fv.f, sv.f, wv.i, xv.i
  For r=0 To 3
    For p=0 To 15
      acc=0
      For i=0 To n-1
        wv=pw+r*ws+i*4 : xv=px+i*xs+p*4
        acc+PeekW(wv)*PeekW(xv)+PeekW(wv+2)*PeekW(xv+2)
      Next
      sv=PeekF(psc+p*4) : fv=PeekF(pf+(r*16+p)*4)
      !cvtsi2ss xmm0,dword [p.v_acc]
      !mulss xmm0,[p.v_sv]
      !addss xmm0,[p.v_fv]
      !movss [p.v_fv],xmm0
      PokeF(pf+(r*16+p)*4,fv)
    Next
  Next
EndProcedure

Procedure PmI8TileAll(*a.PmI8TileArgs)
  Protected ap.i=*a, tap.i, pair.i, n.i, pw.i, px.i, psc.i
  If *a\Taps<=0 Or *a\Pairs<=0 : ProcedureReturn : EndIf
  If PmI8Avx2=0
    pw=*a\W
    For tap=0 To *a\Taps-1
      px=*a\X+PeekI(*a\XTap+tap*8) : psc=*a\S+PeekI(*a\STap+tap*8) : pair=0
      While pair<*a\Pairs
        n=*a\Pairs-pair : If n>*a\Flush : n=*a\Flush : EndIf
        PmI8ChunkPlain(pw,*a\WStride,px,*a\XStride,n,psc,*a\F)
        pw+n*4 : px+n* *a\XStride : pair+n
      Wend
    Next
    ProcedureReturn
  EndIf
  !mov rax,[p.v_ap]
  !push rbx
  !push rsi
  !push rdi
  !push r12
  !push r13
  !push r14
  !push r15
  !mov r15,rax
  !sub rsp,96
  !movdqu [rsp],xmm6
  !movdqu [rsp+16],xmm7
  !movdqu [rsp+32],xmm8
  !movdqu [rsp+48],xmm9
  !movdqu [rsp+64],xmm10
  !movdqu [rsp+80],xmm11
  !mov rcx,[r15]
  !mov r10,[r15+8]
  !lea r11,[r10+r10*2]
  !add r11,rcx
  !mov r8,[r15+24]
  !mov r12,[r15+80]
  !mov r13,[r15+56]
  !xor rsi,rsi
  !pmi8all_tap:
  !mov rbx,[r15+32]
  !mov rax,[rbx+rsi]
  !add rax,[r15+16]
  !mov rbx,[r15+48]
  !mov r9,[rbx+rsi]
  !add r9,[r15+40]
  !mov r14,[r15+64]
  !pmi8all_chunk:
  !mov rdx,[r15+72]
  !cmp r14,rdx
  !cmovb rdx,r14
  !sub r14,rdx
  !vpxor ymm0,ymm0,ymm0
  !vpxor ymm1,ymm1,ymm1
  !vpxor ymm2,ymm2,ymm2
  !vpxor ymm3,ymm3,ymm3
  !vpxor ymm4,ymm4,ymm4
  !vpxor ymm5,ymm5,ymm5
  !vpxor ymm6,ymm6,ymm6
  !vpxor ymm7,ymm7,ymm7
  !test rdx,1
  !jz pmi8all_pairs
  !vmovdqu ymm8,[rax]
  !vmovdqu ymm9,[rax+32]
  !vpbroadcastd ymm10,[rcx]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm0,ymm0,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm1,ymm1,ymm11
  !vpbroadcastd ymm10,[rcx+r10]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm2,ymm2,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm3,ymm3,ymm11
  !vpbroadcastd ymm10,[rcx+r10*2]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm4,ymm4,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm5,ymm5,ymm11
  !vpbroadcastd ymm10,[r11]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm6,ymm6,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm7,ymm7,ymm11
  !add rax,r8
  !add rcx,4
  !add r11,4
  !pmi8all_pairs:
  !shr rdx,1
  !jz pmi8all_flush
  !pmi8all_loop:
  !vmovdqu ymm8,[rax]
  !vmovdqu ymm9,[rax+32]
  !vpbroadcastd ymm10,[rcx]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm0,ymm0,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm1,ymm1,ymm11
  !vpbroadcastd ymm10,[rcx+r10]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm2,ymm2,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm3,ymm3,ymm11
  !vpbroadcastd ymm10,[rcx+r10*2]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm4,ymm4,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm5,ymm5,ymm11
  !vpbroadcastd ymm10,[r11]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm6,ymm6,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm7,ymm7,ymm11
  !vmovdqu ymm8,[rax+r8]
  !vmovdqu ymm9,[rax+r8+32]
  !vpbroadcastd ymm10,[rcx+4]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm0,ymm0,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm1,ymm1,ymm11
  !vpbroadcastd ymm10,[rcx+r10+4]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm2,ymm2,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm3,ymm3,ymm11
  !vpbroadcastd ymm10,[rcx+r10*2+4]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm4,ymm4,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm5,ymm5,ymm11
  !vpbroadcastd ymm10,[r11+4]
  !vpmaddwd ymm11,ymm10,ymm8
  !vpaddd ymm6,ymm6,ymm11
  !vpmaddwd ymm11,ymm10,ymm9
  !vpaddd ymm7,ymm7,ymm11
  !lea rax,[rax+r8*2]
  !add rcx,8
  !add r11,8
  !dec rdx
  !jnz pmi8all_loop
  !pmi8all_flush:
  !vmovups ymm8,[r9]
  !vmovups ymm9,[r9+32]
  !vcvtdq2ps ymm0,ymm0
  !vmulps ymm0,ymm0,ymm8
  !vaddps ymm0,ymm0,[r12]
  !vmovups [r12],ymm0
  !vcvtdq2ps ymm1,ymm1
  !vmulps ymm1,ymm1,ymm9
  !vaddps ymm1,ymm1,[r12+32]
  !vmovups [r12+32],ymm1
  !vcvtdq2ps ymm2,ymm2
  !vmulps ymm2,ymm2,ymm8
  !vaddps ymm2,ymm2,[r12+64]
  !vmovups [r12+64],ymm2
  !vcvtdq2ps ymm3,ymm3
  !vmulps ymm3,ymm3,ymm9
  !vaddps ymm3,ymm3,[r12+96]
  !vmovups [r12+96],ymm3
  !vcvtdq2ps ymm4,ymm4
  !vmulps ymm4,ymm4,ymm8
  !vaddps ymm4,ymm4,[r12+128]
  !vmovups [r12+128],ymm4
  !vcvtdq2ps ymm5,ymm5
  !vmulps ymm5,ymm5,ymm9
  !vaddps ymm5,ymm5,[r12+160]
  !vmovups [r12+160],ymm5
  !vcvtdq2ps ymm6,ymm6
  !vmulps ymm6,ymm6,ymm8
  !vaddps ymm6,ymm6,[r12+192]
  !vmovups [r12+192],ymm6
  !vcvtdq2ps ymm7,ymm7
  !vmulps ymm7,ymm7,ymm9
  !vaddps ymm7,ymm7,[r12+224]
  !vmovups [r12+224],ymm7
  !test r14,r14
  !jnz pmi8all_chunk
  !add rsi,8
  !dec r13
  !jnz pmi8all_tap
  !vzeroupper
  !movdqu xmm6,[rsp]
  !movdqu xmm7,[rsp+16]
  !movdqu xmm8,[rsp+32]
  !movdqu xmm9,[rsp+48]
  !movdqu xmm10,[rsp+64]
  !movdqu xmm11,[rsp+80]
  !add rsp,96
  !pop r15
  !pop r14
  !pop r13
  !pop r12
  !pop rdi
  !pop rsi
  !pop rbx
EndProcedure

; E[c][p] = F[c][p] * wScale(c) (+ bias(c)); wide: (F[2c] + F[2c+1] / 254) * wScale(c).
; `rows` output channels (4 narrow, 2 wide), sixteen positions each, in single
; precision with the division by 254 done as a division, on both paths.
Procedure PmI8Epilogue(*f, *e, rows.i, wide.i, *ws, *bias)
  Protected c.i, p.i, v.f, v2.f, w.f, b.f, k.f=#PMI8_WIDE_RESIDUAL
  Protected pf.i=*f, pe.i=*e, pw.i=*ws, pb.i=*bias, n.i=rows, wd.i=wide
  If rows<=0 : ProcedureReturn : EndIf
  If PmI8Avx2
    !mov rax,[p.v_pf]
    !mov rdx,[p.v_pe]
    !mov r8,[p.v_pw]
    !mov r9,[p.v_pb]
    !mov rcx,[p.v_n]
    !mov r10,[p.v_wd]
    !vbroadcastss ymm5,[p.v_k]
    !pmi8epi_row:
    !vmovups ymm0,[rax]
    !vmovups ymm1,[rax+32]
    !test r10,r10
    !jz pmi8epi_narrow
    !vmovups ymm2,[rax+64]
    !vmovups ymm3,[rax+96]
    !vdivps ymm2,ymm2,ymm5
    !vdivps ymm3,ymm3,ymm5
    !vaddps ymm0,ymm0,ymm2
    !vaddps ymm1,ymm1,ymm3
    !add rax,64
    !pmi8epi_narrow:
    !vbroadcastss ymm4,[r8]
    !vmulps ymm0,ymm0,ymm4
    !vmulps ymm1,ymm1,ymm4
    !test r9,r9
    !jz pmi8epi_store
    !vbroadcastss ymm4,[r9]
    !vaddps ymm0,ymm4,ymm0
    !vaddps ymm1,ymm4,ymm1
    !add r9,4
    !pmi8epi_store:
    !vmovups [rdx],ymm0
    !vmovups [rdx+32],ymm1
    !add rax,64
    !add rdx,64
    !add r8,4
    !dec rcx
    !jnz pmi8epi_row
    !vzeroupper
    ProcedureReturn
  EndIf
  For c=0 To rows-1
    w=PeekF(*ws+c*4) : b=0.0 : If *bias : b=PeekF(*bias+c*4) : EndIf
    For p=0 To 15
      If wide
        v=PeekF(*f+(2*c*16+p)*4) : v2=PeekF(*f+((2*c+1)*16+p)*4)
        !movss xmm1,[p.v_v2]
        !divss xmm1,[p.v_k]
        !movss xmm0,[p.v_v]
        !addss xmm0,xmm1
        !movss [p.v_v],xmm0
      Else
        v=PeekF(*f+(c*16+p)*4)
      EndIf
      !movss xmm0,[p.v_v]
      !mulss xmm0,[p.v_w]
      !movss [p.v_v],xmm0
      If *bias
        !movss xmm0,[p.v_b]
        !addss xmm0,[p.v_v]
        !movss [p.v_v],xmm0
      EndIf
      PokeF(*e+(c*16+p)*4,v)
    Next
  Next
EndProcedure


; Exact dot of `rows` INT8 weight rows (rowbytes apart) with one INT16 vector
; of `count` elements (count <= 512 on the wide path), into INT32 out[].
Procedure PmI8DotRows(*w, rowbytes.i, rows.i, *x, count.i, *out)
  Protected r.i, k.i, blocks.i=count/16, pw.i=*w, px.i=*x, po.i=*out, tail.i=0, total.l
  If PmI8Avx2 And blocks>0 And rows>0
    !mov r8,[p.v_pw]
    !mov r9,[p.v_po]
    !mov r10,[p.v_rows]
    !mov r11,[p.v_rowbytes]
    !pmi8dot_row:
    !mov rcx,r8
    !mov rax,[p.v_px]
    !mov rdx,[p.v_blocks]
    !vpxor ymm0,ymm0,ymm0
    !pmi8dot_loop:
    !vpmovsxbw ymm1,[rcx]
    !vpmaddwd ymm1,ymm1,[rax]
    !vpaddd ymm0,ymm0,ymm1
    !add rcx,16
    !add rax,32
    !dec rdx
    !jnz pmi8dot_loop
    !vextracti128 xmm1,ymm0,1
    !vpaddd xmm0,xmm0,xmm1
    !vpshufd xmm1,xmm0,78
    !vpaddd xmm0,xmm0,xmm1
    !vpshufd xmm1,xmm0,177
    !vpaddd xmm0,xmm0,xmm1
    !vmovd dword [r9],xmm0
    !add r8,r11
    !add r9,4
    !dec r10
    !jnz pmi8dot_row
    !vzeroupper
    tail=blocks*16
  EndIf
  For r=0 To rows-1
    If tail=0 : total=0 : Else : total=PeekL(*out+r*4) : EndIf
    pw=*w+r*rowbytes
    For k=tail To count-1
      total+PeekB(pw+k)*PeekW(px+k*2)
    Next
    PokeL(*out+r*4,total)
  Next
EndProcedure

; ---- prepared weight layouts: INT16 pairs along the reduction ----------------
; Row r of a prepared weight is 4 * taps * pairs bytes. Narrow: row = output
; channel. Wide: row 2*o + plane, so a block of four rows is two channels with
; both planes. Each group's rows start a new block of four (zero rows pad).
Procedure.i PmI8GroupRows(outPerGroup.i, wide.i)
  ProcedureReturn (outPerGroup*(1+wide)+3)&~3
EndProcedure

Procedure.i PmI8PrepareConv(*w, outCh.i, groups.i, chPerGroup.i, kernel.i, wide.i, elements.i)
  Protected key.s=Hex(*w)+":c"+Str(outCh)+"/"+Str(groups)+"/"+Str(chPerGroup)+"/"+Str(kernel)+"/"+Str(wide), pairs.i=(chPerGroup+1)/2, og.i=outCh/groups, rpg.i=PmI8GroupRows(og,wide)
  Protected *p, o.i, plane.i, j.i, i.i, row.i, lo.i, hi.i, src.i, rowBytes.i=kernel*pairs*4
  LockMutex(PmI8PrepareMutex)
  If FindMapElement(PmI8Prepared(),key) : *p=PmI8Prepared() : UnlockMutex(PmI8PrepareMutex) : ProcedureReturn *p : EndIf
  *p=AllocateMemory(groups*rpg*rowBytes+64)
  If *p
    For o=0 To outCh-1
      For plane=0 To wide
        row=(o/og)*rpg+(o % og)*(1+wide)+plane
        src=*w+plane*elements+o*chPerGroup*kernel
        For j=0 To kernel-1
          For i=0 To pairs-1
            lo=PeekB(src+(2*i)*kernel+j) : hi=0
            If 2*i+1<chPerGroup : hi=PeekB(src+(2*i+1)*kernel+j) : EndIf
            PokeW(*p+row*rowBytes+(j*pairs+i)*4,lo) : PokeW(*p+row*rowBytes+(j*pairs+i)*4+2,hi)
          Next
        Next
      Next
    Next
    PmI8Prepared(key)=*p
  EndIf
  UnlockMutex(PmI8PrepareMutex)
  ProcedureReturn *p
EndProcedure

; MatMul B is [K][N] (one batch slice), or [N][K] rows when bRows (an LSTM
; input weight); prepared rows are the N output columns either way.
Procedure.i PmI8PrepareMatMul(*w, k.i, n.i, wide.i, elements.i, bRows.i=0)
  Protected key.s=Hex(*w)+":m"+Str(bRows)+"/"+Str(k)+"/"+Str(n)+"/"+Str(wide), pairs.i=(k+1)/2, rows.i=n*(1+wide), padded.i=(rows+3)&~3
  Protected *p, c.i, plane.i, i.i, row.i, lo.i, hi.i, src.i, rowBytes.i=pairs*4
  LockMutex(PmI8PrepareMutex)
  If FindMapElement(PmI8Prepared(),key) : *p=PmI8Prepared() : UnlockMutex(PmI8PrepareMutex) : ProcedureReturn *p : EndIf
  *p=AllocateMemory(padded*rowBytes+64)
  If *p
    For c=0 To n-1
      For plane=0 To wide
        row=c : If wide : row=2*c+plane : EndIf
        If bRows
          src=*w+plane*elements+c*k
          For i=0 To pairs-1
            lo=PeekB(src+2*i) : hi=0
            If 2*i+1<k : hi=PeekB(src+2*i+1) : EndIf
            PokeW(*p+row*rowBytes+i*4,lo) : PokeW(*p+row*rowBytes+i*4+2,hi)
          Next
          Continue
        EndIf
        src=*w+plane*elements+c
        For i=0 To pairs-1
          lo=PeekB(src+(2*i)*n) : hi=0
          If 2*i+1<k : hi=PeekB(src+(2*i+1)*n) : EndIf
          PokeW(*p+row*rowBytes+i*4,lo) : PokeW(*p+row*rowBytes+i*4+2,hi)
        Next
      Next
    Next
    PmI8Prepared(key)=*p
  EndIf
  UnlockMutex(PmI8PrepareMutex)
  ProcedureReturn *p
EndProcedure

; Gemm B with transB=0 is [K][N]; the dot path wants rows of K, so keep a
; transposed INT8 copy (both planes).
Procedure.i PmI8PrepareRows(*w, k.i, n.i, wide.i, elements.i)
  Protected key.s=Hex(*w)+":t"+Str(k)+"/"+Str(n)+"/"+Str(wide), *p, c.i, i.i, plane.i
  LockMutex(PmI8PrepareMutex)
  If FindMapElement(PmI8Prepared(),key) : *p=PmI8Prepared() : UnlockMutex(PmI8PrepareMutex) : ProcedureReturn *p : EndIf
  *p=AllocateMemory(k*n*(1+wide)+64)
  If *p
    For plane=0 To wide
      For c=0 To n-1
        For i=0 To k-1
          PokeB(*p+plane*k*n+c*k+i,PeekB(*w+plane*elements+i*n+c))
        Next
      Next
    Next
    PmI8Prepared(key)=*p
  EndIf
  UnlockMutex(PmI8PrepareMutex)
  ProcedureReturn *p
EndProcedure

; ---- the tiled integer product ---------------------------------------------
; One job computes prepared rows [RowFirst, RowLast) (multiples of four) at
; position tiles [TileFirst, TileLast) of sixteen. Tap j reads activation
; pairs at X + XTap(j) and position scales at S + STap(j).
Structure PmI8TileJob
  W.i
  WRowBytes.i
  Pairs.i
  Taps.i
  X.i
  XStride.i
  XTap.i
  S.i
  STap.i
  Flush.i
  RowFirst.i
  RowLast.i
  TileCount.i
  ByTiles.i
  Counter.i
  Mode.i
  Wide.i
  Dst.i
  Scales.i
  Bias.i
  Rows.i
  RowBase.i
  ChannelBase.i
  Positions.i
  DstRowStride.i
EndStructure

#PMI8_MODE_CONV = 0
#PMI8_MODE_MATMUL = 1

; One task: a position tile (every row block) or a row block (every tile),
; so a slower core simply takes fewer of them. Its outputs are the task's
; own tile or rows; f and e are this call's own scratch.
Procedure PmI8TileTask(*j.PmI8TileJob, task.i, worker.i)
  Protected Dim f.f(63)
  Protected Dim e.f(63)
  Protected a.PmI8TileArgs
  Protected t.i, t0.i, t1.i, rb.i, r0.i, r1.i, c.i, p.i, ch0.i, local.i, valid.i, channels.i, n.i
  a\WStride=*j\WRowBytes : a\XStride=*j\XStride : a\XTap=*j\XTap : a\STap=*j\STap
  a\Taps=*j\Taps : a\Pairs=*j\Pairs : a\Flush=*j\Flush : a\F=@f(0)
  If *j\ByTiles
    If task>=*j\TileCount : ProcedureReturn : EndIf
    t0=task : t1=task+1 : r0=*j\RowFirst : r1=*j\RowLast
  Else
    If task>=(*j\RowLast-*j\RowFirst)/4 : ProcedureReturn : EndIf
    t0=0 : t1=*j\TileCount : r0=*j\RowFirst+task*4 : r1=r0+4
  EndIf
  For t=t0 To t1-1
    valid=*j\Positions-t*16 : If valid>16 : valid=16 : EndIf
    rb=r0
    While rb<r1
      local=(rb-*j\RowBase) >> *j\Wide
      channels=*j\Rows-local : If channels>(4 >> *j\Wide) : channels=4 >> *j\Wide : EndIf
      If channels>0
        FillMemory(@f(0),256,0)
        a\W=*j\W+rb* *j\WRowBytes : a\X=*j\X+t*64 : a\S=*j\S+t*64
        PmI8TileAll(@a)
        ch0=*j\ChannelBase+local
        If *j\Bias : n=*j\Bias+ch0*4 : Else : n=0 : EndIf
        PmI8Epilogue(@f(0),@e(0),channels,*j\Wide,*j\Scales+ch0*4,n)
        If *j\Mode=#PMI8_MODE_CONV
          For c=0 To channels-1
            CopyMemory(@e(c*16),*j\Dst+((ch0+c)* *j\DstRowStride+t*16)*4,valid*4)
          Next
        Else
          For c=0 To channels-1
            For p=0 To valid-1
              PokeF(*j\Dst+((t*16+p)* *j\DstRowStride+ch0+c)*4,e(c*16+p))
            Next
          Next
        EndIf
      EndIf
      rb+4
    Wend
  Next
EndProcedure

; Below two million multiply-accumulates the product stays on the calling
; thread (the pool's hand-off costs more than it saves).
Procedure PmI8RunTiles(*proto.PmI8TileJob, rowBase.i, rowsPadded.i, tiles.i)
  Protected job.PmI8TileJob, tasks.i, t.i
  Protected macs.q=rowsPadded
  macs*tiles*16 : macs* *proto\Pairs* *proto\Taps*2
  CopyStructure(*proto,@job,PmI8TileJob)
  job\RowFirst=rowBase : job\RowLast=rowBase+rowsPadded : job\TileCount=tiles
  job\ByTiles=Bool(tiles>=PmPoolThreads*4 Or tiles>=rowsPadded/4)
  If job\ByTiles : tasks=tiles : Else : tasks=rowsPadded/4 : EndIf
  If PmPoolThreads<=1 Or (macs<2000000 And PmPoolForceSplit=0)
    For t=0 To tasks-1 : PmI8TileTask(@job,t,0) : Next
  Else
    PmPoolRun(@PmI8TileTask(),@job,tasks)
  EndIf
EndProcedure
; ---- operators ------------------------------------------------------------
; MatMul: A is [m][k] (one batch slice), B is INT8 [k][n], dst [m][n].
; Returns 0 for a NaN or an infinity in A, or when memory runs out.
Procedure.i PmI8MatMul(*a, *b, *dst, m.i, k.i, n.i, *weightScales, wide.i, elements.i, bRows.i=0)
  Protected pairs.i=(k+1)/2, tiles.i=(m+15)/16, positions.i=tiles*16, row.i, qmax.i=127
  Protected *x, *s, *inv, *bits, *wp, ok.i=0, inv.f
  Protected job.PmI8TileJob
  Protected zero.i=0
  If m<=0 Or n<=0 : ProcedureReturn 1 : EndIf
  If wide : qmax=32767 : EndIf
  *wp=PmI8PrepareMatMul(*b,k,n,wide,elements,bRows)
  *x=AllocateMemory(pairs*positions*4+64) : *s=AllocateMemory(positions*4+64)
  *inv=AllocateMemory(positions*4+64) : *bits=AllocateMemory(positions*4+64)
  If *wp And *x And *s And *inv And *bits
    For row=0 To m-1 : PokeL(*bits+row*4,PmI8MaxBitsContig(*a+row*k*4,k)) : Next
    If PmI8ScalesFromBits(*bits,m,qmax,*s,*inv)=0
      PmTensorInt8Fault=1
    Else
      For row=0 To m-1
        inv=PeekF(*inv+row*4)
        PmI8QuantRowPairs(*a+row*k*4,k,inv,qmax,*x+row*4,positions*4)
      Next
      job\W=*wp : job\WRowBytes=pairs*4 : job\Pairs=pairs : job\Taps=1
      job\X=*x : job\XStride=positions*4 : job\XTap=@zero : job\S=*s : job\STap=@zero
      job\Flush=pairs : If wide : job\Flush=#PMI8_FLUSH_PAIRS : EndIf
      job\Mode=#PMI8_MODE_MATMUL : job\Wide=wide : job\Dst=*dst : job\Scales=*weightScales
      job\Rows=n : job\RowBase=0 : job\ChannelBase=0 : job\Positions=m : job\DstRowStride=n
      PmI8RunTiles(@job,0,PmI8GroupRows(n,wide),tiles)
      ok=1
    EndIf
  EndIf
  If *x : FreeMemory(*x) : EndIf
  If *s : FreeMemory(*s) : EndIf
  If *inv : FreeMemory(*inv) : EndIf
  If *bits : FreeMemory(*bits) : EndIf
  ProcedureReturn ok
EndProcedure

; ---- the row-dot path: Gemm and LSTM -----------------------------------------
; Quantize one contiguous row of `count` floats to INT16 (dst) with its scale.
; Returns 0 for a NaN or an infinity.
Procedure.i PmI8QuantizeVector(*src, count.i, qmax.i, *dst, *scale)
  Protected bits.i=PmI8MaxBitsContig(*src,count), m.f, inv.f
  If bits>=$7F800000 : ProcedureReturn 0 : EndIf
  PokeL(@m,bits)
  If m<#PMI8_TINY
    FillMemory(*dst,count*2,0) : PokeF(*scale,0.0) : ProcedureReturn 1
  EndIf
  PokeF(*scale,m/qmax) : inv=qmax/m
  PmI8QuantRowPairs(*src,count,inv,qmax,*dst,4)
  ProcedureReturn 1
EndProcedure

; F[r] = sum over chunks of float(dot(row r, x)) * xScale, for `rows` INT8
; rows `rowbytes` apart; chunks of 512 elements when wide, else one chunk.
Procedure PmI8DotRowsScaled(*w, rowbytes.i, rows.i, *x, count.i, xScale.f, wide.i, *acc, *f)
  Protected k.i=0, chunk.i, r.i, v.f
  FillMemory(*f,rows*4,0)
  While k<count
    chunk=count-k
    If wide And chunk>512 : chunk=512 : EndIf
    PmI8DotRows(*w+k,rowbytes,rows,*x+k*2,chunk,*acc)
    For r=0 To rows-1
      v=PeekL(*acc+r*4) : v=v*xScale : v=PeekF(*f+r*4)+v
      PokeF(*f+r*4,v)
    Next
    k+chunk
  Wend
EndProcedure


CompilerEndIf

; A is MxK, B is KxN and dst is MxN, all row-major.
Procedure PmTensorMatMul2Serial(*a, *b, *dst, m.i, k.i, n.i)
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

; Rows of A (and of dst) split across the pool; each row whole in one task.
Procedure PmTensorMatMul2(*a, *b, *dst, m.i, k.i, n.i)
  Protected j.PmRowJob
  j\Kind = 5 : j\Src = *a : j\Dst = *dst : j\Rows = m : j\Width = k * n : j\Args = *b
  j\I1 = k : j\I2 = n
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure

CompilerIf #PMO_USE_INT8 = 1
; The fixed-shape entry: narrow weights, one scale per row of A.
Procedure PmTensorMatMul2Int8(*a, *b, *dst, m.i, k.i, n.i, *weightScales)
  If PmI8MatMul(*a,*b,*dst,m,k,n,*weightScales,0,k*n)=0 And PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
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
  Wide.i
EndStructure

CompilerIf #PMO_USE_INT8 = 1

; ONNX Gemm with an INT8 B: one scale per row of A' (per token). Rows of A'
; are split across the pool; each task has its own row buffers, and reports
; a fault in its own slot (1 not finite, 2 no memory).
Structure PmI8GemmJob
  G.i
  Rows.i
  Chunk.i
  Faults.i
EndStructure

Procedure PmI8GemmTask(*j.PmI8GemmJob, task.i, worker.i)
  Protected *g.PmTensorGemmArgs=*j\G, *rows=*j\Rows
  Protected wide.i=*g\Wide, qmax.i=127, row.i, col.i, i.i, ic.i, *xq, *acc, *f1, *f2, *arow, r0.i, r1.i
  Protected xs.f, v.f, t.f, elements.i=*g\K * *g\N, rowbytes.i=*g\K
  If wide : qmax=32767 : EndIf
  r0=task* *j\Chunk : r1=r0+ *j\Chunk : If r1>*g\M : r1=*g\M : EndIf
  *xq=AllocateMemory(*g\K*2+64) : *acc=AllocateMemory(*g\N*4+64)
  *f1=AllocateMemory(*g\N*4+64) : *f2=AllocateMemory(*g\N*4+64) : *arow=AllocateMemory(*g\K*4+64)
  If *xq=0 Or *acc=0 Or *f1=0 Or *f2=0 Or *arow=0
    PokeI(*j\Faults+task*8,2)
  Else
    For row=r0 To r1-1
      If *g\TransA
        For i=0 To *g\K-1 : PokeL(*arow+i*4,PeekL(*g\A+(i* *g\M+row)*4)) : Next
      Else
        CopyMemory(*g\A+row* *g\K*4,*arow,*g\K*4)
      EndIf
      If PmI8QuantizeVector(*arow,*g\K,qmax,*xq,@xs)=0 : PokeI(*j\Faults+task*8,1) : Break : EndIf
      PmI8DotRowsScaled(*rows,rowbytes,*g\N,*xq,*g\K,xs,wide,*acc,*f1)
      If wide : PmI8DotRowsScaled(*rows+elements,rowbytes,*g\N,*xq,*g\K,xs,wide,*acc,*f2) : EndIf
      For col=0 To *g\N-1
        ; One rounding per operation, on every target: this host evaluates a
        ; longer expression on the x87 stack at extended precision.
        v=PeekF(*f1+col*4)
        If wide : t=PeekF(*f2+col*4) : t=t/#PMI8_WIDE_RESIDUAL : v=v+t : EndIf
        v=v*PeekF(*g\WeightScales+col*4)
        v=v* *g\Alpha
        If *g\C<>0 And *g\CCount>0
          ic=row* *g\N+col
          If *g\CCount=1 : ic=0 : ElseIf *g\CCount=*g\N : ic=col : EndIf
          t=PeekF(*g\C+ic*4) : t=*g\Beta*t : v=v+t
        EndIf
        PokeF(*g\Dst+(row* *g\N+col)*4,v)
      Next
    Next
  EndIf
  If *xq : FreeMemory(*xq) : EndIf
  If *acc : FreeMemory(*acc) : EndIf
  If *f1 : FreeMemory(*f1) : EndIf
  If *f2 : FreeMemory(*f2) : EndIf
  If *arow : FreeMemory(*arow) : EndIf
EndProcedure

Procedure.i PmI8Gemm(*g.PmTensorGemmArgs)
  Protected j.PmI8GemmJob, tasks.i, t.i, fault.i, elements.i=*g\K * *g\N
  If *g\TransB : j\Rows=*g\B : Else : j\Rows=PmI8PrepareRows(*g\B,*g\K,*g\N,*g\Wide,elements) : EndIf
  If j\Rows=0 : ProcedureReturn 0 : EndIf
  j\G=*g
  tasks=PmPoolTasks(*g\M,(65536+elements-1)/PmPoolMax(elements,1),1,@j\Chunk)
  If tasks<1 : ProcedureReturn 1 : EndIf
  j\Faults=AllocateMemory(tasks*8+8)
  If j\Faults=0 : ProcedureReturn 0 : EndIf
  If tasks=1 : j\Chunk=*g\M : PmI8GemmTask(@j,0,0) : Else : PmPoolRun(@PmI8GemmTask(),@j,tasks) : EndIf
  For t=0 To tasks-1
    If PeekI(j\Faults+t*8)=1 : fault=1 : ElseIf PeekI(j\Faults+t*8)=2 And fault=0 : fault=2 : EndIf
  Next
  FreeMemory(j\Faults)
  If fault=1 : PmTensorInt8Fault=1 : EndIf
  ProcedureReturn Bool(fault=0)
EndProcedure
CompilerEndIf

; ONNX Gemm: Y = alpha * A' * B' + beta * C. C may be absent (0), a
; scalar (cCount=1), one value per column (cCount=n), or a full MxN matrix.
; One argument block is used because the A64 calling convention implemented
; by Forge currently has eight register arguments and no stack arguments.
; Rows [first, last) of the FLOAT form; the pool gives each task its rows.
Procedure PmTensorGemmRows(*g.PmTensorGemmArgs, first.i, last.i)
  Protected row.i
  Protected col.i
  Protected inner.i
  Protected ia.i
  Protected ib.i
  Protected ic.i
  Protected accumulator.i
  Protected sum.f
  row = first
  While row < last
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

Procedure PmTensorGemm(*g.PmTensorGemmArgs)
  Protected j.PmRowJob
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    If PmI8Gemm(*g)=0 And PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
    ProcedureReturn
  EndIf
  CompilerEndIf
  j\Kind = 6 : j\Rows = *g\M : j\Width = *g\K * *g\N : j\Args = *g
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure

; Softmax over the final dimension. outer is the product of every earlier
; dimension and width is the final dimension.
Procedure PmTensorSoftmaxLastSerial(*src, *dst, outer.i, width.i)
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

Procedure PmTensorReduceMeanLastSerial(*src, *dst, outer.i, width.i)
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

Procedure PmTensorReduceSumLastSerial(*src, *dst, outer.i, width.i)
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

; Softmax and the last-axis reductions split by rows: a row's sum is never
; divided between tasks.
Procedure PmTensorSoftmaxLast(*src, *dst, outer.i, width.i)
  Protected j.PmRowJob
  j\Kind = 0 : j\Src = *src : j\Dst = *dst : j\Rows = outer : j\Width = width
  PmRowRun(@j, #PMELEM_DEAR)
EndProcedure
Procedure PmTensorReduceMeanLast(*src, *dst, outer.i, width.i)
  Protected j.PmRowJob
  j\Kind = 1 : j\Src = *src : j\Dst = *dst : j\Rows = outer : j\Width = width
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure
Procedure PmTensorReduceSumLast(*src, *dst, outer.i, width.i)
  Protected j.PmRowJob
  j\Kind = 2 : j\Src = *src : j\Dst = *dst : j\Rows = outer : j\Width = width
  PmRowRun(@j, #PMELEM_CHEAP)
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

; LayerNormalization-17 (https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
; over outer blocks of width elements: the normalised axes [axis, rank - 1]
; flattened, so every axis is one width and one outer count. Scale and Bias
; each contain width elements; Bias may be 0. Mean and InvStdDev, when their
; addresses are not 0, receive one element per block: the mean and
; 1 / Sqrt(variance + epsilon) that block was normalised with (stash_type 1).
Structure PmTensorLayerNormArgs
  Src.i
  Scale.i
  Bias.i
  Dst.i
  Mean.i
  InvStdDev.i
  Outer.i
  Width.i
  Epsilon.f
EndStructure

Procedure PmTensorLayerNormSerial(*g.PmTensorLayerNormArgs)
  Protected o.i
  Protected j.i
  Protected base.i
  Protected v.f
  Protected mean.f
  Protected variance.f
  Protected inv.f
  Protected divisor.f
  Protected *src = *g\Src
  Protected *scale = *g\Scale
  Protected *bias = *g\Bias
  Protected *dst = *g\Dst
  Protected outer.i = *g\Outer
  Protected width.i = *g\Width
  Protected epsilon.f = *g\Epsilon
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
    If *g\Mean <> 0 : PmTensorPut(*g\Mean, o, mean) : EndIf
    If *g\InvStdDev <> 0 : PmTensorPut(*g\InvStdDev, o, inv) : EndIf
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

; Blocks split across the pool; each block's mean and variance are summed
; whole by one task.
Procedure PmTensorLayerNorm(*g.PmTensorLayerNormArgs)
  Protected j.PmRowJob
  j\Kind = 3 : j\Rows = *g\Outer : j\Width = *g\Width : j\Args = *g
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure

; The same kernel without the Mean and InvStdDev outputs.
Procedure PmTensorLayerNormLast(*src, *scale, *bias, *dst, outer.i, width.i, epsilon.f)
  Protected g.PmTensorLayerNormArgs
  g\Src = *src : g\Scale = *scale : g\Bias = *bias : g\Dst = *dst
  g\Mean = 0 : g\InvStdDev = 0
  g\Outer = outer : g\Width = width : g\Epsilon = epsilon
  PmTensorLayerNorm(@g)
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
  RunningMean.i
  RunningVariance.i
  Epsilon.f
  Momentum.f
EndStructure

CompilerIf #PMO_USE_BATCHNORM = 1
; Planes [first, last) of the N x C planes, plane = bn * C + ch, each
; normalised by the statements the whole-tensor loop used.
Procedure PmTensorBatchNormPlanes(*g.PmTensorBatchNormArgs, first.i, last.i)
  ; one binary32 operation per expression, the square root correctly rounded
  ; (C6c-1): the same bits as every target. The host evaluates an expression
  ; wider than binary32 and rounds once where it is stored, which is the
  ; correctly rounded binary32 result for ONE operation; x * mul + add in one
  ; expression would round once for two.
  Protected bn.i
  Protected ch.i
  Protected s.i
  Protected idx.i
  Protected plane.i
  Protected ps.i
  Protected pd.i
  Protected pe.i
  Protected mul.f
  Protected add.f
  Protected t.f
  Protected v.f
  plane = first
  While plane < last
    bn = plane / *g\C
    ch = plane % *g\C
    t = PmTensorGet(*g\Variance, ch)
    t = t + *g\Epsilon
    t = PmTensorSqrtF(t)
    mul = PmTensorGet(*g\Scale, ch)
    mul = mul / t
    add = PmTensorGet(*g\Mean, ch)
    add = add * mul
    t = PmTensorGet(*g\Bias, ch)
    add = t - add
    ; the plane's elements are contiguous: walk them by address
    idx = (bn * *g\C + ch) * *g\Spatial * 4
    ps = *g\Src + idx
    pd = *g\Dst + idx
    pe = ps + *g\Spatial * 4
    While ps < pe
      v = PeekF(ps) * mul
      PokeF(pd, v + add)
      ps = ps + 4 : pd = pd + 4
    Wend
    plane = plane + 1
  Wend
EndProcedure

Procedure PmTensorBatchNorm(*g.PmTensorBatchNormArgs)
  Protected j.PmRowJob
  If *g\C <= 0 : ProcedureReturn : EndIf
  j\Kind = 7 : j\Rows = *g\N * *g\C : j\Width = *g\Spatial : j\Args = *g
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure

; BatchNormalization-15 with training_mode = 1
; (https://onnx.ai/onnx/operators/onnx__BatchNormalization.html): each channel
; is normalised with the batch's own statistics, current_mean and current_var
; (the population variance over every axis but the channel axis),
;   Y = (X - current_mean) / Sqrt(current_var + epsilon) * scale + B
; and, when their addresses are not 0,
;   running_mean = input_mean * momentum + current_mean * (1 - momentum)
;   running_var  = input_var * momentum + current_var * (1 - momentum).
; Mean and Variance hold input_mean and input_var.
; Channels [first, last): each channel's statistics are summed whole by one task.
Procedure PmTensorBatchNormTrainingChannels(*g.PmTensorBatchNormArgs, first.i, last.i)
  ; one binary32 operation per statement, the square root correctly rounded
  ; (C6c-1): the same bits on every target
  Protected bn.i
  Protected ch.i
  Protected s.i
  Protected idx.i
  Protected ps.i
  Protected pd.i
  Protected pe.i
  Protected mean.f
  Protected variance.f
  Protected v.f
  Protected mul.f
  Protected add.f
  Protected divisor.f
  Protected keep.f
  Protected t.f
  Protected u.f
  If *g\N * *g\Spatial <= 0
    ProcedureReturn
  EndIf
  divisor = *g\N * *g\Spatial
  keep = 1.0
  keep = keep - *g\Momentum
  ch = first
  While ch < last
    ; each (batch, channel) plane is contiguous: walk it by address
    mean = 0.0
    bn = 0
    While bn < *g\N
      ps = *g\Src + (bn * *g\C + ch) * *g\Spatial * 4
      pe = ps + *g\Spatial * 4
      While ps < pe
        mean = mean + PeekF(ps)
        ps = ps + 4
      Wend
      bn = bn + 1
    Wend
    mean = mean / divisor
    variance = 0.0
    bn = 0
    While bn < *g\N
      ps = *g\Src + (bn * *g\C + ch) * *g\Spatial * 4
      pe = ps + *g\Spatial * 4
      While ps < pe
        v = PeekF(ps) - mean
        v = v * v
        variance = variance + v
        ps = ps + 4
      Wend
      bn = bn + 1
    Wend
    variance = variance / divisor
    t = variance + *g\Epsilon
    t = PmTensorSqrtF(t)
    mul = PmTensorGet(*g\Scale, ch)
    mul = mul / t
    add = PmTensorGet(*g\Bias, ch)
    If *g\RunningMean <> 0
      t = PmTensorGet(*g\Mean, ch)
      t = t * *g\Momentum
      u = mean * keep
      t = t + u
      PmTensorPut(*g\RunningMean, ch, t)
    EndIf
    If *g\RunningVariance <> 0
      t = PmTensorGet(*g\Variance, ch)
      t = t * *g\Momentum
      u = variance * keep
      t = t + u
      PmTensorPut(*g\RunningVariance, ch, t)
    EndIf
    bn = 0
    While bn < *g\N
      idx = (bn * *g\C + ch) * *g\Spatial * 4
      ps = *g\Src + idx
      pd = *g\Dst + idx
      pe = ps + *g\Spatial * 4
      While ps < pe
        v = PeekF(ps) - mean
        v = v * mul
        PokeF(pd, v + add)
        ps = ps + 4 : pd = pd + 4
      Wend
      bn = bn + 1
    Wend
    ch = ch + 1
  Wend
EndProcedure

Procedure PmTensorBatchNormTraining(*g.PmTensorBatchNormArgs)
  Protected j.PmRowJob
  If *g\N * *g\Spatial <= 0 : ProcedureReturn : EndIf
  j\Kind = 8 : j\Rows = *g\C : j\Width = *g\N * *g\Spatial : j\Args = *g
  PmRowRun(@j, #PMELEM_CHEAP)
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

Structure PmStftJob
  G.i
  ComplexCount.i
  Total.i
  Chunk.i
  BR.i
  BI.i
  ChirpR.i
  ChirpI.i
  RootsR.i
  RootsI.i
  Work.i
  Failed.i
EndStructure

; Frames [first, last) of every batch (index = bn * Frames + frame), with the
; four working arrays at *work, by the statements of the single-threaded loop.
Procedure PmStftFrames(*j.PmStftJob, first.i, last.i, *work)
  Protected *g.PmTensorStftArgs = *j\G
  Protected complexCount.i = *j\ComplexCount
  Protected *work1R = *work
  Protected *work1I = *work1R + complexCount * 4
  Protected *work2R = *work1I + complexCount * 4
  Protected *work2I = *work2R + complexCount * 4
  Protected *bR = *j\BR, *bI = *j\BI, *chirpR = *j\ChirpR, *chirpI = *j\ChirpI, *rootsR = *j\RootsR, *rootsI = *j\RootsI
  Protected index.i, bn.i, frame.i, bin.i, sample.i, signalIndex.i, outputIndex.i, mirror.i
  Protected value.f, sourceR.f, sourceI.f, weightR.f, weightI.f, resultR.f, resultI.f
  index = first
  While index < last
      bn = index / *g\Frames
      frame = index % *g\Frames
      PmTensorZeroSerial(*work1R, complexCount)
      PmTensorZeroSerial(*work1I, complexCount)
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
      index = index + 1
  Wend
EndProcedure

Procedure PmStftTask(*j.PmStftJob, task.i, worker.i)
  Protected first.i = task * *j\Chunk, last.i = first + *j\Chunk, *work, *g.PmTensorStftArgs = *j\G
  If last > *j\Total : last = *j\Total : EndIf
  If worker = 0
    *work = *g\Scratch
  Else
    *work = PeekI(*j\Work + worker * 8)
    If *work = 0
      *work = AllocateMemory(*j\ComplexCount * 16 + 64)
      If *work = 0 : PokeI(*j\Failed + task * 8, 1) : ProcedureReturn : EndIf
      PokeI(*j\Work + worker * 8, *work)
    EndIf
  EndIf
  PmStftFrames(*j, first, last, *work)
EndProcedure

Procedure.i PmTensorStft(*g.PmTensorStftArgs)
  Protected job.PmStftJob, tasks.i, t.i
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

  ; The frames, split across the pool. Worker 0 uses the planned working
  ; arrays; every other worker its own four, made on its first frame. The
  ; tables above are read-only from here on.
  job\G = *g : job\ComplexCount = complexCount : job\Total = *g\Batches * *g\Frames
  job\BR = *bR : job\BI = *bI : job\ChirpR = *chirpR : job\ChirpI = *chirpI
  job\RootsR = *rootsR : job\RootsI = *rootsI
  tasks = PmPoolTasks(job\Total, PmPoolMax(1, 262144 / (complexCount * 8)), 1, @job\Chunk)
  If tasks <= 1
    PmStftFrames(@job, 0, job\Total, *g\Scratch)
    ProcedureReturn 1
  EndIf
  job\Work = AllocateMemory(PmPoolThreads * 8 + 8)
  job\Failed = AllocateMemory(tasks * 8 + 8)
  If job\Work = 0 Or job\Failed = 0
    If job\Work : FreeMemory(job\Work) : EndIf
    If job\Failed : FreeMemory(job\Failed) : EndIf
    PmStftFrames(@job, 0, job\Total, *g\Scratch)
    ProcedureReturn 1
  EndIf
  PmPoolRun(@PmStftTask(), @job, tasks)
  For t = 0 To tasks - 1
    ; A worker that could not get working arrays left its frames undone.
    If PeekI(job\Failed + t * 8)
      frame = t * job\Chunk : sample = frame + job\Chunk : If sample > job\Total : sample = job\Total : EndIf
      PmStftFrames(@job, frame, sample, *g\Scratch)
    EndIf
  Next
  For t = 1 To PmPoolThreads - 1
    If PeekI(job\Work + t * 8) : FreeMemory(PeekI(job\Work + t * 8)) : EndIf
  Next
  FreeMemory(job\Work) : FreeMemory(job\Failed)
  ProcedureReturn 1
EndProcedure
CompilerEndIf

; Resize the final axis of a contiguous FLOAT tensor. mode 0 is nearest/floor,
; mode 1 is linear. coordinateMode 0 is asymmetric, 1 is half_pixel.
CompilerIf #PMO_USE_RESIZE = 1
Procedure PmTensorResize1DSerial(*src, *dst, outer.i, inWidth.i, outWidth.i, mode.i, coordinateMode.i, scale.f)
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

Procedure PmTensorResize1D(*src, *dst, outer.i, inWidth.i, outWidth.i, mode.i, coordinateMode.i, scale.f)
  Protected j.PmRowJob
  j\Kind = 4 : j\Src = *src : j\Dst = *dst : j\Rows = outer : j\Width = outWidth
  j\I1 = mode : j\I2 = coordinateMode : j\I3 = inWidth : j\F1 = scale
  PmRowRun(@j, #PMELEM_CHEAP)
EndProcedure
CompilerEndIf

; The row kinds of PmRowRun: the serial kernel over rows [first, first + n).
Procedure PmRowTask(*j.PmRowJob, task.i, worker.i)
  Protected first.i = task * *j\Chunk, n.i = *j\Rows - first, w.i = *j\Width
  Protected g.PmTensorLayerNormArgs, *ln.PmTensorLayerNormArgs
  If n > *j\Chunk : n = *j\Chunk : EndIf
  If n <= 0 : ProcedureReturn : EndIf
  Select *j\Kind
    Case 0 : PmTensorSoftmaxLastSerial(*j\Src + first * w * 4, *j\Dst + first * w * 4, n, w)
    Case 1 : PmTensorReduceMeanLastSerial(*j\Src + first * w * 4, *j\Dst + first * 4, n, w)
    Case 2 : PmTensorReduceSumLastSerial(*j\Src + first * w * 4, *j\Dst + first * 4, n, w)
    Case 3
      *ln = *j\Args
      CopyStructure(*ln, @g, PmTensorLayerNormArgs)
      g\Src + first * w * 4 : g\Dst + first * w * 4 : g\Outer = n
      If g\Mean : g\Mean + first * 4 : EndIf
      If g\InvStdDev : g\InvStdDev + first * 4 : EndIf
      PmTensorLayerNormSerial(@g)
    CompilerIf #PMO_USE_RESIZE = 1
    Case 4 : PmTensorResize1DSerial(*j\Src + first * *j\I3 * 4, *j\Dst + first * w * 4, n, *j\I3, w, *j\I1, *j\I2, *j\F1)
    CompilerEndIf
    Case 5 : PmTensorMatMul2Serial(*j\Src + first * *j\I1 * 4, *j\Args, *j\Dst + first * *j\I2 * 4, n, *j\I1, *j\I2)
    Case 6 : PmTensorGemmRows(*j\Args, first, first + n)
    CompilerIf #PMO_USE_BATCHNORM = 1
    Case 7 : PmTensorBatchNormPlanes(*j\Args, first, first + n)
    Case 8 : PmTensorBatchNormTrainingChannels(*j\Args, first, first + n)
    CompilerEndIf
  EndSelect
EndProcedure

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

Structure PmConvRowsJob
  Args.i
  Rows.i
  Chunk.i
  Kind.i
EndStructure

Declare PmConvRowsTask(*j.PmConvRowsJob, task.i, worker.i)

; Output rows split across the pool; `rowMacs` is one row's work.
Procedure PmConvRowsRun(kind.i, *g, rows.i, rowMacs.q)
  Protected j.PmConvRowsJob, tasks.i, grain.i
  j\Kind = kind : j\Args = *g : j\Rows = rows
  If rowMacs < 1 : rowMacs = 1 : EndIf
  grain = (65536 + rowMacs - 1) / rowMacs
  tasks = PmPoolTasks(rows, grain, 1, @j\Chunk)
  If tasks <= 1
    j\Chunk = rows : PmConvRowsTask(@j, 0, 0)
  Else
    PmPoolRun(@PmConvRowsTask(), @j, tasks)
  EndIf
EndProcedure
CompilerIf #PMO_USE_CONVTRANSPOSE = 1
; Output rows [first, last), row = bn * OutChannels + oc. A row starts at its
; bias and takes its contributions in the order the whole-tensor loop gave
; them: input channel, then input position, then kernel tap, ascending.
Procedure PmTensorConvTranspose1DRows(*g.PmTensorConvTranspose1DArgs, first.i, last.i)
  Protected work.i
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
  work = first
  While work < last
    bn = work / *g\OutChannels
    oc = work % *g\OutChannels
    group = oc / outPerGroup
    ocg = oc % outPerGroup
    ox = 0
    While ox < *g\OutWidth
      value = 0.0
      If *g\Bias <> 0 : value = PmTensorGet(*g\Bias, oc) : EndIf
      PmTensorPut(*g\Dst, (bn * *g\OutChannels + oc) * *g\OutWidth + ox, value)
      ox = ox + 1
    Wend
    ic = group * inPerGroup
    While ic < (group + 1) * inPerGroup
      ix = 0
      While ix < *g\InWidth
        srcIndex = (bn * *g\InChannels + ic) * *g\InWidth + ix
        value = PmTensorGet(*g\Src, srcIndex)
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
        ix = ix + 1
      Wend
      ic = ic + 1
    Wend
    work = work + 1
  Wend
EndProcedure


Procedure PmTensorConvTranspose1D(*g.PmTensorConvTranspose1DArgs)
  Protected macs.q = *g\InWidth
  If *g\Groups <= 0 Or *g\OutChannels <= 0 : ProcedureReturn : EndIf
  macs * *g\Kernel * (*g\InChannels / *g\Groups)
  PmConvRowsRun(0, *g, *g\Batches * *g\OutChannels, macs)
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
  Wide.i
EndStructure

CompilerIf #PMO_USE_CONV = 1
CompilerIf #PMO_USE_INT8 = 1
; Quantizing a convolution input is split by position ranges over the same
; threads as the product: each position's scale needs every channel, and
; positions are independent.
Structure PmI8QuantJob
  Src.i
  Channels.i
  Width.i
  First.i
  Last.i
  PadLeft.i
  QMax.i
  Bits.i
  Scales.i
  Inv.i
  Xs.i
  Qp.i
  Groups.i
  PerGroup.i
  Pairs.i
  Fault.i
EndStructure

Procedure PmI8QuantWorker(*q.PmI8QuantJob)
  Protected n.i=*q\Last-*q\First, grp.i, i.i, c.i, at.i=*q\PadLeft+*q\First, src1.i
  If n<=0 : ProcedureReturn : EndIf
  PmI8MaxBitsColumns(*q\Src+*q\First*4,*q\Channels,*q\Width*4,n,*q\Bits+at*4)
  If PmI8ScalesFromBits(*q\Bits+at*4,n,*q\QMax,*q\Scales+at*4,*q\Inv+at*4)=0 : *q\Fault=1 : ProcedureReturn : EndIf
  For grp=0 To *q\Groups-1
    For i=0 To *q\Pairs-1
      c=grp* *q\PerGroup+2*i
      src1=0 : If 2*i+1<*q\PerGroup : src1=*q\Src+((c+1)* *q\Width+*q\First)*4 : EndIf
      PmI8QuantColumnPair(*q\Src+(c* *q\Width+*q\First)*4,src1,n,*q\Inv+at*4,*q\QMax,*q\Xs+((grp* *q\Pairs+i)* *q\Qp+at)*4)
    Next
  Next
EndProcedure

; Position ranges (multiples of eight) split across the pool; each task
; reports a NaN or an infinity in its own slot.
Structure PmI8QuantRun
  Proto.i
  Chunk.i
  Faults.i
EndStructure

Procedure PmI8QuantTask(*r.PmI8QuantRun, task.i, worker.i)
  Protected q.PmI8QuantJob
  CopyStructure(*r\Proto, @q, PmI8QuantJob)
  q\First = task * *r\Chunk : q\Last = q\First + *r\Chunk : q\Fault = 0
  If q\Last > q\Width : q\Last = q\Width : EndIf
  If q\First > q\Width : q\First = q\Width : EndIf
  PmI8QuantWorker(@q)
  If q\Fault : PokeI(*r\Faults + task * 8, 1) : EndIf
EndProcedure

Procedure.i PmI8QuantConvInput(*proto.PmI8QuantJob)
  Protected r.PmI8QuantRun, tasks.i, t.i, fault.i
  r\Proto = *proto
  tasks = PmPoolTasks(*proto\Width, PmPoolMax(8, 65536 / PmPoolMax(*proto\Channels, 1)), 8, @r\Chunk)
  If tasks <= 1
    *proto\First = 0 : *proto\Last = *proto\Width : *proto\Fault = 0
    PmI8QuantWorker(*proto)
    ProcedureReturn 1 - Bool(*proto\Fault <> 0)
  EndIf
  r\Faults = AllocateMemory(tasks * 8 + 8)
  If r\Faults = 0
    *proto\First = 0 : *proto\Last = *proto\Width : *proto\Fault = 0
    PmI8QuantWorker(*proto)
    ProcedureReturn 1 - Bool(*proto\Fault <> 0)
  EndIf
  PmPoolRun(@PmI8QuantTask(), @r, tasks)
  For t = 0 To tasks - 1 : If PeekI(r\Faults + t * 8) : fault = 1 : EndIf : Next
  FreeMemory(r\Faults)
  ProcedureReturn 1 - fault
EndProcedure

; 1-D convolution, NCW, W [out][in/groups][kernel] INT8 (both planes if wide).
; Activations are quantized once for the whole input: one scale per input
; position over every channel. Stride 1 reads the shared padded layout at an
; offset per tap; any other stride gathers each tap's positions first.
Procedure.i PmI8Conv1D(*g.PmTensorConv1DArgs, wide.i, elements.i)
  Protected cg.i=*g\InChannels / *g\Groups, og.i=*g\OutChannels / *g\Groups, pairs.i=(cg+1)/2
  Protected tiles.i=(*g\OutWidth+15)/16, owr.i=tiles*16, span.i=(*g\Kernel-1)* *g\Dilation
  Protected qp.i, padR.i, bn.i, grp.i, c.i, j.i, i.i, t.i, qmax.i=127, ok.i=1, rowsPadded.i
  Protected *xs, *ss, *inv, *bits, *xg, *sg, *wp, *src, *xtap, *stap, xbase.i, sbase.i
  Protected job.PmI8TileJob, qj.PmI8QuantJob
  If wide : qmax=32767 : EndIf
  ; padded positions: q = ix + PadLeft, enough that every tile of every tap stays inside
  qp=*g\PadLeft+*g\InWidth
  If *g\Stride=1
    If owr+span>qp : qp=owr+span : EndIf
  Else
    If (owr-1)* *g\Stride+span+1>qp : qp=(owr-1)* *g\Stride+span+1 : EndIf
  EndIf
  qp=(qp+15)&~15
  *xs=AllocateMemory(*g\Groups*pairs*qp*4+64) : *ss=AllocateMemory(qp*4+64)
  *inv=AllocateMemory(qp*4+64) : *bits=AllocateMemory(qp*4+64)
  *xtap=AllocateMemory(*g\Kernel*8) : *stap=AllocateMemory(*g\Kernel*8)
  If *g\Stride<>1
    *xg=AllocateMemory(*g\Kernel* *g\Groups*pairs*owr*4+64) : *sg=AllocateMemory(*g\Kernel*owr*4+64)
  EndIf
  *wp=PmI8PrepareConv(*g\Weight,*g\OutChannels,*g\Groups,cg,*g\Kernel,wide,elements)
  If *xs=0 Or *ss=0 Or *inv=0 Or *bits=0 Or *xtap=0 Or *stap=0 Or *wp=0 Or (*g\Stride<>1 And (*xg=0 Or *sg=0))
    ok=0
  EndIf
  bn=0
  While ok And bn<*g\Batches
    FillMemory(*xs,*g\Groups*pairs*qp*4,0)
    *src=*g\Src+bn* *g\InChannels* *g\InWidth*4
    qj\Src=*src : qj\Channels=*g\InChannels : qj\Width=*g\InWidth : qj\PadLeft=*g\PadLeft
    qj\QMax=qmax : qj\Bits=*bits : qj\Scales=*ss : qj\Inv=*inv : qj\Xs=*xs : qj\Qp=qp
    qj\Groups=*g\Groups : qj\PerGroup=cg : qj\Pairs=pairs : qj\Fault=0
    If PmI8QuantConvInput(@qj)=0 : PmTensorInt8Fault=1 : ok=0 : Break : EndIf
    If *g\Stride=1
      For j=0 To *g\Kernel-1 : PokeI(*xtap+j*8,j* *g\Dilation*4) : PokeI(*stap+j*8,j* *g\Dilation*4) : Next
      xbase=*xs : sbase=*ss : job\XStride=qp*4
    Else
      For j=0 To *g\Kernel-1
        For t=0 To owr-1
          PokeF(*sg+(j*owr+t)*4,PeekF(*ss+(t* *g\Stride+j* *g\Dilation)*4))
        Next
        For i=0 To *g\Groups*pairs-1
          For t=0 To owr-1
            PokeL(*xg+((j* *g\Groups*pairs+i)*owr+t)*4,PeekL(*xs+(i*qp+t* *g\Stride+j* *g\Dilation)*4))
          Next
        Next
        PokeI(*xtap+j*8,j* *g\Groups*pairs*owr*4) : PokeI(*stap+j*8,j*owr*4)
      Next
      xbase=*xg : sbase=*sg : job\XStride=owr*4
    EndIf
    rowsPadded=PmI8GroupRows(og,wide)
    For grp=0 To *g\Groups-1
      job\W=*wp : job\WRowBytes=*g\Kernel*pairs*4 : job\Pairs=pairs : job\Taps=*g\Kernel
      job\X=xbase+grp*pairs*job\XStride : job\XTap=*xtap : job\S=sbase : job\STap=*stap
      job\Flush=pairs : If wide : job\Flush=#PMI8_FLUSH_PAIRS : EndIf
      job\Mode=#PMI8_MODE_CONV : job\Wide=wide
      job\Scales=*g\WeightScales : job\Bias=*g\Bias
      job\Dst=*g\Dst+bn* *g\OutChannels* *g\OutWidth*4
      job\Rows=og : job\RowBase=grp*rowsPadded : job\ChannelBase=grp*og
      job\Positions=*g\OutWidth : job\DstRowStride=*g\OutWidth
      PmI8RunTiles(@job,grp*rowsPadded,rowsPadded,tiles)
    Next
    bn+1
  Wend
  If *xs : FreeMemory(*xs) : EndIf
  If *ss : FreeMemory(*ss) : EndIf
  If *inv : FreeMemory(*inv) : EndIf
  If *bits : FreeMemory(*bits) : EndIf
  If *xtap : FreeMemory(*xtap) : EndIf
  If *stap : FreeMemory(*stap) : EndIf
  If *xg : FreeMemory(*xg) : EndIf
  If *sg : FreeMemory(*sg) : EndIf
  ProcedureReturn ok
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
  Protected workCount.i = *g\Batches * *g\OutChannels
  Protected chunk.i
  Protected first.i
  Protected index.i
  Protected totalMacs.q
  Protected aScale.f
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    If PmI8Conv1D(*g,*g\Wide,*g\OutChannels*(*g\InChannels / *g\Groups)* *g\Kernel)=0 And PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
    ProcedureReturn
  EndIf
  CompilerEndIf
  ; Output rows (bn, oc) split across the pool; a row's sums are whole in one task.
  totalMacs = *g\OutWidth : totalMacs * (*g\InChannels / *g\Groups) : totalMacs * *g\Kernel
  PmConvRowsRun(1, *g, workCount, totalMacs)
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
; 2-D convolution with INT8 weights (the fixed-shape path; narrow only): one
; scale per input position (y, x) over every channel, one FP32 partial per tap.
; Output channels [first, last) of batch Bn, from the quantized input.
Structure PmI8Conv2DJob
  G.i
  Q.i
  S.i
  Positions.i
  Bn.i
EndStructure

Procedure PmI8Conv2DRows(*j.PmI8Conv2DJob, first.i, last.i)
  Protected *g.PmTensorConv2DArgs=*j\G, positions.i=*j\Positions, cg.i=*g\InChannels / *g\Groups, og.i=*g\OutChannels / *g\Groups
  Protected bn.i=*j\Bn, oc.i, oy.i, ox.i, ky.i, kx.i, iy.i, ix.i, c.i, grp.i, pos.i, acc.l
  Protected *s=*j\S, *q=*j\Q, w.i, f.f, v.f, bias.f
  For oc=first To last-1
    grp=oc/og
    bias=0.0 : If *g\Bias : bias=PeekF(*g\Bias+oc*4) : EndIf
    For oy=0 To *g\OutH-1
      For ox=0 To *g\OutW-1
        f=0.0
        For ky=0 To *g\KernelH-1
          iy=oy* *g\StrideH-*g\PadTop+ky* *g\DilationH
          If iy<0 Or iy>=*g\InH : Continue : EndIf
          For kx=0 To *g\KernelW-1
            ix=ox* *g\StrideW-*g\PadLeft+kx* *g\DilationW
            If ix<0 Or ix>=*g\InW : Continue : EndIf
            pos=iy* *g\InW+ix : acc=0
            For c=0 To cg-1
              w=*g\Weight+((oc*cg+c)* *g\KernelH+ky)* *g\KernelW+kx
              acc+PeekB(w)*PeekB(*q+(grp*cg+c)*positions+pos)
            Next
            v=acc : v=v*PeekF(*s+pos*4) : f=f+v
          Next
        Next
        v=f*PeekF(*g\WeightScales+oc*4) : v=bias+v
        PokeF(*g\Dst+(((bn* *g\OutChannels+oc)* *g\OutH+oy)* *g\OutW+ox)*4,v)
      Next
    Next
  Next
EndProcedure

Procedure.i PmI8Conv2D(*g.PmTensorConv2DArgs)
  Protected positions.i=*g\InH* *g\InW, cg.i=*g\InChannels / *g\Groups
  Protected bn.i, c.i, pos.i, q.i, ok.i=1
  Protected *bits, *s, *inv, *q, src.i
  Protected j.PmI8Conv2DJob, macs.q
  *bits=AllocateMemory(positions*4+64) : *s=AllocateMemory(positions*4+64)
  *inv=AllocateMemory(positions*4+64) : *q=AllocateMemory(*g\InChannels*positions+64)
  If *bits=0 Or *s=0 Or *inv=0 Or *q=0 : ok=0 : EndIf
  bn=0
  While ok And bn<*g\Batches
    src=*g\Src+bn* *g\InChannels*positions*4
    PmI8MaxBitsColumns(src,*g\InChannels,positions*4,positions,*bits)
    If PmI8ScalesFromBits(*bits,positions,127,*s,*inv)=0 : PmTensorInt8Fault=1 : ok=0 : Break : EndIf
    For c=0 To *g\InChannels-1
      For pos=0 To positions-1
        q=PmI8RoundProduct(PeekF(src+(c*positions+pos)*4),PeekF(*inv+pos*4))
        If q>127 : q=127 : ElseIf q<-127 : q=-127 : EndIf
        PokeB(*q+c*positions+pos,q)
      Next
    Next
    j\G=*g : j\Q=*q : j\S=*s : j\Positions=positions : j\Bn=bn
    macs=*g\OutH : macs* *g\OutW* *g\KernelH* *g\KernelW*cg
    PmConvRowsRun(3,@j,*g\OutChannels,macs)
    bn+1
  Wend
  If *bits : FreeMemory(*bits) : EndIf
  If *s : FreeMemory(*s) : EndIf
  If *inv : FreeMemory(*inv) : EndIf
  If *q : FreeMemory(*q) : EndIf
  ProcedureReturn ok
EndProcedure
CompilerEndIf

; Output rows [first, last), row = bn * OutChannels + oc, each row's sums whole.
Procedure PmTensorConv2DRows(*g.PmTensorConv2DArgs, first.i, last.i)
  Protected work.i
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
  work = first
  While work < last
    bn = work / *g\OutChannels
    oc = work % *g\OutChannels
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
    work = work + 1
  Wend
EndProcedure

Procedure PmTensorConv2D(*g.PmTensorConv2DArgs)
  Protected macs.q
  CompilerIf #PMO_USE_INT8 = 1
  If *g\WeightScales <> 0
    If PmI8Conv2D(*g)=0 And PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
    ProcedureReturn
  EndIf
  CompilerEndIf
  If *g\Groups <= 0 : ProcedureReturn : EndIf
  macs = *g\OutH : macs * *g\OutW * *g\KernelH * *g\KernelW * (*g\InChannels / *g\Groups)
  PmConvRowsRun(2, *g, *g\Batches * *g\OutChannels, macs)
EndProcedure
CompilerEndIf

; The conv row kinds of PmConvRowsRun.
Procedure PmConvRowsTask(*j.PmConvRowsJob, task.i, worker.i)
  Protected first.i = task * *j\Chunk, last.i = first + *j\Chunk
  CompilerIf #PMO_USE_CONV = 1
  Protected w.PmTensorConv1DWorker
  CompilerEndIf
  If last > *j\Rows : last = *j\Rows : EndIf
  If first >= last : ProcedureReturn : EndIf
  Select *j\Kind
    CompilerIf #PMO_USE_CONVTRANSPOSE = 1
    Case 0 : PmTensorConvTranspose1DRows(*j\Args, first, last)
    CompilerEndIf
    CompilerIf #PMO_USE_CONV = 1
    Case 1
      w\Args = *j\Args : w\WorkStart = first : w\WorkEnd = last
      PmTensorConv1DWorker(@w)
    Case 2 : PmTensorConv2DRows(*j\Args, first, last)
    CompilerIf #PMO_USE_INT8 = 1
    Case 3 : PmI8Conv2DRows(*j\Args, first, last)
    CompilerEndIf
    CompilerEndIf
  EndSelect
EndProcedure

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
  WWide.i
  RWide.i
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

CompilerIf #PMO_USE_INT8 = 1
; One gate's pre-activation from the scaled row sums: wScale * (F1 + F2 / 254).
; One rounding per operation (see PmI8Gemm).
Procedure.f PmI8GateValue(*f1, *f2, row.i, wide.i, *scales, scaleBase.i)
  Protected v.f=PeekF(*f1+row*4), t.f
  If wide : t=PeekF(*f2+row*4) : t=t/#PMI8_WIDE_RESIDUAL : v=v+t : EndIf
  v=v*PeekF(*scales+(scaleBase+row)*4)
  ProcedureReturn v
EndProcedure
CompilerEndIf

; One recurrent step of one batch row: units [u0, u1), each unit's four
; gates, cell and output computed whole by one task, from H(t-1) and C(t-1),
; which no task writes until the step is complete (YC(unit) is each unit's own).
Structure PmLstmStep
  G.i
  Dir.i
  T.i
  Bn.i
  XIndex.i
  Rows4.i
  GW1.i
  GW2.i
  GR1.i
  GR2.i
  Chunk.i
EndStructure

Procedure PmTensorLstmUnits(*s.PmLstmStep, u0.i, u1.i)
  Protected *g.PmTensorLstmArgs = *s\G
  Protected dir.i = *s\Dir, t.i = *s\T, bn.i = *s\Bn, xIndex.i = *s\XIndex, rows4.i = *s\Rows4
  Protected *gw1 = *s\GW1, *gw2 = *s\GW2, *gr1 = *s\GR1, *gr2 = *s\GR2
  Protected unit.i, k.i, stateIndex.i, wBase.i, rBase.i, bBase.i, yIndex.i, leftPointer.i, rightPointer.i
  Protected iv.f, ov.f, fv.f, cv.f, previousC.f, newC.f
  Protected floatFour.PmTensorLstmFloatFourArgs
  unit = u0
  While unit < u1
            iv = 0.0 : ov = 0.0 : fv = 0.0 : cv = 0.0
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
              iv = iv + PmI8GateValue(*gw1, *gw2, unit, *g\WWide, *g\WScales, dir * rows4)
              ov = ov + PmI8GateValue(*gw1, *gw2, *g\Hidden + unit, *g\WWide, *g\WScales, dir * rows4)
              fv = fv + PmI8GateValue(*gw1, *gw2, 2 * *g\Hidden + unit, *g\WWide, *g\WScales, dir * rows4)
              cv = cv + PmI8GateValue(*gw1, *gw2, 3 * *g\Hidden + unit, *g\WWide, *g\WScales, dir * rows4)
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
              iv = iv + PmI8GateValue(*gr1, *gr2, unit, *g\RWide, *g\RScales, dir * rows4)
              ov = ov + PmI8GateValue(*gr1, *gr2, *g\Hidden + unit, *g\RWide, *g\RScales, dir * rows4)
              fv = fv + PmI8GateValue(*gr1, *gr2, 2 * *g\Hidden + unit, *g\RWide, *g\RScales, dir * rows4)
              cv = cv + PmI8GateValue(*gr1, *gr2, 3 * *g\Hidden + unit, *g\RWide, *g\RScales, dir * rows4)
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
EndProcedure

Procedure PmLstmUnitsTask(*s.PmLstmStep, task.i, worker.i)
  Protected u0.i = task * *s\Chunk, u1.i = u0 + *s\Chunk, *g.PmTensorLstmArgs = *s\G
  If u1 > *g\Hidden : u1 = *g\Hidden : EndIf
  If u0 < u1 : PmTensorLstmUnits(*s, u0, u1) : EndIf
EndProcedure

Procedure PmTensorLstm(*g.PmTensorLstmArgs)
  Protected st.PmLstmStep, stepTasks.i
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
  CompilerIf #PMO_USE_INT8 = 1
  Protected rows4.i=4 * *g\Hidden, *xq, *hq, *acc, *gw1, *gw2, *gr1, *gr2, qx.i=127, qh.i=127
  Protected wElements.i=*g\Directions*rows4* *g\InputSize, rElements.i=*g\Directions*rows4* *g\Hidden
  If *g\WScales <> 0 Or *g\RScales <> 0
    If *g\WWide : qx=32767 : EndIf
    If *g\RWide : qh=32767 : EndIf
    *xq=AllocateMemory(*g\InputSize*2+64) : *hq=AllocateMemory(*g\Hidden*2+64) : *acc=AllocateMemory(rows4*4+64)
    *gw1=AllocateMemory(rows4*4+64) : *gw2=AllocateMemory(rows4*4+64)
    *gr1=AllocateMemory(rows4*4+64) : *gr2=AllocateMemory(rows4*4+64)
    If *xq=0 Or *hq=0 Or *acc=0 Or *gw1=0 Or *gw2=0 Or *gr1=0 Or *gr2=0
      PmTensorInt8Fault=2 : Goto PmTensorLstmInt8Done
    EndIf
  EndIf
  CompilerEndIf

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
          ; One scale for x(t), one for h(t-1); every gate row at once.
          If *g\WScales <> 0
            leftPointer = *g\X : leftPointer = leftPointer + xIndex * 4
            If PmI8QuantizeVector(leftPointer, *g\InputSize, qx, *xq, @xScale)=0 : PmTensorInt8Fault=1 : Goto PmTensorLstmInt8Done : EndIf
            rightPointer = *g\W + dir * rows4 * *g\InputSize
            PmI8DotRowsScaled(rightPointer, *g\InputSize, rows4, *xq, *g\InputSize, xScale, *g\WWide, *acc, *gw1)
            If *g\WWide : PmI8DotRowsScaled(rightPointer + wElements, *g\InputSize, rows4, *xq, *g\InputSize, xScale, 1, *acc, *gw2) : EndIf
          EndIf
          If *g\RScales <> 0
            leftPointer = *g\YH : leftPointer = leftPointer + stateIndex * 4
            If PmI8QuantizeVector(leftPointer, *g\Hidden, qh, *hq, @hScale)=0 : PmTensorInt8Fault=1 : Goto PmTensorLstmInt8Done : EndIf
            rightPointer = *g\R + dir * rows4 * *g\Hidden
            PmI8DotRowsScaled(rightPointer, *g\Hidden, rows4, *hq, *g\Hidden, hScale, *g\RWide, *acc, *gr1)
            If *g\RWide : PmI8DotRowsScaled(rightPointer + rElements, *g\Hidden, rows4, *hq, *g\Hidden, hScale, 1, *acc, *gr2) : EndIf
          EndIf
          CompilerEndIf
          st\G = *g : st\Dir = dir : st\T = t : st\Bn = bn : st\XIndex = xIndex
          CompilerIf #PMO_USE_INT8 = 1
          st\Rows4 = rows4 : st\GW1 = *gw1 : st\GW2 = *gw2 : st\GR1 = *gr1 : st\GR2 = *gr2
          CompilerEndIf
          stepTasks = PmPoolTasks(*g\Hidden, PmPoolMax(4, 2048 / PmPoolMax(*g\InputSize + *g\Hidden, 1)), 4, @st\Chunk)
          If stepTasks <= 1
            PmTensorLstmUnits(@st, 0, *g\Hidden)
          Else
            PmPoolRun(@PmLstmUnitsTask(), @st, stepTasks)
          EndIf
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
  CompilerIf #PMO_USE_INT8 = 1
  PmTensorLstmInt8Done:
  If *xq : FreeMemory(*xq) : EndIf
  If *hq : FreeMemory(*hq) : EndIf
  If *acc : FreeMemory(*acc) : EndIf
  If *gw1 : FreeMemory(*gw1) : EndIf
  If *gw2 : FreeMemory(*gw2) : EndIf
  If *gr1 : FreeMemory(*gr1) : EndIf
  If *gr2 : FreeMemory(*gr2) : EndIf
  CompilerEndIf
EndProcedure

CompilerIf #PMO_USE_INT8 = 1
; INT8 LSTM with both weights quantized, in the host's fast shape: every x(t)
; projected at once by the INT8 product (one scale per step, as the scheme
; says), then per step one scale for h(t-1) and every recurrent row at once.
; Returns 0 on a fault (PmTensorInt8Fault says which).
; One recurrent step of the INT8 LSTM: units [u0, u1). The four gate rows of
; each unit are dotted with the quantized H(t-1) (rows g*H + u, each whole in
; this task, into this task's own slices of Acc, R1 and R2), then the gates.
Structure PmI8LstmStep
  G.i
  Dir.i
  Hq.i
  Hs.f
  Acc.i
  R1.i
  R2.i
  XProj.i
  InputRow.i
  State.i
  OutRow.i
  Chunk.i
EndStructure

Procedure PmI8LstmUnits(*s.PmI8LstmStep, u0.i, u1.i)
  Protected *g.PmTensorLstmArgs=*s\G, width.i=4* *g\Hidden, dir.i=*s\Dir, gate.i, row.i, unit.i, n.i=u1-u0
  Protected rElements.i=*g\Directions*width* *g\Hidden, bbase.i=dir*8* *g\Hidden
  Protected *r1=*s\R1, *r2=*s\R2, *xproj=*s\XProj, inputRow.i=*s\InputRow, state.i=*s\State, outRow.i=*s\OutRow
  Protected iv.f, ov.f, fv.f, cv.f, previous.f
  If n<=0 : ProcedureReturn : EndIf
  For gate=0 To 3
    row=gate* *g\Hidden+u0
    PmI8DotRowsScaled(*g\R+dir*width* *g\Hidden+row* *g\Hidden,*g\Hidden,n,*s\Hq,*g\Hidden,*s\Hs,*g\RWide,*s\Acc+row*4,*r1+row*4)
    If *g\RWide : PmI8DotRowsScaled(*g\R+rElements+dir*width* *g\Hidden+row* *g\Hidden,*g\Hidden,n,*s\Hq,*g\Hidden,*s\Hs,1,*s\Acc+row*4,*r2+row*4) : EndIf
  Next
  For unit=u0 To u1-1
    iv=0 : ov=0 : fv=0 : cv=0
    If *g\B
      iv=PeekF(*g\B+(bbase+unit)*4)+PeekF(*g\B+(bbase+4* *g\Hidden+unit)*4)
      ov=PeekF(*g\B+(bbase+ *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+5* *g\Hidden+unit)*4)
      fv=PeekF(*g\B+(bbase+2* *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+6* *g\Hidden+unit)*4)
      cv=PeekF(*g\B+(bbase+3* *g\Hidden+unit)*4)+PeekF(*g\B+(bbase+7* *g\Hidden+unit)*4)
    EndIf
    iv+PeekF(*xproj+(inputRow+unit)*4) : iv+PmI8GateValue(*r1,*r2,unit,*g\RWide,*g\RScales,dir*width)
    ov+PeekF(*xproj+(inputRow+ *g\Hidden+unit)*4) : ov+PmI8GateValue(*r1,*r2,*g\Hidden+unit,*g\RWide,*g\RScales,dir*width)
    fv+PeekF(*xproj+(inputRow+2* *g\Hidden+unit)*4) : fv+PmI8GateValue(*r1,*r2,2* *g\Hidden+unit,*g\RWide,*g\RScales,dir*width)
    cv+PeekF(*xproj+(inputRow+3* *g\Hidden+unit)*4) : cv+PmI8GateValue(*r1,*r2,3* *g\Hidden+unit,*g\RWide,*g\RScales,dir*width)
    previous=PeekF(*g\YC+(state+unit)*4)
    cv=PmTensorSigmoidValue(fv)*previous+PmTensorSigmoidValue(iv)*PmTensorTanhValue(cv)
    ov=PmTensorSigmoidValue(ov)*PmTensorTanhValue(cv)
    PokeF(*g\YC+(state+unit)*4,cv) : PokeF(*g\Y+(outRow+unit)*4,ov)
  Next
EndProcedure

Procedure PmI8LstmUnitsTask(*s.PmI8LstmStep, task.i, worker.i)
  Protected u0.i=task* *s\Chunk, u1.i=u0+ *s\Chunk, *g.PmTensorLstmArgs=*s\G
  If u1>*g\Hidden : u1=*g\Hidden : EndIf
  PmI8LstmUnits(*s,u0,u1)
EndProcedure
Procedure.i PmI8Lstm(*g.PmTensorLstmArgs)
  Protected st.PmI8LstmStep, stepTasks.i
  Protected width.i=4* *g\Hidden, dir.i, i.i, t.i, bn.i, unit.i, stepIndex.i, valid.i, state.i, inputRow.i, outRow.i, bbase.i
  Protected wElements.i=*g\Directions*width* *g\InputSize, rElements.i=*g\Directions*width* *g\Hidden, qh.i=127
  Protected *xproj, *hq, *acc, *r1, *r2, hs.f, iv.f, ov.f, fv.f, cv.f, previous.f, ok.i=1
  If *g\RWide : qh=32767 : EndIf
  *xproj=AllocateMemory(width* *g\Sequence* *g\Batch*4+64) : *hq=AllocateMemory(*g\Hidden*2+64)
  *acc=AllocateMemory(width*4+64) : *r1=AllocateMemory(width*4+64) : *r2=AllocateMemory(width*4+64)
  If *xproj=0 Or *hq=0 Or *acc=0 Or *r1=0 Or *r2=0 : PmTensorInt8Fault=2 : ok=0 : EndIf
  If ok
    For i=0 To *g\Directions* *g\Batch* *g\Hidden-1
      iv=0 : cv=0
      If *g\InitialH : iv=PeekF(*g\InitialH+i*4) : EndIf
      If *g\InitialC : cv=PeekF(*g\InitialC+i*4) : EndIf
      PokeF(*g\YH+i*4,iv) : PokeF(*g\YC+i*4,cv)
    Next
  EndIf
  stepTasks=PmPoolTasks(*g\Hidden,PmPoolMax(4,4096/PmPoolMax(*g\Hidden,1)),4,@st\Chunk)
  dir=0
  While ok And dir<*g\Directions
    If PmI8MatMul(*g\X,*g\W+dir*width* *g\InputSize,*xproj,*g\Sequence* *g\Batch,*g\InputSize,width,*g\WScales+dir*width*4,*g\WWide,wElements,1)=0
      If PmTensorInt8Fault=0 : PmTensorInt8Fault=2 : EndIf
      ok=0 : Break
    EndIf
    For stepIndex=0 To *g\Sequence-1
      For bn=0 To *g\Batch-1
        valid=*g\Sequence
        If *g\SeqLens : valid=PeekQ(*g\SeqLens+bn*8) : EndIf
        If stepIndex>=valid : Continue : EndIf
        t=stepIndex : If dir=1 : t=valid-1-stepIndex : EndIf
        inputRow=(t* *g\Batch+bn)*width
        state=(dir* *g\Batch+bn)* *g\Hidden
        outRow=((t* *g\Directions+dir)* *g\Batch+bn)* *g\Hidden
        bbase=dir*8* *g\Hidden
        If PmI8QuantizeVector(*g\YH+state*4,*g\Hidden,qh,*hq,@hs)=0 : PmTensorInt8Fault=1 : ok=0 : Break 2 : EndIf
        st\G=*g : st\Dir=dir : st\Hq=*hq : st\Hs=hs : st\Acc=*acc : st\R1=*r1 : st\R2=*r2 : st\XProj=*xproj
        st\InputRow=inputRow : st\State=state : st\OutRow=outRow
        If stepTasks<=1
          PmI8LstmUnits(@st,0,*g\Hidden)
        Else
          PmPoolRun(@PmI8LstmUnitsTask(),@st,stepTasks)
        EndIf
        CopyMemory(*g\Y+outRow*4,*g\YH+state*4,*g\Hidden*4)
      Next
    Next
    dir+1
  Wend
  If *xproj : FreeMemory(*xproj) : EndIf
  If *hq : FreeMemory(*hq) : EndIf
  If *acc : FreeMemory(*acc) : EndIf
  If *r1 : FreeMemory(*r1) : EndIf
  If *r2 : FreeMemory(*r2) : EndIf
  ProcedureReturn ok
EndProcedure
CompilerEndIf

CompilerEndIf
