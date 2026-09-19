; ======================================================================
; tensor_norm_small_windows.pbi - Windows x64 kernels for
; InstanceNormalization, TopK, ScatterElements, ReduceMax, ReduceProd, Not
; and Pad (constant, reflect, edge, wrap).
; ----------------------------------------------------------------------
; Windows counterpart of tensor_norm_small.pmi; keep the two synchronized.
; Each kernel is written from the ONNX operator specification,
; https://onnx.ai/onnx/operators/, and names the section it implements.
; Procedures take byte addresses and do no shape inference: the fixed-shape
; emitter and the runtime-dimension wrappers have already checked shapes,
; element types and attributes before any of these run.
;
; Element kinds are ONNX TensorProto codes: 1 FLOAT (4 bytes), 6 INT32
; (4 bytes), 7 INT64 (8 bytes), 9 BOOL (1 byte).
; ======================================================================

; ----------------------------------------------------------------------
; InstanceNormalization-6 (the definition current through opset 20):
;   y = scale * (x - mean) / sqrt(variance + epsilon) + B
; with mean and variance computed per instance per channel over the
; spatial elements D1..Dn of an (N x C x D1 ... Dn) input.
;
; THE REFERENCE ORDER OF EVALUATION, which the vector body below and the
; A64 body in tensor_norm_small.pmi reproduce bit for bit:
;   * mean: four running sums, element k of the plane going to lane k mod 4
;     in index order; sum = (lane0 + lane1) + (lane2 + lane3); mean = sum / n
;   * variance: the same four lanes over d = x - mean, d = d * d; the same
;     combination; variance = sum / n
;   * den = sqrt(variance + epsilon)
;   * per element, one binary32 operation per step:
;     t = x - mean, t = scale * t, t = t / den, y = t + B
; Every step is a single IEEE binary32 operation, so a lane of four performs
; exactly the operations the scalar form performs, in the same order, and
; signed zeros, infinities, subnormals and NaN operands give the same bits.
;
; Every floating-point step is written in scalar SSE here rather than in
; host-language arithmetic, because the host's float code runs on the x87
; stack, which (for example) turns a signalling NaN operand into a quiet one
; on load - the reference must be a definition, not an approximation of one.
; ----------------------------------------------------------------------
Structure PmTensorInstanceNormWork
  Lane0.f
  Lane1.f
  Lane2.f
  Lane3.f
  Mean.f
  Scale.f
  Den.f
  Bias.f
  Count.f
  Epsilon.f
EndStructure

CompilerIf #PB_Compiler_Backend = #PB_Backend_Asm And #PB_Compiler_Processor = #PB_Processor_x64

; Four lane sums of the plane (squares=0) or of (x - mean)^2 (squares=1),
; one element at a time: the scalar reference.
Procedure PmTensorInstanceNormLanes(*src, count.i, *work, squares.i)
  Protected ps.i = *src, n.i = count, pw.i = *work, sq.i = squares
  !mov rax,[p.v_ps]
  !mov rcx,[p.v_n]
  !mov rdx,[p.v_pw]
  !mov r10,[p.v_sq]
  !xorps xmm0,xmm0
  !movss [rdx],xmm0
  !movss [rdx+4],xmm0
  !movss [rdx+8],xmm0
  !movss [rdx+12],xmm0
  !movss xmm2,[rdx+16]
  !xor r8,r8
  !test rcx,rcx
  !jle pmnsinl_done
  !pmnsinl_loop:
  !movss xmm1,[rax+r8*4]
  !test r10,r10
  !jz pmnsinl_plain
  !subss xmm1,xmm2
  !mulss xmm1,xmm1
  !pmnsinl_plain:
  !mov r9,r8
  !and r9,3
  !movss xmm0,[rdx+r9*4]
  !addss xmm0,xmm1
  !movss [rdx+r9*4],xmm0
  !inc r8
  !cmp r8,rcx
  !jl pmnsinl_loop
  !pmnsinl_done:
EndProcedure

; The same lanes, four elements per SSE instruction. Lane j receives
; elements j, j+4, j+8 ... in index order, exactly as the reference does;
; the one to three trailing elements are added by the reference's own
; scalar instructions.
Procedure PmFastInstanceNormLanes(*src, count.i, *work, squares.i)
  Protected ps.i = *src, n.i = count, pw.i = *work, sq.i = squares
  !mov rax,[p.v_ps]
  !mov rcx,[p.v_n]
  !mov rdx,[p.v_pw]
  !mov r10,[p.v_sq]
  !xorps xmm0,xmm0
  !movss xmm2,[rdx+16]
  !shufps xmm2,xmm2,0
  !xor r8,r8
  !pmnsfnl_four:
  !lea r9,[r8+4]
  !cmp r9,rcx
  !jg pmnsfnl_tailstart
  !movups xmm1,[rax+r8*4]
  !test r10,r10
  !jz pmnsfnl_plain
  !subps xmm1,xmm2
  !mulps xmm1,xmm1
  !pmnsfnl_plain:
  !addps xmm0,xmm1
  !mov r8,r9
  !jmp pmnsfnl_four
  !pmnsfnl_tailstart:
  !movups [rdx],xmm0
  !cmp r8,rcx
  !jge pmnsfnl_done
  !pmnsfnl_tail:
  !movss xmm1,[rax+r8*4]
  !test r10,r10
  !jz pmnsfnl_tailplain
  !subss xmm1,xmm2
  !mulss xmm1,xmm1
  !pmnsfnl_tailplain:
  !mov r9,r8
  !and r9,3
  !movss xmm0,[rdx+r9*4]
  !addss xmm0,xmm1
  !movss [rdx+r9*4],xmm0
  !inc r8
  !cmp r8,rcx
  !jl pmnsfnl_tail
  !pmnsfnl_done:
EndProcedure

; Combines the lanes and divides by the element count. Stage 0 stores the
; mean; stage 1 stores den = sqrt(variance + epsilon). Shared by both forms.
Procedure PmTensorInstanceNormStats(*work, stage.i)
  Protected pw.i = *work, st.i = stage
  !mov rdx,[p.v_pw]
  !movss xmm0,[rdx]
  !addss xmm0,[rdx+4]
  !movss xmm1,[rdx+8]
  !addss xmm1,[rdx+12]
  !addss xmm0,xmm1
  !divss xmm0,[rdx+32]
  !cmp qword [p.v_st],0
  !jne pmnsins_den
  !movss [rdx+16],xmm0
  !jmp pmnsins_done
  !pmnsins_den:
  !addss xmm0,[rdx+36]
  !sqrtss xmm0,xmm0
  !movss [rdx+24],xmm0
  !pmnsins_done:
EndProcedure

; y = ((scale * (x - mean)) / den) + B, one element at a time.
Procedure PmTensorInstanceNormAffine(*src, *dst, count.i, *work)
  Protected ps.i = *src, pd.i = *dst, n.i = count, pw.i = *work
  !mov rax,[p.v_ps]
  !mov r11,[p.v_pd]
  !mov rcx,[p.v_n]
  !mov rdx,[p.v_pw]
  !xor r8,r8
  !test rcx,rcx
  !jle pmnsina_done
  !pmnsina_loop:
  !movss xmm0,[rax+r8*4]
  !subss xmm0,[rdx+16]
  !movss xmm1,[rdx+20]
  !mulss xmm1,xmm0
  !divss xmm1,[rdx+24]
  !addss xmm1,[rdx+28]
  !movss [r11+r8*4],xmm1
  !inc r8
  !cmp r8,rcx
  !jl pmnsina_loop
  !pmnsina_done:
EndProcedure

; The affine step four elements per instruction, in the same operand order.
Procedure PmFastInstanceNormAffine(*src, *dst, count.i, *work)
  Protected ps.i = *src, pd.i = *dst, n.i = count, pw.i = *work
  !mov rax,[p.v_ps]
  !mov r11,[p.v_pd]
  !mov rcx,[p.v_n]
  !mov rdx,[p.v_pw]
  !movss xmm2,[rdx+16]
  !shufps xmm2,xmm2,0
  !movss xmm3,[rdx+20]
  !shufps xmm3,xmm3,0
  !movss xmm4,[rdx+24]
  !shufps xmm4,xmm4,0
  !movss xmm5,[rdx+28]
  !shufps xmm5,xmm5,0
  !xor r8,r8
  !pmnsfna_four:
  !lea r9,[r8+4]
  !cmp r9,rcx
  !jg pmnsfna_tail
  !movups xmm0,[rax+r8*4]
  !subps xmm0,xmm2
  !movaps xmm1,xmm3
  !mulps xmm1,xmm0
  !divps xmm1,xmm4
  !addps xmm1,xmm5
  !movups [r11+r8*4],xmm1
  !mov r8,r9
  !jmp pmnsfna_four
  !pmnsfna_tail:
  !cmp r8,rcx
  !jge pmnsfna_done
  !movss xmm0,[rax+r8*4]
  !subss xmm0,[rdx+16]
  !movss xmm1,[rdx+20]
  !mulss xmm1,xmm0
  !divss xmm1,[rdx+24]
  !addss xmm1,[rdx+28]
  !movss [r11+r8*4],xmm1
  !inc r8
  !jmp pmnsfna_tail
  !pmnsfna_done:
EndProcedure

CompilerElse

; Without the assembly backend there is no binary32-exact scalar path to
; define the reference by, so both forms are this one loop and the vector
; claim is not made.
Procedure PmTensorInstanceNormLanes(*src, count.i, *work.PmTensorInstanceNormWork, squares.i)
  Protected k.i, x.f, lane.i
  Protected Dim acc.f(3)
  For k = 0 To count - 1
    x = PeekF(*src + k * 4)
    If squares : x = x - *work\Mean : x = x * x : EndIf
    lane = k & 3 : acc(lane) = acc(lane) + x
  Next
  *work\Lane0 = acc(0) : *work\Lane1 = acc(1) : *work\Lane2 = acc(2) : *work\Lane3 = acc(3)
EndProcedure
Procedure PmFastInstanceNormLanes(*src, count.i, *work, squares.i)
  PmTensorInstanceNormLanes(*src, count, *work, squares)
EndProcedure
Procedure PmTensorInstanceNormStats(*work.PmTensorInstanceNormWork, stage.i)
  Protected a.f, b.f
  a = *work\Lane0 + *work\Lane1 : b = *work\Lane2 + *work\Lane3 : a = a + b : a = a / *work\Count
  If stage = 0 : *work\Mean = a : Else : a = a + *work\Epsilon : *work\Den = Sqr(a) : EndIf
EndProcedure
Procedure PmTensorInstanceNormAffine(*src, *dst, count.i, *work.PmTensorInstanceNormWork)
  Protected k.i, t.f
  For k = 0 To count - 1
    t = PeekF(*src + k * 4) - *work\Mean : t = *work\Scale * t : t = t / *work\Den : t = t + *work\Bias
    PokeF(*dst + k * 4, t)
  Next
EndProcedure
Procedure PmFastInstanceNormAffine(*src, *dst, count.i, *work)
  PmTensorInstanceNormAffine(*src, *dst, count, *work)
EndProcedure

CompilerEndIf

; The whole operator. fast=0 runs the scalar reference, fast=1 the vector
; bodies; the two write identical bits. scale and bias hold `channels`
; elements each; *dst may equal *src.
Procedure PmTensorInstanceNormRun(*src, *scale, *bias, *dst, batches.i, channels.i, spatial.i, epsilon.f, fast.i)
  Protected work.PmTensorInstanceNormWork
  Protected n.i, c.i, plane.i
  If spatial <= 0 Or batches <= 0 Or channels <= 0 : ProcedureReturn : EndIf
  ; Integer-to-float of the count and the epsilon attribute's bits are the
  ; only host-language float moves, and both are shared by the two forms.
  work\Count = spatial
  PokeL(@work\Epsilon, PeekL(@epsilon))
  For n = 0 To batches - 1
    For c = 0 To channels - 1
      plane = (n * channels + c) * spatial * 4
      PokeL(@work\Scale, PeekL(*scale + c * 4))
      PokeL(@work\Bias, PeekL(*bias + c * 4))
      If fast
        PmFastInstanceNormLanes(*src + plane, spatial, @work, 0)
        PmTensorInstanceNormStats(@work, 0)
        PmFastInstanceNormLanes(*src + plane, spatial, @work, 1)
        PmTensorInstanceNormStats(@work, 1)
        PmFastInstanceNormAffine(*src + plane, *dst + plane, spatial, @work)
      Else
        PmTensorInstanceNormLanes(*src + plane, spatial, @work, 0)
        PmTensorInstanceNormStats(@work, 0)
        PmTensorInstanceNormLanes(*src + plane, spatial, @work, 1)
        PmTensorInstanceNormStats(@work, 1)
        PmTensorInstanceNormAffine(*src + plane, *dst + plane, spatial, @work)
      EndIf
    Next
  Next
EndProcedure

Procedure PmTensorInstanceNorm(*src, *scale, *bias, *dst, batches.i, channels.i, spatial.i, epsilon.f)
  PmTensorInstanceNormRun(*src, *scale, *bias, *dst, batches, channels, spatial, epsilon, 0)
EndProcedure

Procedure PmFastInstanceNorm(*src, *scale, *bias, *dst, batches.i, channels.i, spatial.i, epsilon.f)
  PmTensorInstanceNormRun(*src, *scale, *bias, *dst, batches, channels, spatial, epsilon, 1)
EndProcedure

; ----------------------------------------------------------------------
; Element helpers shared by the selection kernels below.
; ----------------------------------------------------------------------
Procedure.i PmNsElementBytes(kind.i)
  Select kind
    Case 1, 6 : ProcedureReturn 4
    Case 7 : ProcedureReturn 8
    Case 9 : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.i PmNsIsNan(*value)
  ProcedureReturn Bool((PeekL(*value) & $7FFFFFFF) > $7F800000)
EndProcedure

Procedure.i PmNsReadInt(*base, index.i, kind.i)
  Select kind
    Case 7 : ProcedureReturn PeekQ(*base + index * 8)
    Case 6 : ProcedureReturn PeekL(*base + index * 4)
    Case 9 : ProcedureReturn PeekA(*base + index) & 255
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure PmNsWriteInt(*base, index.i, kind.i, value.i)
  Select kind
    Case 7 : PokeQ(*base + index * 8, value)
    Case 6 : PokeL(*base + index * 4, value)
    Case 9 : PokeA(*base + index, Bool(value <> 0))
  EndSelect
EndProcedure

; 1 when element *a orders strictly above element *b for the given kind.
; FLOAT: a NaN orders above every number and equal to another NaN, which is
; the order numpy's sort uses; the specification leaves NaN unordered.
Procedure.i PmNsGreater(*a, *b, kind.i)
  Protected an.i, bn.i
  If kind = 1
    an = PmNsIsNan(*a) : bn = PmNsIsNan(*b)
    If an <> 0 Or bn <> 0 : ProcedureReturn Bool(an <> 0 And bn = 0) : EndIf
    ProcedureReturn Bool(PeekF(*a) > PeekF(*b))
  EndIf
  ProcedureReturn Bool(PmNsReadInt(*a, 0, kind) > PmNsReadInt(*b, 0, kind))
EndProcedure

; ----------------------------------------------------------------------
; TopK-11 (current through opset 20). For every slice along the axis the
; k largest (Largest=1) or smallest elements, in sorted order; "given two
; equivalent values, ... the element with the lower index will appear
; first". The slice is ordered by a stable bottom-up merge sort of its
; indices, so equal elements keep index order, and the first k are written.
; sorted=0 leaves the order unspecified; this kernel writes sorted order
; for it too. Order must address 2 * Width native integers of scratch.
; Kinds 1, 6 and 7; Indices are INT64.
; ----------------------------------------------------------------------
Structure PmTensorTopKArgs
  Src.i
  Values.i
  Indices.i
  Order.i
  Outer.i
  Width.i
  Inner.i
  K.i
  Largest.i
  Kind.i
EndStructure

Procedure PmTensorTopK(*g.PmTensorTopKArgs)
  Protected o.i, i.i, j.i, size.i, base.i, span.i, lo.i, mid.i, hi.i, a.i, b.i, out.i
  Protected *from, *into, *swap, ia.i, ib.i, takeRight.i
  size = PmNsElementBytes(*g\Kind)
  If size = 0 Or *g\Width <= 0 Or *g\K <= 0 : ProcedureReturn : EndIf
  For o = 0 To *g\Outer - 1
    For i = 0 To *g\Inner - 1
      base = o * *g\Width * *g\Inner + i
      *from = *g\Order : *into = *g\Order + *g\Width * SizeOf(Integer)
      For j = 0 To *g\Width - 1 : PokeI(*from + j * SizeOf(Integer), j) : Next
      span = 1
      While span < *g\Width
        lo = 0
        While lo < *g\Width
          mid = lo + span : If mid > *g\Width : mid = *g\Width : EndIf
          hi = lo + 2 * span : If hi > *g\Width : hi = *g\Width : EndIf
          a = lo : b = mid : out = lo
          While a < mid Or b < hi
            If a >= mid
              takeRight = 1
            ElseIf b >= hi
              takeRight = 0
            Else
              ia = PeekI(*from + a * SizeOf(Integer)) : ib = PeekI(*from + b * SizeOf(Integer))
              ; The right element goes first only when it orders strictly
              ; before the left one, which keeps equal elements in index order.
              If *g\Largest
                takeRight = PmNsGreater(*g\Src + (base + ib * *g\Inner) * size, *g\Src + (base + ia * *g\Inner) * size, *g\Kind)
              Else
                takeRight = PmNsGreater(*g\Src + (base + ia * *g\Inner) * size, *g\Src + (base + ib * *g\Inner) * size, *g\Kind)
              EndIf
            EndIf
            If takeRight
              PokeI(*into + out * SizeOf(Integer), PeekI(*from + b * SizeOf(Integer))) : b + 1
            Else
              PokeI(*into + out * SizeOf(Integer), PeekI(*from + a * SizeOf(Integer))) : a + 1
            EndIf
            out + 1
          Wend
          lo = hi
        Wend
        *swap = *from : *from = *into : *into = *swap
        span * 2
      Wend
      For j = 0 To *g\K - 1
        ia = PeekI(*from + j * SizeOf(Integer))
        CopyMemory(*g\Src + (base + ia * *g\Inner) * size, *g\Values + ((o * *g\K + j) * *g\Inner + i) * size, size)
        PokeQ(*g\Indices + ((o * *g\K + j) * *g\Inner + i) * 8, ia)
      Next
    Next
  Next
EndProcedure

; ----------------------------------------------------------------------
; ScatterElements-18 (current through opset 20). Dst starts as a copy of
; Data; every element of Indices, visited in row-major order, names the
; target whose coordinate on Axis is that index value (negative counts from
; the end) and whose other coordinates are the element's own; the matching
; Updates element is stored (Reduction 0) or combined with the target by
; 1 add, 2 mul, 3 max, 4 min. FLOAT max/min propagate a NaN from either
; side, as numpy.maximum/minimum do. BOOL takes none, max (or) and min (and);
; add and mul on BOOL are refused. Integer add and mul wrap.
; DataDims and IndexDims address Rank native integers each.
; Returns 0, or 1 for an index outside [-s, s-1], or 2 for a refused
; reduction.
; ----------------------------------------------------------------------
Structure PmTensorScatterElementsArgs
  Data.i
  Indices.i
  Updates.i
  Dst.i
  Rank.i
  Axis.i
  Reduction.i
  Kind.i
  IndexKind.i
  DataDims.i
  IndexDims.i
EndStructure

Procedure.i PmTensorScatterElements(*g.PmTensorScatterElementsArgs)
  Protected Dim stride.i(8)
  Protected i.i, d.i, count.i, dataCount.i, size.i, n.i, coord.i, target.i, index.i, extent.i
  Protected cur.f, upd.f, ci.i, ui.i, *t, *u
  size = PmNsElementBytes(*g\Kind)
  If size = 0 : ProcedureReturn 2 : EndIf
  If *g\Kind = 9 And (*g\Reduction = 1 Or *g\Reduction = 2) : ProcedureReturn 2 : EndIf
  count = 1 : dataCount = 1
  For d = *g\Rank - 1 To 0 Step -1
    stride(d) = dataCount
    dataCount * PeekI(*g\DataDims + d * SizeOf(Integer))
    count * PeekI(*g\IndexDims + d * SizeOf(Integer))
  Next
  If *g\Dst <> *g\Data : CopyMemory(*g\Data, *g\Dst, dataCount * size) : EndIf
  extent = PeekI(*g\DataDims + *g\Axis * SizeOf(Integer))
  For i = 0 To count - 1
    n = i : target = 0
    For d = *g\Rank - 1 To 0 Step -1
      coord = n % PeekI(*g\IndexDims + d * SizeOf(Integer))
      n / PeekI(*g\IndexDims + d * SizeOf(Integer))
      If d = *g\Axis
        index = PmNsReadInt(*g\Indices, i, *g\IndexKind)
        If index < 0 : index + extent : EndIf
        If index < 0 Or index >= extent : ProcedureReturn 1 : EndIf
        coord = index
      EndIf
      target + coord * stride(d)
    Next
    *t = *g\Dst + target * size : *u = *g\Updates + i * size
    If *g\Reduction = 0
      CopyMemory(*u, *t, size)
    ElseIf *g\Kind = 1
      Select *g\Reduction
        Case 1
          cur = PeekF(*t) : upd = PeekF(*u) : cur = cur + upd : PokeF(*t, cur)
        Case 2
          cur = PeekF(*t) : upd = PeekF(*u) : cur = cur * upd : PokeF(*t, cur)
        Case 3
          If PmNsIsNan(*t) = 0 And (PmNsIsNan(*u) Or PeekF(*u) > PeekF(*t)) : PokeL(*t, PeekL(*u)) : EndIf
        Case 4
          If PmNsIsNan(*t) = 0 And (PmNsIsNan(*u) Or PeekF(*u) < PeekF(*t)) : PokeL(*t, PeekL(*u)) : EndIf
      EndSelect
    Else
      ci = PmNsReadInt(*t, 0, *g\Kind) : ui = PmNsReadInt(*u, 0, *g\Kind)
      Select *g\Reduction
        Case 1 : ci = ci + ui
        Case 2 : ci = ci * ui
        Case 3 : If ui > ci : ci = ui : EndIf
        Case 4 : If ui < ci : ci = ui : EndIf
      EndSelect
      PmNsWriteInt(*t, 0, *g\Kind, ci)
    EndIf
  Next
  ProcedureReturn 0
EndProcedure

; ----------------------------------------------------------------------
; ReduceMax-20 and ReduceProd-18 (and their earlier attribute forms, which
; reduce the same way). Mask has bit d set for every reduced axis; the
; output holds the product of the kept extents in row-major order whether
; or not keepdims kept the reduced axes as ones. Every output starts at the
; identity of the reduction over an empty set - ReduceMax: minus infinity
; for FLOAT, the type's minimum for integers, False for BOOL; ReduceProd: 1
; - and input elements are combined in row-major order. A FLOAT NaN wins
; ReduceMax, as numpy.maximum.reduce does. Op 0 max, 1 prod. Dims addresses
; Rank native integers.
; ----------------------------------------------------------------------
Structure PmTensorReduceArgs
  Src.i
  Dst.i
  Rank.i
  Dims.i
  Mask.i
  Op.i
  Kind.i
EndStructure

Procedure PmTensorReduce(*g.PmTensorReduceArgs)
  Protected Dim outStride.i(8)
  Protected i.i, d.i, n.i, coord.i, target.i, size.i, srcCount.i, outCount.i, extent.i
  Protected cur.f, value.f, ci.i, vi.i, *t, *s
  size = PmNsElementBytes(*g\Kind)
  If size = 0 : ProcedureReturn : EndIf
  srcCount = 1 : outCount = 1
  For d = *g\Rank - 1 To 0 Step -1
    extent = PeekI(*g\Dims + d * SizeOf(Integer))
    srcCount * extent
    If (*g\Mask >> d) & 1
      outStride(d) = 0
    Else
      outStride(d) = outCount : outCount * extent
    EndIf
  Next
  For i = 0 To outCount - 1
    *t = *g\Dst + i * size
    If *g\Op = 1
      If *g\Kind = 1 : PokeL(*t, $3F800000) : Else : PmNsWriteInt(*t, 0, *g\Kind, 1) : EndIf
    Else
      Select *g\Kind
        Case 1 : PokeL(*t, $FF800000)
        Case 6 : PokeL(*t, $80000000)
        Case 7 : PokeQ(*t, -9223372036854775807 - 1)
        Case 9 : PokeA(*t, 0)
      EndSelect
    EndIf
  Next
  For i = 0 To srcCount - 1
    n = i : target = 0
    For d = *g\Rank - 1 To 0 Step -1
      extent = PeekI(*g\Dims + d * SizeOf(Integer))
      coord = n % extent : n / extent
      target + coord * outStride(d)
    Next
    *t = *g\Dst + target * size : *s = *g\Src + i * size
    If *g\Kind = 1
      If *g\Op = 1
        cur = PeekF(*t) : value = PeekF(*s) : cur = cur * value : PokeF(*t, cur)
      ElseIf PmNsIsNan(*t) = 0 And (PmNsIsNan(*s) Or PeekF(*s) > PeekF(*t))
        PokeL(*t, PeekL(*s))
      EndIf
    Else
      ci = PmNsReadInt(*t, 0, *g\Kind) : vi = PmNsReadInt(*s, 0, *g\Kind)
      If *g\Op = 1
        ci = ci * vi
      ElseIf vi > ci
        ci = vi
      EndIf
      PmNsWriteInt(*t, 0, *g\Kind, ci)
    EndIf
  Next
EndProcedure

; ----------------------------------------------------------------------
; Pad-19 (current through opset 20; Pad-11 and Pad-18 are its constant,
; reflect and edge subsets). Begins holds, per axis, the number of elements
; added (or, when negative, removed) at the start; the end count follows
; from the source and destination extents. Negative pads crop first, and the
; three copying modes then work on the cropped data, as numpy.pad does on
; the sliced array:
;   constant  Value (one element, or zeros when Value is 0) outside the data
;   reflect   mirrored on the first and last element, period 2(n-1); a
;             single element reflects to itself
;   edge      the nearest edge element
;   wrap      the data repeated as a torus, period n
; Map must address DstDims[Rank-1] native integers of scratch.
; Returns 0, or 1 when a copying mode needs an element from an axis that
; cropping left empty.
; ----------------------------------------------------------------------
Structure PmTensorPadArgs
  Src.i
  Dst.i
  Rank.i
  SrcDims.i
  DstDims.i
  Begins.i
  Mode.i
  Kind.i
  Value.i
  Map.i
EndStructure

Procedure.i PmNsPadCoordinate(c.i, before.i, srcExtent.i, dstExtent.i, mode.i)
  Protected lo.i, kept.i, afterPad.i, r.i, period.i
  afterPad = dstExtent - srcExtent - before
  lo = 0 : If before < 0 : lo = -before : EndIf
  kept = srcExtent
  If before < 0 : kept + before : EndIf
  If afterPad < 0 : kept + afterPad : EndIf
  r = c : If before > 0 : r - before : EndIf
  If r >= 0 And r < kept : ProcedureReturn lo + r : EndIf
  Select mode
    Case 0
      ProcedureReturn -1
    Case 1
      If kept <= 0 : ProcedureReturn -2 : EndIf
      If kept = 1 : ProcedureReturn lo : EndIf
      period = 2 * (kept - 1)
      r = r % period : If r < 0 : r + period : EndIf
      If r >= kept : r = period - r : EndIf
    Case 2
      If kept <= 0 : ProcedureReturn -2 : EndIf
      If r < 0 : r = 0 : Else : r = kept - 1 : EndIf
    Case 3
      If kept <= 0 : ProcedureReturn -2 : EndIf
      r = r % kept : If r < 0 : r + kept : EndIf
  EndSelect
  ProcedureReturn lo + r
EndProcedure

Procedure.i PmTensorPad(*g.PmTensorPadArgs)
  Protected Dim srcStride.i(8)
  Protected i.i, d.i, n.i, coord.i, source.i, outside.i, size.i, count.i, width.i, rows.i, row.i, last.i, x.i
  Protected k.i
  size = PmNsElementBytes(*g\Kind)
  If size = 0 : ProcedureReturn 1 : EndIf
  count = 1 : n = 1
  For d = *g\Rank - 1 To 0 Step -1
    srcStride(d) = n : n * PeekI(*g\SrcDims + d * SizeOf(Integer))
    count * PeekI(*g\DstDims + d * SizeOf(Integer))
  Next
  If count = 0 : ProcedureReturn 0 : EndIf
  If *g\Rank = 0
    CopyMemory(*g\Src, *g\Dst, size) : ProcedureReturn 0
  EndIf
  last = *g\Rank - 1
  width = PeekI(*g\DstDims + last * SizeOf(Integer))
  rows = count / width
  For x = 0 To width - 1
    coord = PmNsPadCoordinate(x, PeekI(*g\Begins + last * SizeOf(Integer)), PeekI(*g\SrcDims + last * SizeOf(Integer)), width, *g\Mode)
    If coord = -2 : ProcedureReturn 1 : EndIf
    PokeI(*g\Map + x * SizeOf(Integer), coord)
  Next
  For row = 0 To rows - 1
    n = row : source = 0 : outside = 0
    For d = last - 1 To 0 Step -1
      coord = n % PeekI(*g\DstDims + d * SizeOf(Integer))
      n / PeekI(*g\DstDims + d * SizeOf(Integer))
      coord = PmNsPadCoordinate(coord, PeekI(*g\Begins + d * SizeOf(Integer)), PeekI(*g\SrcDims + d * SizeOf(Integer)), PeekI(*g\DstDims + d * SizeOf(Integer)), *g\Mode)
      If coord = -2 : ProcedureReturn 1 : EndIf
      If coord = -1 : outside = 1 : Else : source + coord * srcStride(d) : EndIf
    Next
    For x = 0 To width - 1
      coord = PeekI(*g\Map + x * SizeOf(Integer))
      If outside Or coord = -1
        If *g\Value
          CopyMemory(*g\Value, *g\Dst + (row * width + x) * size, size)
        Else
          For k = 0 To size - 1 : PokeA(*g\Dst + (row * width + x) * size + k, 0) : Next
        EndIf
      Else
        CopyMemory(*g\Src + (source + coord) * size, *g\Dst + (row * width + x) * size, size)
      EndIf
    Next
  Next
  ProcedureReturn 0
EndProcedure

; ----------------------------------------------------------------------
; Not-1 (current through opset 20): the element-wise negation of a BOOL
; tensor.
; ----------------------------------------------------------------------
Procedure PmTensorNot(*src, *dst, count.i)
  Protected i.i
  For i = 0 To count - 1
    PokeA(*dst + i, Bool((PeekA(*src + i) & 255) = 0))
  Next
EndProcedure
