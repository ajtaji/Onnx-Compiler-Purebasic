; ============================================================================
; tensor_control_windows.pbi - sequences and control-flow transfers for the
; runtime-dimension path on Windows
; ----------------------------------------------------------------------------
; Included after tensor_dynamic_windows.pbi, and only by a model that uses If,
; Loop or a sequence operator. Reference semantics: the ONNX operator
; specification, https://onnx.ai/onnx/operators/ - SequenceEmpty-11,
; SequenceConstruct-11, SequenceInsert-11, SequenceAt-11, SequenceLength-11,
; SplitToSequence-11, ConcatFromSequence-11, If-11..19 (cond holds exactly one
; element) and Loop-11..19 (trip count and condition semantics, scan outputs
; stacked along a new first axis).
;
; A SEQUENCE IS ONE BLOCK. Dt(id)\Kind = #PMD_KIND_SEQUENCE, \Data is the
; block, \Bytes its size, \Count the element count, \Owned = 1. DRelease,
; request reset and close therefore free a sequence completely, and the
; working-memory accounting (DLive against DLimit) covers it, without any
; change to the tensor runtime.
;
; Block layout, every field a 32-bit little-endian integer:
;   +0 'PMSQ'  +4 element kind  +8 element count  +12 record capacity
;   +16 data bytes used  +20 data capacity
;   +24 records, 48 bytes each: rank, count, bytes, data offset, dims[8]
;   +24+48*capacity: data area; element data at 8-byte aligned offsets
; The same layout is used on every target (tensor_control_portable.pmi).
; ============================================================================

#PMD_KIND_SEQUENCE = 1024
#PMD_SEQ_MAGIC = $51534D50
#PMD_SEQ_HEADER = 24
#PMD_SEQ_RECORD = 48

Procedure.i DSeqAlign8(Value.i)
  ProcedureReturn (Value + 7) & -8
EndProcedure

Procedure.i DSeqIs(Id.i)
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : ProcedureReturn 0 : EndIf
  If Dt(Id)\Kind <> #PMD_KIND_SEQUENCE Or Dt(Id)\Data = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i DSeqRequire(Id.i)
  If DSeqIs(Id) = 0 : ProcedureReturn DFail("A sequence operator was given a value that is not a sequence.") : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i DSeqCount(Id.i)
  If DSeqIs(Id) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(Dt(Id)\Data + 8)
EndProcedure

Procedure.i DSeqElementKind(Id.i)
  If DSeqIs(Id) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(Dt(Id)\Data + 4)
EndProcedure

Procedure.i DSeqRecord(Id.i, Index.i)
  ProcedureReturn Dt(Id)\Data + #PMD_SEQ_HEADER + Index * #PMD_SEQ_RECORD
EndProcedure

Procedure.i DSeqItemValid(Id.i, Index.i)
  If DSeqIs(Id) = 0 : ProcedureReturn 0 : EndIf
  If Index < 0 Or Index >= PeekL(Dt(Id)\Data + 8) : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i DSeqItemRank(Id.i, Index.i)
  If DSeqItemValid(Id, Index) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(DSeqRecord(Id, Index))
EndProcedure

Procedure.i DSeqItemCount(Id.i, Index.i)
  If DSeqItemValid(Id, Index) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(DSeqRecord(Id, Index) + 4)
EndProcedure

Procedure.i DSeqItemBytes(Id.i, Index.i)
  If DSeqItemValid(Id, Index) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(DSeqRecord(Id, Index) + 8)
EndProcedure

Procedure.i DSeqItemDim(Id.i, Index.i, Axis.i)
  If DSeqItemValid(Id, Index) = 0 : ProcedureReturn 0 : EndIf
  If Axis < 0 Or Axis >= PeekL(DSeqRecord(Id, Index)) : ProcedureReturn 0 : EndIf
  ProcedureReturn PeekL(DSeqRecord(Id, Index) + 16 + Axis * 4)
EndProcedure

; The address is valid until the sequence next grows.
Procedure.i DSeqItemData(Id.i, Index.i)
  If DSeqItemValid(Id, Index) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn Dt(Id)\Data + #PMD_SEQ_HEADER + PeekL(Dt(Id)\Data + 12) * #PMD_SEQ_RECORD + PeekL(DSeqRecord(Id, Index) + 12)
EndProcedure

; Bytes for a block with Capacity records and at least DataBytes of data, or
; -1 when that cannot fit in the working-memory limit at all.
Procedure.i DSeqBlockBytes(Capacity.i, DataBytes.i)
  Protected records.i
  If Capacity < 0 Or DataBytes < 0 Or Capacity > (DLimit - #PMD_SEQ_HEADER) / #PMD_SEQ_RECORD : ProcedureReturn -1 : EndIf
  records = #PMD_SEQ_HEADER + Capacity * #PMD_SEQ_RECORD
  If DataBytes > DLimit - records - 8 : ProcedureReturn -1 : EndIf
  ProcedureReturn DSeqAlign8(records + DataBytes)
EndProcedure

Procedure.i DSeqKindCarried(Kind.i)
  Select Kind
    Case 1, 6, 7, 9 : ProcedureReturn 1
  EndSelect
  ProcedureReturn 0
EndProcedure

; Makes Id an empty sequence of ElementKind with room for Capacity elements
; and DataBytes of element data.
Procedure.i DSeqMake(Id.i, ElementKind.i, Capacity.i, DataBytes.i)
  Protected bytes.i, *block
  If DError <> "" Or DCancel : ProcedureReturn DFail("Request cancelled or an earlier operation failed.") : EndIf
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : ProcedureReturn DFail("Invalid sequence tensor ID.") : EndIf
  If DSeqKindCarried(ElementKind) = 0
    ProcedureReturn DFail("A sequence carries FLOAT, INT32, INT64 or BOOL tensors; element type " + Str(ElementKind) + " is not carried.")
  EndIf
  bytes = DSeqBlockBytes(Capacity, DataBytes)
  DRelease(Id)
  If bytes < 0 Or bytes > DLimit - DLive
    ProcedureReturn DFail("A sequence of " + Str(Capacity) + " elements and " + Str(DataBytes) + " data bytes exceeds the working-memory limit; nothing was truncated.")
  EndIf
  *block = AllocateMemory(bytes)
  If *block = 0 : ProcedureReturn DFail("Sequence allocation failed.") : EndIf
  PokeL(*block, #PMD_SEQ_MAGIC)
  PokeL(*block + 4, ElementKind)
  PokeL(*block + 8, 0)
  PokeL(*block + 12, Capacity)
  PokeL(*block + 16, 0)
  PokeL(*block + 20, bytes - #PMD_SEQ_HEADER - Capacity * #PMD_SEQ_RECORD)
  Dt(Id)\Data = *block : Dt(Id)\Kind = #PMD_KIND_SEQUENCE : Dt(Id)\Rank = 0 : Dt(Id)\Count = 0
  Dt(Id)\Bytes = bytes : Dt(Id)\Owned = 1 : Dt(Id)\Scales = 0
  DLive + bytes : DPeak = DMax(DPeak, DLive)
  ProcedureReturn 1
EndProcedure

; Ensures room for Records more elements and Bytes more element data, growing
; the block geometrically inside the working memory.
Procedure.i DSeqReserve(Id.i, MoreRecords.i, MoreBytes.i)
  Protected *old, *block, count.i, capacity.i, used.i, dataCapacity.i
  Protected newCapacity.i, newData.i, bytes.i, oldBytes.i, need.i
  If DSeqRequire(Id) = 0 : ProcedureReturn 0 : EndIf
  If MoreRecords < 0 Or MoreBytes < 0 Or MoreRecords > DLimit Or MoreBytes > DLimit
    ProcedureReturn DFail("A sequence element exceeds the working-memory limit.")
  EndIf
  *old = Dt(Id)\Data
  count = PeekL(*old + 8) : capacity = PeekL(*old + 12) : used = PeekL(*old + 16) : dataCapacity = PeekL(*old + 20)
  need = DSeqAlign8(used) + MoreBytes
  If count + MoreRecords <= capacity And need <= dataCapacity : ProcedureReturn 1 : EndIf
  newCapacity = capacity
  If count + MoreRecords > capacity : newCapacity = DMax(DMax(capacity * 2, count + MoreRecords), 4) : EndIf
  newData = dataCapacity
  If need > dataCapacity : newData = DMax(DMax(dataCapacity * 2, need), 64) : EndIf
  bytes = DSeqBlockBytes(newCapacity, newData)
  oldBytes = Dt(Id)\Bytes
  If bytes < 0 Or bytes - oldBytes > DLimit - DLive
    ProcedureReturn DFail("Sequence growth to " + Str(count + MoreRecords) + " elements and " + Str(need) + " data bytes exceeds the working-memory limit; nothing was truncated.")
  EndIf
  *block = AllocateMemory(bytes)
  If *block = 0 : ProcedureReturn DFail("Sequence growth allocation failed.") : EndIf
  CopyMemory(*old, *block, #PMD_SEQ_HEADER)
  If count : CopyMemory(*old + #PMD_SEQ_HEADER, *block + #PMD_SEQ_HEADER, count * #PMD_SEQ_RECORD) : EndIf
  If used : CopyMemory(*old + #PMD_SEQ_HEADER + capacity * #PMD_SEQ_RECORD, *block + #PMD_SEQ_HEADER + newCapacity * #PMD_SEQ_RECORD, used) : EndIf
  PokeL(*block + 12, newCapacity)
  PokeL(*block + 20, bytes - #PMD_SEQ_HEADER - newCapacity * #PMD_SEQ_RECORD)
  FreeMemory(*old)
  Dt(Id)\Data = *block : Dt(Id)\Bytes = bytes
  DLive + bytes - oldBytes : DPeak = DMax(DPeak, DLive)
  ProcedureReturn 1
EndProcedure

; Adds an element record at Position (0..count) with room for its data and
; returns the data address, zero-filled; zero after a refusal. *Dims holds
; Rank native integers.
Procedure.i DSeqPlace(Id.i, Position.i, Rank.i, *Dims, ElementCount.i, ElementBytes.i)
  Protected *block, *record, count.i, capacity.i, offset.i, i.i, *data
  If Rank < 0 Or Rank > 8 : ProcedureReturn DFail("A sequence element has a rank above eight.") : EndIf
  If DSeqReserve(Id, 1, ElementBytes) = 0 : ProcedureReturn 0 : EndIf
  *block = Dt(Id)\Data
  count = PeekL(*block + 8) : capacity = PeekL(*block + 12)
  If Position < 0 Or Position > count : ProcedureReturn DFail("A sequence insert position is outside the sequence.") : EndIf
  If Position < count
    MoveMemory(*block + #PMD_SEQ_HEADER + Position * #PMD_SEQ_RECORD, *block + #PMD_SEQ_HEADER + (Position + 1) * #PMD_SEQ_RECORD, (count - Position) * #PMD_SEQ_RECORD)
  EndIf
  offset = DSeqAlign8(PeekL(*block + 16))
  *record = *block + #PMD_SEQ_HEADER + Position * #PMD_SEQ_RECORD
  FillMemory(*record, #PMD_SEQ_RECORD, 0)
  PokeL(*record, Rank) : PokeL(*record + 4, ElementCount) : PokeL(*record + 8, ElementBytes) : PokeL(*record + 12, offset)
  For i = 0 To Rank - 1 : PokeL(*record + 16 + i * 4, PeekI(*Dims + i * 8)) : Next
  PokeL(*block + 16, offset + ElementBytes)
  PokeL(*block + 8, count + 1)
  Dt(Id)\Count = count + 1
  *data = *block + #PMD_SEQ_HEADER + capacity * #PMD_SEQ_RECORD + offset
  If ElementBytes : FillMemory(*data, ElementBytes, 0) : EndIf
  ProcedureReturn *data
EndProcedure

Procedure.i DSeqTensorOk(T.i)
  If T <= 0 Or T > #PMD_TENSOR_COUNT : ProcedureReturn 0 : EndIf
  If Dt(T)\Kind = #PMD_KIND_SEQUENCE Or Dt(T)\Data = 0 Or DSeqKindCarried(Dt(T)\Kind) = 0 : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i DSeqPlaceTensor(Id.i, Position.i, T.i)
  Protected *data
  If DSeqTensorOk(T) = 0 : ProcedureReturn DFail("A sequence element must be a FLOAT, INT32, INT64 or BOOL tensor.") : EndIf
  If Dt(T)\Kind <> DSeqElementKind(Id) : ProcedureReturn DFail("A tensor's element type does not match the element type of the sequence it joins.") : EndIf
  *data = DSeqPlace(Id, Position, Dt(T)\Rank, @Dt(T)\D[0], Dt(T)\Count, Dt(T)\Bytes)
  If *data = 0 : ProcedureReturn 0 : EndIf
  If Dt(T)\Bytes : CopyMemory(Dt(T)\Data, *data, Dt(T)\Bytes) : EndIf
  ProcedureReturn 1
EndProcedure

; ---------------------------------------------------------------------------
; Value transfers between tensor ids (If outputs, Loop carried values)
; ---------------------------------------------------------------------------

; Hands Src's storage to Dst without copying; Src is left empty. The emitter
; moves only a value nothing reads afterwards.
Procedure DValueMove(Dst.i, Src.i)
  If Dst = Src : ProcedureReturn : EndIf
  If Dst <= 0 Or Dst > #PMD_TENSOR_COUNT Or Src <= 0 Or Src > #PMD_TENSOR_COUNT : DFail("Invalid control-flow transfer ID.") : ProcedureReturn : EndIf
  If Dt(Src)\Data = 0 : DFail("A control-flow value was read before anything produced it.") : ProcedureReturn : EndIf
  DRelease(Dst)
  CopyMemory(@Dt(Src), @Dt(Dst), SizeOf(PmDynamicTensor))
  FillMemory(@Dt(Src), SizeOf(PmDynamicTensor), 0)
EndProcedure

Procedure DValueCopy(Dst.i, Src.i)
  Protected *block
  If Dst = Src : ProcedureReturn : EndIf
  If Dst <= 0 Or Dst > #PMD_TENSOR_COUNT Or Src <= 0 Or Src > #PMD_TENSOR_COUNT : DFail("Invalid control-flow transfer ID.") : ProcedureReturn : EndIf
  If Dt(Src)\Data = 0 : DFail("A control-flow value was read before anything produced it.") : ProcedureReturn : EndIf
  If Dt(Src)\Kind = #PMD_KIND_SEQUENCE
    If DError <> "" Or DCancel : ProcedureReturn : EndIf
    DRelease(Dst)
    If Dt(Src)\Bytes > DLimit - DLive : DFail("Copying a sequence exceeds the working-memory limit; nothing was truncated.") : ProcedureReturn : EndIf
    *block = AllocateMemory(Dt(Src)\Bytes)
    If *block = 0 : DFail("Sequence copy allocation failed.") : ProcedureReturn : EndIf
    CopyMemory(Dt(Src)\Data, *block, Dt(Src)\Bytes)
    Dt(Dst)\Data = *block : Dt(Dst)\Kind = #PMD_KIND_SEQUENCE : Dt(Dst)\Rank = 0
    Dt(Dst)\Count = Dt(Src)\Count : Dt(Dst)\Bytes = Dt(Src)\Bytes : Dt(Dst)\Owned = 1 : Dt(Dst)\Scales = 0
    DLive + Dt(Dst)\Bytes : DPeak = DMax(DPeak, DLive)
    ProcedureReturn
  EndIf
  If DAlloc(Dst, Dt(Src)\Kind, Dt(Src)\Rank, @Dt(Src)\D[0])
    If Dt(Src)\Bytes : CopyMemory(Dt(Src)\Data, Dt(Dst)\Data, Dt(Src)\Bytes) : EndIf
    Dt(Dst)\Scales = Dt(Src)\Scales
  EndIf
EndProcedure

Procedure DValueTake(Dst.i, Src.i, Move.i)
  If Move : DValueMove(Dst, Src) : Else : DValueCopy(Dst, Src) : EndIf
EndProcedure

; If's cond and Loop's cond: a BOOL tensor holding exactly one element.
Procedure.i DScalarTruth(Id.i)
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : DFail("A condition input is missing.") : ProcedureReturn 0 : EndIf
  If Dt(Id)\Data = 0 Or Dt(Id)\Kind <> 9 Or Dt(Id)\Count <> 1
    DFail("A condition must be a BOOL tensor holding exactly one element.") : ProcedureReturn 0
  EndIf
  ProcedureReturn Bool(PeekA(Dt(Id)\Data) <> 0)
EndProcedure

; Loop's trip count M: an INT64 tensor holding exactly one element.
Procedure.i DScalarCount(Id.i)
  If Id <= 0 Or Id > #PMD_TENSOR_COUNT : DFail("A Loop trip count is missing.") : ProcedureReturn 0 : EndIf
  If Dt(Id)\Data = 0 Or Dt(Id)\Kind <> 7 Or Dt(Id)\Count <> 1
    DFail("A Loop trip count must be an INT64 tensor holding exactly one element.") : ProcedureReturn 0
  EndIf
  ProcedureReturn PeekQ(Dt(Id)\Data)
EndProcedure

Procedure DScalarSet(Id.i, Kind.i, Value.i)
  Protected Dim dims.i(7)
  If DAlloc(Id, Kind, 0, @dims(0)) = 0 : ProcedureReturn : EndIf
  Select Kind
    Case 7 : PokeQ(Dt(Id)\Data, Value)
    Case 6 : PokeL(Dt(Id)\Data, Value)
    Case 9 : PokeA(Dt(Id)\Data, Bool(Value <> 0))
    Default : DFail("A loop counter or condition has an unsupported element type.")
  EndSelect
EndProcedure

; ---------------------------------------------------------------------------
; The sequence operators
; ---------------------------------------------------------------------------

Procedure DSeqEmpty(Y.i, Kind.i)
  DSeqMake(Y, Kind, 0, 0)
EndProcedure

Procedure DSeqConstruct(Y.i, Inputs.s)
  Protected count.i = CountString(Inputs, ",") + 1, i.i, a.i, kind.i, total.i
  a = Val(StringField(Inputs, 1, ","))
  If DSeqTensorOk(a) = 0 : DFail("SequenceConstruct inputs must be FLOAT, INT32, INT64 or BOOL tensors.") : ProcedureReturn : EndIf
  kind = Dt(a)\Kind
  For i = 1 To count
    a = Val(StringField(Inputs, i, ","))
    If DSeqTensorOk(a) = 0 Or Dt(a)\Kind <> kind : DFail("SequenceConstruct inputs must all be tensors of one element type.") : ProcedureReturn : EndIf
    If Dt(a)\Bytes > DLimit - total : DFail("SequenceConstruct exceeds the working-memory limit; nothing was truncated.") : ProcedureReturn : EndIf
    total + DSeqAlign8(Dt(a)\Bytes)
  Next
  If DSeqMake(Y, kind, count, total) = 0 : ProcedureReturn : EndIf
  For i = 1 To count
    If DSeqPlaceTensor(Y, i - 1, Val(StringField(Inputs, i, ","))) = 0 : ProcedureReturn : EndIf
  Next
EndProcedure

Procedure.i DSeqPosition(Pos.i, *Value.Integer)
  If Pos <= 0 Or Pos > #PMD_TENSOR_COUNT Or Dt(Pos)\Data = 0 Or Dt(Pos)\Count <> 1 Or (Dt(Pos)\Kind <> 6 And Dt(Pos)\Kind <> 7)
    ProcedureReturn DFail("A sequence position must be an INT32 or INT64 tensor holding exactly one element.")
  EndIf
  If Dt(Pos)\Kind = 7 : *Value\i = PeekQ(Dt(Pos)\Data) : Else : *Value\i = PeekL(Dt(Pos)\Data) : EndIf
  ProcedureReturn 1
EndProcedure

; SequenceInsert: a new sequence with T at Pos (default: the end). Move=1
; takes S's block over because the emitter proved nothing reads S again.
Procedure DSeqInsert(Y.i, S.i, T.i, Pos.i, Move.i)
  Protected n.i, p.i, raw.Integer
  If DSeqRequire(S) = 0 : ProcedureReturn : EndIf
  n = DSeqCount(S) : p = n
  If Pos
    If DSeqPosition(Pos, @raw) = 0 : ProcedureReturn : EndIf
    p = raw\i
    If p < 0 : p + n : EndIf
    If p < 0 Or p > n
      DFail("SequenceInsert position " + Str(raw\i) + " is outside [-" + Str(n) + ", " + Str(n) + "] for a sequence of " + Str(n) + " elements.") : ProcedureReturn
    EndIf
  EndIf
  If DSeqTensorOk(T) = 0 : DFail("SequenceInsert tensor must be a FLOAT, INT32, INT64 or BOOL tensor.") : ProcedureReturn : EndIf
  If Dt(T)\Kind <> DSeqElementKind(S) : DFail("SequenceInsert tensor element type " + Str(Dt(T)\Kind) + " does not match the sequence element type " + Str(DSeqElementKind(S)) + ".") : ProcedureReturn : EndIf
  DValueTake(Y, S, Move)
  If DError <> "" Or DCancel : ProcedureReturn : EndIf
  DSeqPlaceTensor(Y, p, T)
EndProcedure

Procedure DSeqAt(Y.i, S.i, Pos.i)
  Protected n.i, p.i, raw.Integer, *record, i.i
  Protected Dim dims.i(7)
  If DSeqRequire(S) = 0 Or DSeqPosition(Pos, @raw) = 0 : ProcedureReturn : EndIf
  n = DSeqCount(S) : p = raw\i
  If p < 0 : p + n : EndIf
  If p < 0 Or p >= n
    DFail("SequenceAt position " + Str(raw\i) + " is outside [-" + Str(n) + ", " + Str(n - 1) + "] for a sequence of " + Str(n) + " elements.") : ProcedureReturn
  EndIf
  *record = DSeqRecord(S, p)
  For i = 0 To PeekL(*record) - 1 : dims(i) = PeekL(*record + 16 + i * 4) : Next
  If DAlloc(Y, DSeqElementKind(S), PeekL(*record), @dims(0)) = 0 : ProcedureReturn : EndIf
  If Dt(Y)\Bytes : CopyMemory(DSeqItemData(S, p), Dt(Y)\Data, Dt(Y)\Bytes) : EndIf
EndProcedure

Procedure DSeqLength(Y.i, S.i)
  Protected Dim dims.i(7)
  If DSeqRequire(S) = 0 : ProcedureReturn : EndIf
  If DAlloc(Y, 7, 0, @dims(0)) : PokeQ(Dt(Y)\Data, DSeqCount(S)) : EndIf
EndProcedure

; SplitToSequence: no split gives chunks of one (keepdims decides whether the
; axis stays); a scalar split gives equal chunks with a shorter last one; a
; 1-D split gives explicit lengths that must sum to the axis extent.
Procedure DSplitToSequence(Y.i, A.i, Split.i, Axis.i, KeepDims.i)
  Protected Dim dims.i(7)
  Protected rank.i, extent.i, outer.i, inner.i, size.i, mode.i, chunk.i, chunks.i, length.i, sum.i
  Protected total.i, start.i, c.i, i.i, o.i, count.i, bytes.i, outRank.i, *data
  If DSeqTensorOk(A) = 0 : DFail("SplitToSequence input must be a FLOAT, INT32, INT64 or BOOL tensor.") : ProcedureReturn : EndIf
  rank = Dt(A)\Rank
  If rank < 1 : DFail("SplitToSequence requires an input of rank one or more.") : ProcedureReturn : EndIf
  If Axis < 0 : Axis + rank : EndIf
  If Axis < 0 Or Axis >= rank : DFail("SplitToSequence attribute axis is outside the input rank of " + Str(rank) + ".") : ProcedureReturn : EndIf
  extent = Dt(A)\D[Axis] : outer = DProduct(A, 0, Axis - 1) : inner = DProduct(A, Axis + 1, rank - 1) : size = DSize(Dt(A)\Kind)
  If Split
    If Dt(Split)\Data = 0 Or (Dt(Split)\Kind <> 6 And Dt(Split)\Kind <> 7) : DFail("SplitToSequence split must be an INT32 or INT64 tensor.") : ProcedureReturn : EndIf
    If Dt(Split)\Rank = 0
      mode = 1 : chunk = DInt(Split, 0)
      If chunk <= 0 : DFail("SplitToSequence scalar split " + Str(chunk) + " must be positive.") : ProcedureReturn : EndIf
      chunks = (extent + chunk - 1) / chunk
    ElseIf Dt(Split)\Rank = 1
      mode = 2 : chunks = Dt(Split)\Count
      For c = 0 To chunks - 1
        length = DInt(Split, c)
        If length < 0 : DFail("SplitToSequence split length " + Str(length) + " is negative.") : ProcedureReturn : EndIf
        sum + length
      Next
      If sum <> extent : DFail("SplitToSequence split lengths sum to " + Str(sum) + " but the input extent along axis " + Str(Axis) + " is " + Str(extent) + ".") : ProcedureReturn : EndIf
    Else
      DFail("SplitToSequence split must be a scalar or a 1-D tensor.") : ProcedureReturn
    EndIf
  Else
    mode = 0 : chunks = extent
  EndIf
  For c = 0 To chunks - 1
    Select mode
      Case 0 : length = 1
      Case 1 : length = DMin(chunk, extent - c * chunk)
      Case 2 : length = DInt(Split, c)
    EndSelect
    total + DSeqAlign8(outer * length * inner * size)
  Next
  If DSeqMake(Y, Dt(A)\Kind, chunks, total) = 0 : ProcedureReturn : EndIf
  For c = 0 To chunks - 1
    Select mode
      Case 0 : length = 1
      Case 1 : length = DMin(chunk, extent - start)
      Case 2 : length = DInt(Split, c)
    EndSelect
    For i = 0 To rank - 1 : dims(i) = Dt(A)\D[i] : Next
    dims(Axis) = length : outRank = rank
    If mode = 0 And KeepDims = 0
      For i = Axis To rank - 2 : dims(i) = dims(i + 1) : Next
      outRank = rank - 1
    EndIf
    count = outer * length * inner : bytes = count * size
    *data = DSeqPlace(Y, c, outRank, @dims(0), count, bytes)
    If *data = 0 : ProcedureReturn : EndIf
    If bytes
      For o = 0 To outer - 1
        CopyMemory(Dt(A)\Data + (o * extent + start) * inner * size, *data + o * length * inner * size, length * inner * size)
      Next
    EndIf
    start + length
  Next
EndProcedure

; ConcatFromSequence: new_axis=0 joins along an existing axis (numpy
; concatenate); new_axis=1 stacks along a new one (numpy stack).
Procedure DConcatFromSequence(Y.i, S.i, Axis.i, NewAxis.i)
  Protected Dim dims.i(8), Dim first.i(7)
  Protected n.i, r.i, outRank.i, e.i, i.i, extent.i, total.i, outer.i, inner.i, size.i, o.i, span.i, offset.i, *record
  If DSeqRequire(S) = 0 : ProcedureReturn : EndIf
  n = DSeqCount(S)
  If n = 0 : DFail("ConcatFromSequence cannot join an empty sequence: the result would have no shape.") : ProcedureReturn : EndIf
  r = DSeqItemRank(S, 0)
  For i = 0 To r - 1 : first(i) = DSeqItemDim(S, 0, i) : Next
  If NewAxis
    outRank = r + 1
    If Axis < -(r + 1) Or Axis > r : DFail("ConcatFromSequence attribute axis " + Str(Axis) + " is outside [-" + Str(r + 1) + ", " + Str(r) + "] with new_axis=1.") : ProcedureReturn : EndIf
  Else
    outRank = r
    If r < 1 : DFail("ConcatFromSequence with new_axis=0 needs elements of rank one or more.") : ProcedureReturn : EndIf
    If Axis < -r Or Axis > r - 1 : DFail("ConcatFromSequence attribute axis " + Str(Axis) + " is outside [-" + Str(r) + ", " + Str(r - 1) + "].") : ProcedureReturn : EndIf
  EndIf
  If Axis < 0 : Axis + outRank : EndIf
  If outRank > 8 : DFail("ConcatFromSequence result rank exceeds eight.") : ProcedureReturn : EndIf
  For e = 0 To n - 1
    If DSeqItemRank(S, e) <> r : DFail("ConcatFromSequence elements differ in rank.") : ProcedureReturn : EndIf
    For i = 0 To r - 1
      extent = DSeqItemDim(S, e, i)
      If (NewAxis Or i <> Axis) And extent <> first(i) : DFail("ConcatFromSequence elements differ in an extent other than the joined axis.") : ProcedureReturn : EndIf
    Next
    If NewAxis : total + 1 : Else : total + DSeqItemDim(S, e, Axis) : EndIf
  Next
  If NewAxis
    For i = 0 To Axis - 1 : dims(i) = first(i) : Next
    dims(Axis) = n
    For i = Axis To r - 1 : dims(i + 1) = first(i) : Next
  Else
    For i = 0 To r - 1 : dims(i) = first(i) : Next
    dims(Axis) = total
  EndIf
  If DAlloc(Y, DSeqElementKind(S), outRank, @dims(0)) = 0 : ProcedureReturn : EndIf
  size = DSize(DSeqElementKind(S))
  outer = 1 : For i = 0 To Axis - 1 : outer * first(i) : Next
  inner = 1
  If NewAxis
    For i = Axis To r - 1 : inner * first(i) : Next
  Else
    For i = Axis + 1 To r - 1 : inner * first(i) : Next
  EndIf
  If Dt(Y)\Bytes = 0 : ProcedureReturn : EndIf
  For o = 0 To outer - 1
    For e = 0 To n - 1
      If NewAxis : span = inner * size : Else : span = DSeqItemDim(S, e, Axis) * inner * size : EndIf
      If span : CopyMemory(DSeqItemData(S, e) + o * span, Dt(Y)\Data + offset, span) : EndIf
      offset + span
    Next
  Next
EndProcedure

; A Loop scan output: one copy per iteration, stacked when the loop ends.
Procedure DSeqAppendTensor(Acc.i, T.i)
  If DSeqTensorOk(T) = 0 : DFail("A Loop scan output must be a FLOAT, INT32, INT64 or BOOL tensor.") : ProcedureReturn : EndIf
  If DSeqIs(Acc) = 0
    If DSeqMake(Acc, Dt(T)\Kind, 4, DSeqAlign8(Dt(T)\Bytes)) = 0 : ProcedureReturn : EndIf
  EndIf
  DSeqPlaceTensor(Acc, DSeqCount(Acc), T)
EndProcedure

; Stacks the accumulated scan values along a new first axis. After zero
; iterations ONNX defines no shape for it; the result is [0] followed by the
; body's declared extents only when all of them are declared, else refused.
Procedure DSeqStack(Y.i, Acc.i, Kind.i, Rank.i, DimsText.s)
  Protected Dim dims.i(8)
  Protected i.i
  If DSeqIs(Acc) And DSeqCount(Acc) > 0
    DConcatFromSequence(Y, Acc, 0, 1)
    ProcedureReturn
  EndIf
  If DSeqKindCarried(Kind) = 0
    DFail("A Loop ran zero iterations and its scan output declares no FLOAT, INT32, INT64 or BOOL element type, so its empty result has no type.") : ProcedureReturn
  EndIf
  If Rank < 0
    DFail("A Loop ran zero iterations and its scan output does not declare every extent, so its empty result has no defined shape.") : ProcedureReturn
  EndIf
  If Rank + 1 > 8 : DFail("A Loop scan output rank exceeds eight.") : ProcedureReturn : EndIf
  dims(0) = 0
  For i = 1 To Rank : dims(i) = Val(StringField(DimsText, i, ",")) : Next
  DAlloc(Y, Kind, Rank + 1, @dims(0))
EndProcedure

; ---------------------------------------------------------------------------
; Host API for sequence graph inputs and outputs
; ---------------------------------------------------------------------------

; Makes an input id an empty sequence of Kind; append its elements next.
Procedure.i DSeqReset(Id.i, Kind.i)
  ProcedureReturn DSeqMake(Id, Kind, 0, 0)
EndProcedure

; Appends one element of Rank extents (*Dims: native integers) and returns
; the address to fill with its data, valid until the next append; zero after
; a refusal.
Procedure.i DSeqAppendShape(Id.i, Kind.i, Rank.i, *Dims)
  Protected i.i, count.i = 1, extent.i, size.i = DSize(Kind)
  If DSeqRequire(Id) = 0 : ProcedureReturn 0 : EndIf
  If Kind <> DSeqElementKind(Id) : ProcedureReturn DFail("An appended element's type does not match the sequence element type.") : EndIf
  If Rank < 0 Or Rank > 8 : ProcedureReturn DFail("An appended element has a rank above eight.") : EndIf
  For i = 0 To Rank - 1
    extent = PeekI(*Dims + i * 8)
    If extent < 0 Or extent > DLimit : ProcedureReturn DFail("An appended element has an invalid extent.") : EndIf
    If extent > 0 And count > DLimit / extent : ProcedureReturn DFail("An appended element exceeds the working-memory limit.") : EndIf
    count * extent
  Next
  If count > DLimit / size : ProcedureReturn DFail("An appended element exceeds the working-memory limit.") : EndIf
  ProcedureReturn DSeqPlace(Id, DSeqCount(Id), Rank, *Dims, count, count * size)
EndProcedure

; SequenceErase: a new sequence without the element at Pos (default: the
; last). Move=1 takes S's block over, as SequenceInsert does. The erased
; element's data stays in the block until the sequence is released.
Procedure DSeqErase(Y.i, S.i, Pos.i, Move.i)
  Protected n.i, p.i, raw.Integer, *block
  If DSeqRequire(S) = 0 : ProcedureReturn : EndIf
  n = DSeqCount(S) : p = n - 1
  If Pos
    If DSeqPosition(Pos, @raw) = 0 : ProcedureReturn : EndIf
    p = raw\i
    If p < 0 : p + n : EndIf
    If p < 0 Or p >= n
      DFail("SequenceErase position " + Str(raw\i) + " is outside [-" + Str(n) + ", " + Str(n - 1) + "] for a sequence of " + Str(n) + " elements.") : ProcedureReturn
    EndIf
  ElseIf n = 0
    DFail("SequenceErase was given an empty sequence; there is no element to erase.") : ProcedureReturn
  EndIf
  DValueTake(Y, S, Move)
  If DError <> "" Or DCancel : ProcedureReturn : EndIf
  *block = Dt(Y)\Data
  If p < n - 1
    MoveMemory(*block + #PMD_SEQ_HEADER + (p + 1) * #PMD_SEQ_RECORD, *block + #PMD_SEQ_HEADER + p * #PMD_SEQ_RECORD, (n - 1 - p) * #PMD_SEQ_RECORD)
  EndIf
  PokeL(*block + 8, n - 1)
  Dt(Y)\Count = n - 1
EndProcedure
