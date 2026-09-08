; ============================================================================
; onnx_wire.pbi - bounded Protocol Buffers wire reader for native ONNX tools
; ----------------------------------------------------------------------------
; No protobuf runtime is linked. Every read is checked against the enclosing
; message before a byte is touched. Unknown fields are skipped by wire type;
; malformed, truncated, overlong and unsupported group encodings fail loudly.
; ============================================================================

Structure PmoWireCursor
  *Base
  Position.q
  Limit.q
EndStructure

Structure PmoWireSlice
  *Data
  Bytes.q
EndStructure

Global PmoWireError.s

Procedure.i PmoWireFail(Message.s)
  If PmoWireError = ""
    PmoWireError = Message
  EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoWireResetError()
  PmoWireError = ""
EndProcedure

Procedure.i PmoWireInit(*Cursor.PmoWireCursor, *Data, Bytes.q)
  If *Cursor = 0
    ProcedureReturn PmoWireFail("internal wire cursor is null")
  EndIf
  If Bytes < 0 Or (Bytes > 0 And *Data = 0)
    ProcedureReturn PmoWireFail("invalid Protocol Buffers input extent")
  EndIf
  *Cursor\Base = *Data
  *Cursor\Position = 0
  *Cursor\Limit = Bytes
  ProcedureReturn #True
EndProcedure

Procedure.q PmoWireRemaining(*Cursor.PmoWireCursor)
  If *Cursor = 0 Or *Cursor\Position < 0 Or *Cursor\Position > *Cursor\Limit
    ProcedureReturn -1
  EndIf
  ProcedureReturn *Cursor\Limit - *Cursor\Position
EndProcedure

Procedure.i PmoWireReadVarint(*Cursor.PmoWireCursor, *Value.Quad)
  Protected ByteValue.i
  Protected Shift.i
  Protected Result.q
  If *Cursor = 0 Or *Value = 0
    ProcedureReturn PmoWireFail("internal varint destination is null")
  EndIf
  Result = 0
  Shift = 0
  While Shift < 70
    If *Cursor\Position >= *Cursor\Limit
      ProcedureReturn PmoWireFail("truncated Protocol Buffers varint")
    EndIf
    ByteValue = PeekA(*Cursor\Base + *Cursor\Position) & $FF
    *Cursor\Position + 1
    If Shift = 63 And (ByteValue & $FE) <> 0
      ProcedureReturn PmoWireFail("Protocol Buffers varint exceeds 64 bits")
    EndIf
    Result | (ByteValue & $7F) << Shift
    If (ByteValue & $80) = 0
      *Value\q = Result
      ProcedureReturn #True
    EndIf
    Shift + 7
  Wend
  ProcedureReturn PmoWireFail("overlong Protocol Buffers varint")
EndProcedure

Procedure.i PmoWireReadKey(*Cursor.PmoWireCursor, *Field.Integer, *Wire.Integer)
  Protected Key.Quad
  If *Field = 0 Or *Wire = 0
    ProcedureReturn PmoWireFail("internal field-key destination is null")
  EndIf
  If PmoWireReadVarint(*Cursor, @Key) = 0
    ProcedureReturn #False
  EndIf
  *Field\i = Key\q >> 3
  *Wire\i = Key\q & 7
  If *Field\i <= 0
    ProcedureReturn PmoWireFail("Protocol Buffers field number zero is invalid")
  EndIf
  Select *Wire\i
    Case 0, 1, 2, 5
      ProcedureReturn #True
  EndSelect
  ProcedureReturn PmoWireFail("unsupported Protocol Buffers wire type " + Str(*Wire\i))
EndProcedure

Procedure.i PmoWireReadSlice(*Cursor.PmoWireCursor, *Slice.PmoWireSlice)
  Protected Length.Quad
  If *Cursor = 0 Or *Slice = 0
    ProcedureReturn PmoWireFail("internal length-delimited destination is null")
  EndIf
  If PmoWireReadVarint(*Cursor, @Length) = 0
    ProcedureReturn #False
  EndIf
  If Length\q < 0 Or Length\q > PmoWireRemaining(*Cursor)
    ProcedureReturn PmoWireFail("length-delimited field exceeds its enclosing message")
  EndIf
  *Slice\Data = *Cursor\Base + *Cursor\Position
  *Slice\Bytes = Length\q
  *Cursor\Position + Length\q
  ProcedureReturn #True
EndProcedure

Procedure.i PmoWireReadSubmessage(*Cursor.PmoWireCursor, *Sub.PmoWireCursor)
  Protected Slice.PmoWireSlice
  If PmoWireReadSlice(*Cursor, @Slice) = 0
    ProcedureReturn #False
  EndIf
  ProcedureReturn PmoWireInit(*Sub, Slice\Data, Slice\Bytes)
EndProcedure

Procedure.i PmoWireReadFixed32(*Cursor.PmoWireCursor, *Value.Long)
  If *Cursor = 0 Or *Value = 0
    ProcedureReturn PmoWireFail("internal fixed32 destination is null")
  EndIf
  If PmoWireRemaining(*Cursor) < 4
    ProcedureReturn PmoWireFail("truncated Protocol Buffers fixed32")
  EndIf
  *Value\l = PeekL(*Cursor\Base + *Cursor\Position)
  *Cursor\Position + 4
  ProcedureReturn #True
EndProcedure

Procedure.i PmoWireReadFixed64(*Cursor.PmoWireCursor, *Value.Quad)
  If *Cursor = 0 Or *Value = 0
    ProcedureReturn PmoWireFail("internal fixed64 destination is null")
  EndIf
  If PmoWireRemaining(*Cursor) < 8
    ProcedureReturn PmoWireFail("truncated Protocol Buffers fixed64")
  EndIf
  *Value\q = PeekQ(*Cursor\Base + *Cursor\Position)
  *Cursor\Position + 8
  ProcedureReturn #True
EndProcedure

Procedure.i PmoWireSkip(*Cursor.PmoWireCursor, Wire.i)
  Protected Value.Quad
  Protected Slice.PmoWireSlice
  Select Wire
    Case 0
      ProcedureReturn PmoWireReadVarint(*Cursor, @Value)
    Case 1
      If PmoWireRemaining(*Cursor) < 8
        ProcedureReturn PmoWireFail("truncated Protocol Buffers fixed64 field")
      EndIf
      *Cursor\Position + 8
      ProcedureReturn #True
    Case 2
      ProcedureReturn PmoWireReadSlice(*Cursor, @Slice)
    Case 5
      If PmoWireRemaining(*Cursor) < 4
        ProcedureReturn PmoWireFail("truncated Protocol Buffers fixed32 field")
      EndIf
      *Cursor\Position + 4
      ProcedureReturn #True
  EndSelect
  ProcedureReturn PmoWireFail("cannot skip Protocol Buffers wire type " + Str(Wire))
EndProcedure

Procedure.s PmoWireUtf8(*Slice.PmoWireSlice)
  If *Slice = 0 Or *Slice\Bytes <= 0
    ProcedureReturn ""
  EndIf
  ProcedureReturn PeekS(*Slice\Data, *Slice\Bytes, #PB_UTF8)
EndProcedure

Procedure.i PmoWireRequire(*Cursor.PmoWireCursor, Field.i, ExpectedWire.i, ActualWire.i)
  Protected Message.s
  If ActualWire <> ExpectedWire
    Message = "field " + Str(Field) + " uses wire type " + Str(ActualWire)
    Message + ", expected " + Str(ExpectedWire)
    ProcedureReturn PmoWireFail(Message)
  EndIf
  ProcedureReturn #True
EndProcedure
