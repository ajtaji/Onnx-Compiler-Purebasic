; ============================================================================
; onnx_proto_writer.pbi - small canonical protobuf writer for trace models
; ----------------------------------------------------------------------------
; A trace model is the original checked ModelProto with selected ValueInfoProto
; records exposed as temporary graph outputs. The source file is never changed.
; ============================================================================

Structure PmoProtoBuffer
  *Data
  Length.q
  Capacity.q
EndStructure

Global PmoProtoError.s

Procedure.i PmoProtoFail(Message.s)
  If PmoProtoError = "" : PmoProtoError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoProtoFree(*Buffer.PmoProtoBuffer)
  If *Buffer = 0 : ProcedureReturn : EndIf
  If *Buffer\Data : FreeMemory(*Buffer\Data) : EndIf
  *Buffer\Data = 0 : *Buffer\Length = 0 : *Buffer\Capacity = 0
EndProcedure

Procedure.i PmoProtoReserve(*Buffer.PmoProtoBuffer, Extra.q)
  Protected Needed.q
  Protected Capacity.q
  Protected *NewData
  If *Buffer = 0 Or Extra < 0 : ProcedureReturn PmoProtoFail("invalid protobuf output extent") : EndIf
  If *Buffer\Length > $7FFFFFFFFFFFFFFF - Extra
    ProcedureReturn PmoProtoFail("protobuf output size overflows 64 bits")
  EndIf
  Needed = *Buffer\Length + Extra
  If Needed <= *Buffer\Capacity : ProcedureReturn #True : EndIf
  Capacity = *Buffer\Capacity
  If Capacity < 4096 : Capacity = 4096 : EndIf
  While Capacity < Needed
    If Capacity > $3FFFFFFFFFFFFFFF : Capacity = Needed : Break : EndIf
    Capacity * 2
  Wend
  If *Buffer\Data
    *NewData = ReAllocateMemory(*Buffer\Data, Capacity)
  Else
    *NewData = AllocateMemory(Capacity)
  EndIf
  If *NewData = 0 : ProcedureReturn PmoProtoFail("cannot allocate trace ModelProto") : EndIf
  *Buffer\Data = *NewData : *Buffer\Capacity = Capacity
  ProcedureReturn #True
EndProcedure

Procedure.i PmoProtoAppend(*Buffer.PmoProtoBuffer, *Data, Bytes.q)
  If Bytes < 0 Or (Bytes > 0 And *Data = 0)
    ProcedureReturn PmoProtoFail("invalid protobuf append")
  EndIf
  If PmoProtoReserve(*Buffer, Bytes) = 0 : ProcedureReturn #False : EndIf
  If Bytes > 0 : CopyMemory(*Data, *Buffer\Data + *Buffer\Length, Bytes) : EndIf
  *Buffer\Length + Bytes
  ProcedureReturn #True
EndProcedure

Procedure.i PmoProtoAppendVarint(*Buffer.PmoProtoBuffer, Value.q)
  Protected ByteValue.i
  Repeat
    ByteValue = Value & $7F
    Value >> 7
    If Value <> 0 : ByteValue | $80 : EndIf
    If PmoProtoReserve(*Buffer, 1) = 0 : ProcedureReturn #False : EndIf
    PokeA(*Buffer\Data + *Buffer\Length, ByteValue)
    *Buffer\Length + 1
  Until Value = 0
  ProcedureReturn #True
EndProcedure

Procedure.i PmoProtoAppendMessage(*Buffer.PmoProtoBuffer, Field.i, *Data, Bytes.q)
  If Field <= 0 Or Bytes < 0 : ProcedureReturn PmoProtoFail("invalid protobuf message field") : EndIf
  If PmoProtoAppendVarint(*Buffer, (Field << 3) | 2) = 0 Or
     PmoProtoAppendVarint(*Buffer, Bytes) = 0
    ProcedureReturn #False
  EndIf
  ProcedureReturn PmoProtoAppend(*Buffer, *Data, Bytes)
EndProcedure

Procedure.i PmoProtoAppendString(*Buffer.PmoProtoBuffer, Field.i, Text.s)
  Protected *Encoded = UTF8(Text)
  Protected Bytes.i
  Protected Result.i
  If *Encoded = 0 : ProcedureReturn PmoProtoFail("cannot encode protobuf string") : EndIf
  Bytes = MemorySize(*Encoded) - 1
  Result = PmoProtoAppendMessage(*Buffer, Field, *Encoded, Bytes)
  FreeMemory(*Encoded)
  ProcedureReturn Result
EndProcedure

Procedure.i PmoProtoEncodeValueInfo(*Value.PmoOnnxValue, *Result.PmoProtoBuffer)
  Protected TypeProto.PmoProtoBuffer
  Protected TensorType.PmoProtoBuffer
  Protected Shape.PmoProtoBuffer
  Protected Dimension.PmoProtoBuffer
  If *Value = 0 Or *Result = 0 Or *Value\Name = "" Or
     *Value\HasTensorType = 0 Or *Value\ElementType <= 0 Or *Value\HasShape = 0
    ProcedureReturn PmoProtoFail("cannot encode an incomplete trace ValueInfoProto")
  EndIf
  PmoProtoFree(*Result)
  If PmoProtoAppendString(*Result, 1, *Value\Name) = 0 Or
     PmoProtoAppendVarint(@TensorType, (1 << 3) | 0) = 0 Or
     PmoProtoAppendVarint(@TensorType, *Value\ElementType) = 0
    Goto PmoProtoEncodeValueInfoFailed
  EndIf
  ForEach *Value\Dims()
    PmoProtoFree(@Dimension)
    If *Value\Dims()\HasValue
      If PmoProtoAppendVarint(@Dimension, (1 << 3) | 0) = 0 Or
         PmoProtoAppendVarint(@Dimension, *Value\Dims()\Value) = 0
        Goto PmoProtoEncodeValueInfoFailed
      EndIf
    ElseIf *Value\Dims()\Symbol <> ""
      If PmoProtoAppendString(@Dimension, 2, *Value\Dims()\Symbol) = 0
        Goto PmoProtoEncodeValueInfoFailed
      EndIf
    EndIf
    If PmoProtoAppendMessage(@Shape, 1, Dimension\Data, Dimension\Length) = 0
      Goto PmoProtoEncodeValueInfoFailed
    EndIf
  Next
  If PmoProtoAppendMessage(@TensorType, 2, Shape\Data, Shape\Length) = 0 Or
     PmoProtoAppendMessage(@TypeProto, 1, TensorType\Data, TensorType\Length) = 0 Or
     PmoProtoAppendMessage(*Result, 2, TypeProto\Data, TypeProto\Length) = 0
    Goto PmoProtoEncodeValueInfoFailed
  EndIf
  PmoProtoFree(@Dimension) : PmoProtoFree(@Shape)
  PmoProtoFree(@TensorType) : PmoProtoFree(@TypeProto)
  ProcedureReturn #True

  PmoProtoEncodeValueInfoFailed:
  PmoProtoFree(@Dimension) : PmoProtoFree(@Shape)
  PmoProtoFree(@TensorType) : PmoProtoFree(@TypeProto) : PmoProtoFree(*Result)
  ProcedureReturn #False
EndProcedure

Procedure.i PmoOnnxValueNeedsTrace(*Value.PmoOnnxValue)
  If *Value = 0 Or *Value\HasTensorType = 0 Or *Value\HasShape = 0
    ProcedureReturn #True
  EndIf
  ForEach *Value\Dims()
    If *Value\Dims()\HasValue = 0 Or *Value\Dims()\Value <= 0
      ProcedureReturn #True
    EndIf
  Next
  ProcedureReturn #False
EndProcedure

Procedure.i PmoBuildTraceModel(*Model.PmoOnnxModel, List TraceNames.s(),
                               Map Checkpoints.i(), *Result.PmoProtoBuffer)
  Protected Cursor.PmoWireCursor
  Protected Field.Integer
  Protected Wire.Integer
  Protected Start.q
  Protected Sub.PmoWireCursor
  Protected Extra.PmoProtoBuffer
  Protected EncodedValue.PmoProtoBuffer
  NewMap Existing.i()
  If *Model = 0 Or *Result = 0 Or *Model\FileData = 0
    ProcedureReturn PmoProtoFail("trace model source is not loaded")
  EndIf
  PmoProtoFree(*Result) : PmoProtoError = ""
  ClearList(TraceNames())
  ForEach *Model\Graph\Outputs()
    Existing(*Model\Graph\Outputs()\Name) = #True
    AddElement(TraceNames()) : TraceNames() = *Model\Graph\Outputs()\Name
  Next
  ForEach *Model\Graph\Values()
    If PmoOnnxValueNeedsTrace(@*Model\Graph\Values()) Or FindMapElement(Checkpoints(), *Model\Graph\Values()\Name)
      If FindMapElement(Existing(), *Model\Graph\Values()\Name) = 0
        If PmoProtoEncodeValueInfo(@*Model\Graph\Values(), @EncodedValue) = 0 Or
           PmoProtoAppendMessage(@Extra, 12, EncodedValue\Data, EncodedValue\Length) = 0
          PmoProtoFree(@EncodedValue) : PmoProtoFree(@Extra) : ProcedureReturn #False
        EndIf
        PmoProtoFree(@EncodedValue)
        Existing(*Model\Graph\Values()\Name) = #True
        AddElement(TraceNames()) : TraceNames() = *Model\Graph\Values()\Name
      EndIf
    EndIf
  Next
  If PmoWireInit(@Cursor, *Model\FileData, *Model\FileBytes) = 0
    PmoProtoFree(@Extra) : ProcedureReturn PmoProtoFail(PmoWireError)
  EndIf
  While PmoWireRemaining(@Cursor) > 0
    Start = Cursor\Position
    If PmoWireReadKey(@Cursor, @Field, @Wire) = 0
      PmoProtoFree(@Extra) : PmoProtoFree(*Result) : ProcedureReturn PmoProtoFail(PmoWireError)
    EndIf
    If Field\i = 7 And Wire\i = 2
      If PmoWireReadSubmessage(@Cursor, @Sub) = 0 Or
         PmoProtoAppendVarint(*Result, (7 << 3) | 2) = 0 Or
         PmoProtoAppendVarint(*Result, Sub\Limit + Extra\Length) = 0 Or
         PmoProtoAppend(*Result, Sub\Base, Sub\Limit) = 0 Or
         PmoProtoAppend(*Result, Extra\Data, Extra\Length) = 0
        PmoProtoFree(@Extra) : PmoProtoFree(*Result)
        ProcedureReturn PmoProtoFail("cannot construct trace graph: " + PmoWireError)
      EndIf
    Else
      If PmoWireSkip(@Cursor, Wire\i) = 0 Or
         PmoProtoAppend(*Result, *Model\FileData + Start, Cursor\Position - Start) = 0
        PmoProtoFree(@Extra) : PmoProtoFree(*Result)
        ProcedureReturn PmoProtoFail("cannot copy trace ModelProto: " + PmoWireError)
      EndIf
    EndIf
  Wend
  PmoProtoFree(@Extra)
  ProcedureReturn #True
EndProcedure
