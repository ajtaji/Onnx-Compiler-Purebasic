; ============================================================================
; onnx_model.pbi - native ONNX ModelProto reader
; ----------------------------------------------------------------------------
; This deliberately parses the ONNX protobuf itself. It retains tensor payload
; slices inside the checked, immutable file buffer, avoiding a second 325 MB
; copy for Kokoro. The owning PmoOnnxModel must remain alive while those slices
; are used.
; ============================================================================

XIncludeFile "onnx_wire.pbi"

Structure PmoOnnxDim
  HasValue.i
  Value.q
  Symbol.s
EndStructure

Structure PmoOnnxValue
  Name.s
  ElementType.i
  HasTensorType.i
  HasShape.i
  Proto.PmoWireSlice
  List Dims.PmoOnnxDim()
EndStructure

Structure PmoOnnxKeyValue
  Key.s
  Value.s
EndStructure

Structure PmoOnnxTensor
  Name.s
  DataType.i
  DataLocation.i
  List Dims.q()
  Raw.PmoWireSlice
  List FloatData.f()
  List DoubleData.d()
  List Int32Data.q()
  List Int64Data.q()
  List UInt64Data.q()
  StringDataCount.i
  List External.PmoOnnxKeyValue()
EndStructure

Structure PmoOnnxAttribute
  Name.s
  AttributeType.i
  FloatValue.f
  IntegerValue.q
  StringValue.s
  HasTensor.i
  Tensor.PmoOnnxTensor
  List Floats.f()
  List Integers.q()
  List Strings.s()
EndStructure

Structure PmoOnnxNode
  Name.s
  Operation.s
  Domain.s
  List Inputs.s()
  List Outputs.s()
  List Attributes.PmoOnnxAttribute()
EndStructure

Structure PmoOnnxGraph
  Name.s
  List Nodes.PmoOnnxNode()
  List Initializers.PmoOnnxTensor()
  List Inputs.PmoOnnxValue()
  List Outputs.PmoOnnxValue()
  List Values.PmoOnnxValue()
EndStructure

Structure PmoOnnxOpset
  Domain.s
  Version.q
EndStructure

Structure PmoOnnxModel
  Path.s
  *FileData
  FileBytes.q
  IrVersion.q
  Producer.s
  ProducerVersion.s
  Domain.s
  ModelVersion.q
  GraphProto.PmoWireSlice
  Graph.PmoOnnxGraph
  List Opsets.PmoOnnxOpset()
EndStructure

Procedure.i PmoOnnxReadString(*Cursor.PmoWireCursor, *Text.String)
  Protected Slice.PmoWireSlice
  If *Text = 0
    ProcedureReturn PmoWireFail("internal ONNX string destination is null")
  EndIf
  If PmoWireReadSlice(*Cursor, @Slice) = 0
    ProcedureReturn #False
  EndIf
  *Text\s = PmoWireUtf8(@Slice)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParsePackedVarints(*Cursor.PmoWireCursor, List Values.q())
  Protected Sub.PmoWireCursor
  Protected Value.Quad
  If PmoWireReadSubmessage(*Cursor, @Sub) = 0
    ProcedureReturn #False
  EndIf
  While PmoWireRemaining(@Sub) > 0
    If PmoWireReadVarint(@Sub, @Value) = 0
      ProcedureReturn #False
    EndIf
    AddElement(Values())
    Values() = Value\q
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseStringEntry(*Cursor.PmoWireCursor, *Entry.PmoOnnxKeyValue)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0
      ProcedureReturn #False
    EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Entry\Key = Text\s
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Entry\Value = Text\s
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0
          ProcedureReturn #False
        EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseTensor(*Cursor.PmoWireCursor, *Tensor.PmoOnnxTensor)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Value.Quad
  Protected Bits.Long
  Protected Bits64.Quad
  Protected Slice.PmoWireSlice
  Protected Sub.PmoWireCursor
  Protected Text.String
  If *Tensor = 0
    ProcedureReturn PmoWireFail("internal TensorProto destination is null")
  EndIf
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0
      ProcedureReturn #False
    EndIf
    Select Field\i
      Case 1 ; dims, packed or unpacked int64
        If Wire\i = 2
          If PmoWireReadSubmessage(*Cursor, @Sub) = 0 : ProcedureReturn #False : EndIf
          While PmoWireRemaining(@Sub) > 0
            If PmoWireReadVarint(@Sub, @Value) = 0 : ProcedureReturn #False : EndIf
            AddElement(*Tensor\Dims()) : *Tensor\Dims() = Value\q
          Wend
        ElseIf Wire\i = 0
          If PmoWireReadVarint(*Cursor, @Value) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\Dims()) : *Tensor\Dims() = Value\q
        Else
          ProcedureReturn PmoWireFail("TensorProto dims has the wrong wire type")
        EndIf
      Case 2 ; data_type
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Value) = 0
          ProcedureReturn #False
        EndIf
        *Tensor\DataType = Value\q
      Case 4 ; float_data
        If Wire\i = 2
          If PmoWireReadSubmessage(*Cursor, @Sub) = 0 : ProcedureReturn #False : EndIf
          If (PmoWireRemaining(@Sub) & 3) <> 0
            ProcedureReturn PmoWireFail("TensorProto float_data byte count is not divisible by four")
          EndIf
          While PmoWireRemaining(@Sub) > 0
            AddElement(*Tensor\FloatData())
            *Tensor\FloatData() = PeekF(Sub\Base + Sub\Position)
            Sub\Position + 4
          Wend
        ElseIf Wire\i = 5
          If PmoWireReadFixed32(*Cursor, @Bits) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\FloatData())
          CopyMemory(@Bits\l, @*Tensor\FloatData(), 4)
        Else
          ProcedureReturn PmoWireFail("TensorProto float_data has the wrong wire type")
        EndIf
      Case 5 ; int32_data
        If Wire\i = 2
          If PmoOnnxParsePackedVarints(*Cursor, *Tensor\Int32Data()) = 0 : ProcedureReturn #False : EndIf
        ElseIf Wire\i = 0
          If PmoWireReadVarint(*Cursor, @Value) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\Int32Data()) : *Tensor\Int32Data() = Value\q
        Else
          ProcedureReturn PmoWireFail("TensorProto int32_data has the wrong wire type")
        EndIf
      Case 6 ; string_data
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSlice(*Cursor, @Slice) = 0
          ProcedureReturn #False
        EndIf
        *Tensor\StringDataCount + 1
      Case 7 ; int64_data
        If Wire\i = 2
          If PmoOnnxParsePackedVarints(*Cursor, *Tensor\Int64Data()) = 0 : ProcedureReturn #False : EndIf
        ElseIf Wire\i = 0
          If PmoWireReadVarint(*Cursor, @Value) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\Int64Data()) : *Tensor\Int64Data() = Value\q
        Else
          ProcedureReturn PmoWireFail("TensorProto int64_data has the wrong wire type")
        EndIf
      Case 8 ; name
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Tensor\Name = Text\s
      Case 9 ; raw_data
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSlice(*Cursor, @*Tensor\Raw) = 0
          ProcedureReturn #False
        EndIf
      Case 10 ; double_data
        If Wire\i = 2
          If PmoWireReadSubmessage(*Cursor, @Sub) = 0 : ProcedureReturn #False : EndIf
          If (PmoWireRemaining(@Sub) & 7) <> 0
            ProcedureReturn PmoWireFail("TensorProto double_data byte count is not divisible by eight")
          EndIf
          While PmoWireRemaining(@Sub) > 0
            AddElement(*Tensor\DoubleData())
            *Tensor\DoubleData() = PeekD(Sub\Base + Sub\Position)
            Sub\Position + 8
          Wend
        ElseIf Wire\i = 1
          If PmoWireReadFixed64(*Cursor, @Bits64) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\DoubleData())
          CopyMemory(@Bits64\q, @*Tensor\DoubleData(), 8)
        Else
          ProcedureReturn PmoWireFail("TensorProto double_data has the wrong wire type")
        EndIf
      Case 11 ; uint64_data
        If Wire\i = 2
          If PmoOnnxParsePackedVarints(*Cursor, *Tensor\UInt64Data()) = 0 : ProcedureReturn #False : EndIf
        ElseIf Wire\i = 0
          If PmoWireReadVarint(*Cursor, @Value) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Tensor\UInt64Data()) : *Tensor\UInt64Data() = Value\q
        Else
          ProcedureReturn PmoWireFail("TensorProto uint64_data has the wrong wire type")
        EndIf
      Case 13 ; external_data
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Tensor\External())
        If PmoOnnxParseStringEntry(@Sub, @*Tensor\External()) = 0 : ProcedureReturn #False : EndIf
      Case 14 ; data_location
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Value) = 0
          ProcedureReturn #False
        EndIf
        *Tensor\DataLocation = Value\q
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseDimension(*Cursor.PmoWireCursor, *Dim.PmoOnnxDim)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Value.Quad
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Value) = 0
          ProcedureReturn #False
        EndIf
        *Dim\HasValue = #True : *Dim\Value = Value\q
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Dim\Symbol = Text\s
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseShape(*Cursor.PmoWireCursor, *Value.PmoOnnxValue)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Sub.PmoWireCursor
  *Value\HasShape = #True
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    If Field\i = 1
      If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
        ProcedureReturn #False
      EndIf
      AddElement(*Value\Dims())
      If PmoOnnxParseDimension(@Sub, @*Value\Dims()) = 0 : ProcedureReturn #False : EndIf
    ElseIf PmoWireSkip(*Cursor, Wire\i) = 0
      ProcedureReturn #False
    EndIf
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseTensorType(*Cursor.PmoWireCursor, *Value.PmoOnnxValue)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Number.Quad
  Protected Sub.PmoWireCursor
  *Value\HasTensorType = #True
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Value\ElementType = Number\q
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        If PmoOnnxParseShape(@Sub, *Value) = 0 : ProcedureReturn #False : EndIf
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseType(*Cursor.PmoWireCursor, *Value.PmoOnnxValue)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Sub.PmoWireCursor
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    If Field\i = 1
      If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
        ProcedureReturn #False
      EndIf
      If PmoOnnxParseTensorType(@Sub, *Value) = 0 : ProcedureReturn #False : EndIf
    ElseIf PmoWireSkip(*Cursor, Wire\i) = 0
      ProcedureReturn #False
    EndIf
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseValue(*Cursor.PmoWireCursor, *Value.PmoOnnxValue)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Sub.PmoWireCursor
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Value\Name = Text\s
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        If PmoOnnxParseType(@Sub, *Value) = 0 : ProcedureReturn #False : EndIf
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseAttribute(*Cursor.PmoWireCursor, *Attribute.PmoOnnxAttribute)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Number.Quad
  Protected Bits.Long
  Protected Slice.PmoWireSlice
  Protected Sub.PmoWireCursor
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Attribute\Name = Text\s
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 5, Wire\i) = 0 Or PmoWireReadFixed32(*Cursor, @Bits) = 0
          ProcedureReturn #False
        EndIf
        CopyMemory(@Bits\l, @*Attribute\FloatValue, 4)
      Case 3
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Attribute\IntegerValue = Number\q
      Case 4
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Attribute\StringValue = Text\s
      Case 5
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        *Attribute\HasTensor = #True
        If PmoOnnxParseTensor(@Sub, @*Attribute\Tensor) = 0 : ProcedureReturn #False : EndIf
      Case 7 ; floats, normally packed fixed32
        If Wire\i = 2
          If PmoWireReadSubmessage(*Cursor, @Sub) = 0 : ProcedureReturn #False : EndIf
          If (PmoWireRemaining(@Sub) & 3) <> 0
            ProcedureReturn PmoWireFail("AttributeProto floats byte count is not divisible by four")
          EndIf
          While PmoWireRemaining(@Sub) > 0
            AddElement(*Attribute\Floats())
            *Attribute\Floats() = PeekF(Sub\Base + Sub\Position)
            Sub\Position + 4
          Wend
        ElseIf Wire\i = 5
          If PmoWireReadFixed32(*Cursor, @Bits) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Attribute\Floats())
          CopyMemory(@Bits\l, @*Attribute\Floats(), 4)
        Else
          ProcedureReturn PmoWireFail("AttributeProto floats has the wrong wire type")
        EndIf
      Case 8 ; ints
        If Wire\i = 2
          If PmoOnnxParsePackedVarints(*Cursor, *Attribute\Integers()) = 0 : ProcedureReturn #False : EndIf
        ElseIf Wire\i = 0
          If PmoWireReadVarint(*Cursor, @Number) = 0 : ProcedureReturn #False : EndIf
          AddElement(*Attribute\Integers()) : *Attribute\Integers() = Number\q
        Else
          ProcedureReturn PmoWireFail("AttributeProto ints has the wrong wire type")
        EndIf
      Case 9 ; strings
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSlice(*Cursor, @Slice) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Attribute\Strings())
        *Attribute\Strings() = PmoWireUtf8(@Slice)
      Case 20 ; type enum
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Attribute\AttributeType = Number\q
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseNode(*Cursor.PmoWireCursor, *Node.PmoOnnxNode)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Sub.PmoWireCursor
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Node\Inputs()) : *Node\Inputs() = Text\s
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Node\Outputs()) : *Node\Outputs() = Text\s
      Case 3
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Node\Name = Text\s
      Case 4
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Node\Operation = Text\s
      Case 5
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Node\Attributes())
        If PmoOnnxParseAttribute(@Sub, @*Node\Attributes()) = 0 : ProcedureReturn #False : EndIf
      Case 7
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Node\Domain = Text\s
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseGraph(*Cursor.PmoWireCursor, *Graph.PmoOnnxGraph)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Sub.PmoWireCursor
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Graph\Nodes())
        If PmoOnnxParseNode(@Sub, @*Graph\Nodes()) = 0 : ProcedureReturn #False : EndIf
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Graph\Name = Text\s
      Case 5
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Graph\Initializers())
        If PmoOnnxParseTensor(@Sub, @*Graph\Initializers()) = 0 : ProcedureReturn #False : EndIf
      Case 11
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Graph\Inputs())
        *Graph\Inputs()\Proto\Data = Sub\Base
        *Graph\Inputs()\Proto\Bytes = Sub\Limit
        If PmoOnnxParseValue(@Sub, @*Graph\Inputs()) = 0 : ProcedureReturn #False : EndIf
      Case 12
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Graph\Outputs())
        *Graph\Outputs()\Proto\Data = Sub\Base
        *Graph\Outputs()\Proto\Bytes = Sub\Limit
        If PmoOnnxParseValue(@Sub, @*Graph\Outputs()) = 0 : ProcedureReturn #False : EndIf
      Case 13
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Graph\Values())
        *Graph\Values()\Proto\Data = Sub\Base
        *Graph\Values()\Proto\Bytes = Sub\Limit
        If PmoOnnxParseValue(@Sub, @*Graph\Values()) = 0 : ProcedureReturn #False : EndIf
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseOpset(*Cursor.PmoWireCursor, *Opset.PmoOnnxOpset)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Number.Quad
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Opset\Domain = Text\s
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Opset\Version = Number\q
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoOnnxParseModel(*Cursor.PmoWireCursor, *Model.PmoOnnxModel)
  Protected Field.Integer
  Protected Wire.Integer
  Protected Number.Quad
  Protected Sub.PmoWireCursor
  Protected Text.String
  While PmoWireRemaining(*Cursor) > 0
    If PmoWireReadKey(*Cursor, @Field, @Wire) = 0 : ProcedureReturn #False : EndIf
    Select Field\i
      Case 1
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Model\IrVersion = Number\q
      Case 2
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Model\Producer = Text\s
      Case 3
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Model\ProducerVersion = Text\s
      Case 4
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoOnnxReadString(*Cursor, @Text) = 0
          ProcedureReturn #False
        EndIf
        *Model\Domain = Text\s
      Case 5
        If PmoWireRequire(*Cursor, Field\i, 0, Wire\i) = 0 Or PmoWireReadVarint(*Cursor, @Number) = 0
          ProcedureReturn #False
        EndIf
        *Model\ModelVersion = Number\q
      Case 7
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        *Model\GraphProto\Data = Sub\Base
        *Model\GraphProto\Bytes = Sub\Limit
        If PmoOnnxParseGraph(@Sub, @*Model\Graph) = 0 : ProcedureReturn #False : EndIf
      Case 8
        If PmoWireRequire(*Cursor, Field\i, 2, Wire\i) = 0 Or PmoWireReadSubmessage(*Cursor, @Sub) = 0
          ProcedureReturn #False
        EndIf
        AddElement(*Model\Opsets())
        If PmoOnnxParseOpset(@Sub, @*Model\Opsets()) = 0 : ProcedureReturn #False : EndIf
      Default
        If PmoWireSkip(*Cursor, Wire\i) = 0 : ProcedureReturn #False : EndIf
    EndSelect
  Wend
  ProcedureReturn #True
EndProcedure

Procedure PmoOnnxFree(*Model.PmoOnnxModel)
  If *Model = 0 : ProcedureReturn : EndIf
  If *Model\FileData
    FreeMemory(*Model\FileData)
  EndIf
  ; ClearStructure releases every nested list/string. A cleared structure is
  ; raw zeroed storage, not an initialized home for the next AddElement(), so
  ; immediately restore its dynamic-list descriptors before it can be reused.
  ClearStructure(*Model, PmoOnnxModel)
  InitializeStructure(*Model, PmoOnnxModel)
EndProcedure

Procedure.i PmoOnnxLoad(Path.s, *Model.PmoOnnxModel)
  Protected File.i
  Protected Bytes.q
  Protected ReadBytes.q
  Protected Cursor.PmoWireCursor
  If *Model = 0
    ProcedureReturn PmoWireFail("internal ModelProto destination is null")
  EndIf
  PmoOnnxFree(*Model)
  PmoWireResetError()
  Bytes = FileSize(Path)
  If Bytes <= 0
    ProcedureReturn PmoWireFail("ONNX model does not exist or is empty: " + Path)
  EndIf
  *Model\FileData = AllocateMemory(Bytes)
  If *Model\FileData = 0
    ProcedureReturn PmoWireFail("cannot allocate " + Str(Bytes) + " bytes for ONNX model")
  EndIf
  File = ReadFile(#PB_Any, Path, #PB_File_SharedRead)
  If File = 0
    PmoOnnxFree(*Model)
    ProcedureReturn PmoWireFail("cannot open ONNX model: " + Path)
  EndIf
  ReadBytes = ReadData(File, *Model\FileData, Bytes)
  CloseFile(File)
  If ReadBytes <> Bytes
    PmoOnnxFree(*Model)
    ProcedureReturn PmoWireFail("short read while loading ONNX model")
  EndIf
  *Model\Path = GetPathPart(Path) + GetFilePart(Path)
  *Model\FileBytes = Bytes
  If PmoWireInit(@Cursor, *Model\FileData, Bytes) = 0 Or PmoOnnxParseModel(@Cursor, *Model) = 0
    PmoOnnxFree(*Model)
    ProcedureReturn #False
  EndIf
  If Cursor\Position <> Cursor\Limit
    PmoOnnxFree(*Model)
    ProcedureReturn PmoWireFail("ONNX parser did not consume the complete ModelProto")
  EndIf
  If ListSize(*Model\Graph\Nodes()) = 0 Or ListSize(*Model\Graph\Outputs()) = 0
    PmoOnnxFree(*Model)
    ProcedureReturn PmoWireFail("ONNX graph has no executable nodes or outputs")
  EndIf
  ProcedureReturn #True
EndProcedure
