; ============================================================================
; onnx_pack.pbi - deterministic PMONNXW weight container writer
; ============================================================================

Structure PmoPackItem
  Name.s
  *Constant.PmoIrConstant
EndStructure

Global PmoPackError.s

Procedure.i PmoPackFail(Message.s)
  If PmoPackError = "" : PmoPackError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoPackShapeEquals(*Value.PmoIrValue, *Array.PmoNpyArray)
  If *Value\ElementType <> *Array\ElementType Or *Value\Bytes <> *Array\DataBytes Or
     ListSize(*Value\Dims()) <> ListSize(*Array\Dims())
    ProcedureReturn #False
  EndIf
  FirstElement(*Array\Dims())
  ForEach *Value\Dims()
    If *Value\Dims() <> *Array\Dims() : ProcedureReturn #False : EndIf
    NextElement(*Array\Dims())
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoPackAddTraceInputs(*Ir.PmoIrModel, List Specs.s())
  Protected Name.String
  Protected Path.String
  Protected Array.PmoNpyArray
  Protected *Value.PmoIrValue
  Protected Key.s
  ForEach Specs()
    If PmoTraceParseFeed(Specs(), @Name, @Path) = 0
      ProcedureReturn PmoPackFail(PmoTraceError)
    EndIf
    If FindMapElement(*Ir\ValueByName(), Name\s) = 0
      ProcedureReturn PmoPackFail("trace input " + Name\s + " is not a graph value")
    EndIf
    *Value = *Ir\ValueByName()
    If PmoNpyLoad(Path\s, @Array) = 0
      ProcedureReturn PmoPackFail(PmoNpyError)
    EndIf
    If PmoPackShapeEquals(*Value, @Array) = 0
      PmoNpyFree(@Array)
      ProcedureReturn PmoPackFail("trace input " + Name\s + " disagrees with its concrete graph tensor")
    EndIf
    Key = "@trace/" + Name\s
    If FindMapElement(*Ir\ConstantByName(), Key)
      PmoNpyFree(@Array)
      ProcedureReturn PmoPackFail("duplicate embedded trace input " + Name\s)
    EndIf
    AddElement(*Ir\Constants())
    *Ir\Constants()\Name = Key
    *Ir\Constants()\ElementType = Array\ElementType
    *Ir\Constants()\Elements = *Value\Elements
    *Ir\Constants()\Bytes = Array\DataBytes
    *Ir\Constants()\Data = AllocateMemory(Array\DataBytes)
    If *Ir\Constants()\Data = 0
      PmoNpyFree(@Array) : ProcedureReturn PmoPackFail("cannot copy embedded trace input " + Name\s)
    EndIf
    *Ir\Constants()\OwnsData = #True
    CopyMemory(Array\Data, *Ir\Constants()\Data, Array\DataBytes)
    ForEach Array\Dims()
      AddElement(*Ir\Constants()\Dims()) : *Ir\Constants()\Dims() = Array\Dims()
    Next
    *Ir\ConstantByName(Key) = @*Ir\Constants()
    PmoNpyFree(@Array)
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoPackNibble(Character.s)
  Protected Code.i = Asc(Character)
  If Code >= '0' And Code <= '9' : ProcedureReturn Code - '0' : EndIf
  If Code >= 'a' And Code <= 'f' : ProcedureReturn Code - 'a' + 10 : EndIf
  If Code >= 'A' And Code <= 'F' : ProcedureReturn Code - 'A' + 10 : EndIf
  ProcedureReturn -1
EndProcedure

Procedure.i PmoPackHexBytes(Hex.s, *Destination, Bytes.i)
  Protected Index.i
  Protected Hi.i
  Protected Lo.i
  If Len(Hex) <> Bytes * 2 : ProcedureReturn #False : EndIf
  For Index = 0 To Bytes - 1
    Hi = PmoPackNibble(Mid(Hex, Index * 2 + 1, 1))
    Lo = PmoPackNibble(Mid(Hex, Index * 2 + 2, 1))
    If Hi < 0 Or Lo < 0 : ProcedureReturn #False : EndIf
    PokeA(*Destination + Index, (Hi << 4) | Lo)
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoPackWriteZeros(File.i, Bytes.q, *ZeroBlock)
  Protected Chunk.i
  While Bytes > 0
    Chunk = Bytes : If Chunk > 65536 : Chunk = 65536 : EndIf
    If WriteData(File, *ZeroBlock, Chunk) <> Chunk : ProcedureReturn #False : EndIf
    Bytes - Chunk
  Wend
  ProcedureReturn #True
EndProcedure

Procedure.i PmoPackWeights(*Ir.PmoIrModel, Path.s)
  Protected *Payload
  NewMap Used.i()
  NewList Items.PmoPackItem()
  Protected *Constant.PmoIrConstant
  Protected Offset.q
  Protected Pad.q
  Protected Count.i
  Protected Hash.i
  Protected CrcHex.s
  Protected ModelHex.s
  Protected CrcValue.q
  Protected *Zeros
  Protected *Header
  Protected File.i
  Protected Written.q
  PmoPackError = ""
  UseCRC32Fingerprint()
  UseSHA2Fingerprint()
  ForEach *Ir\Outputs() : Used(*Ir\Outputs()) = #True : Next
  ForEach *Ir\Nodes()
    ForEach *Ir\Nodes()\Node\Inputs()
      If *Ir\Nodes()\Node\Inputs() <> "" : Used(*Ir\Nodes()\Node\Inputs()) = #True : EndIf
    Next
  Next
  ForEach *Ir\Constants()
    If (Left(*Ir\Constants()\Name, 7) = "@trace/" Or Left(*Ir\Constants()\Name, 7) = "@quant/" Or
        FindMapElement(Used(), *Ir\Constants()\Name)) And
       (*Ir\Constants()\ElementType = 1 Or *Ir\Constants()\ElementType = 2 Or *Ir\Constants()\ElementType = 3 Or *Ir\Constants()\ElementType = #PMO_ELEMENT_INT8_WIDE Or *Ir\Constants()\ElementType = 6 Or *Ir\Constants()\ElementType = 7 Or
        *Ir\Constants()\ElementType = 9)
      AddElement(Items())
      Items()\Name = *Ir\Constants()\Name
      Items()\Constant = @*Ir\Constants()
    EndIf
  Next
  SortStructuredList(Items(), #PB_Sort_Ascending, OffsetOf(PmoPackItem\Name), TypeOf(PmoPackItem\Name))
  Offset = 0
  ForEach Items()
    Offset = PmoIrAlign(Offset)
    Items()\Constant\PackedOffset = #PMO_WEIGHT_HEADER_BYTES + Offset
    Offset + Items()\Constant\Bytes
    Count + 1
  Next
  *Ir\WeightDataBytes = Offset
  *Ir\WeightBytes = #PMO_WEIGHT_HEADER_BYTES + Offset

  *Zeros = AllocateMemory(65536)
  If *Zeros = 0 : ProcedureReturn PmoPackFail("cannot allocate packer padding") : EndIf
  Hash = StartFingerprint(#PB_Any, #PB_Cipher_CRC32)
  If Hash = 0 : FreeMemory(*Zeros) : ProcedureReturn PmoPackFail("cannot start weight CRC") : EndIf
  Offset = 0
  ForEach Items()
    Pad = PmoIrAlign(Offset) - Offset
    While Pad > 0
      Written = Pad : If Written > 65536 : Written = 65536 : EndIf
      AddFingerprintBuffer(Hash, *Zeros, Written)
      Pad - Written
    Wend
    If Items()\Constant\Bytes > 0
      *Payload=Items()\Constant\Data
      If Items()\Constant\StorageData : *Payload=Items()\Constant\StorageData : EndIf
      AddFingerprintBuffer(Hash, *Payload, Items()\Constant\Bytes)
    EndIf
    Offset = PmoIrAlign(Offset) + Items()\Constant\Bytes
  Next
  CrcHex = FinishFingerprint(Hash)
  CrcValue = Val("$" + CrcHex)

  *Header = AllocateMemory(#PMO_WEIGHT_HEADER_BYTES)
  If *Header = 0 : FreeMemory(*Zeros) : ProcedureReturn PmoPackFail("cannot allocate weight header") : EndIf
  CopyMemory(?PmoWeightMagic, *Header, 8)
  PokeL(*Header + 8, 1)
  PokeL(*Header + 12, #PMO_WEIGHT_HEADER_BYTES)
  PokeQ(*Header + 16, *Ir\WeightBytes)
  PokeQ(*Header + 24, #PMO_WEIGHT_HEADER_BYTES)
  PokeQ(*Header + 32, *Ir\WeightDataBytes)
  PokeL(*Header + 40, CrcValue)
  PokeL(*Header + 44, Count)
  PokeL(*Header + 48, *Ir\Source\IrVersion)
  Protected DefaultOpset.i
  ForEach *Ir\Source\Opsets()
    If *Ir\Source\Opsets()\Domain = "" Or *Ir\Source\Opsets()\Domain = "ai.onnx"
      DefaultOpset = *Ir\Source\Opsets()\Version
    EndIf
  Next
  PokeL(*Header + 52, DefaultOpset)
  ModelHex = Fingerprint(*Ir\Source\FileData, *Ir\Source\FileBytes, #PB_Cipher_SHA2, 256)
  If PmoPackHexBytes(ModelHex, *Header + 56, 32) = 0
    FreeMemory(*Header) : FreeMemory(*Zeros) : ProcedureReturn PmoPackFail("cannot encode model SHA-256")
  EndIf

  File = CreateFile(#PB_Any, Path)
  If File = 0
    FreeMemory(*Header) : FreeMemory(*Zeros) : ProcedureReturn PmoPackFail("cannot create weight file: " + Path)
  EndIf
  If WriteData(File, *Header, #PMO_WEIGHT_HEADER_BYTES) <> #PMO_WEIGHT_HEADER_BYTES
    CloseFile(File) : FreeMemory(*Header) : FreeMemory(*Zeros)
    ProcedureReturn PmoPackFail("short write in weight header")
  EndIf
  Offset = 0
  ForEach Items()
    Pad = PmoIrAlign(Offset) - Offset
    If PmoPackWriteZeros(File, Pad, *Zeros) = 0
      CloseFile(File) : FreeMemory(*Header) : FreeMemory(*Zeros)
      ProcedureReturn PmoPackFail("short write in weight padding")
    EndIf
    *Payload=Items()\Constant\Data
    If Items()\Constant\StorageData : *Payload=Items()\Constant\StorageData : EndIf
    If Items()\Constant\Bytes > 0 And
       WriteData(File, *Payload, Items()\Constant\Bytes) <> Items()\Constant\Bytes
      CloseFile(File) : FreeMemory(*Header) : FreeMemory(*Zeros)
      ProcedureReturn PmoPackFail("short write in tensor " + Items()\Name)
    EndIf
    Offset = PmoIrAlign(Offset) + Items()\Constant\Bytes
  Next
  CloseFile(File)
  FreeMemory(*Header) : FreeMemory(*Zeros)
  ProcedureReturn #True

  DataSection
    PmoWeightMagic:
    Data.a 'P','M','O','N','N','X','W',0
  EndDataSection
EndProcedure
