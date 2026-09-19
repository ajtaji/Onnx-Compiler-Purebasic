; Host-side construction and verification of checked Kokoro PMVOICE files.
; The caller supplies the immutable upstream raw voice. This module performs
; no download and contains no voice or model data.

#PMO_KAV_HEADER_BYTES = 128
#PMO_KAV_ROWS = 510
#PMO_KAV_WIDTH = 256
#PMO_KAV_RAW_BYTES = #PMO_KAV_ROWS * #PMO_KAV_WIDTH * 4
#PMO_KAV_FILE_BYTES = #PMO_KAV_HEADER_BYTES + #PMO_KAV_RAW_BYTES
#PMO_KAV_MODEL_SHA256$ = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"

Global PmoKokoroVoicePackError.s
Global Dim PmoKavCrcTable.i(255)
Global PmoKavCrcReady.i

Procedure.i PmoKavFail(Message.s)
  If PmoKokoroVoicePackError = "" : PmoKokoroVoicePackError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoKavCrcInit()
  Protected i.i, bit.i
  Protected value.i
  If PmoKavCrcReady : ProcedureReturn : EndIf
  For i = 0 To 255
    value = i
    For bit = 0 To 7
      If value & 1 : value = ((value >> 1) & $7FFFFFFF) ! $EDB88320 : Else : value = (value >> 1) & $7FFFFFFF : EndIf
    Next
    PmoKavCrcTable(i) = value & $FFFFFFFF
  Next
  PmoKavCrcReady = #True
EndProcedure

Procedure.i PmoKavCrc32(*Data, Bytes.i)
  Protected i.i
  Protected crc.i = $FFFFFFFF
  If *Data = 0 Or Bytes < 0 : ProcedureReturn 0 : EndIf
  PmoKavCrcInit()
  While i < Bytes
    crc = ((crc >> 8) & $00FFFFFF) ! PmoKavCrcTable((crc ! PeekA(*Data + i)) & $FF)
    i + 1
  Wend
  ProcedureReturn (crc ! $FFFFFFFF) & $FFFFFFFF
EndProcedure

Procedure.i PmoKavHexToBytes(Text.s, *Destination, Bytes.i)
  Protected i.i, value.i
  If Len(Text) <> Bytes * 2 Or *Destination = 0 : ProcedureReturn #False : EndIf
  For i = 0 To Bytes - 1
    value = Val("$" + Mid(Text, i * 2 + 1, 2))
    If LCase(Mid(Text, i * 2 + 1, 2)) <> LCase(RSet(Hex(value), 2, "0"))
      ProcedureReturn #False
    EndIf
    PokeA(*Destination + i, value)
  Next
  ProcedureReturn #True
EndProcedure

Procedure.s PmoKavMemorySha256(*Data, Bytes.i)
  Protected fingerprint.i
  UseSHA2Fingerprint()
  fingerprint = StartFingerprint(#PB_Any, #PB_Cipher_SHA2, 256)
  If fingerprint = 0 : ProcedureReturn "" : EndIf
  AddFingerprintBuffer(fingerprint, *Data, Bytes)
  ProcedureReturn LCase(FinishFingerprint(fingerprint))
EndProcedure

Procedure.i PmoKokoroVoiceVerifyMemory(*Pack, Bytes.i)
  Protected i.i
  Protected sourceHash.s
  PmoKokoroVoicePackError = ""
  If *Pack = 0 Or Bytes <> #PMO_KAV_FILE_BYTES : ProcedureReturn PmoKavFail("PMVOICE length is not 522368 bytes") : EndIf
  If PeekS(*Pack, 8, #PB_Ascii) <> "PMVOICE" : ProcedureReturn PmoKavFail("PMVOICE magic is invalid") : EndIf
  If PeekL(*Pack + 8) <> 1 Or PeekL(*Pack + 12) <> #PMO_KAV_HEADER_BYTES : ProcedureReturn PmoKavFail("PMVOICE version or header length is invalid") : EndIf
  If PeekQ(*Pack + 16) <> Bytes Or PeekL(*Pack + 24) <> #PMO_KAV_ROWS Or PeekL(*Pack + 28) <> #PMO_KAV_WIDTH
    ProcedureReturn PmoKavFail("PMVOICE extent or shape is invalid")
  EndIf
  If PeekL(*Pack + 36) <> 0 : ProcedureReturn PmoKavFail("PMVOICE reserved field is not zero") : EndIf
  For i = 104 To #PMO_KAV_HEADER_BYTES - 1
    If PeekA(*Pack + i) <> 0 : ProcedureReturn PmoKavFail("PMVOICE reserved header bytes are not zero") : EndIf
  Next
  If PmoKavCrc32(*Pack + #PMO_KAV_HEADER_BYTES, #PMO_KAV_RAW_BYTES) <> (PeekL(*Pack + 32) & $FFFFFFFF)
    ProcedureReturn PmoKavFail("PMVOICE payload CRC-32 is invalid")
  EndIf
  sourceHash = PmoKavMemorySha256(*Pack + #PMO_KAV_HEADER_BYTES, #PMO_KAV_RAW_BYTES)
  If sourceHash = "" : ProcedureReturn PmoKavFail("cannot compute PMVOICE payload SHA-256") : EndIf
  For i = 0 To 31
    If PeekA(*Pack + 40 + i) <> Val("$" + Mid(sourceHash, i * 2 + 1, 2))
      ProcedureReturn PmoKavFail("PMVOICE source SHA-256 does not match its payload")
    EndIf
    If PeekA(*Pack + 72 + i) <> Val("$" + Mid(#PMO_KAV_MODEL_SHA256$, i * 2 + 1, 2))
      ProcedureReturn PmoKavFail("PMVOICE names a different Kokoro model")
    EndIf
  Next
  For i = 0 To #PMO_KAV_ROWS * #PMO_KAV_WIDTH - 1
    If (PeekL(*Pack + #PMO_KAV_HEADER_BYTES + i * 4) & $7F800000) = $7F800000
      ProcedureReturn PmoKavFail("PMVOICE payload contains NaN or infinity")
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKokoroVoiceVerifyFile(Path.s)
  Protected file.i, bytes.q, readBytes.q, *pack
  PmoKokoroVoicePackError = ""
  bytes = FileSize(Path)
  If bytes <> #PMO_KAV_FILE_BYTES : ProcedureReturn PmoKavFail("PMVOICE file is missing or has the wrong length") : EndIf
  *pack = AllocateMemory(bytes)
  If *pack = 0 : ProcedureReturn PmoKavFail("cannot allocate the PMVOICE verification buffer") : EndIf
  file = ReadFile(#PB_Any, Path, #PB_File_SharedRead)
  If file : readBytes = ReadData(file, *pack, bytes) : CloseFile(file) : EndIf
  If readBytes <> bytes : FreeMemory(*pack) : ProcedureReturn PmoKavFail("cannot read the complete PMVOICE file") : EndIf
  file = PmoKokoroVoiceVerifyMemory(*pack, bytes)
  FreeMemory(*pack)
  ProcedureReturn file
EndProcedure

; Refuses an existing destination. Publishing is a rename of a fully written,
; closed and independently verified sibling temporary file.
Procedure.i PmoKokoroVoicePack(Source.s, Destination.s, ExpectedSourceSha256.s = "")
  Protected sourceFile.i, outputFile.i, bytes.q, readBytes.q, written.q
  Protected *raw, *pack
  Protected sourceHash.s, temporary.s
  PmoKokoroVoicePackError = ""
  If Source = "" Or Destination = "" : ProcedureReturn PmoKavFail("voice source and destination are required") : EndIf
  If FileSize(Destination) >= 0 : ProcedureReturn PmoKavFail("destination already exists; refusing to replace it") : EndIf
  bytes = FileSize(Source)
  If bytes <> #PMO_KAV_RAW_BYTES : ProcedureReturn PmoKavFail("raw voice length is not 522240 bytes") : EndIf
  *raw = AllocateMemory(bytes)
  *pack = AllocateMemory(#PMO_KAV_FILE_BYTES)
  If *raw = 0 Or *pack = 0 : PmoKavFail("cannot allocate voice packing buffers") : Goto PmoKavPackFailed : EndIf
  sourceFile = ReadFile(#PB_Any, Source, #PB_File_SharedRead)
  If sourceFile : readBytes = ReadData(sourceFile, *raw, bytes) : CloseFile(sourceFile) : EndIf
  If readBytes <> bytes : PmoKavFail("cannot read the complete raw voice") : Goto PmoKavPackFailed : EndIf
  sourceHash = PmoKavMemorySha256(*raw, bytes)
  If sourceHash = "" : PmoKavFail("cannot compute raw voice SHA-256") : Goto PmoKavPackFailed : EndIf
  If ExpectedSourceSha256 <> "" And LCase(ExpectedSourceSha256) <> sourceHash
    PmoKavFail("raw voice SHA-256 does not match the expected identity") : Goto PmoKavPackFailed
  EndIf
  FillMemory(*pack, #PMO_KAV_FILE_BYTES, 0)
  PokeS(*pack, "PMVOICE", 7, #PB_Ascii) : PokeL(*pack + 8, 1) : PokeL(*pack + 12, #PMO_KAV_HEADER_BYTES)
  PokeQ(*pack + 16, #PMO_KAV_FILE_BYTES) : PokeL(*pack + 24, #PMO_KAV_ROWS) : PokeL(*pack + 28, #PMO_KAV_WIDTH)
  PokeL(*pack + 32, PmoKavCrc32(*raw, bytes))
  If PmoKavHexToBytes(sourceHash, *pack + 40, 32) = 0 Or PmoKavHexToBytes(#PMO_KAV_MODEL_SHA256$, *pack + 72, 32) = 0
    PmoKavFail("internal PMVOICE identity encoding failed") : Goto PmoKavPackFailed
  EndIf
  CopyMemory(*raw, *pack + #PMO_KAV_HEADER_BYTES, bytes)
  If PmoKokoroVoiceVerifyMemory(*pack, #PMO_KAV_FILE_BYTES) = 0 : Goto PmoKavPackFailed : EndIf
  temporary = Destination + ".tmp-" + Hex(ElapsedMilliseconds())
  If FileSize(temporary) >= 0 : PmoKavFail("temporary output already exists") : Goto PmoKavPackFailed : EndIf
  outputFile = CreateFile(#PB_Any, temporary)
  If outputFile = 0 : PmoKavFail("cannot create temporary PMVOICE output") : Goto PmoKavPackFailed : EndIf
  written = WriteData(outputFile, *pack, #PMO_KAV_FILE_BYTES)
  CloseFile(outputFile)
  If written <> #PMO_KAV_FILE_BYTES : DeleteFile(temporary) : PmoKavFail("short write while creating PMVOICE") : Goto PmoKavPackFailed : EndIf
  If PmoKokoroVoiceVerifyFile(temporary) = 0 : DeleteFile(temporary) : Goto PmoKavPackFailed : EndIf
  If FileSize(Destination) >= 0 : DeleteFile(temporary) : PmoKavFail("destination appeared during packing; refusing replacement") : Goto PmoKavPackFailed : EndIf
  If RenameFile(temporary, Destination) = 0 : DeleteFile(temporary) : PmoKavFail("cannot publish verified PMVOICE output") : Goto PmoKavPackFailed : EndIf
  FreeMemory(*raw) : FreeMemory(*pack)
  ProcedureReturn #True
  PmoKavPackFailed:
  If *raw : FreeMemory(*raw) : EndIf
  If *pack : FreeMemory(*pack) : EndIf
  ProcedureReturn #False
EndProcedure
