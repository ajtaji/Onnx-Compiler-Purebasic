; ============================================================================
; onnx_kokoro.pbi - native Kokoro text/voice input preparation
; ----------------------------------------------------------------------------
; This optional host adapter produces ordinary named NumPy inputs, then hands
; them to the same target-neutral compiler path as every other ONNX model.
; ============================================================================

#PMO_KOKORO_MODEL_SHA256$ = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"

Global Dim PmoKokoroCrcTable.i(255)
Global PmoKokoroCrcReady.i
Global PmoKokoroError.s

; The portable asset code uses PeekN for an explicit 32-bit native word.
; The host toolchain names that operation PeekL.
Macro PeekN(Address)
  (PeekL(Address) & $FFFFFFFF)
EndMacro

Procedure PmoKokoroCrcInit()
  Protected i.i
  Protected bit.i
  Protected value.i
  If PmoKokoroCrcReady : ProcedureReturn : EndIf
  i = 0
  While i < 256
    value = i
    bit = 0
    While bit < 8
      If value & 1
        value = ((value >> 1) & $7FFFFFFF) ! $EDB88320
      Else
        value = (value >> 1) & $7FFFFFFF
      EndIf
      bit = bit + 1
    Wend
    PmoKokoroCrcTable(i) = value & $FFFFFFFF
    i = i + 1
  Wend
  PmoKokoroCrcReady = #True
EndProcedure

Procedure.i PmTensorCrc32(*data, bytes.i)
  Protected i.i
  Protected index.i
  Protected crc.i = $FFFFFFFF
  If *data = 0 Or bytes < 0 : ProcedureReturn 0 : EndIf
  PmoKokoroCrcInit()
  While i < bytes
    index = (crc ! (PeekA(*data + i) & 255)) & 255
    crc = ((crc >> 8) & $00FFFFFF) ! PmoKokoroCrcTable(index)
    i = i + 1
  Wend
  ProcedureReturn (crc ! $FFFFFFFF) & $FFFFFFFF
EndProcedure

Procedure.i PmTensorGetI64(*base, index.i)
  ProcedureReturn PeekQ(*base + index * 8)
EndProcedure

Procedure PmTensorPutI64(*base, index.i, value.i)
  PokeQ(*base + index * 8, value)
EndProcedure

Procedure PmTensorCopy(*src, *dst, count.i)
  Protected i.i
  While i < count
    PokeF(*dst + i * 4, PeekF(*src + i * 4))
    i = i + 1
  Wend
EndProcedure

XIncludeFile "../../runtime/kokoro_assets.pmi"
XIncludeFile "../../runtime/kokoro_g2p.pmi"

Structure PmoKokoroPrepared
  InputIds.s
  Style.s
  Speed.s
  Phonemes.s
  TokenCount.i
  VoiceRow.i
EndStructure

Procedure.i PmoKokoroFail(Message.s)
  If PmoKokoroError = "" : PmoKokoroError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoKokoroFreeFile(*Storage.Integer, *Bytes.Integer)
  If *Storage And *Storage\i : FreeMemory(*Storage\i) : *Storage\i = 0 : EndIf
  If *Bytes : *Bytes\i = 0 : EndIf
EndProcedure

Procedure.i PmoKokoroLoadFile(Path.s, *Storage.Integer, *Bytes.Integer)
  Protected Size.q = FileSize(Path)
  Protected File.i
  Protected ReadBytes.q
  If *Storage = 0 Or *Bytes = 0 : ProcedureReturn PmoKokoroFail("internal file destination is null") : EndIf
  *Storage\i = 0 : *Bytes\i = 0
  If Size <= 0 Or Size > $7FFFFFFF : ProcedureReturn PmoKokoroFail("asset is missing or too large: " + Path) : EndIf
  *Storage\i = AllocateMemory(Size)
  If *Storage\i = 0 : ProcedureReturn PmoKokoroFail("cannot allocate asset: " + Path) : EndIf
  File = ReadFile(#PB_Any, Path, #PB_File_SharedRead)
  If File = 0 : PmoKokoroFreeFile(*Storage, *Bytes) : ProcedureReturn PmoKokoroFail("cannot open asset: " + Path) : EndIf
  ReadBytes = ReadData(File, *Storage\i, Size)
  CloseFile(File)
  If ReadBytes <> Size : PmoKokoroFreeFile(*Storage, *Bytes) : ProcedureReturn PmoKokoroFail("short read in asset: " + Path) : EndIf
  *Bytes\i = Size
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKokoroWriteNpy(Path.s, Descriptor.s, Shape.s, *Data, Bytes.i)
  Protected Header.s = "{'descr': '" + Descriptor + "', 'fortran_order': False, 'shape': " + Shape + ", }"
  Protected HeaderBytes.i
  Protected Temp.s = Path + ".tmp"
  Protected File.i
  If *Data = 0 Or Bytes < 0 : ProcedureReturn PmoKokoroFail("invalid NumPy payload") : EndIf
  While (10 + StringByteLength(Header, #PB_Ascii) + 1) % 16 <> 0 : Header + " " : Wend
  Header + Chr(10)
  HeaderBytes = StringByteLength(Header, #PB_Ascii)
  If HeaderBytes > $FFFF : ProcedureReturn PmoKokoroFail("NumPy header is too large") : EndIf
  If FileSize(Temp) >= 0 : DeleteFile(Temp) : EndIf
  File = CreateFile(#PB_Any, Temp)
  If File = 0 : ProcedureReturn PmoKokoroFail("cannot create NumPy input: " + Temp) : EndIf
  WriteByte(File, $93)
  WriteString(File, "NUMPY", #PB_Ascii)
  WriteByte(File, 1) : WriteByte(File, 0)
  WriteWord(File, HeaderBytes)
  WriteString(File, Header, #PB_Ascii)
  If Bytes > 0 And WriteData(File, *Data, Bytes) <> Bytes
    CloseFile(File) : DeleteFile(Temp) : ProcedureReturn PmoKokoroFail("short write in NumPy input")
  EndIf
  CloseFile(File)
  If FileSize(Path) >= 0 And DeleteFile(Path) = 0
    DeleteFile(Temp) : ProcedureReturn PmoKokoroFail("cannot replace NumPy input: " + Path)
  EndIf
  If RenameFile(Temp, Path) = 0 : ProcedureReturn PmoKokoroFail("cannot publish NumPy input: " + Path) : EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKokoroPrepareInputs(ModelPath.s, Text.s, G2pPath.s, VoicePath.s,
                                   Speed.f, OutputPrefix.s, *Prepared.PmoKokoroPrepared)
  Protected ModelHash.s
  Protected TextBytes.i
  Protected PhonemeBytes.i
  Protected TokenCount.i
  Protected VoiceRow.i
  Protected G2pStorage.Integer
  Protected G2pBytes.Integer
  Protected VoiceStorage.Integer
  Protected VoiceBytes.Integer
  Protected *TextUtf8
  Protected *Phonemes
  Protected *Scratch
  Protected *Ids
  Protected *Style
  Protected *VoiceRowPointer
  Protected IdPath.s = OutputPrefix + ".trace-input_ids.npy"
  Protected StylePath.s = OutputPrefix + ".trace-style.npy"
  Protected SpeedPath.s = OutputPrefix + ".trace-speed.npy"
  PmoKokoroError = ""
  If *Prepared = 0 : ProcedureReturn PmoKokoroFail("internal Kokoro result is null") : EndIf
  ClearStructure(*Prepared, PmoKokoroPrepared) : InitializeStructure(*Prepared, PmoKokoroPrepared)
  If FileSize(ModelPath) <= 0 : ProcedureReturn PmoKokoroFail("Kokoro model is missing: " + ModelPath) : EndIf
  If Text = "" : ProcedureReturn PmoKokoroFail("Kokoro text is empty") : EndIf
  If OutputPrefix = "" Or GetPathPart(OutputPrefix) = "" Or FileSize(GetPathPart(OutputPrefix)) <> -2
    ProcedureReturn PmoKokoroFail("Kokoro output prefix must name an existing folder")
  EndIf
  UseSHA2Fingerprint()
  ModelHash = LCase(FileFingerprint(ModelPath, #PB_Cipher_SHA2, 256))
  If ModelHash <> #PMO_KOKORO_MODEL_SHA256$
    ProcedureReturn PmoKokoroFail("selected model is not the pinned full Kokoro 82M model (SHA-256 " + ModelHash + ")")
  EndIf
  If PmKokoroF32PositiveFinite(@Speed) = 0 : ProcedureReturn PmoKokoroFail("Kokoro speed must be finite and positive") : EndIf
  If PmoKokoroLoadFile(G2pPath, @G2pStorage, @G2pBytes) = 0 : Goto PmoKokoroPrepareFailed : EndIf
  If PmKokoroG2pValidate(G2pStorage\i, G2pBytes\i) = 0
    PmoKokoroFail("Kokoro pronunciation pack failed its identity, extent or CRC check") : Goto PmoKokoroPrepareFailed
  EndIf
  If PmoKokoroLoadFile(VoicePath, @VoiceStorage, @VoiceBytes) = 0 : Goto PmoKokoroPrepareFailed : EndIf
  If PmKokoroValidateVoice(VoiceStorage\i, VoiceBytes\i) = 0
    PmoKokoroFail("Kokoro voice pack failed its model identity, extent, CRC or finite-value check") : Goto PmoKokoroPrepareFailed
  EndIf
  TextBytes = StringByteLength(Text, #PB_UTF8)
  *TextUtf8 = UTF8(Text)
  *Phonemes = AllocateMemory(#PMK_MAX_PHONEMES * 4)
  *Scratch = AllocateMemory(#PMK_G2P_MAX_WORD)
  *Ids = AllocateMemory(#PMK_MAX_IDS * 8)
  *Style = AllocateMemory(#PMK_VOICE_WIDTH * 4)
  If *TextUtf8 = 0 Or *Phonemes = 0 Or *Scratch = 0 Or *Ids = 0 Or *Style = 0
    PmoKokoroFail("cannot allocate Kokoro input preparation buffers") : Goto PmoKokoroPrepareFailed
  EndIf
  PhonemeBytes = PmKokoroG2pEnglish(G2pStorage\i, G2pBytes\i, *TextUtf8, TextBytes,
                                    *Phonemes, #PMK_MAX_PHONEMES * 4, *Scratch, #PMK_G2P_MAX_WORD)
  If PhonemeBytes <= 0
    PmoKokoroFail("text could not be converted to the pinned Kokoro vocabulary; use /phonemes/ for an explicit override")
    Goto PmoKokoroPrepareFailed
  EndIf
  TokenCount = PmKokoroEncodePhonemes(*Phonemes, PhonemeBytes, *Ids, #PMK_MAX_IDS)
  If TokenCount < 3 : PmoKokoroFail("Kokoro tokenization failed") : Goto PmoKokoroPrepareFailed : EndIf
  VoiceRow = TokenCount - 3
  *VoiceRowPointer = PmKokoroVoiceRow(VoiceStorage\i, VoiceBytes\i, VoiceRow)
  If *VoiceRowPointer = 0 : PmoKokoroFail("Kokoro voice has no row for " + Str(TokenCount - 2) + " phonemes") : Goto PmoKokoroPrepareFailed : EndIf
  PmTensorCopy(*VoiceRowPointer, *Style, #PMK_VOICE_WIDTH)
  If PmoKokoroWriteNpy(IdPath, "<i8", "(1, " + Str(TokenCount) + ")", *Ids, TokenCount * 8) = 0 : Goto PmoKokoroPrepareFailed : EndIf
  If PmoKokoroWriteNpy(StylePath, "<f4", "(1, 256)", *Style, #PMK_VOICE_WIDTH * 4) = 0 : Goto PmoKokoroPrepareFailed : EndIf
  If PmoKokoroWriteNpy(SpeedPath, "<f4", "(1,)", @Speed, 4) = 0 : Goto PmoKokoroPrepareFailed : EndIf
  *Prepared\InputIds = IdPath : *Prepared\Style = StylePath : *Prepared\Speed = SpeedPath
  *Prepared\Phonemes = PeekS(*Phonemes, PhonemeBytes, #PB_UTF8)
  *Prepared\TokenCount = TokenCount : *Prepared\VoiceRow = VoiceRow
  If *TextUtf8 : FreeMemory(*TextUtf8) : EndIf
  If *Phonemes : FreeMemory(*Phonemes) : EndIf
  If *Scratch : FreeMemory(*Scratch) : EndIf
  If *Ids : FreeMemory(*Ids) : EndIf
  If *Style : FreeMemory(*Style) : EndIf
  PmoKokoroFreeFile(@G2pStorage, @G2pBytes) : PmoKokoroFreeFile(@VoiceStorage, @VoiceBytes)
  ProcedureReturn #True

  PmoKokoroPrepareFailed:
  If *TextUtf8 : FreeMemory(*TextUtf8) : EndIf
  If *Phonemes : FreeMemory(*Phonemes) : EndIf
  If *Scratch : FreeMemory(*Scratch) : EndIf
  If *Ids : FreeMemory(*Ids) : EndIf
  If *Style : FreeMemory(*Style) : EndIf
  PmoKokoroFreeFile(@G2pStorage, @G2pBytes) : PmoKokoroFreeFile(@VoiceStorage, @VoiceBytes)
  ProcedureReturn #False
EndProcedure

Procedure.i PmoKokoroPrepareCommand(ModelPath.s)
  Protected Text.s
  Protected TextFile.s
  Protected G2p.s
  Protected Voice.s
  Protected Output.s
  Protected Speed.f = 1.0
  Protected OptionName.s
  Protected Index.i = 2
  Protected File.i
  Protected Prepared.PmoKokoroPrepared
  While Index < CountProgramParameters()
    OptionName = ProgramParameter(Index)
    If OptionName = "--text" And Index + 1 < CountProgramParameters()
      Index + 1 : Text = ProgramParameter(Index)
    ElseIf OptionName = "--text-file" And Index + 1 < CountProgramParameters()
      Index + 1 : TextFile = ProgramParameter(Index)
    ElseIf OptionName = "--g2p" And Index + 1 < CountProgramParameters()
      Index + 1 : G2p = ProgramParameter(Index)
    ElseIf OptionName = "--voice" And Index + 1 < CountProgramParameters()
      Index + 1 : Voice = ProgramParameter(Index)
    ElseIf OptionName = "--speed" And Index + 1 < CountProgramParameters()
      Index + 1 : Speed = ValF(ProgramParameter(Index))
    ElseIf OptionName = "--output" And Index + 1 < CountProgramParameters()
      Index + 1 : Output = ProgramParameter(Index)
    Else
      ProcedureReturn PmoKokoroFail("unknown or incomplete Kokoro option " + OptionName)
    EndIf
    Index + 1
  Wend
  If Text <> "" And TextFile <> "" : ProcedureReturn PmoKokoroFail("use --text or --text-file, not both") : EndIf
  If TextFile <> ""
    File = ReadFile(#PB_Any, TextFile, #PB_File_SharedRead)
    If File = 0 : ProcedureReturn PmoKokoroFail("cannot open Kokoro text file: " + TextFile) : EndIf
    Text = ReadString(File, #PB_UTF8 | #PB_File_IgnoreEOL)
    CloseFile(File)
  EndIf
  If G2p = "" Or Voice = "" Or Output = "" : ProcedureReturn PmoKokoroFail("--g2p, --voice and --output are required") : EndIf
  If PmoKokoroPrepareInputs(ModelPath, Text, G2p, Voice, Speed, Output, @Prepared) = 0 : ProcedureReturn #False : EndIf
  PrintN("PASS: prepared native Kokoro inputs")
  PrintN("Phonemes: " + Prepared\Phonemes)
  PrintN("Tokens: " + Str(Prepared\TokenCount) + ", voice row: " + Str(Prepared\VoiceRow))
  PrintN("Trace input: input_ids=" + Prepared\InputIds)
  PrintN("Trace input: style=" + Prepared\Style)
  PrintN("Trace input: speed=" + Prepared\Speed)
  ProcedureReturn #True
EndProcedure
