; Speech application around the generated reusable model, in the host language.
; The compiled application loads PMW/G2P/voice data once. Speak has no build path.
Macro PeekN(Address)
  (PeekL(Address) & $FFFFFFFF)
EndMacro
XIncludeFile "kokoro_assets.pmi"
XIncludeFile "kokoro_g2p.pmi"
XIncludeFile "kokoro_reader_windows.pbi"
XIncludeFile "kokoro_voices_windows.pbi"

Enumeration
  #KS_TEXT
  #KS_SPEAK
  #KS_STOP
  #KS_REPLAY
  #KS_SAVE
  #KS_SPEED
  #KS_STATUS
  #KS_VOICE
  #KS_VOICE_PICK
  #KS_G2P
  #KS_G2P_PICK
  #KS_PAUSE
  #KS_OPEN
  #KS_PROGRESS
  #KS_HINT
EndEnumeration
#KS_DONE = #PB_Event_FirstCustomValue+214
Global KsWindow.i,KsThread.i,KsClosing.i,KsActive.i,KsPaused.i,KsIsReplay.i
Global KsTotalPosition.i,KsGenerated.i,KsFrames.i
Global KsStarted.i,KsStage.s,KsError.s,KsResult.s,KsText.s,KsSpeed.f
Global KsVoicePath.s,KsG2pPath.s,KsLoadedVoice.s,KsLoadedG2p.s,KsHome.s,KsMutex.i
Global KsVoice.i,KsVoiceBytes.i,KsG2p.i,KsG2pBytes.i,KsSequence.i
Global KsExtra.i,KsExtraBytes.i,KsPronunciationNotes.s

Procedure KsStatus(Text.s)
  LockMutex(KsMutex) : KsStage=Text : UnlockMutex(KsMutex)
EndProcedure

Procedure.i KsLoad(Path.s,*Address.Integer,*Bytes.Integer)
  Protected f.i,bytes.i=FileSize(Path),buffer.i
  If bytes<=0 Or bytes>32*1024*1024 : KsError="Cannot read the selected asset. Check the file path and size." : ProcedureReturn 0 : EndIf
  f=ReadFile(#PB_Any,Path)
  If f=0 : KsError="Cannot open "+Path : ProcedureReturn 0 : EndIf
  buffer=AllocateMemory(bytes)
  If buffer=0 : CloseFile(f) : KsError="Cannot allocate memory for the selected asset." : ProcedureReturn 0 : EndIf
  If ReadData(f,buffer,bytes)<>bytes
    FreeMemory(buffer) : CloseFile(f) : KsError="The selected asset could not be read completely." : ProcedureReturn 0
  EndIf
  CloseFile(f)
  If *Address\i : FreeMemory(*Address\i) : EndIf
  *Address\i=buffer : *Bytes\i=bytes
  ProcedureReturn 1
EndProcedure

Procedure.i KsAssets()
  Protected extraPath.s
  If KsLoadedG2p<>KsG2pPath
    If KsLoad(KsG2pPath,@KsG2p,@KsG2pBytes)=0 : ProcedureReturn 0 : EndIf
    If PmKokoroG2pValidate(KsG2p,KsG2pBytes)=0
      KsLoadedG2p="" : KsError="Pronunciation pack validation failed. Select a valid PMG2P file." : ProcedureReturn 0
    EndIf
    KsLoadedG2p=KsG2pPath
  EndIf
  If PmKokoroG2pExtraPack=0
    extraPath=GetPathPart(KsG2pPath)+"kokoro_us_extra.pmg2p"
    If FileSize(extraPath)<0 : extraPath=#PM_SPEECH_ASSET_ROOT$+"Kokoro\Assets\kokoro_us_extra.pmg2p" : EndIf
    If KsLoad(extraPath,@KsExtra,@KsExtraBytes)=0 : ProcedureReturn 0 : EndIf
    If PmKokoroG2pUseExtra(KsExtra,KsExtraBytes)=0
      KsError="[G2P 11] Supplementary pronunciation pack validation failed. Restore kokoro_us_extra.pmg2p beside the main pronunciation pack."
      ProcedureReturn 0
    EndIf
  EndIf
  If KsLoadedVoice<>KsVoicePath
    If KsLoad(KsVoicePath,@KsVoice,@KsVoiceBytes)=0 : ProcedureReturn 0 : EndIf
    If PmKokoroValidateVoice(KsVoice,KsVoiceBytes)=0
      KsLoadedVoice="" : KsError="Voice validation failed. Select a valid PMVOICE for the full Kokoro model." : ProcedureReturn 0
    EndIf
    KsLoadedVoice=KsVoicePath
  EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i KsInfer(Text.s)
  Protected *utf=UTF8(Text),bytes.i=StringByteLength(Text,#PB_UTF8),phonemes.i,count.i,row.i,a.i,cp.i,nextAt.i,detail.s
  Protected missed.i,at.i,finish.i,word.s
  Protected Dim ph.a(#PMK_MAX_PHONEMES*4),Dim scratch.a(#PMK_G2P_MAX_WORD),Dim ids.q(#PMK_MAX_IDS)
  If *utf=0 : KsError="Text allocation failed." : ProcedureReturn 0 : EndIf
  phonemes=PmKokoroG2pEnglish(KsG2p,KsG2pBytes,*utf,bytes,@ph(0),#PMK_MAX_PHONEMES*4,@scratch(0),#PMK_G2P_MAX_WORD)
  If phonemes<=0
    cp=PmKokoroUtf8Next(*utf,PmKokoroG2pErrorAt,bytes,@nextAt)
    Select PmKokoroG2pError
      Case 10 : detail="Pronunciation pack validation failed. Select the checked pronunciation pack."
      Case 20 : detail="The text or conversion buffer is empty. Enter a nonempty passage."
      Case 21 : detail="A word exceeds 63 characters. Check for a pasted address or missing word boundary."
      Case 30 : detail="Unsupported character U+"+RSet(Hex(cp),4,"0")+". Replace this character or write its spoken name."
      Case 31 : detail="The /phoneme/ override contains an invalid pronunciation symbol. Check the text inside the slashes."
      Default : detail="The pronunciation exceeds the passage buffer. Split this passage at a word boundary."
    EndSelect
    KsError="[G2P "+Str(PmKokoroG2pError)+"] "+detail+" UTF-8 position "+Str(PmKokoroG2pErrorAt+1)+" in: "+Left(Text,180)
    FreeMemory(*utf) : ProcedureReturn 0
  EndIf
  For missed=0 To PmKokoroG2pSpelledCount-1
    If missed>=8 : Break : EndIf
    at=PmKokoroG2pSpelledAt(missed) : finish=at
    While finish<bytes
      cp=PeekA(*utf+finish)
      If Not ((cp>=65 And cp<=90) Or (cp>=97 And cp<=122) Or cp=39) : Break : EndIf
      finish+1
    Wend
    word=PeekS(*utf+at,finish-at,#PB_Ascii)
    ; Initialisms such as GPIO are supposed to be spoken as letters.
    If Len(word)>1 And word<>UCase(word)
      LockMutex(KsMutex)
      If Len(KsPronunciationNotes)<200 And FindString(","+KsPronunciationNotes+",",","+word+",")=0
        If KsPronunciationNotes<>"" : KsPronunciationNotes+"," : EndIf
        KsPronunciationNotes+word
      EndIf
      UnlockMutex(KsMutex)
    EndIf
  Next
  FreeMemory(*utf)
  count=PmKokoroEncodePhonemes(@ph(0),phonemes,@ids(0),#PMK_MAX_IDS)
  If count<3 : KsError="The text could not be tokenized." : ProcedureReturn 0 : EndIf
  row=PmKokoroVoiceRow(KsVoice,KsVoiceBytes,count-3)
  If row=0 : KsError="This text chunk is too long for the voice. Use shorter words or sentences." : ProcedureReturn 0 : EndIf
  PmModelResetRequest()
  a=PmModelInput(0)
  If DShape(a,7,2,1,count)=0 : KsError=DError : ProcedureReturn 0 : EndIf
  CopyMemory(@ids(0),Dt(a)\Data,count*8)
  a=PmModelInput(1)
  If DShape(a,1,2,1,256)=0 : KsError=DError : ProcedureReturn 0 : EndIf
  CopyMemory(row,Dt(a)\Data,256*4)
  a=PmModelInput(2)
  If DShape(a,1,1,1)=0 : KsError=DError : ProcedureReturn 0 : EndIf
  PokeF(Dt(a)\Data,KsSpeed)
  PmModelExecute()
  If DCancel : KsError="Speech cancelled." : ProcedureReturn 0 : EndIf
  If DError<>"" : KsError=DError : ProcedureReturn 0 : EndIf
  ProcedureReturn PmModelOutput(0)
EndProcedure

Procedure.i KsWaveHeader(File.i,Bytes.i)
  Protected Dim header.a(43)
  PokeS(@header(0),"RIFF",4,#PB_Ascii) : PokeL(@header(4),36+Bytes)
  PokeS(@header(8),"WAVEfmt ",8,#PB_Ascii) : PokeL(@header(16),16)
  PokeW(@header(20),1) : PokeW(@header(22),1) : PokeL(@header(24),24000)
  PokeL(@header(28),48000) : PokeW(@header(32),2) : PokeW(@header(34),16)
  PokeS(@header(36),"data",4,#PB_Ascii) : PokeL(@header(40),Bytes)
  FileSeek(File,0)
  ProcedureReturn Bool(WriteData(File,@header(0),44)=44)
EndProcedure

Procedure KsWorker(*Unused)
  Protected t.i,f.i,i.i,bits.i,frames.i,bytes.i,chunk.s,cursor.i,chunkIndex.i,buffer.i
  Protected sample.f,pcm.i
  KsError="" : DError=""
  KsStatus("Loading voice and pronunciation data")
  If KsAssets()=0 : Goto Finished : EndIf
  KsStatus("Loading model weights (once)")
  If PmModelInitialize(GetPathPart(ProgramFilename())+#PM_SPEECH_WEIGHT_NAME$)=0
    KsError=DError : If DCancel : KsError="Reading stopped." : EndIf
    Goto Finished
  EndIf
  KsSequence+1
  KsResult=KsHome+"speech-"+FormatDate("%yyyy%mm%dd-%hh%ii%ss",Date())+"-"+Str(KsSequence)+".wav"
  f=CreateFile(#PB_Any,KsResult)
  If f=0 : KsResult="" : KsError="Cannot create a recording. Check free space and access to "+KsHome : Goto Finished : EndIf
  If KsWaveHeader(f,0)=0 : KsError="Cannot write the recording header. Check free disk space." : Goto RecordingDone : EndIf
  While cursor<Len(KsText)
    If KsAudioWaitSlot()=0 : Break : EndIf
    chunk=KsNextPassage(@KsText,Len(KsText),@cursor)
    If KsTextError<>"" : KsError=KsTextError : Break : EndIf
    If chunk="" : Break : EndIf
    chunkIndex+1 : KsStatus("Generating passage "+Str(chunkIndex))
    t=KsInfer(chunk)
    If t=0 : Break : EndIf
    If Dt(t)\Count<=0 Or Dt(t)\Count>#KS_MAX_PCM_BYTES/2
      KsError="A passage produced empty audio or more than 60 seconds. Shorten the passage or check the model." : Break
    EndIf
    If frames+Dt(t)\Count>1073741800 : KsError="This recording exceeds the WAV size limit. Read the book in smaller sections." : Break : EndIf
    bytes=Dt(t)\Count*2 : buffer=AllocateMemory(bytes)
    If buffer=0 : KsError="Cannot allocate the next audio block. Close other applications and try again." : Break : EndIf
    For i=0 To Dt(t)\Count-1
      bits=PeekL(Dt(t)\Data+i*4)
      If (bits & $7F800000)=$7F800000 : KsError="The model produced non-finite audio. This passage was not queued." : Break : EndIf
      sample=PeekF(Dt(t)\Data+i*4)
      If sample>1 : sample=1 : ElseIf sample< -1 : sample=-1 : EndIf
      If sample<0 : pcm=Int(sample*32768.0) : Else : pcm=Int(sample*32767.0) : EndIf
      PokeW(buffer+i*2,pcm & $FFFF)
    Next
    If KsError<>"" Or DCancel : FreeMemory(buffer) : buffer=0 : Break : EndIf
    If WriteData(f,buffer,bytes)<>bytes
      FreeMemory(buffer) : buffer=0 : KsError="The recording could not be written completely. Check free disk space." : Break
    EndIf
    frames+Dt(t)\Count
    If KsAudioPublish(buffer,bytes,chunkIndex,cursor)=0
      FreeMemory(buffer) : buffer=0
      If DCancel=0 : KsError="The audio queue rejected a passage. Stop and restart reading." : EndIf
      Break
    EndIf
    buffer=0
    LockMutex(KsMutex) : KsGenerated=chunkIndex : KsFrames=frames : UnlockMutex(KsMutex)
    PmModelResetRequest()
  Wend
  RecordingDone:
  ; Keep a valid completed prefix on Stop, without retaining book-sized audio.
  If KsWaveHeader(f,frames*2)=0
    KsError="The recording header could not be finalized. Check free disk space." : KsResult=""
  EndIf
  CloseFile(f)
  If frames=0 : KsResult="" : EndIf
  Finished:
  PmModelResetRequest()
  PostEvent(#KS_DONE,KsWindow,0)
EndProcedure

; Replay uses the same bounded queue, not LoadSound on a book-sized WAV.
Procedure KsReplayWorker(*Unused)
  Protected f.i=ReadFile(#PB_Any,KsResult),remaining.i,bytes.i,buffer.i,position.i,passage.i
  Protected Dim header.a(43)
  KsError=""
  If f=0 : KsError="Cannot open the recording. Check that the WAV file still exists." : Goto Finished : EndIf
  If ReadData(f,@header(0),44)<>44 Or PeekS(@header(8),4,#PB_Ascii)<>"WAVE" Or PeekL(@header(24))<>24000 Or PeekW(@header(22))<>1 Or PeekW(@header(34))<>16 Or PeekS(@header(36),4,#PB_Ascii)<>"data"
    KsError="The recording is not the expected mono 24 kHz PCM16 WAV. Select a completed recording." : Goto Finished
  EndIf
  remaining=PeekL(@header(40)) & $FFFFFFFF
  If remaining<=0 Or remaining & 1 Or remaining>Lof(f)-44
    KsError="The recording is empty or truncated. Generate a new recording." : Goto Finished
  EndIf
  LockMutex(KsMutex) : KsTotalPosition=remaining : UnlockMutex(KsMutex)
  While remaining>0
    If KsAudioWaitSlot()=0 : Break : EndIf
    bytes=96000 : If bytes>remaining : bytes=remaining : EndIf
    buffer=AllocateMemory(bytes)
    If buffer=0 : KsError="Cannot allocate replay audio. Close other applications and try again." : Break : EndIf
    If ReadData(f,buffer,bytes)<>bytes
      FreeMemory(buffer) : KsError="The recording could not be read completely. Check the WAV file." : Break
    EndIf
    passage+1 : position+bytes : remaining-bytes
    If KsAudioPublish(buffer,bytes,passage,position)=0
      FreeMemory(buffer)
      If DCancel=0 : KsError="The replay queue rejected audio. Stop and retry playback." : EndIf
      Break
    EndIf
  Wend
  Finished:
  If f : CloseFile(f) : EndIf
  PostEvent(#KS_DONE,KsWindow,0)
EndProcedure

Procedure KsLayout()
  Protected w.i=WindowWidth(KsWindow),h.i=WindowHeight(KsWindow)
  ResizeGadget(#KS_HINT,18,16,w-180,32) : ResizeGadget(#KS_OPEN,w-150,14,132,32)
  ResizeGadget(#KS_TEXT,18,56,w-36,h-300)
  ResizeGadget(#KS_PROGRESS,18,h-232,w-36,16)
  ResizeGadget(#KS_SPEAK,18,h-202,100,34)
  ResizeGadget(#KS_PAUSE,128,h-202,100,34)
  ResizeGadget(#KS_STOP,238,h-202,80,34)
  ResizeGadget(#KS_REPLAY,328,h-202,90,34)
  ResizeGadget(#KS_SAVE,428,h-202,100,34)
  ResizeGadget(#KS_SPEED,538,h-202,80,30)
  ResizeGadget(#KS_VOICE,18,h-148,w-138,28)
  ResizeGadget(#KS_VOICE_PICK,w-110,h-148,92,28)
  ResizeGadget(#KS_G2P,18,h-104,w-138,28)
  ResizeGadget(#KS_G2P_PICK,w-110,h-104,92,28)
  ResizeGadget(#KS_STATUS,18,h-58,w-36,48)
EndProcedure

Procedure KsBusy(Value.i)
  DisableGadget(#KS_SPEAK,Value) : DisableGadget(#KS_OPEN,Value)
  SendMessage_(GadgetID(#KS_TEXT),#EM_SETREADONLY,Value,0)
  DisableGadget(#KS_VOICE,Value) : DisableGadget(#KS_VOICE_PICK,Value)
  DisableGadget(#KS_G2P,Value) : DisableGadget(#KS_G2P_PICK,Value)
  DisableGadget(#KS_SPEED,Value)
  DisableGadget(#KS_PAUSE,1-Value) : DisableGadget(#KS_STOP,1-Value)
  DisableGadget(#KS_REPLAY,Bool(Value Or KsResult=""))
  DisableGadget(#KS_SAVE,Bool(Value Or KsResult=""))
EndProcedure

Procedure KsVoiceMenu(Selected.i)
  ClearGadgetItems(#KS_VOICE)
  ForEach KsVoices() : AddGadgetItem(#KS_VOICE,-1,KsVoices()\Label) : Next
  SetGadgetState(#KS_VOICE,Selected)
  GadgetToolTip(#KS_VOICE,KsVoicePathAt(Selected))
EndProcedure

Procedure KsStart(Replay.i)
  If KsActive : ProcedureReturn : EndIf
  If Replay=0
    KsText=Trim(GetGadgetText(#KS_TEXT))
    If KsText="" : SetGadgetText(#KS_STATUS,"Open a plain-text book or paste text, then press Read.") : ProcedureReturn : EndIf
    If Len(KsText)>#KS_MAX_TEXT : SetGadgetText(#KS_STATUS,"The text exceeds 4 million characters. Open a smaller section of the book.") : ProcedureReturn : EndIf
  ElseIf KsResult=""
    ProcedureReturn
  EndIf
  If KsAudioOpen()=0 : SetGadgetText(#KS_STATUS,KsAudioError) : ProcedureReturn : EndIf
  KsVoicePath=KsVoicePathAt(GetGadgetState(#KS_VOICE)) : KsG2pPath=GetGadgetText(#KS_G2P)
  KsSpeed=ValF(GetGadgetText(#KS_SPEED)) : DCancel=0 : KsError=""
  KsStarted=ElapsedMilliseconds() : KsPaused=0 : KsIsReplay=Replay : KsPronunciationNotes=""
  KsGenerated=0 : KsFrames=0 : KsTotalPosition=Len(KsText)
  KsActive=1 : KsBusy(1) : SetGadgetText(#KS_PAUSE,"Pause") : SetGadgetState(#KS_PROGRESS,0)
  If Replay
    KsStatus("Replaying recording") : KsThread=CreateThread(@KsReplayWorker(),0)
  Else
    KsResult="" : KsStatus("Preparing the first passage") : KsThread=CreateThread(@KsWorker(),0)
  EndIf
  If KsThread=0
    KsAudioClose() : KsActive=0 : KsBusy(0)
    SetGadgetText(#KS_STATUS,"Cannot start the reading worker. Close other applications and try again.")
  EndIf
EndProcedure

Procedure KsUpdate()
  Protected stage.s,queued.i,generated.i,total.i,progress.i,notes.s
  If KsActive=0 : ProcedureReturn : EndIf
  If KsAudioPump()=0 : KsAudioStop() : EndIf
  LockMutex(KsMutex) : stage=KsStage : generated=KsGenerated : total=KsTotalPosition : notes=KsPronunciationNotes : UnlockMutex(KsMutex)
  LockMutex(KsQueueMutex) : queued=KsQueued : UnlockMutex(KsQueueMutex)
  If total>0
    progress=KsPlayedPosition*1000/total : If progress>1000 : progress=1000 : EndIf
    SetGadgetState(#KS_PROGRESS,progress)
  EndIf
  If KsThread=0 And (DCancel Or queued=0)
    If KsAudioClose()=0 : SetGadgetText(#KS_STATUS,KsAudioError) : ProcedureReturn : EndIf
    KsActive=0 : KsPaused=0 : SetGadgetText(#KS_PAUSE,"Pause") : KsBusy(0)
    If KsAudioError<>"" : stage=KsAudioError
    ElseIf KsError<>"" : stage=KsError
    ElseIf DCancel : stage="Reading stopped."
    Else : stage="Finished reading." : SetGadgetState(#KS_PROGRESS,1000)
    EndIf
    If KsResult<>"" And DCancel And KsIsReplay=0 : stage+" Completed passages are available with Save WAV." : EndIf
    If notes<>"" And KsError="" And KsAudioError="" : stage+" Spelled unknown words: "+notes+". Check their spelling or use /phonemes/." : EndIf
    SetGadgetText(#KS_STATUS,stage)
    ProcedureReturn
  EndIf
  If DCancel
    stage="Playback stopped. Waiting for the active model operation to finish."
  ElseIf KsPaused
    stage="Paused. "+Str(queued)+" audio blocks buffered; press Resume to continue."
  ElseIf queued=0 And KsFirstAudio
    stage="Buffering the next passage. "+stage+"."
  ElseIf KsIsReplay
    stage="Replaying the recording."
  ElseIf KsFirstAudio
    stage="Reading passage "+Str(KsPlayed+1)+". "+Str(queued)+" blocks queued; "+Str(generated)+" generated."
  EndIf
  If KsFirstAudio : stage+" First audio queued in "+StrD((KsFirstAudio-KsStarted)/1000.0,1)+" s." : EndIf
  If notes<>"" : stage+" Spelled unknown words: "+notes+"." : EndIf
  SetGadgetText(#KS_STATUS,stage)
EndProcedure

Procedure KsOpenText()
  Protected path.s=OpenFileRequester("Open a plain-text book","","Text (*.txt)|*.txt",0)
  Protected file.i,bytes.i,buffer.i,text.s
  If path="" : ProcedureReturn : EndIf
  bytes=FileSize(path)
  If bytes<=0 Or bytes>8*1024*1024 : SetGadgetText(#KS_STATUS,"Choose a nonempty UTF-8 or UTF-16LE text file no larger than 8 MiB.") : ProcedureReturn : EndIf
  file=ReadFile(#PB_Any,path) : buffer=AllocateMemory(bytes+2)
  If file=0 Or buffer=0
    If file : CloseFile(file) : EndIf
    If buffer : FreeMemory(buffer) : EndIf
    SetGadgetText(#KS_STATUS,"Cannot load this text file. Check the path and available memory.") : ProcedureReturn
  EndIf
  If ReadData(file,buffer,bytes)=bytes
    If bytes>=2 And PeekU(buffer)=$FEFF
      text=PeekS(buffer+2,(bytes-2)/2,#PB_Unicode)
    Else
      text=PeekS(buffer,bytes,#PB_UTF8)
      If Left(text,1)=Chr($FEFF) : text=Mid(text,2) : EndIf
    EndIf
    If Len(text)>#KS_MAX_TEXT
      SetGadgetText(#KS_STATUS,"The text exceeds 4 million characters. Open a smaller section.")
    Else
      SetGadgetText(#KS_TEXT,text) : SetGadgetState(#KS_PROGRESS,0)
      SetGadgetText(#KS_STATUS,"Loaded "+GetFilePart(path)+". Press Read; playback begins after the first passage.")
    EndIf
  Else
    SetGadgetText(#KS_STATUS,"The text file could not be read completely. Check the file and try again.")
  EndIf
  CloseFile(file) : FreeMemory(buffer)
EndProcedure

Procedure PmSpeechRun()
  Protected event.i,gadget.i,path.s,font.i,i.i,voiceIndex.i,savedText.s="Hello, I hope you have a wonderful day."
  KsHome=GetEnvironmentVariable("LOCALAPPDATA")+"\PureMetalKokoro\"
  CreateDirectory(KsHome)
  KsG2pPath=#PM_SPEECH_ASSET_ROOT$+"Kokoro\Assets\kokoro_us_english.pmg2p"
  KsVoicePath=GetPathPart(ProgramFilename())+"voice.pmvoice"
  If OpenPreferences(KsHome+"settings.ini")
    KsG2pPath=ReadPreferenceString("pronunciation",KsG2pPath)
    KsVoicePath=ReadPreferenceString("voice",KsVoicePath)
    savedText=ReadPreferenceString("text",savedText)
    ClosePreferences()
  EndIf
  voiceIndex=KsVoiceDiscover(GetPathPart(ProgramFilename())+"voices\",KsVoicePath)
  KsMutex=CreateMutex()
  KsWindow=OpenWindow(#PB_Any,0,0,880,650,"Kokoro - book reader",
    #PB_Window_SystemMenu|#PB_Window_SizeGadget|#PB_Window_MinimizeGadget|#PB_Window_ScreenCentered)
  WindowBounds(KsWindow,680,480,#PB_Ignore,#PB_Ignore)
  font=LoadFont(#PB_Any,"Segoe UI",12)
  TextGadget(#KS_HINT,0,0,0,0,"Open or paste a book. Listen while the next passages are generated.")
  ButtonGadget(#KS_OPEN,0,0,0,0,"Open text...")
  EditorGadget(#KS_TEXT,0,0,0,0)
  SendMessage_(GadgetID(#KS_TEXT),#EM_EXLIMITTEXT,0,#KS_MAX_TEXT)
  SendMessage_(GadgetID(#KS_TEXT),#EM_SETTARGETDEVICE,0,0)
  SetGadgetFont(#KS_TEXT,FontID(font)) : SetGadgetText(#KS_TEXT,savedText)
  ButtonGadget(#KS_SPEAK,0,0,0,0,"Read")
  ButtonGadget(#KS_PAUSE,0,0,0,0,"Pause")
  ButtonGadget(#KS_STOP,0,0,0,0,"Stop")
  ButtonGadget(#KS_REPLAY,0,0,0,0,"Replay")
  ButtonGadget(#KS_SAVE,0,0,0,0,"Save WAV")
  ProgressBarGadget(#KS_PROGRESS,0,0,0,0,0,1000)
  ComboBoxGadget(#KS_SPEED,0,0,0,0)
  AddGadgetItem(#KS_SPEED,-1,"0.8") : AddGadgetItem(#KS_SPEED,-1,"1.0")
  AddGadgetItem(#KS_SPEED,-1,"1.2") : SetGadgetState(#KS_SPEED,1)
  GadgetToolTip(#KS_SPEED,"Speech speed: 1.0 is normal")
  ComboBoxGadget(#KS_VOICE,0,0,0,0)
  KsVoiceMenu(voiceIndex)
  ButtonGadget(#KS_VOICE_PICK,0,0,0,0,"Browse...")
  StringGadget(#KS_G2P,0,0,0,0,KsG2pPath)
  ButtonGadget(#KS_G2P_PICK,0,0,0,0,"Language...")
  TextGadget(#KS_STATUS,0,0,0,0,"Ready. The first Read loads the model; later passages reuse it.")
  KsLayout() : KsBusy(0)
  AddWindowTimer(KsWindow,1,50)
  Repeat
    event=WaitWindowEvent()
    Select event
      Case #PB_Event_SizeWindow : KsLayout()
      Case #PB_Event_Timer : KsUpdate()
      Case #KS_DONE
        If KsThread : WaitThread(KsThread) : KsThread=0 : EndIf
        If KsError<>"" : KsAudioStop() : EndIf
        KsUpdate()
      Case #PB_Event_CloseWindow
        If KsActive : KsClosing=1 : KsAudioStop() : KsUpdate() : Else : Break : EndIf
      Case #PB_Event_Gadget
        gadget=EventGadget()
        Select gadget
          Case #KS_SPEAK : KsStart(0)
          Case #KS_PAUSE
            If KsActive And DCancel=0
              If KsAudioPause(1-KsPaused)
                KsPaused=1-KsPaused
                If KsPaused : SetGadgetText(#KS_PAUSE,"Resume") : Else : SetGadgetText(#KS_PAUSE,"Pause") : EndIf
              Else
                KsAudioStop()
              EndIf
              KsUpdate()
            EndIf
          Case #KS_STOP : KsAudioStop() : KsUpdate()
          Case #KS_REPLAY : KsStart(1)
          Case #KS_OPEN : If KsActive=0 : KsOpenText() : EndIf
          Case #KS_SAVE
            path=SaveFileRequester("Save speech",GetUserDirectory(#PB_Directory_Documents)+"kokoro.wav","WAV (*.wav)|*.wav",0)
            If path<>"" And KsResult<>""
              If CopyFile(KsResult,path)=0 : SetGadgetText(#KS_STATUS,"Could not save the recording to "+path+". Check destination access and free space.") : EndIf
            EndIf
          Case #KS_VOICE_PICK
            path=OpenFileRequester("Choose voice",KsVoicePath,"Kokoro voice (*.pmvoice)|*.pmvoice",0)
            If path<>"" : KsVoicePath=path : KsVoiceMenu(KsVoiceAdd(path)) : EndIf
          Case #KS_VOICE
            KsVoicePath=KsVoicePathAt(GetGadgetState(#KS_VOICE))
            GadgetToolTip(#KS_VOICE,KsVoicePath)
            SetGadgetText(#KS_STATUS,"Selected "+GetGadgetText(#KS_VOICE)+". Press Read to use this voice.")
          Case #KS_G2P_PICK
            path=OpenFileRequester("Choose pronunciation pack",KsG2pPath,"Pronunciation (*.pmg2p)|*.pmg2p",0)
            If path<>"" : SetGadgetText(#KS_G2P,path) : EndIf
        EndSelect
    EndSelect
    If KsClosing And KsActive=0 : Break : EndIf
  ForEver
  If CreatePreferences(KsHome+"settings.ini")
    ; Do not duplicate a whole book into the preference file.
    savedText=GetGadgetText(#KS_TEXT)
    If Len(savedText)<=4096 : WritePreferenceString("text",savedText) : EndIf
    WritePreferenceString("pronunciation",GetGadgetText(#KS_G2P))
    WritePreferenceString("voice",KsVoicePathAt(GetGadgetState(#KS_VOICE))) : ClosePreferences()
  EndIf
  PmModelClose()
  If KsVoice : FreeMemory(KsVoice) : EndIf
  If KsG2p : FreeMemory(KsG2p) : EndIf
  PmKokoroG2pUseExtra(0,0)
  If KsExtra : FreeMemory(KsExtra) : EndIf
  If KsQueueMutex : FreeMutex(KsQueueMutex) : EndIf
  FreeMutex(KsMutex) : CloseWindow(KsWindow)
EndProcedure
