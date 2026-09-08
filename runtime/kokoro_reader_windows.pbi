; Bounded producer/consumer playback for the resident speech application.
; Only the UI thread owns the wave device and prepared headers. The producer
; publishes immutable PCM blocks under a mutex and waits when three are live.
#KS_AUDIO_SLOTS=3
#KS_MAX_PASSAGE=180
#KS_MAX_PCM_BYTES=24000*2*60
#KS_MAX_TEXT=4*1024*1024
Structure KsAudioBlock
  Header.WAVEHDR
  State.i ; 0 free, 1 ready, 2 submitted
  Passage.i
  EndPosition.i
EndStructure
Global Dim KsBlocks.KsAudioBlock(#KS_AUDIO_SLOTS-1)
Global KsQueueMutex.i,KsWave.i,KsQueued.i,KsWriteIndex.i,KsSubmitIndex.i
Global KsPlayed.i,KsPlayedPosition.i,KsFirstAudio.i,KsQueuePeak.i,KsAudioError.s
Global KsTextError.s

Procedure.s KsAudioFailure(Action.s,Code.i)
  Protected detail.s=Space(256)
  waveOutGetErrorText_(Code,@detail,256)
  ProcedureReturn Action+" failed (audio code "+Str(Code)+"): "+Trim(detail)+". Check the Windows audio output device."
EndProcedure

Procedure.i KsAudioOpen()
  Protected format.WAVEFORMATEX,code.i
  KsAudioError=""
  If KsQueueMutex=0 : KsQueueMutex=CreateMutex() : EndIf
  If KsQueueMutex=0 : KsAudioError="Cannot allocate the playback lock. Close other applications and try again." : ProcedureReturn 0 : EndIf
  format\wFormatTag=#WAVE_FORMAT_PCM : format\nChannels=1
  format\nSamplesPerSec=24000 : format\wBitsPerSample=16
  format\nBlockAlign=2 : format\nAvgBytesPerSec=48000
  code=waveOutOpen_(@KsWave,#WAVE_MAPPER,@format,0,0,#CALLBACK_NULL)
  If code : KsAudioError=KsAudioFailure("Opening playback",code) : ProcedureReturn 0 : EndIf
  KsQueued=0 : KsWriteIndex=0 : KsSubmitIndex=0 : KsPlayed=0
  KsPlayedPosition=0 : KsFirstAudio=0 : KsQueuePeak=0
  ProcedureReturn 1
EndProcedure

; Called by the sole producer BEFORE inference or replay allocation. The free
; slot cannot be taken by another producer. Stop also wakes a full/paused queue.
Procedure.i KsAudioWaitSlot()
  Protected free.i
  Repeat
    If DCancel : ProcedureReturn 0 : EndIf
    LockMutex(KsQueueMutex)
    free=Bool(KsBlocks(KsWriteIndex % #KS_AUDIO_SLOTS)\State=0)
    UnlockMutex(KsQueueMutex)
    If free : ProcedureReturn 1 : EndIf
    Delay(10)
  ForEver
EndProcedure

; Ownership transfers only on success. No model tensor address reaches the
; audio driver: request storage may be reset immediately after publishing.
Procedure.i KsAudioPublish(*Pcm,Bytes.i,Passage.i,EndPosition.i)
  Protected slot.i,result.i
  If *Pcm=0 Or Bytes<=0 Or Bytes>#KS_MAX_PCM_BYTES Or Bytes & 1 : ProcedureReturn 0 : EndIf
  LockMutex(KsQueueMutex)
  slot=KsWriteIndex % #KS_AUDIO_SLOTS
  If DCancel=0 And KsBlocks(slot)\State=0
    ClearStructure(@KsBlocks(slot),KsAudioBlock)
    KsBlocks(slot)\Header\lpData=*Pcm
    KsBlocks(slot)\Header\dwBufferLength=Bytes
    KsBlocks(slot)\Passage=Passage : KsBlocks(slot)\EndPosition=EndPosition
    KsBlocks(slot)\State=1 : KsWriteIndex+1 : KsQueued+1
    If KsQueued>KsQueuePeak : KsQueuePeak=KsQueued : EndIf
    result=1
  EndIf
  UnlockMutex(KsQueueMutex)
  ProcedureReturn result
EndProcedure

; Poll on the UI timer. Pre-submit all available blocks so audio boundaries
; are scheduled by the device, not by the next timer tick.
Procedure.i KsAudioPump()
  Protected slot.i,code.i
  If KsWave=0 : ProcedureReturn 1 : EndIf
  LockMutex(KsQueueMutex)
  For slot=0 To #KS_AUDIO_SLOTS-1
    If KsBlocks(slot)\State=2 And KsBlocks(slot)\Header\dwFlags & #WHDR_DONE
      code=waveOutUnprepareHeader_(KsWave,@KsBlocks(slot)\Header,SizeOf(WAVEHDR))
      If code
        KsAudioError=KsAudioFailure("Releasing an audio block",code)
        UnlockMutex(KsQueueMutex) : ProcedureReturn 0
      EndIf
      If DCancel=0 And KsBlocks(slot)\Passage>KsPlayed
        KsPlayed=KsBlocks(slot)\Passage : KsPlayedPosition=KsBlocks(slot)\EndPosition
      EndIf
      FreeMemory(KsBlocks(slot)\Header\lpData)
      ClearStructure(@KsBlocks(slot),KsAudioBlock) : KsQueued-1
    EndIf
  Next
  While DCancel=0 And KsSubmitIndex<KsWriteIndex
    slot=KsSubmitIndex % #KS_AUDIO_SLOTS
    If KsBlocks(slot)\State<>1 : Break : EndIf
    code=waveOutPrepareHeader_(KsWave,@KsBlocks(slot)\Header,SizeOf(WAVEHDR))
    If code=0 : code=waveOutWrite_(KsWave,@KsBlocks(slot)\Header,SizeOf(WAVEHDR)) : EndIf
    If code
      KsAudioError=KsAudioFailure("Queuing playback",code)
      UnlockMutex(KsQueueMutex) : ProcedureReturn 0
    EndIf
    KsBlocks(slot)\State=2 : KsSubmitIndex+1
    If KsFirstAudio=0 : KsFirstAudio=ElapsedMilliseconds() : EndIf
  Wend
  UnlockMutex(KsQueueMutex)
  ProcedureReturn 1
EndProcedure

Procedure.i KsAudioPause(Paused.i)
  Protected code.i
  If Paused : code=waveOutPause_(KsWave) : Else : code=waveOutRestart_(KsWave) : EndIf
  If code : KsAudioError=KsAudioFailure("Changing playback pause",code) : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure KsAudioStop()
  Protected code.i
  DCancel=1
  If KsWave
    code=waveOutReset_(KsWave)
    If code : KsAudioError=KsAudioFailure("Stopping playback",code) : EndIf
  EndIf
EndProcedure

; Call only after the producer has exited. Keep buffers owned if the device
; refuses reset/unprepare; never free memory still referenced by a driver.
Procedure.i KsAudioClose()
  Protected slot.i,code.i
  If KsWave=0 : ProcedureReturn 1 : EndIf
  code=waveOutReset_(KsWave)
  If code : KsAudioError=KsAudioFailure("Resetting playback",code) : ProcedureReturn 0 : EndIf
  For slot=0 To #KS_AUDIO_SLOTS-1
    If KsBlocks(slot)\Header\dwFlags & #WHDR_PREPARED
      code=waveOutUnprepareHeader_(KsWave,@KsBlocks(slot)\Header,SizeOf(WAVEHDR))
      If code : KsAudioError=KsAudioFailure("Releasing playback",code) : ProcedureReturn 0 : EndIf
    EndIf
    If KsBlocks(slot)\Header\lpData : FreeMemory(KsBlocks(slot)\Header\lpData) : EndIf
    ClearStructure(@KsBlocks(slot),KsAudioBlock)
  Next
  code=waveOutClose_(KsWave)
  If code : KsAudioError=KsAudioFailure("Closing playback",code) : ProcedureReturn 0 : EndIf
  KsWave=0 : KsQueued=0
  ProcedureReturn 1
EndProcedure

Procedure.s KsBookCharacter(Code.i)
  Select Code
    Case 9,10,13,160 : ProcedureReturn " "
    Case $2018,$2019 : ProcedureReturn "'"
    Case $201C,$201D : ProcedureReturn Chr(34)
    Case $2013,$2014 : ProcedureReturn "-"
    Case $2026 : ProcedureReturn "."
  EndSelect
  ProcedureReturn Chr(Code)
EndProcedure

; A cursor advances through UTF-16 text once. At most one short passage is
; rescanned at a word boundary; no word list or per-word scan of a whole book.
Procedure.s KsNextPassage(*Text,length.i,*Cursor.Integer)
  Protected pos.i=*Cursor\i,lastSpace.i,lastLength.i,code.i
  Protected chunk.s,ch.s,word.s,nextCode.i
  KsTextError=""
  While pos<length
    code=PeekU(*Text+pos*2) : ch=KsBookCharacter(code) : pos+1
    If ch=" "
      If chunk="" : Continue : EndIf
      lastSpace=pos : lastLength=Len(chunk)
      If Right(chunk,1)<>" " : chunk+" " : EndIf
    Else
      chunk+ch
    EndIf
    If ch="." Or ch="!" Or ch="?"
      nextCode=0 : If pos<length : nextCode=PeekU(*Text+pos*2) : EndIf
      word=LCase(StringField(chunk,CountString(chunk," ")+1," "))
      If FindString("|mr.|mrs.|ms.|dr.|prof.|st.|vs.|e.g.|i.e.|","|"+word+"|")=0 And (nextCode=0 Or nextCode<=32 Or nextCode=34 Or nextCode=$201D Or nextCode=39 Or nextCode=$2019)
        While pos<length And (nextCode=34 Or nextCode=$201D Or nextCode=39 Or nextCode=$2019)
          chunk+KsBookCharacter(nextCode) : pos+1
          nextCode=0 : If pos<length : nextCode=PeekU(*Text+pos*2) : EndIf
        Wend
        Break
      EndIf
    EndIf
    If Len(chunk)>=#KS_MAX_PASSAGE
      If lastSpace
        pos=lastSpace : chunk=Left(chunk,lastLength)
      ElseIf pos<length And PeekU(*Text+pos*2)>32
        KsTextError="A word exceeds the passage limit. Add a word boundary or correct the text."
        *Cursor\i=pos : ProcedureReturn ""
      EndIf
      Break
    EndIf
  Wend
  *Cursor\i=pos
  ProcedureReturn Trim(chunk)
EndProcedure
