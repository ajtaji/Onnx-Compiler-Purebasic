; ============================================================================
; onnx_ui.pbi - native desktop front end for the native ONNX compiler
; ----------------------------------------------------------------------------
; The UI launches the sibling native CLI build of the SAME source. It does not
; invoke Python and contains no graph/compiler logic of its own.
; ============================================================================

Enumeration PmoUiGadget
  #PMO_UI_MODEL
  #PMO_UI_MODEL_BROWSE
  #PMO_UI_OUTPUT
  #PMO_UI_OUTPUT_BROWSE
  #PMO_UI_TARGET
  #PMO_UI_PRECISION
  #PMO_UI_ORT
  #PMO_UI_ORT_BROWSE
  #PMO_UI_SIMPLIFY
  #PMO_UI_SHAPES
  #PMO_UI_TRACES
  #PMO_UI_CHECKPOINTS
  #PMO_UI_COMPILE
  #PMO_UI_CANCEL
  #PMO_UI_OPEN_FOLDER
  #PMO_UI_KOKORO
  #PMO_UI_PROGRESS
  #PMO_UI_LOG
  #PMO_UI_REUSABLE
  #PMO_UI_RANDOM_INPUTS
EndEnumeration

#PMO_UI_DONE = #PB_Event_FirstCustomValue + 171

Structure PmoUiCompileJob
  Arguments.s
  OutputPrefix.s
  TargetIndex.i
  ExitCode.i
  Complete.i
EndStructure

Global PmoUiWindow.i
Global PmoUiThread.i
Global PmoUiProgram.i
Global PmoUiCancel.i
Global PmoUiMutex.i
Global PmoUiPendingLog.s
Global PmoUiJob.PmoUiCompileJob
Global PmoUiKokoroG2p.s
Global PmoUiKokoroVoice.s
Global PmoUiKokoroText.s = "Hello from PureMetal."
Global PmoUiKokoroSpeed.s = "1.0"

Procedure.s PmoUiQuote(Value.s)
  If FindString(Value, Chr(34))
    ProcedureReturn ""
  EndIf
  ProcedureReturn Chr(34) + Value + Chr(34)
EndProcedure

Procedure PmoUiAppendLog(Text.s)
  LockMutex(PmoUiMutex)
  PmoUiPendingLog + Text
  UnlockMutex(PmoUiMutex)
EndProcedure

Procedure PmoUiDrainLog()
  Protected Text.s
  LockMutex(PmoUiMutex)
  Text = PmoUiPendingLog
  PmoUiPendingLog = ""
  UnlockMutex(PmoUiMutex)
  If Text <> ""
    AddGadgetItem(#PMO_UI_LOG, -1, Text)
  EndIf
EndProcedure

Procedure.s PmoUiAddRepeated(Arguments.s, OptionName.s, Text.s)
  Protected Index.i
  Protected Line.s
  Text = ReplaceString(Text, Chr(13), "")
  For Index = 1 To CountString(Text, Chr(10)) + 1
    Line = Trim(StringField(Text, Index, Chr(10)))
    If Line <> ""
      If FindString(Line, Chr(34))
        ProcedureReturn ""
      EndIf
      Arguments + " " + OptionName + " " + PmoUiQuote(Line)
    EndIf
  Next
  ProcedureReturn Arguments
EndProcedure

Procedure.s PmoUiCompilerCli()
  ProcedureReturn GetPathPart(ProgramFilename()) + "PureMetalOnnxCompilerCLI.exe"
EndProcedure

Procedure.s PmoUiFindKokoroG2p()
  Protected Candidate.s = GetPathPart(ProgramFilename())
  Protected Index.i
  For Index = 0 To 8
    If FileSize(Candidate + "Kokoro\Assets\kokoro_us_english.pmg2p") > 0
      ProcedureReturn Candidate + "Kokoro\Assets\kokoro_us_english.pmg2p"
    EndIf
    Candidate = GetPathPart(RTrim(Candidate, "\/"))
    If Candidate = "" : Break : EndIf
  Next
  ProcedureReturn ""
EndProcedure

Procedure.i PmoUiRunCli(Arguments.s, *Text.String)
  Protected Cli.s = PmoUiCompilerCli()
  Protected Process.i
  Protected Line.s
  If *Text = 0 Or FileSize(Cli) <= 0 : ProcedureReturn 2 : EndIf
  *Text\s = ""
  Process = RunProgram(Cli, Arguments, GetPathPart(Cli),
                       #PB_Program_Open | #PB_Program_Read | #PB_Program_Error | #PB_Program_Hide)
  If Process = 0 : ProcedureReturn 2 : EndIf
  While ProgramRunning(Process)
    While AvailableProgramOutput(Process)
      *Text\s + ReadProgramString(Process) + Chr(10)
    Wend
    Line = ReadProgramError(Process)
    If Line <> "" : *Text\s + Line + Chr(10) : EndIf
    Delay(15)
  Wend
  While AvailableProgramOutput(Process) : *Text\s + ReadProgramString(Process) + Chr(10) : Wend
  Repeat
    Line = ReadProgramError(Process)
    If Line = "" : Break : EndIf
    *Text\s + Line + Chr(10)
  ForEver
  Protected ExitCode.i = ProgramExitCode(Process)
  CloseProgram(Process)
  ProcedureReturn ExitCode
EndProcedure

Procedure PmoUiSetRunning(Running.i)
  DisableGadget(#PMO_UI_MODEL, Running)
  DisableGadget(#PMO_UI_MODEL_BROWSE, Running)
  DisableGadget(#PMO_UI_OUTPUT, Running)
  DisableGadget(#PMO_UI_OUTPUT_BROWSE, Running)
  DisableGadget(#PMO_UI_TARGET, Running)
  DisableGadget(#PMO_UI_PRECISION, Running)
  DisableGadget(#PMO_UI_ORT, Running)
  DisableGadget(#PMO_UI_ORT_BROWSE, Running)
  DisableGadget(#PMO_UI_SIMPLIFY, Running)
  DisableGadget(#PMO_UI_REUSABLE, Running)
  DisableGadget(#PMO_UI_RANDOM_INPUTS, Running)
  DisableGadget(#PMO_UI_SHAPES, Running)
  DisableGadget(#PMO_UI_TRACES, Running)
  DisableGadget(#PMO_UI_CHECKPOINTS, Running)
  DisableGadget(#PMO_UI_COMPILE, Running)
  DisableGadget(#PMO_UI_KOKORO, Running)
  DisableGadget(#PMO_UI_CANCEL, 1 - Running)
EndProcedure

Procedure PmoUiPrepareKokoro()
  Protected Model.s = Trim(GetGadgetText(#PMO_UI_MODEL))
  Protected Output.s = Trim(GetGadgetText(#PMO_UI_OUTPUT))
  Protected Text.s
  Protected SpeedText.s
  Protected Speed.f
  Protected DefaultPath.s
  Protected Arguments.s
  Protected Result.s
  Protected ExitCode.i
  Protected IdPath.s
  Protected StylePath.s
  Protected SpeedPath.s
  If FileSize(Model) <= 0
    MessageRequester("Kokoro inputs", "Choose the pinned full Kokoro ONNX model first.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If Output = "" Or GetPathPart(Output) = "" Or FileSize(GetPathPart(Output)) <> -2
    MessageRequester("Kokoro inputs", "Choose an output prefix in an existing folder first.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  Text = InputRequester("Kokoro text", "Text to speak (or /phonemes/ for an explicit phoneme sequence):", PmoUiKokoroText)
  If Text = "" : ProcedureReturn : EndIf
  PmoUiKokoroText = Text
  If PmoUiKokoroG2p = ""
    DefaultPath = PmoUiFindKokoroG2p()
  Else
    DefaultPath = PmoUiKokoroG2p
  EndIf
  PmoUiKokoroG2p = OpenFileRequester("Choose the Kokoro pronunciation pack", DefaultPath,
                                    "PureMetal G2P (*.pmg2p)|*.pmg2p|All files (*.*)|*.*", 0)
  If PmoUiKokoroG2p = "" : ProcedureReturn : EndIf
  If PmoUiKokoroVoice = "" : PmoUiKokoroVoice = GetPathPart(Model) + "af_heart.pmvoice" : EndIf
  PmoUiKokoroVoice = OpenFileRequester("Choose a Kokoro voice", PmoUiKokoroVoice,
                                      "PureMetal voices (*.pmvoice)|*.pmvoice|All files (*.*)|*.*", 0)
  If PmoUiKokoroVoice = "" : ProcedureReturn : EndIf
  SpeedText = InputRequester("Kokoro speed", "Positive playback speed (1.0 is normal):", PmoUiKokoroSpeed)
  If SpeedText = "" : ProcedureReturn : EndIf
  Speed = ValF(SpeedText)
  If Speed <= 0.0
    MessageRequester("Kokoro inputs", "Speed must be a positive number.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If FindString(Model + Output + Text + PmoUiKokoroG2p + PmoUiKokoroVoice + SpeedText, Chr(34))
    MessageRequester("Kokoro inputs", "Text and filenames cannot contain a quote character.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  Arguments = "--kokoro-prepare " + PmoUiQuote(Model) +
              " --text " + PmoUiQuote(Text) +
              " --g2p " + PmoUiQuote(PmoUiKokoroG2p) +
              " --voice " + PmoUiQuote(PmoUiKokoroVoice) +
              " --speed " + PmoUiQuote(SpeedText) +
              " --output " + PmoUiQuote(Output)
  PmoUiSetRunning(#True)
  SetGadgetState(#PMO_UI_PROGRESS, 15)
  ExitCode = PmoUiRunCli(Arguments, @Result)
  PmoUiSetRunning(#False)
  If ExitCode <> 0
    SetGadgetState(#PMO_UI_PROGRESS, 0)
    If Result = "" : Result = "The native compiler engine could not prepare the inputs." : EndIf
    MessageRequester("Kokoro inputs", Result, #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  PmoUiKokoroSpeed = SpeedText
  IdPath = Output + ".trace-input_ids.npy"
  StylePath = Output + ".trace-style.npy"
  SpeedPath = Output + ".trace-speed.npy"
  SetGadgetText(#PMO_UI_TRACES,
                "input_ids=" + IdPath + Chr(10) +
                "style=" + StylePath + Chr(10) +
                "speed=" + SpeedPath)
  AddGadgetItem(#PMO_UI_LOG, -1, Result)
  SetGadgetState(#PMO_UI_PROGRESS, 25)
EndProcedure

Procedure PmoUiCompileWorker(*Unused)
  Protected Cli.s = PmoUiCompilerCli()
  Protected Line.s
  If FileSize(Cli) <= 0
    PmoUiAppendLog("ERROR: native compiler engine is missing: " + Cli + Chr(10))
    PmoUiJob\ExitCode = 2
    PmoUiJob\Complete = #True
    PostEvent(#PMO_UI_DONE)
    ProcedureReturn
  EndIf
  PmoUiProgram = RunProgram(Cli, PmoUiJob\Arguments, GetPathPart(Cli),
                            #PB_Program_Open | #PB_Program_Read | #PB_Program_Error | #PB_Program_Hide)
  If PmoUiProgram = 0
    PmoUiAppendLog("ERROR: Windows could not start the native compiler engine." + Chr(10))
    PmoUiJob\ExitCode = 2
    PmoUiJob\Complete = #True
    PostEvent(#PMO_UI_DONE)
    ProcedureReturn
  EndIf
  While ProgramRunning(PmoUiProgram)
    While AvailableProgramOutput(PmoUiProgram)
      Line = ReadProgramString(PmoUiProgram)
      PmoUiAppendLog(Line + Chr(10))
    Wend
    Line = ReadProgramError(PmoUiProgram)
    If Line <> ""
      PmoUiAppendLog(Line + Chr(10))
    EndIf
    Delay(15)
  Wend
  While AvailableProgramOutput(PmoUiProgram)
    PmoUiAppendLog(ReadProgramString(PmoUiProgram) + Chr(10))
  Wend
  Repeat
    Line = ReadProgramError(PmoUiProgram)
    If Line = "" : Break : EndIf
    PmoUiAppendLog(Line + Chr(10))
  ForEver
  PmoUiJob\ExitCode = ProgramExitCode(PmoUiProgram)
  CloseProgram(PmoUiProgram)
  PmoUiProgram = 0
  If PmoUiCancel
    PmoUiJob\ExitCode = 130
    PmoUiAppendLog("Compilation cancelled." + Chr(10))
  EndIf
  PmoUiJob\Complete = #True
  PostEvent(#PMO_UI_DONE)
EndProcedure

Procedure PmoUiStartCompile()
  Protected Model.s = Trim(GetGadgetText(#PMO_UI_MODEL))
  Protected Output.s = Trim(GetGadgetText(#PMO_UI_OUTPUT))
  Protected Arguments.s
  Protected TargetIndex.i = GetGadgetState(#PMO_UI_TARGET)
  Protected Precision.s = LCase(GetGadgetText(#PMO_UI_PRECISION))
  Protected OrtPath.s = Trim(GetGadgetText(#PMO_UI_ORT))
  If FileSize(Model) <= 0
    MessageRequester("PureMetal ONNX Compiler", "Choose an ONNX model first.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If Output = "" Or GetPathPart(Output) = ""
    MessageRequester("PureMetal ONNX Compiler", "Choose an output prefix with a folder.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If TargetIndex < 0 Or TargetIndex >= #PMO_TARGET_COUNT
    MessageRequester("PureMetal ONNX Compiler", "Choose a target.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If GetGadgetState(#PMO_UI_REUSABLE)
    If TargetIndex<>#PMO_TARGET_WINDOWS And TargetIndex<>#PMO_TARGET_PI4 And TargetIndex<>#PMO_TARGET_UNOQ
      MessageRequester("Kokoro text adapter", "Kokoro requires Windows, Pi 4 or UNO Q RAM. Other models can still target Pico or Pico 2 with this option unchecked.", #PB_MessageRequester_Error)
      ProcedureReturn
    EndIf
  EndIf
  If Precision <> "fp32"
    If MessageRequester("Reduced precision",
                        "Reduced precision is lossy and can change model quality. INT8 uses integer kernels where supported, but is not guaranteed faster. FP16/BF16/INT4 reduce weight storage and decode once into resident FP32 RAM; they do not reduce activation memory. Continue?",
                        #PB_MessageRequester_YesNo | #PB_MessageRequester_Warning) <> #PB_MessageRequester_Yes
      ProcedureReturn
    EndIf
  EndIf
  If FindString(Model, Chr(34)) Or FindString(Output, Chr(34))
    MessageRequester("PureMetal ONNX Compiler", "A filename cannot contain a quote character.", #PB_MessageRequester_Error)
    ProcedureReturn
  EndIf
  If GetGadgetState(#PMO_UI_REUSABLE)
    Arguments="--compile "+PmoUiQuote(Model)+" --output "+PmoUiQuote(Output)+" --target "+PmoTargets(TargetIndex)\Id+" --precision "+Precision
    If TargetIndex=#PMO_TARGET_WINDOWS
      Arguments+" --speech-ui"
    Else
      Arguments+" --kokoro-text"
    EndIf
  Else
  Arguments = "--compile " + PmoUiQuote(Model) + " --output " + PmoUiQuote(Output)
  Arguments + " --target " + PmoTargets(TargetIndex)\Id
  Arguments + " --precision " + Precision
  If OrtPath <> ""
    If FileSize(OrtPath) <= 0 Or FindString(OrtPath, Chr(34))
      MessageRequester("PureMetal ONNX Compiler", "The ONNX Runtime library path is not a file.", #PB_MessageRequester_Error)
      ProcedureReturn
    EndIf
    Arguments + " --ort-dll " + PmoUiQuote(OrtPath)
  EndIf
  If GetGadgetState(#PMO_UI_SIMPLIFY) : Arguments + " --simplify" : EndIf
  Arguments = PmoUiAddRepeated(Arguments, "--shape", GetGadgetText(#PMO_UI_SHAPES))
  If Arguments = "" : MessageRequester("PureMetal ONNX Compiler", "A shape contains a quote character.", #PB_MessageRequester_Error) : ProcedureReturn : EndIf
  Arguments = PmoUiAddRepeated(Arguments, "--trace-input", GetGadgetText(#PMO_UI_TRACES))
  If Arguments = "" : MessageRequester("PureMetal ONNX Compiler", "A trace input contains a quote character.", #PB_MessageRequester_Error) : ProcedureReturn : EndIf
  Arguments = PmoUiAddRepeated(Arguments, "--checkpoint", GetGadgetText(#PMO_UI_CHECKPOINTS))
  If Arguments = "" : MessageRequester("PureMetal ONNX Compiler", "A checkpoint contains a quote character.", #PB_MessageRequester_Error) : ProcedureReturn : EndIf
  EndIf
  ; Applies to either form; the compiler refuses it for a model with no random node.
  If GetGadgetState(#PMO_UI_RANDOM_INPUTS) : Arguments + " --random-inputs" : EndIf

  PmoUiJob\Arguments = Arguments
  PmoUiJob\OutputPrefix = Output
  PmoUiJob\TargetIndex = TargetIndex
  PmoUiJob\ExitCode = -1
  PmoUiJob\Complete = #False
  PmoUiCancel = #False
  SetGadgetText(#PMO_UI_LOG, "")
  AddGadgetItem(#PMO_UI_LOG, -1, "Generating source for " + PmoTargets(TargetIndex)\Label + "..." + Chr(10))
  SetGadgetState(#PMO_UI_PROGRESS, 35)
  DisableGadget(#PMO_UI_OPEN_FOLDER, #True)
  PmoUiSetRunning(#True)
  PmoUiThread = CreateThread(@PmoUiCompileWorker(), 0)
  If PmoUiThread = 0
    PmoUiSetRunning(#False)
    SetGadgetState(#PMO_UI_PROGRESS, 0)
    MessageRequester("PureMetal ONNX Compiler", "Could not create the compiler worker thread.", #PB_MessageRequester_Error)
  EndIf
EndProcedure

Procedure PmoUiResize()
  Protected Width.i = WindowWidth(PmoUiWindow)
  Protected Height.i = WindowHeight(PmoUiWindow)
  Protected LogTop.i = 462
  ResizeGadget(#PMO_UI_MODEL, #PB_Ignore, #PB_Ignore, Width - 190, #PB_Ignore)
  ResizeGadget(#PMO_UI_MODEL_BROWSE, Width - 118, #PB_Ignore, #PB_Ignore, #PB_Ignore)
  ResizeGadget(#PMO_UI_OUTPUT, #PB_Ignore, #PB_Ignore, Width - 190, #PB_Ignore)
  ResizeGadget(#PMO_UI_OUTPUT_BROWSE, Width - 118, #PB_Ignore, #PB_Ignore, #PB_Ignore)
  ResizeGadget(#PMO_UI_ORT, #PB_Ignore, #PB_Ignore, Width - 190, #PB_Ignore)
  ResizeGadget(#PMO_UI_ORT_BROWSE, Width - 118, #PB_Ignore, #PB_Ignore, #PB_Ignore)
  ResizeGadget(#PMO_UI_SHAPES, #PB_Ignore, #PB_Ignore, (Width - 64) / 3, #PB_Ignore)
  ResizeGadget(#PMO_UI_TRACES, 24 + (Width - 64) / 3 + 8, #PB_Ignore, (Width - 64) / 3, #PB_Ignore)
  ResizeGadget(#PMO_UI_CHECKPOINTS, 24 + 2 * ((Width - 64) / 3 + 8), #PB_Ignore, (Width - 64) / 3, #PB_Ignore)
  ResizeGadget(#PMO_UI_PROGRESS, #PB_Ignore, #PB_Ignore, Width - 48, #PB_Ignore)
  ResizeGadget(#PMO_UI_LOG, #PB_Ignore, LogTop, Width - 48, Height - LogTop - 24)
EndProcedure

Procedure PmoRunUi()
  Protected Event.i
  Protected Gadget.i
  Protected Path.s
  Protected Index.i
  PmoInitTargets()
  PmoUiMutex = CreateMutex()
  PmoUiWindow = OpenWindow(#PB_Any, 0, 0, 1040, 820, "PureMetal ONNX Compiler",
                           #PB_Window_SystemMenu | #PB_Window_SizeGadget |
                           #PB_Window_MinimizeGadget | #PB_Window_MaximizeGadget |
                           #PB_Window_ScreenCentered)
  If PmoUiWindow = 0 : ProcedureReturn : EndIf
  WindowBounds(PmoUiWindow, 820, 720, #PB_Ignore, #PB_Ignore)

  TextGadget(#PB_Any, 24, 20, 120, 24, "ONNX model")
  StringGadget(#PMO_UI_MODEL, 24, 44, 850, 28, "")
  ButtonGadget(#PMO_UI_MODEL_BROWSE, 922, 44, 94, 28, "Browse...")
  TextGadget(#PB_Any, 24, 82, 120, 24, "Output prefix")
  StringGadget(#PMO_UI_OUTPUT, 24, 106, 850, 28, "")
  ButtonGadget(#PMO_UI_OUTPUT_BROWSE, 922, 106, 94, 28, "Browse...")
  TextGadget(#PB_Any, 24, 144, 120, 24, "Target")
  ComboBoxGadget(#PMO_UI_TARGET, 24, 168, 390, 30)
  For Index = 0 To #PMO_TARGET_COUNT - 1
    AddGadgetItem(#PMO_UI_TARGET, -1, PmoTargets(Index)\Label)
  Next
  SetGadgetState(#PMO_UI_TARGET, #PMO_TARGET_WINDOWS)
  TextGadget(#PB_Any, 442, 144, 120, 24, "Precision")
  ComboBoxGadget(#PMO_UI_PRECISION, 442, 168, 170, 30)
  AddGadgetItem(#PMO_UI_PRECISION, -1, "FP32")
  AddGadgetItem(#PMO_UI_PRECISION, -1, "INT8")
  AddGadgetItem(#PMO_UI_PRECISION, -1, "FP16")
  AddGadgetItem(#PMO_UI_PRECISION, -1, "BF16")
  AddGadgetItem(#PMO_UI_PRECISION, -1, "INT4")
  SetGadgetState(#PMO_UI_PRECISION, 0)
  CheckBoxGadget(#PMO_UI_SIMPLIFY, 636, 171, 118, 26, "Simplify")
  CheckBoxGadget(#PMO_UI_RANDOM_INPUTS, 636, 204, 380, 24, "Random nodes as model inputs (comparison)")
  GadgetToolTip(#PMO_UI_RANDOM_INPUTS, "Compile every RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Bernoulli and Multinomial node as an extra model input that the caller fills, so a run can be compared sample for sample with a reference's noise. This changes the model's inputs; leave it off for deployment.")
  TextGadget(#PB_Any, 770, 174, 242, 24, "Resident model (always)")

  TextGadget(#PB_Any, 24, 208, 280, 24, "ONNX Runtime library (fixed-extent tracing only)")
  StringGadget(#PMO_UI_ORT, 24, 232, 850, 28, "")
  ButtonGadget(#PMO_UI_ORT_BROWSE, 922, 232, 94, 28, "Browse...")

  TextGadget(#PB_Any, 24, 274, 300, 24, "Shapes - one NAME=DIM,DIM per line")
  TextGadget(#PB_Any, 357, 274, 300, 24, "Trace inputs - one NAME=file.npy per line")
  TextGadget(#PB_Any, 690, 274, 300, 24, "Checkpoints - one tensor name per line")
  EditorGadget(#PMO_UI_SHAPES, 24, 298, 317, 92)
  EditorGadget(#PMO_UI_TRACES, 349, 298, 317, 92)
  EditorGadget(#PMO_UI_CHECKPOINTS, 674, 298, 342, 92)

  ButtonGadget(#PMO_UI_COMPILE, 24, 406, 132, 34, "Generate source")
  ButtonGadget(#PMO_UI_CANCEL, 164, 406, 110, 34, "Cancel")
  ButtonGadget(#PMO_UI_OPEN_FOLDER, 282, 406, 142, 34, "Open output folder")
  ButtonGadget(#PMO_UI_KOKORO, 432, 406, 184, 34, "Prepare Kokoro inputs")
  CheckBoxGadget(#PMO_UI_REUSABLE, 632, 410, 230, 26, "Add Kokoro text adapter")
  GadgetToolTip(#PMO_UI_REUSABLE,"Windows: speech window. Pi 4 / UNO Q: resident text-to-audio API with caller-owned dictionaries and voice. Full Kokoro cannot fit Pico RAM. All models are already reusable without this option.")
  DisableGadget(#PMO_UI_CANCEL, #True)
  DisableGadget(#PMO_UI_OPEN_FOLDER, #True)
  ProgressBarGadget(#PMO_UI_PROGRESS, 24, 446, 992, 8, 0, 100)
  EditorGadget(#PMO_UI_LOG, 24, 462, 992, 334, #PB_Editor_ReadOnly)
  AddWindowTimer(PmoUiWindow, 1, 100)
  PmoUiResize()

  Repeat
    Event = WaitWindowEvent()
    Select Event
      Case #PB_Event_SizeWindow
        PmoUiResize()
      Case #PB_Event_Timer
        PmoUiDrainLog()
      Case #PMO_UI_DONE
        PmoUiDrainLog()
        PmoUiSetRunning(#False)
        If PmoUiJob\ExitCode = 0
          SetGadgetState(#PMO_UI_PROGRESS, 100)
          DisableGadget(#PMO_UI_OPEN_FOLDER, #False)
        Else
          SetGadgetState(#PMO_UI_PROGRESS, 0)
        EndIf
        PmoUiThread = 0
      Case #PB_Event_Gadget
        Gadget = EventGadget()
        Select Gadget
          Case #PMO_UI_MODEL_BROWSE
            Path = OpenFileRequester("Choose an ONNX model", GetGadgetText(#PMO_UI_MODEL),
                                     "ONNX models (*.onnx)|*.onnx|All files (*.*)|*.*", 0)
            If Path <> ""
              SetGadgetText(#PMO_UI_MODEL, Path)
              If GetGadgetText(#PMO_UI_OUTPUT) = ""
                SetGadgetText(#PMO_UI_OUTPUT, GetPathPart(Path) + GetFilePart(Path, #PB_FileSystem_NoExtension))
              EndIf
            EndIf
          Case #PMO_UI_OUTPUT_BROWSE
            Path = SaveFileRequester("Choose the output prefix", GetGadgetText(#PMO_UI_OUTPUT),
                                     "Output prefix (*.*)|*.*", 0)
            If Path <> "" : SetGadgetText(#PMO_UI_OUTPUT, Path) : EndIf
          Case #PMO_UI_ORT_BROWSE
            Path = OpenFileRequester("Choose ONNX Runtime", GetGadgetText(#PMO_UI_ORT),
                                     "ONNX Runtime (onnxruntime.dll)|onnxruntime.dll|Libraries (*.dll)|*.dll|All files (*.*)|*.*", 0)
            If Path <> "" : SetGadgetText(#PMO_UI_ORT, Path) : EndIf
          Case #PMO_UI_COMPILE
            PmoUiStartCompile()
          Case #PMO_UI_KOKORO
            PmoUiPrepareKokoro()
          Case #PMO_UI_CANCEL
            PmoUiCancel = #True
            If PmoUiProgram : KillProgram(PmoUiProgram) : EndIf
            DisableGadget(#PMO_UI_CANCEL, #True)
          Case #PMO_UI_OPEN_FOLDER
            RunProgram("explorer.exe", PmoUiQuote(GetPathPart(PmoUiJob\OutputPrefix)), "")
        EndSelect
      Case #PB_Event_CloseWindow
        If PmoUiThread
          If MessageRequester("PureMetal ONNX Compiler", "Cancel the active compilation and close?",
                              #PB_MessageRequester_YesNo | #PB_MessageRequester_Warning) = #PB_MessageRequester_Yes
            PmoUiCancel = #True
            If PmoUiProgram : KillProgram(PmoUiProgram) : EndIf
            While PmoUiThread And IsThread(PmoUiThread) : WaitThread(PmoUiThread, 50) : Wend
            Break
          EndIf
        Else
          Break
        EndIf
    EndSelect
  ForEver
  RemoveWindowTimer(PmoUiWindow, 1)
  If PmoUiMutex : FreeMutex(PmoUiMutex) : EndIf
EndProcedure
