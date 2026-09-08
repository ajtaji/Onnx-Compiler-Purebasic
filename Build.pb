; Developer build helper. This is NOT part of the ONNX source generator.
; Open this file in the Windows x64 PureBasic IDE and Run (F5).
EnableExplicit
CompilerIf #PB_Compiler_OS <> #PB_OS_Windows
  CompilerError "This build helper is verified for Windows x64 PureBasic. See docs/BUILD.md."
CompilerEndIf
CompilerIf #PB_Compiler_Processor <> #PB_Processor_x64
  CompilerError "Select the x64 PureBasic compiler."
CompilerEndIf
OpenConsole()
Define root.s=GetPathPart(#PB_Compiler_File)
Define compiler.s=#PB_Compiler_Home+"Compilers\pbcompiler.exe"
Define process.i,index.i,arguments.s,line.s,code.i
Define source.s,output.s,flags.s
Define deployment.s,releaseHost.s,settings.s=root+"build.local.ini"
If FileSize(compiler)<=0
  PrintN("Cannot find the PureBasic compiler: "+compiler)
  End 1
EndIf
; Optional private release integration. Never publish machine-specific paths.
; This developer hook builds/deploys the tools, not generated model programs.
If FileSize(settings)>=0
  If OpenPreferences(settings)=0 : PrintN("Cannot read build.local.ini.") : End 1 : EndIf
  PreferenceGroup("Build")
  deployment=ReadPreferenceString("ReleaseScript","")
  releaseHost=ReadPreferenceString("ReleaseHost",GetEnvironmentVariable("SystemRoot")+"\System32\WindowsPowerShell\v1.0\powershell.exe")
  ClosePreferences()
  If deployment="" Or FileSize(deployment)<=0 Or FileSize(releaseHost)<=0
    PrintN("build.local.ini must name an existing ReleaseScript and PowerShell ReleaseHost. No build was deployed.")
    End 1
  EndIf
  arguments="-NoProfile -NonInteractive -File "+Chr(34)+deployment+Chr(34)+" -SourceRoot "+Chr(34)+RTrim(root,"\/")+Chr(34)+" -Compiler "+Chr(34)+compiler+Chr(34)
  process=RunProgram(releaseHost,arguments,root,#PB_Program_Open|#PB_Program_Read|#PB_Program_Error|#PB_Program_Hide)
  If process=0 : PrintN("Cannot start the configured release build.") : End 1 : EndIf
  While ProgramRunning(process)
    While AvailableProgramOutput(process) : PrintN(ReadProgramString(process)) : Wend
    line=ReadProgramError(process) : If line<>"" : PrintN(line) : EndIf
    Delay(10)
  Wend
  While AvailableProgramOutput(process) : PrintN(ReadProgramString(process)) : Wend
  Repeat
    line=ReadProgramError(process) : If line="" : Break : EndIf
    PrintN(line)
  ForEver
  code=ProgramExitCode(process) : CloseProgram(process)
  End code
EndIf
If FileSize(root+"bin")<>-2 And CreateDirectory(root+"bin")=0
  PrintN("Cannot create the bin directory.")
  End 1
EndIf
For index=0 To 1
  If index=0
    source="src\PureMetalOnnxCompiler.pb"
    output="bin\PureMetalOnnxCompilerCLI.exe"
    flags=" /CONSOLE /THREAD /OPTIMIZER"
  Else
    source="src\PureMetalOnnxCompilerUI.pb"
    output="bin\PureMetalOnnxCompiler.exe"
    flags=" /THREAD /OPTIMIZER /DPIAWARE"
  EndIf
  PrintN("Building "+output)
  arguments=Chr(34)+root+source+Chr(34)+flags+" /OUTPUT "+Chr(34)+root+output+Chr(34)
  process=RunProgram(compiler,arguments,root,#PB_Program_Open|#PB_Program_Read|#PB_Program_Error|#PB_Program_Hide)
  If process=0 : PrintN("Cannot start the build compiler.") : End 1 : EndIf
  While ProgramRunning(process)
    While AvailableProgramOutput(process) : PrintN(ReadProgramString(process)) : Wend
    line=ReadProgramError(process)
    If line<>"" : PrintN(line) : EndIf
    Delay(10)
  Wend
  While AvailableProgramOutput(process) : PrintN(ReadProgramString(process)) : Wend
  Repeat
    line=ReadProgramError(process)
    If line="" : Break : EndIf
    PrintN(line)
  ForEver
  code=ProgramExitCode(process)
  CloseProgram(process)
  If code<>0 Or FileSize(root+output)<=0
    PrintN("BUILD FAILED: "+source+" (exit "+Str(code)+")")
    End 1
  EndIf
Next
PrintN("Build complete. Open bin\PureMetalOnnxCompiler.exe.")
End 0
