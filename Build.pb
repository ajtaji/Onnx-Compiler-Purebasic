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
If FileSize(compiler)<=0
  PrintN("Cannot find the PureBasic compiler: "+compiler)
  End 1
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
