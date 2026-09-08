; First generate output/twice.pb; see docs/QUICKSTART.md.
; This is a separately built consumer, not part of the ONNX compiler.
EnableExplicit
XIncludeFile "../output/twice.pb"
OpenConsole()
Define root.s=GetPathPart(#PB_Compiler_File)+"../output/"
Define request.i,n.i,i.i,input.i,output.i,baseline.i
If PmModelInitialize(root+"twice.pmw")=0
  PrintN("Model initialization failed: "+DError)
  End 1
EndIf
baseline=DLive
For request=1 To 3
  n=request*3
  PmModelResetRequest()
  DError=""
  input=PmModelInput(0)
  If DShape(input,1,2,1,n)=0 : PrintN(DError) : PmModelClose() : End 1 : EndIf
  For i=0 To n-1 : PokeF(Dt(input)\Data+i*4,i+request) : Next
  If PmModelExecute()=0 : PrintN(DError) : PmModelClose() : End 1 : EndIf
  output=PmModelOutput(0)
  If Dt(output)\Count<>n : PmModelClose() : End 1 : EndIf
  For i=0 To n-1
    If PeekF(Dt(output)\Data+i*4)<>2*(i+request)
      PrintN("Numerical mismatch.") : PmModelClose() : End 1
    EndIf
  Next
  PrintN("PASS request "+Str(request)+": "+Str(n)+" values doubled by one resident model.")
  PmModelResetRequest()
  If DLive<>baseline : PrintN("Request storage was not released.") : PmModelClose() : End 1 : EndIf
Next
PmModelClose()
If DLive<>0 : End 1 : EndIf
PrintN("All requests passed; model closed. No inference runtime DLL was loaded.")
End 0
