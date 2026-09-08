; Run this PureBasic source once to create two tiny, self-contained ONNX models.
; No Python, network, ONNX Runtime, or third-party model download is needed.
EnableExplicit
OpenConsole()
Define root.s=GetPathPart(#PB_Compiler_File)+"../output/"
Define file.i
If FileSize(root)<>-2 And CreateDirectory(root)=0 : PrintN("Cannot create output folder.") : End 1 : EndIf
file=CreateFile(#PB_Any,root+"twice.onnx")
If file=0 : End 1 : EndIf
If WriteData(file,?Twice,?TwiceEnd-?Twice)<>?TwiceEnd-?Twice : CloseFile(file) : End 1 : EndIf
CloseFile(file)
file=CreateFile(#PB_Any,root+"dense.onnx")
If file=0 : End 1 : EndIf
If WriteData(file,?Dense,?DenseEnd-?Dense)<>?DenseEnd-?Dense : CloseFile(file) : End 1 : EndIf
CloseFile(file)
PrintN("Created output/twice.onnx: y = x + x, shape [1,N].")
PrintN("Created output/dense.onnx: y = x * diag(2,3), shape [1,2].")
PrintN("These are ONNX IR 8 / opset 20 model fixtures, not executable code.")
End 0
DataSection
Twice:
  Data.a 8,8,18,31,79,78,78,88,32,67,111,109,112,105,108,101
  Data.a 114,32,80,117,114,101,66,97,115,105,99,32,101,120,97,109
  Data.a 112,108,101,58,67,10,14,10,1,120,10,1,120,18,1,121
  Data.a 34,3,65,100,100,18,5,116,119,105,99,101,90,20,10,1
  Data.a 120,18,15,10,13,8,1,18,9,10,2,8,1,10,3,18
  Data.a 1,78,98,20,10,1,121,18,15,10,13,8,1,18,9,10
  Data.a 2,8,1,10,3,18,1,78,66,2,16,20
TwiceEnd:
Dense:
  Data.a 8,8,18,31,79,78,78,88,32,67,111,109,112,105,108,101
  Data.a 114,32,80,117,114,101,66,97,115,105,99,32,101,120,97,109
  Data.a 112,108,101,58,97,10,17,10,1,120,10,1,119,18,1,121
  Data.a 34,6,77,97,116,77,117,108,18,5,100,101,110,115,101,42
  Data.a 27,8,2,8,2,16,1,66,1,119,74,16,0,0,0,64
  Data.a 0,0,0,0,0,0,0,0,0,0,64,64,90,19,10,1
  Data.a 120,18,14,10,12,8,1,18,8,10,2,8,1,10,2,8
  Data.a 2,98,19,10,1,121,18,14,10,12,8,1,18,8,10,2
  Data.a 8,1,10,2,8,2,66,2,16,20
DenseEnd:
EndDataSection
