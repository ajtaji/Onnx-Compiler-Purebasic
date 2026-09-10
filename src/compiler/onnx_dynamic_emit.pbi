; A second lowering path retains runtime dimensions and emits a fixed sequence
; of tensor calls. It never executes an ONNX graph interpreter or invokes a compiler.
Global PmdPortable.i

; Source-dialect formatting only: dimensions have a flat native array in the
; portable descriptor, and errors are pointers to immutable ASCII literals.
Procedure PmdLine(File.i,Text.s="")
  Protected i.i,quote.i,character.s,output.s,start.i,finish.i,axisEnd.i,id.s,axis.s
  If PmdPortable
    If Left(LTrim(Text),1)=";" : PmoEmitLine(File,Text) : ProcedureReturn : EndIf
    If FindString(Text,"CompilerIf #PMD_PROFILE") : ProcedureReturn : EndIf
    Text=ReplaceString(Text,"DError="+Chr(34)+Chr(34),"DError=0")
    Text=ReplaceString(Text,"DError<>"+Chr(34)+Chr(34),"DError<>0")
    Text=ReplaceString(Text,"Or DCancel :","Or DCancel<>0 :")
    Text=ReplaceString(Text,"Global PmModelWeights.i, PmModelReady.i","Global PmModelWeights.i"+Chr(10)+"Global PmModelReady.i")
    start=FindString(Text,"Dt(")
    While start
      finish=FindString(Text,")\D[",start)
      If finish=0 : Break : EndIf
      id=Mid(Text,start+3,finish-start-3)
      If PmoCompileDigits(id)=0 : start=FindString(Text,"Dt(",start+3) : Continue : EndIf
      axisEnd=FindString(Text,"]",finish+4)
      axis=Mid(Text,finish+4,axisEnd-finish-4)
      Text=Left(Text,start-1)+"DDims("+id+"*8+"+axis+")"+Mid(Text,axisEnd+1)
      start=FindString(Text,"Dt(",start)
    Wend
    For i=1 To Len(Text)
      character=Mid(Text,i,1)
      If character=Chr(34) : quote=1-quote : EndIf
      If character=":" And quote=0 : output+Chr(10) : Else : output+character : EndIf
    Next
    Text=output
  EndIf
  PmoEmitLine(File,Text)
EndProcedure

; These are public generated-program support sources, not compiler sources.
; Embed them in the native tool, then emit a self-contained source closure.
; The release still ships no model, voice, pronunciation data or loose loader.
DataSection
  PmdScalarStart:
  IncludeBinary "../../runtime/tensor_fp32_windows.pbi"
  PmdScalarEnd:
  PmdDynamicStart:
  IncludeBinary "../../runtime/tensor_dynamic_windows.pbi"
  PmdDynamicEnd:
  PmdSimdStart:
  IncludeBinary "../../runtime/tensor_simd_windows.pbi"
  PmdSimdEnd:
  PmdSpeechStart:
  IncludeBinary "../../runtime/kokoro_speech_windows.pbi"
  PmdSpeechEnd:
  PmdReaderStart:
  IncludeBinary "../../runtime/kokoro_reader_windows.pbi"
  PmdReaderEnd:
  PmdVoicesStart:
  IncludeBinary "../../runtime/kokoro_voices_windows.pbi"
  PmdVoicesEnd:
  PmdAssetsStart:
  IncludeBinary "../../runtime/kokoro_assets.pmi"
  PmdAssetsEnd:
  PmdG2pStart:
  IncludeBinary "../../runtime/kokoro_g2p.pmi"
  PmdG2pEnd:
  PmdG2pExtraStart:
  IncludeBinary "../../runtime/kokoro_g2p_extra.pmi"
  PmdG2pExtraEnd:
  PmdKokoroTextStart:
  IncludeBinary "../../runtime/kokoro_text_portable.pmi"
  PmdKokoroTextEnd:
  PmdPortableStart:
  IncludeBinary "../../runtime/tensor_dynamic_portable.pmi"
  PmdPortableEnd:
  PmdTensorStart:
  IncludeBinary "../../runtime/tensor_fp32.pmi"
  PmdTensorEnd:
  PmdNeonStart:
  IncludeBinary "../../runtime/tensor_fp32_neon.pmi"
  PmdNeonEnd:
  PmdMathA64Start:
  IncludeBinary "../../runtime/math/math_a64.pmi"
  PmdMathA64End:
  PmdMathPicoStart:
  IncludeBinary "../../runtime/math/math_m0.pmi"
  PmdMathPicoEnd:
  PmdMathPico2Start:
  IncludeBinary "../../runtime/math/math_m33.pmi"
  PmdMathPico2End:
EndDataSection

Procedure.i PmdWriteSupport(Path.s,*Data,Bytes.i)
  Protected file.i=CreateFile(#PB_Any,Path),written.i
  If file=0 : PmoDynamicError="Cannot create generated runtime source: "+Path : ProcedureReturn 0 : EndIf
  written=WriteData(file,*Data,Bytes) : CloseFile(file)
  If written<>Bytes : PmoDynamicError="Could not write the complete generated runtime source: "+Path : ProcedureReturn 0 : EndIf
  ProcedureReturn 1
EndProcedure

Procedure.i PmdExportRuntime(Folder.s,Speech.i,TargetIndex.i)
  If FileSize(Folder)<>-2 And CreateDirectory(Folder)=0 : PmoDynamicError="Cannot create the generated runtime source folder." : ProcedureReturn 0 : EndIf
  If PmdPortable
    If Speech
      If PmdWriteSupport(Folder+"kokoro_text_portable.pmi",?PmdKokoroTextStart,?PmdKokoroTextEnd-?PmdKokoroTextStart)=0 : ProcedureReturn 0 : EndIf
      If PmdWriteSupport(Folder+"kokoro_assets.pmi",?PmdAssetsStart,?PmdAssetsEnd-?PmdAssetsStart)=0 : ProcedureReturn 0 : EndIf
      If PmdWriteSupport(Folder+"kokoro_g2p.pmi",?PmdG2pStart,?PmdG2pEnd-?PmdG2pStart)=0 : ProcedureReturn 0 : EndIf
      If PmdWriteSupport(Folder+"kokoro_g2p_extra.pmi",?PmdG2pExtraStart,?PmdG2pExtraEnd-?PmdG2pExtraStart)=0 : ProcedureReturn 0 : EndIf
    EndIf
    If PmdWriteSupport(Folder+"tensor_dynamic_portable.pmi",?PmdPortableStart,?PmdPortableEnd-?PmdPortableStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"tensor_fp32.pmi",?PmdTensorStart,?PmdTensorEnd-?PmdTensorStart)=0 : ProcedureReturn 0 : EndIf
    Select TargetIndex
      Case #PMO_TARGET_PI4,#PMO_TARGET_UNOQ
        If PmdWriteSupport(Folder+"math.pmi",?PmdMathA64Start,?PmdMathA64End-?PmdMathA64Start)=0 : ProcedureReturn 0 : EndIf
        If PmdWriteSupport(Folder+"tensor_fp32_neon.pmi",?PmdNeonStart,?PmdNeonEnd-?PmdNeonStart)=0 : ProcedureReturn 0 : EndIf
      Case #PMO_TARGET_PICO
        If PmdWriteSupport(Folder+"math.pmi",?PmdMathPicoStart,?PmdMathPicoEnd-?PmdMathPicoStart)=0 : ProcedureReturn 0 : EndIf
      Case #PMO_TARGET_PICO2
        If PmdWriteSupport(Folder+"math.pmi",?PmdMathPico2Start,?PmdMathPico2End-?PmdMathPico2Start)=0 : ProcedureReturn 0 : EndIf
    EndSelect
    ProcedureReturn 1
  EndIf
  If PmdWriteSupport(Folder+"tensor_fp32_windows.pbi",?PmdScalarStart,?PmdScalarEnd-?PmdScalarStart)=0 : ProcedureReturn 0 : EndIf
  If PmdWriteSupport(Folder+"tensor_dynamic_windows.pbi",?PmdDynamicStart,?PmdDynamicEnd-?PmdDynamicStart)=0 : ProcedureReturn 0 : EndIf
  If PmdWriteSupport(Folder+"tensor_simd_windows.pbi",?PmdSimdStart,?PmdSimdEnd-?PmdSimdStart)=0 : ProcedureReturn 0 : EndIf
  If Speech
    If PmdWriteSupport(Folder+"kokoro_reader_windows.pbi",?PmdReaderStart,?PmdReaderEnd-?PmdReaderStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"kokoro_voices_windows.pbi",?PmdVoicesStart,?PmdVoicesEnd-?PmdVoicesStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"kokoro_speech_windows.pbi",?PmdSpeechStart,?PmdSpeechEnd-?PmdSpeechStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"kokoro_assets.pmi",?PmdAssetsStart,?PmdAssetsEnd-?PmdAssetsStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"kokoro_g2p.pmi",?PmdG2pStart,?PmdG2pEnd-?PmdG2pStart)=0 : ProcedureReturn 0 : EndIf
    If PmdWriteSupport(Folder+"kokoro_g2p_extra.pmi",?PmdG2pExtraStart,?PmdG2pExtraEnd-?PmdG2pExtraStart)=0 : ProcedureReturn 0 : EndIf
  EndIf
  ProcedureReturn 1
EndProcedure

; Reject attributes that this lowering does not implement; never silently
Procedure.i PmdExportTargetRuntime(Folder.s,TargetIndex.i)
  PmdPortable=Bool(TargetIndex<>#PMO_TARGET_WINDOWS)
  ProcedureReturn PmdExportRuntime(Folder,0,TargetIndex)
EndProcedure

; Reject attributes that this lowering does not implement; never silently
; treat a different operator variant as the one exercised by Kokoro.
Procedure.i PmdValidate(*Node.PmoOnnxNode)
  Protected allowed.s="|",op.s=*Node\Operation,mode.s,i.i
  Select op
    Case "Cast" : allowed="|to|"
    Case "Reshape" : allowed="|allowzero|"
    Case "Transpose" : allowed="|perm|"
    Case "Gather","Concat","Softmax" : allowed="|axis|"
    Case "ConstantOfShape" : allowed="|value|"
    Case "LeakyRelu" : allowed="|alpha|"
    Case "Gemm" : allowed="|transA|transB|alpha|beta|"
    Case "ReduceMean","ReduceSum" : allowed="|keepdims|"
    Case "CumSum" : allowed="|exclusive|reverse|"
    Case "STFT" : allowed="|onesided|"
    Case "LayerNormalization"
      allowed="|axis|epsilon|"
      If PmoEmitAttrI(*Node,"axis",-1)<>-1 Or ListSize(*Node\Outputs())<>1 : Goto Unsupported : EndIf
    Case "Pad"
      allowed="|mode|" : mode=PmoEmitAttrS(*Node,"mode","constant")
      If (mode<>"constant" And mode<>"reflect") Or PmoEmitInput(*Node,3)<>"" : Goto Unsupported : EndIf
    Case "LSTM"
      allowed="|direction|hidden_size|" : mode=PmoEmitAttrS(*Node,"direction","forward")
      If (mode<>"forward" And mode<>"bidirectional") Or PmoEmitInput(*Node,7)<>"" Or ListSize(*Node\Outputs())<>3 : Goto Unsupported : EndIf
    Case "Conv","ConvTranspose"
      allowed="|pads|strides|dilations|kernel_shape|group|"
      If op="ConvTranspose" : allowed+"output_padding|" : EndIf
      If PmoEmitAttrListCount(*Node,"pads")>2 Or PmoEmitAttrListCount(*Node,"strides")>1 Or PmoEmitAttrListCount(*Node,"dilations")>1 Or PmoEmitAttrListCount(*Node,"kernel_shape")>1 Or PmoEmitAttrListCount(*Node,"output_padding")>1 : Goto Unsupported : EndIf
    Case "Resize"
      allowed="|coordinate_transformation_mode|mode|nearest_mode|cubic_coeff_a|"
      mode=PmoEmitAttrS(*Node,"mode","nearest")
      If mode<>"nearest" And mode<>"linear" : Goto Unsupported : EndIf
      If mode="nearest" And PmoEmitAttrS(*Node,"nearest_mode","round_prefer_floor")<>"floor" : Goto Unsupported : EndIf
      mode=PmoEmitAttrS(*Node,"coordinate_transformation_mode","half_pixel")
      If (mode<>"asymmetric" And mode<>"half_pixel") Or PmoEmitInput(*Node,1)<>"" : Goto Unsupported : EndIf
  EndSelect
  ForEach *Node\Attributes()
    If FindString(allowed,"|"+*Node\Attributes()\Name+"|")=0 : Goto Unsupported : EndIf
  Next
  If op<>"LSTM" And ListSize(*Node\Outputs())<>1 : Goto Unsupported : EndIf
  ProcedureReturn 1
  Unsupported:
  PmoDynamicError="Reusable emission does not implement this "+op+" attribute/output variant. Choose a supported operator form."
  ProcedureReturn 0
EndProcedure

Procedure.s PmdId(Map Ids.i(),Name.s)
  If Name="" : ProcedureReturn "0" : EndIf
  If FindMapElement(Ids(),Name)=0 : PmoDynamicError="Tensor has no producer: "+Name : ProcedureReturn "0" : EndIf
  ProcedureReturn Str(Ids())
EndProcedure

Procedure.s PmdCall(*Node.PmoOnnxNode, Map Ids.i())
  Protected Dim a.s(7),Dim y.s(2)
  Protected i.i,op.s=*Node\Operation,call.s,args.s,attr.s,kind.i,value.d
  If PmdValidate(*Node)=0 : ProcedureReturn "" : EndIf
  For i=0 To 7 : a(i)=PmdId(Ids(),PmoEmitInput(*Node,i)) : Next
  For i=0 To 2 : y(i)=PmdId(Ids(),PmoEmitOutput(*Node,i)) : Next
  Select op
    Case "Add","Sub","Mul","Div","Pow","Equal","Greater","Less","GreaterOrEqual","And"
      Select op
        Case "Add":i=0
        Case "Sub":i=1
        Case "Mul":i=2
        Case "Div":i=3
        Case "Pow":i=4
        Case "Equal":i=5
        Case "Greater":i=6
        Case "Less":i=7
        Case "GreaterOrEqual":i=8
        Case "And":i=9
      EndSelect
      call="DBinary("+y(0)+","+a(0)+","+a(1)+","+Str(i)+")"
    Case "Exp","Log","Sqrt","Abs","Neg","Sin","Cos","Atan","Sigmoid","Tanh","Floor","Round","LeakyRelu"
      Select op
        Case "Exp":i=0
        Case "Log":i=1
        Case "Sqrt":i=2
        Case "Abs":i=3
        Case "Neg":i=4
        Case "Sin":i=5
        Case "Cos":i=6
        Case "Atan":i=7
        Case "Sigmoid":i=8
        Case "Tanh":i=9
        Case "Floor":i=10
        Case "Round":i=11
        Case "LeakyRelu":i=12
      EndSelect
      call="DUnary("+y(0)+","+a(0)+","+Str(i)+","+PmoEmitFloat(PmoEmitAttrF(*Node,"alpha",0.01))+")"
    Case "Cast"
      call="DCast("+y(0)+","+a(0)+","+Str(PmoEmitAttrI(*Node,"to",1))+")"
    Case "Shape"
      call="DShapeOf("+y(0)+","+a(0)+")"
    Case "Reshape"
      call="DReshape("+y(0)+","+a(0)+","+a(1)+","+Str(PmoEmitAttrI(*Node,"allowzero",0))+")"
    Case "Squeeze","Unsqueeze"
      call="DViewAxes("+y(0)+","+a(0)+","+a(1)+","+Str(Bool(op="Unsqueeze"))+")"
    Case "Transpose"
      For i=0 To PmoEmitAttrListCount(*Node,"perm")-1
        If i : attr+"," : EndIf
        attr+Str(PmoEmitAttrListI(*Node,"perm",i,0))
      Next
      call="DTranspose("+y(0)+","+a(0)+","+Chr(34)+attr+Chr(34)+")"
    Case "Gather"
      call="DGather("+y(0)+","+a(0)+","+a(1)+","+Str(PmoEmitAttrI(*Node,"axis",0))+")"
    Case "Concat"
      ForEach *Node\Inputs()
        If args<>"" : args+"," : EndIf
        args+PmdId(Ids(),*Node\Inputs())
      Next
      call="DConcat("+y(0)+","+Chr(34)+args+Chr(34)+","+Str(PmoEmitAttrI(*Node,"axis",0))+")"
    Case "Slice"
      call="DSlice("+y(0)+","+a(0)+","+a(1)+","+a(2)+","+a(3)+","+a(4)+")"
    Case "Expand"
      call="DExpand("+y(0)+","+a(0)+","+a(1)+")"
    Case "ConstantOfShape"
      kind=1 : value=0
      ForEach *Node\Attributes()
        If *Node\Attributes()\Name="value" And *Node\Attributes()\HasTensor
          kind=*Node\Attributes()\Tensor\DataType
          If FirstElement(*Node\Attributes()\Tensor\FloatData()) : value=*Node\Attributes()\Tensor\FloatData() : EndIf
          If FirstElement(*Node\Attributes()\Tensor\Int32Data()) : value=*Node\Attributes()\Tensor\Int32Data() : EndIf
          If FirstElement(*Node\Attributes()\Tensor\Int64Data()) : value=*Node\Attributes()\Tensor\Int64Data() : EndIf
          If *Node\Attributes()\Tensor\Raw\Bytes
            Select kind
              Case 1 : value=PeekF(*Node\Attributes()\Tensor\Raw\Data)
              Case 6 : value=PeekL(*Node\Attributes()\Tensor\Raw\Data)
              Case 7 : value=PeekQ(*Node\Attributes()\Tensor\Raw\Data)
              Case 9 : value=PeekA(*Node\Attributes()\Tensor\Raw\Data)
              Default : PmoDynamicError="Unsupported ConstantOfShape element type." : ProcedureReturn ""
            EndSelect
          EndIf
        EndIf
      Next
      call="DConstantShape("+y(0)+","+a(0)+","+Str(kind)+","+StrD(value,12)+")"
    Case "Range"
      call="DRange("+y(0)+","+a(0)+","+a(1)+","+a(2)+")"
    Case "Where"
      call="DWhere("+y(0)+","+a(0)+","+a(1)+","+a(2)+")"
    Case "NonZero"
      call="DNonZero("+y(0)+","+a(0)+")"
    Case "ScatterND"
      call="DScatter("+y(0)+","+a(0)+","+a(1)+","+a(2)+")"
    Case "Pad"
      call="DPad("+y(0)+","+a(0)+","+a(1)+","+a(2)+","+Str(Bool(PmoEmitAttrS(*Node,"mode","constant")="reflect"))+")"
    Case "MatMul"
      call="DMatMul("+y(0)+","+a(0)+","+a(1)+")"
    Case "Gemm"
      call="DGemm("+y(0)+","+a(0)+","+a(1)+","+a(2)+","+Str(PmoEmitAttrI(*Node,"transA",0))+","+Str(PmoEmitAttrI(*Node,"transB",0))+","+PmoEmitFloat(PmoEmitAttrF(*Node,"alpha",1))+","+PmoEmitFloat(PmoEmitAttrF(*Node,"beta",1))+")"
    Case "ReduceMean","ReduceSum"
      call="DReduce("+y(0)+","+a(0)+","+a(1)+","+Str(PmoEmitAttrI(*Node,"keepdims",1))+","+Str(Bool(op="ReduceMean"))+")"
    Case "Softmax"
      call="DSoftmax("+y(0)+","+a(0)+","+Str(PmoEmitAttrI(*Node,"axis",-1))+")"
    Case "LayerNormalization"
      call="DLayerNorm("+y(0)+","+a(0)+","+a(1)+","+a(2)+","+PmoEmitFloat(PmoEmitAttrF(*Node,"epsilon",0.00001))+")"
    Case "CumSum"
      call="DCumSum("+y(0)+","+a(0)+","+a(1)+","+Str(PmoEmitAttrI(*Node,"exclusive",0))+","+Str(PmoEmitAttrI(*Node,"reverse",0))+")"
    Case "Clip"
      call="DClip("+y(0)+","+a(0)+","+a(1)+","+a(2)+")"
    Case "Conv","ConvTranspose"
      call="D"+op+"("+y(0)+","+a(0)+","+a(1)+","+a(2)
      call+"," +Str(PmoEmitAttrListI(*Node,"pads",0,0))+","+Str(PmoEmitAttrListI(*Node,"pads",1,0))
      call+"," +Str(PmoEmitAttrListI(*Node,"strides",0,1))+","+Str(PmoEmitAttrListI(*Node,"dilations",0,1))+","+Str(PmoEmitAttrI(*Node,"group",1))
      If op="ConvTranspose" : call+"," +Str(PmoEmitAttrListI(*Node,"output_padding",0,0)) : EndIf
      call+")"
    Case "Resize"
      call="DResize("+y(0)+","+a(0)+","+a(2)+","+a(3)+","+Str(Bool(PmoEmitAttrS(*Node,"mode","nearest")="linear"))+","+Str(Bool(PmoEmitAttrS(*Node,"coordinate_transformation_mode","half_pixel")<>"asymmetric"))+")"
    Case "LSTM"
      call="DLstm("+y(0)+","+y(1)+","+y(2)
      For i=0 To 6 : call+"," +a(i) : Next
      call+"," +Str(PmoEmitAttrI(*Node,"hidden_size",0))+","+Str(1+Bool(PmoEmitAttrS(*Node,"direction","forward")="bidirectional"))+")"
    Case "STFT"
      call="DStft("+y(0)+","+a(0)+","+a(1)+","+a(2)+","+a(3)+","+Str(PmoEmitAttrI(*Node,"onesided",1))+")"
    Default
      PmoDynamicError="Reusable source emission does not yet support "+op+"."
  EndSelect
  If PmdPortable And (op="Conv" Or op="ConvTranspose" Or op="LSTM")
    args=Mid(call,FindString(call,"(")+1) : args=Left(args,Len(args)-1)
    call=""
    For i=1 To CountString(args,",")+1
      call+"DCall("+Str(i-1)+")="+StringField(args,i,",")+Chr(10)
    Next
    If op="LSTM" : call+"DLstm()" : Else : call+"D"+op+"()" : EndIf
  EndIf
  ProcedureReturn call
EndProcedure

Procedure.i PmoDynamicCommand(ModelPath.s)
  Protected model.PmoOnnxModel,ir.PmoIrModel,root.s=PmoCompileProjectRoot()
  Protected prefix.s,index.i=2,n.i,id.i,file.i,i.i,j.i,constantNode.i,pass.i,seq.i,chunk.i,args.s,name.s,call.s,outid.s,code.s,hash.s,speech.i,manifest.i,object.i
  NewMap ids.i() : NewMap known.i() : NewMap persistent.i() : NewMap last.i()
  NewList calls.s() : NewList isstatic.i()
  NewList shapes.s()
  Protected precision.s="fp32",kokoroText.i
  Protected target.s="windows",targetIndex.i,workingLimit.q=1024*1024*1024,metadataBytes.q,crc.i,pending.s,sourcePath.s
  PmdPortable=0
  PmoDynamicError=""
  UseSHA2Fingerprint()
  While index<CountProgramParameters()
    Select ProgramParameter(index)
      Case "--speech-ui" : speech=1
      Case "--kokoro-text" : kokoroText=1
      Case "--library", "--simplify"
        ; All models are resident; constant scheduling is always enabled.
      Case "--precision"
        index+1
        precision=LCase(ProgramParameter(index))
        If PmoStoragePrecision(precision)=0 : PmoDynamicError="Choose FP32, INT8, FP16, BF16, or INT4 precision." : ProcedureReturn 0 : EndIf
      Case "--shape"
        index+1 : AddElement(shapes()) : shapes()=ProgramParameter(index)
      Case "--trace-input", "--ort-dll"
        index+1
        PrintN("Runtime dimensions are generated from the model; the tracing option is not used or embedded.")
      Case "--checkpoint"
        PmoDynamicError="Runtime-dimension checkpoint export is not implemented. Remove --checkpoint; graph outputs remain available after every request." : ProcedureReturn 0
      Case "--output"
        If index+1>=CountProgramParameters() : PmoDynamicError="The --output option requires a path prefix." : ProcedureReturn 0 : EndIf
        index+1 : prefix=ProgramParameter(index)
      Case "--target"
        If index+1>=CountProgramParameters() : PmoDynamicError="The --target option requires a target name." : ProcedureReturn 0 : EndIf
        index+1
        target=LCase(ProgramParameter(index))
      Default
        PmoDynamicError="Unknown reusable-source option: "+ProgramParameter(index) : ProcedureReturn 0
    EndSelect
    index+1
  Wend
  targetIndex=PmoTargetIndex(target)
  If targetIndex<0 : PmoDynamicError="Unknown target: "+target : ProcedureReturn 0 : EndIf
  PmdPortable=Bool(target<>"windows")
  If speech And PmdPortable : PmoDynamicError="The speech window requires Windows." : ProcedureReturn 0 : EndIf
  If kokoroText And targetIndex<>#PMO_TARGET_PI4 And targetIndex<>#PMO_TARGET_UNOQ
    PmoDynamicError="--kokoro-text requires pi4 or unoq; use --speech-ui for Windows. The full model and dictionaries cannot fit Pico RAM." : ProcedureReturn 0
  EndIf
  If prefix="" Or FileSize(GetPathPart(prefix))<>-2 : PmoDynamicError="Choose an output prefix in an existing folder." : ProcedureReturn 0 : EndIf
  prefix=PmoCompileAbsolute(prefix)
  If (speech Or kokoroText) And LCase(FileFingerprint(ModelPath,#PB_Cipher_SHA2,256))<>"8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
    PmoDynamicError="The speech adapter requires the documented full Kokoro-82M model. Check its SHA-256 identity." : ProcedureReturn 0
  EndIf
  If PmoOnnxLoad(ModelPath,@model)=0 : PmoDynamicError=PmoWireError : ProcedureReturn 0 : EndIf
  ForEach shapes()
    If PmoCompileApplyShape(@model,shapes())=0 : PmoDynamicError=PmoCompileError : Goto Failed : EndIf
  Next
  ir\Source=@model
  ForEach model\Opsets()
    If (model\Opsets()\Domain<>"" And model\Opsets()\Domain<>"ai.onnx") Or model\Opsets()\Version<>20
      PmoDynamicError="Reusable emission currently validates the ONNX opset 20 surface." : Goto Failed
    EndIf
  Next
  ForEach model\Graph\Initializers()
    If ListSize(model\Graph\Initializers()\Dims())>8 : PmoDynamicError="Initializer rank exceeds the reusable backend's eight-axis limit." : Goto Failed : EndIf
    Select model\Graph\Initializers()\DataType
      Case 1,6,7,9
      Default : PmoDynamicError="Reusable FP32 emission does not implement this initializer element type." : Goto Failed
    EndSelect
    id+1 : ids(model\Graph\Initializers()\Name)=id : known(Str(id))=1 : persistent(Str(id))=1
    If PmoIrAddInitializer(@ir,@model\Graph\Initializers())=0 : PmoDynamicError=PmoIrError : Goto Failed : EndIf
  Next
  ForEach model\Graph\Inputs()
    If model\Graph\Inputs()\HasShape=0 Or ListSize(model\Graph\Inputs()\Dims())>8
      PmoDynamicError="Runtime-dimension inputs require a declared rank of at most eight. Check the model's input shape metadata." : Goto Failed
    EndIf
    If FindMapElement(ids(),model\Graph\Inputs()\Name)=0 : id+1 : ids(model\Graph\Inputs()\Name)=id : EndIf
  Next
  ForEach model\Graph\Nodes()
    If model\Graph\Nodes()\Domain<>"" And model\Graph\Nodes()\Domain<>"ai.onnx" : PmoDynamicError="Unsupported node domain." : Goto Failed : EndIf
    constantNode=1
    ForEach model\Graph\Nodes()\Inputs()
      name=model\Graph\Nodes()\Inputs()
      If name<>"" And FindMapElement(ids(),name)=0 : PmoDynamicError="Missing producer: "+name : Goto Failed : EndIf
      If name<>"" And known(Str(ids(name)))=0 : constantNode=0 : EndIf
    Next
    ForEach model\Graph\Nodes()\Outputs()
      name=model\Graph\Nodes()\Outputs()
      If name<>"" And FindMapElement(ids(),name) : PmoDynamicError="Duplicate tensor producer: "+name : Goto Failed : EndIf
      If name<>"" : id+1 : ids(name)=id : known(Str(id))=constantNode : EndIf
    Next
    AddElement(isstatic()) : isstatic()=constantNode
    call=PmdCall(@model\Graph\Nodes(),ids())
    If PmoDynamicError<>"" : Goto Failed : EndIf
    AddElement(calls()) : calls()=call
    AddElement(ir\Nodes()) : ir\Nodes()\Node=@model\Graph\Nodes() : ir\Nodes()\Index=n
    n+1
  Next
  n=0
  ForEach model\Graph\Nodes()
    SelectElement(isstatic(),n)
    ForEach model\Graph\Nodes()\Inputs()
      name=model\Graph\Nodes()\Inputs()
      If name<>""
        last(Str(ids(name)))=n
        If isstatic()=0 And known(Str(ids(name))) : persistent(Str(ids(name)))=1 : EndIf
      EndIf
    Next
    n+1
  Next
  ForEach model\Graph\Outputs()
    AddElement(ir\Outputs()) : ir\Outputs()=model\Graph\Outputs()\Name
    last(Str(ids(ir\Outputs())))=n+1
  Next
  If precision="int8"
    If PmoQuantizeDynamicWeights(@ir)=0 : PmoDynamicError=PmoQuantError : Goto Failed : EndIf
    ForEach ir\QuantWeights()
      id+1 : ids(ir\QuantWeights()\ScaleName)=id : known(Str(id))=1 : persistent(Str(id))=1
    Next
  EndIf
  If PmoStorageReduce(@ir,precision)=0 : PmoDynamicError=PmoQuantError : Goto Failed : EndIf
  If ir\DecodedWeightBytes>1024*1024*1024 : PmoDynamicError="Decoded resident weights exceed the working-memory limit. Reduced storage is not reduced execution RAM." : Goto Failed : EndIf
  If PmdPortable
    metadataBytes=(id+1)*15*PmoTargets(targetIndex)\NativeIntegerBytes+4096
    workingLimit=1024*1024*1024
    If PmoTargets(targetIndex)\RamBytes
      workingLimit=(PmoTargets(targetIndex)\RamBytes-PmoTargets(targetIndex)\RamReserveBytes-metadataBytes) & -16
      If workingLimit<32 Or ir\DecodedWeightBytes>workingLimit
        PmoDynamicError="Model does not fit target "+target+": decoded resident weights require "+PmoCompileBytes(ir\DecodedWeightBytes)+", tensor metadata reserve "+PmoCompileBytes(metadataBytes)+", runtime/stack reserve "+PmoCompileBytes(PmoTargets(targetIndex)\RamReserveBytes)+", available SRAM "+PmoCompileBytes(PmoTargets(targetIndex)\RamBytes)+". Reduced storage does not remove FP32 decode memory." : Goto Failed
      EndIf
    EndIf
  EndIf
  pending=prefix+".pmw.pending"
  If PmoPackWeights(@ir,pending)=0 : PmoDynamicError=PmoPackError : Goto Failed : EndIf
  If PmoTargets(targetIndex)\FlashBytes
    If ir\WeightBytes+PmoTargets(targetIndex)\FlashReserveBytes+n*128>PmoTargets(targetIndex)\FlashBytes
      PmoDynamicError="Model does not fit target "+target+": packed weights "+PmoCompileBytes(ir\WeightBytes)+" plus code exceed flash "+PmoCompileBytes(PmoTargets(targetIndex)\FlashBytes)+". Nothing was truncated; choose a smaller model or larger target." : Goto Failed
    EndIf
  EndIf
  If PmdPortable And PmoCompileValidateTargetIntegers(@ir,@PmoTargets(targetIndex))=0 : PmoDynamicError=PmoCompileError : Goto Failed : EndIf
  file=ReadFile(#PB_Any,pending)
  If file=0 : PmoDynamicError="Cannot verify packed weights." : Goto Failed : EndIf
  FileSeek(file,40) : crc=ReadLong(file) : CloseFile(file) : file=0
  If FileSize(prefix+".pmw")>=0 And DeleteFile(prefix+".pmw")=0 : PmoDynamicError="Cannot replace packed weights." : Goto Failed : EndIf
  If RenameFile(pending,prefix+".pmw")=0 : PmoDynamicError="Cannot publish packed weights." : Goto Failed : EndIf
  pending=""
  hash=LCase(FileFingerprint(prefix+".pmw",#PB_Cipher_SHA2,256))
  If PmdExportRuntime(prefix+".runtime\",Bool(speech Or kokoroText),targetIndex)=0 : Goto Failed : EndIf
  sourcePath=prefix+PmoTargets(targetIndex)\SourceSuffix
  file=CreateFile(#PB_Any,sourcePath)
  If file=0 : PmoDynamicError="Cannot create reusable model source." : Goto Failed : EndIf
  PmdLine(file,"; Reusable model: emitted tensor calls with runtime dimensions.")
  If PmdPortable : PmdLine(file,"EnableFloatingPoint") : Else : PmdLine(file,"EnableExplicit") : EndIf
  PmdLine(file,"#PMD_TENSOR_COUNT = "+Str(id))
  PmdLine(file,"#PMD_NODE_COUNT = "+Str(ListSize(model\Graph\Nodes())))
  If PmdPortable
    PmdLine(file,"#PMD_NATIVE_BYTES="+Str(PmoTargets(targetIndex)\NativeIntegerBytes))
    PmdLine(file,"#PMO_INT8_BACKEND="+Str(1+Bool(PmoTargets(targetIndex)\NativeIntegerBytes=4)+Bool(targetIndex=#PMO_TARGET_PICO2)))
    PmdLine(file,"#PMO_INT64_SPLIT="+Str(Bool(PmoTargets(targetIndex)\NativeIntegerBytes=4)))
    PmdLine(file,"#PMD_WORKING_MEMORY_LIMIT="+Str(workingLimit))
    PmdLine(file,"#PMD_METADATA_RESERVE="+Str(metadataBytes))
    PmdLine(file,"#PMO_WEIGHT_FILE_BYTES="+Str(ir\WeightBytes))
    PmdLine(file,"#PMO_USE_BATCHNORM=1 : #PMO_USE_STFT=1 : #PMO_USE_RESIZE=1 : #PMO_USE_CONVTRANSPOSE=1 : #PMO_USE_CONV=1 : #PMO_USE_LSTM=1 : #PMO_USE_INT8=1")
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\math.pmi"+Chr(34))
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\tensor_fp32.pmi"+Chr(34))
    If PmoTargets(targetIndex)\NativeIntegerBytes=8 : PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\tensor_fp32_neon.pmi"+Chr(34)) : EndIf
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\tensor_dynamic_portable.pmi"+Chr(34))
  Else
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\tensor_dynamic_windows.pbi"+Chr(34))
  EndIf
  PmdLine(file,"Global PmModelWeights.i, PmModelReady.i")
  If ir\ReducedWeightCount : PmoStorageEmit(file,Bool(PmdPortable And PmoTargets(targetIndex)\NativeIntegerBytes=8)) : EndIf
  ; Separate fixed schedules for constant preparation and request inference.
  For pass=0 To 1
    seq=0 : chunk=0 : n=0
    ForEach calls()
      SelectElement(isstatic(),n)
      If isstatic()=1-pass
        If seq % 80=0
          If seq : PmdLine(file,"EndProcedure") : EndIf
          PmdLine(file,"Procedure PmModelStage"+Str(pass)+"_"+Str(chunk)+"()")
          PmdLine(file,"  If DRejectProgressReentry()")
          PmdLine(file,"    ProcedureReturn")
          PmdLine(file,"  EndIf") : chunk+1
        EndIf
        PmdLine(file,"  DNode="+Str(n))
        PmdLine(file,"  If DPoll(#PMD_PROGRESS_NODE_BEFORE,"+Str(n)+")=0 : ProcedureReturn : EndIf")
        PmdLine(file,"  CompilerIf #PMD_PROFILE : DStarted=ElapsedMilliseconds() : CompilerEndIf")
        PmdLine(file,"  "+calls())
        PmdLine(file,"  CompilerIf #PMD_PROFILE : DTimes("+Str(n)+")+ElapsedMilliseconds()-DStarted : CompilerEndIf")
        PmdLine(file,"  If DPoll(#PMD_PROGRESS_NODE_AFTER,"+Str(n)+")=0 : ProcedureReturn : EndIf")
        PmdLine(file,"  If DError<>"+Chr(34)+Chr(34)+" Or DCancel : ProcedureReturn : EndIf")
        ForEach last()
          If last()=n And persistent(MapKey(last()))=0
            PmdLine(file,"  DRelease("+MapKey(last())+")")
          EndIf
        Next
        seq+1
      EndIf
      n+1
    Next
    If seq : PmdLine(file,"EndProcedure") : EndIf
    PmdLine(file,"Procedure PmModelStage"+Str(pass)+"()")
    PmdLine(file,"  If DRejectProgressReentry()")
    PmdLine(file,"    ProcedureReturn")
    PmdLine(file,"  EndIf")
    For j=0 To chunk-1
      PmdLine(file,"  PmModelStage"+Str(pass)+"_"+Str(j)+"()")
      PmdLine(file,"  If DError<>"+Chr(34)+Chr(34)+" Or DCancel : ProcedureReturn : EndIf")
    Next
    PmdLine(file,"EndProcedure")
  Next
  PmdLine(file,"Procedure PmModelResetRequest()")
  PmdLine(file,"  If DRejectProgressReentry()")
  PmdLine(file,"    ProcedureReturn")
  PmdLine(file,"  EndIf")
  ForEach ids()
    If persistent(Str(ids()))=0 : PmdLine(file,"  DRelease("+Str(ids())+")") : EndIf
  Next
  PmdLine(file,"EndProcedure")
  PmdLine(file,"Procedure PmModelClose()")
  PmdLine(file,"  Protected i.i")
  PmdLine(file,"  If DRejectProgressReentry()")
  PmdLine(file,"    ProcedureReturn")
  PmdLine(file,"  EndIf")
  PmdLine(file,"  For i=1 To #PMD_TENSOR_COUNT : DRelease(i) : Next")
  If PmdPortable
    PmdLine(file,"  DHeap=0 : DHeapBytes=0 : DHeapUsed=0")
  Else
    PmdLine(file,"  If PmModelWeights : FreeMemory(PmModelWeights) : EndIf")
  EndIf
  PmdLine(file,"  PmModelWeights=0 : PmModelReady=0")
  PmdLine(file,"EndProcedure")
  If PmdPortable
    PmdLine(file,"; Caller owns weights and aligned arena for the complete model lifetime.")
    PmdLine(file,"Procedure.i PmModelBindMemory(weights.i,weightBytes.i,arena.i,arenaBytes.i)")
    PmdLine(file,"  Protected Dim initialDims.i(8)")
    PmdLine(file,"  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf")
    PmdLine(file,"  PmModelClose() : DError=0 : DCancel=0 : PmTensorInt64Ok=1")
    PmdLine(file,"  If DPoll(#PMD_PROGRESS_BIND_BEFORE,-1)=0 : ProcedureReturn 0 : EndIf")
    PmdLine(file,"  If weights=0 Or weightBytes<>#PMO_WEIGHT_FILE_BYTES : ProcedureReturn DFail("+Chr(34)+"Weight file length mismatch."+Chr(34)+") : EndIf")
    PmdLine(file,"  If weights<arena : If weightBytes>arena-weights : ProcedureReturn DFail("+Chr(34)+"Weights overlap the working arena."+Chr(34)+") : EndIf : Else : If arenaBytes>weights-arena : ProcedureReturn DFail("+Chr(34)+"Working arena overlaps weights."+Chr(34)+") : EndIf : EndIf")
    PmdLine(file,"  If PeekL(weights)<>$4E4F4D50 Or PeekL(weights+4)<>$0057584E Or PeekL(weights+8)<>1 Or PeekL(weights+12)<>128 : ProcedureReturn DFail("+Chr(34)+"Weight header mismatch."+Chr(34)+") : EndIf")
    PmdLine(file,"  If PmTensorCrc32(weights+128,weightBytes-128)<>"+PmoEmitHex32(crc)+" : ProcedureReturn DFail("+Chr(34)+"Weight payload identity mismatch."+Chr(34)+") : EndIf")
    PmdLine(file,"  If arenaBytes>#PMD_WORKING_MEMORY_LIMIT : arenaBytes=#PMD_WORKING_MEMORY_LIMIT : EndIf")
    PmdLine(file,"  If DHeapBind(arena,arenaBytes)=0 : ProcedureReturn 0 : EndIf")
    PmdLine(file,"  PmModelWeights=weights")
  Else
  PmdLine(file,"Procedure.i PmModelInitialize(path.s)")
  PmdLine(file,"  Protected file.i")
  PmdLine(file,"  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf")
  PmdLine(file,"  If PmModelReady : ProcedureReturn 1 : EndIf")
  PmdLine(file,"  PmModelClose()")
  PmdLine(file,"  If DPoll(#PMD_PROGRESS_BIND_BEFORE,-1)=0 : ProcedureReturn 0 : EndIf")
  PmdLine(file,"  UseSHA2Fingerprint()")
  PmdLine(file,"  If LCase(FileFingerprint(path,#PB_Cipher_SHA2,256))<>"+Chr(34)+hash+Chr(34)+" : ProcedureReturn DFail("+Chr(34)+"Weight file identity mismatch."+Chr(34)+") : EndIf")
  PmdLine(file,"  file=ReadFile(#PB_Any,path)")
  PmdLine(file,"  If file=0 : ProcedureReturn DFail("+Chr(34)+"Cannot open weights."+Chr(34)+") : EndIf")
  PmdLine(file,"  PmModelWeights=AllocateMemory("+Str(ir\WeightBytes)+")")
  PmdLine(file,"  If PmModelWeights=0 : CloseFile(file) : ProcedureReturn DFail("+Chr(34)+"Cannot allocate weights."+Chr(34)+") : EndIf")
  PmdLine(file,"  If ReadData(file,PmModelWeights,"+Str(ir\WeightBytes)+")<>"+Str(ir\WeightBytes)+" : CloseFile(file) : PmModelClose() : ProcedureReturn DFail("+Chr(34)+"Short weight read."+Chr(34)+") : EndIf")
  PmdLine(file,"  CloseFile(file)")
  EndIf
  ForEach ir\Constants()
    If ir\Constants()\PackedOffset<=0 : Continue : EndIf
    name=Str(ids(ir\Constants()\Name))
    PmdLine(file,"  Dt("+name+")\Data=PmModelWeights+"+Str(ir\Constants()\PackedOffset))
    PmdLine(file,"  Dt("+name+")\Kind="+Str(ir\Constants()\ElementType))
    PmdLine(file,"  Dt("+name+")\Count="+Str(ir\Constants()\Elements))
    PmdLine(file,"  Dt("+name+")\Bytes="+Str(ir\Constants()\Bytes))
    PmdLine(file,"  Dt("+name+")\Rank="+Str(ListSize(ir\Constants()\Dims())))
    i=0
    ForEach ir\Constants()\Dims()
      PmdLine(file,"  Dt("+name+")\D["+Str(i)+"]="+Str(ir\Constants()\Dims())) : i+1
    Next
    If ir\Constants()\StorageKind
      args=""
      ForEach ir\Constants()\Dims() : args+","+Str(ir\Constants()\Dims()) : Next
      If PmdPortable
        i=0
        ForEach ir\Constants()\Dims() : PmdLine(file,"  initialDims("+Str(i)+")="+Str(ir\Constants()\Dims())) : i+1 : Next
        PmdLine(file,"  If DAlloc("+name+",1,"+Str(ListSize(ir\Constants()\Dims()))+",@initialDims(0))=0 : PmModelClose() : ProcedureReturn 0 : EndIf")
      Else
        PmdLine(file,"  If DShape("+name+",1,"+Str(ListSize(ir\Constants()\Dims()))+args+")=0 : PmModelClose() : ProcedureReturn 0 : EndIf")
      EndIf
      PmdLine(file,"  PmWeightDecode(PmModelWeights+"+Str(ir\Constants()\PackedOffset)+",Dt("+name+")\Data,"+Str(ir\Constants()\Elements)+","+Str(ir\Constants()\StorageKind)+")")
    EndIf
  Next
  ForEach ir\QuantWeights()
    PmdLine(file,"  Dt("+Str(ids(ir\QuantWeights()\Name))+")\Scales=Dt("+Str(ids(ir\QuantWeights()\ScaleName))+")\Data")
  Next
  PmdLine(file,"  PmModelStage0()")
  PmdLine(file,"  DPoll(#PMD_PROGRESS_BIND_AFTER,-1)")
  PmdLine(file,"  PmModelReady=Bool(DError="+Chr(34)+Chr(34)+" And DCancel=0)")
  PmdLine(file,"  If PmModelReady=0 : PmModelClose() : EndIf")
  PmdLine(file,"  ProcedureReturn PmModelReady")
  PmdLine(file,"EndProcedure")
  PmdLine(file,"Procedure.i PmModelInput(index.i)")
  PmdLine(file,"  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf")
  PmdLine(file,"  Select index")
  i=0
  ForEach model\Graph\Inputs()
    PmdLine(file,"    Case "+Str(i)+" : ProcedureReturn "+Str(ids(model\Graph\Inputs()\Name))) : i+1
  Next
  PmdLine(file,"  EndSelect")
  PmdLine(file,"EndProcedure")
  PmdLine(file,"Procedure.i PmModelOutput(index.i)")
  PmdLine(file,"  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf")
  PmdLine(file,"  Select index")
  i=0
  ForEach model\Graph\Outputs()
    PmdLine(file,"    Case "+Str(i)+" : ProcedureReturn "+Str(ids(model\Graph\Outputs()\Name))) : i+1
  Next
  PmdLine(file,"  EndSelect")
  PmdLine(file,"EndProcedure")
  PmdLine(file,"Procedure.i PmModelExecute()")
  PmdLine(file,"  If DRejectProgressReentry() : ProcedureReturn 0 : EndIf")
  PmdLine(file,"  If PmModelReady=0 : ProcedureReturn DFail("+Chr(34)+"Initialize the model before submitting inputs."+Chr(34)+") : EndIf")
  ForEach model\Graph\Inputs()
    name=Str(ids(model\Graph\Inputs()\Name))
    PmdLine(file,"  If Dt("+name+")\Data=0 Or Dt("+name+")\Kind<>"+Str(model\Graph\Inputs()\ElementType)+" Or Dt("+name+")\Rank<>"+Str(ListSize(model\Graph\Inputs()\Dims()))+" : ProcedureReturn DFail("+Chr(34)+"Input type or rank does not match the model contract."+Chr(34)+") : EndIf")
    i=0
    ForEach model\Graph\Inputs()\Dims()
      If model\Graph\Inputs()\Dims()\HasValue
        PmdLine(file,"  If Dt("+name+")\D["+Str(i)+"]<>"+Str(model\Graph\Inputs()\Dims()\Value)+" : ProcedureReturn DFail("+Chr(34)+"Input extent does not match a fixed model dimension."+Chr(34)+") : EndIf")
      EndIf
      i+1
    Next
  Next
  PmdLine(file,"  PmModelStage1()")
  If PmdPortable : PmdLine(file,"  If PmTensorInt64Ok=0 : ProcedureReturn DFail("+Chr(34)+"An INT64 value exceeds this target's checked execution range."+Chr(34)+") : EndIf") : EndIf
  PmdLine(file,"  ProcedureReturn Bool(DError="+Chr(34)+Chr(34)+" And DCancel=0)")
  PmdLine(file,"EndProcedure")
  If PmoTargets(targetIndex)\Launch=#PMO_LAUNCH_EMBEDDED_WEIGHTS
    If PmoEmitEmbeddedWeights(file,prefix+".pmw")=0 : PmoDynamicError=PmoEmitError : Goto Failed : EndIf
  EndIf
  CloseFile(file) : file=0
  If speech
    file=CreateFile(#PB_Any,prefix+".speech.pb")
    If file=0 : PmoDynamicError="Cannot create the speech application source." : Goto Failed : EndIf
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".pb"+Chr(34))
    PmdLine(file,"#PM_SPEECH_WEIGHT_NAME$ = "+Chr(34)+GetFilePart(prefix)+".pmw"+Chr(34))
    PmdLine(file,"#PM_SPEECH_ASSET_ROOT$ = "+Chr(34)+root+Chr(34))
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\kokoro_speech_windows.pbi"+Chr(34))
    PmdLine(file,"PmSpeechRun()")
    CloseFile(file) : file=0
  EndIf
  If kokoroText
    file=CreateFile(#PB_Any,prefix+".kokoro.pi4")
    If file=0 : PmoDynamicError="Cannot create the resident Kokoro text adapter entry point." : Goto Failed : EndIf
    PmdLine(file,"; Include this source in the host application; bind assets and model once.")
    PmdLine(file,"; New text goes to PmKokoroTextRender at runtime. No board I/O is assumed.")
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(sourcePath)+Chr(34))
    PmdLine(file,"XIncludeFile "+Chr(34)+GetFilePart(prefix)+".runtime\kokoro_text_portable.pmi"+Chr(34))
    CloseFile(file) : file=0
  EndIf
  manifest=CreateJSON(#PB_Any)
  If manifest=0 : PmoDynamicError="Cannot allocate the source manifest." : Goto Failed : EndIf
  object=SetJSONObject(JSONValue(manifest))
  SetJSONString(AddJSONMember(object,"backend"),"reusable-runtime-dimensions")
  SetJSONString(AddJSONMember(object,"target"),target)
  SetJSONString(AddJSONMember(object,"precision"),precision)
  SetJSONInteger(AddJSONMember(object,"quantized_weight_count"),ListSize(ir\QuantWeights()))
  If precision="int8"
    SetJSONString(AddJSONMember(object,"computation"),"FP32 tensors; selected INT8 linear weights use dynamic INT8 activations and INT32 accumulation")
  Else
    SetJSONString(AddJSONMember(object,"computation"),"FP32 arithmetic; reduced storage weights are decoded once into resident FP32 working memory")
  EndIf
  SetJSONInteger(AddJSONMember(object,"reduced_storage_weight_count"),ir\ReducedWeightCount)
  SetJSONInteger(AddJSONMember(object,"decoded_weight_reserve_bytes"),ir\DecodedWeightBytes)
  SetJSONBoolean(AddJSONMember(object,"source_only"),#True)
  SetJSONString(AddJSONMember(object,"execution_contract"),"resident-reusable")
  SetJSONString(AddJSONMember(object,"shape_contract"),"runtime-dimensions")
  SetJSONBoolean(AddJSONMember(object,"embedded_request_inputs"),#False)
  SetJSONBoolean(AddJSONMember(object,"speech_ui"),speech)
  SetJSONBoolean(AddJSONMember(object,"kokoro_text"),kokoroText)
  If kokoroText
    SetJSONString(AddJSONMember(object,"kokoro_entry_source"),prefix+".kokoro.pi4")
    SetJSONString(AddJSONMember(object,"kokoro_assets"),"Caller-owned immutable base and optional extra pronunciation packs plus one PMVOICE; not embedded in source or included in model arena")
    SetJSONInteger(AddJSONMember(object,"kokoro_sample_rate"),24000)
    SetJSONInteger(AddJSONMember(object,"kokoro_max_phonemes"),510)
  EndIf
  SetJSONString(AddJSONMember(object,"weights_sha256"),hash)
  SetJSONInteger(AddJSONMember(object,"weights_bytes"),ir\WeightBytes)
  SetJSONInteger(AddJSONMember(object,"working_memory_limit_bytes"),workingLimit)
  SetJSONInteger(AddJSONMember(object,"metadata_reserve_bytes"),metadataBytes)
  SetJSONString(AddJSONMember(object,"runtime_capacity_contract"),"Each request is checked against the bound memory budget, including allocator overhead on portable targets; arbitrary future input sizes are not guaranteed to fit")
  SetJSONString(AddJSONMember(object,"source"),sourcePath)
  SetJSONInteger(AddJSONMember(object,"node_count"),ListSize(model\Graph\Nodes()))
  If SaveJSON(manifest,prefix+".json",#PB_JSON_PrettyPrint)=0
    FreeJSON(manifest) : PmoDynamicError="Cannot write the source manifest." : Goto Failed
  EndIf
  FreeJSON(manifest)
  PrintN("PASS: reusable source emitted to "+sourcePath)
  PrintN("Weights: "+prefix+".pmw")
  PrintN("Model requires one downstream build; requests use runtime dimensions.")
  PmoIrFree(@ir) : PmoOnnxFree(@model)
  ProcedureReturn 1
  Failed:
  If file : CloseFile(file) : EndIf
  If pending<>"" And FileSize(pending)>=0 : DeleteFile(pending) : EndIf
  PmoIrFree(@ir) : PmoOnnxFree(@model)
  ProcedureReturn 0
EndProcedure
