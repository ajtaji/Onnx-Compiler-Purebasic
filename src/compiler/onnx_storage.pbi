; Reduced WEIGHT STORAGE. Decode once into explicitly budgeted resident FP32
; memory. This is not reduced activation storage or integer arithmetic.
; Private pack encodings: 10=IEEE binary16, 16=bfloat16, 22=block-32 INT4.
; INT4 blocks: little-endian FP32 symmetric scale followed by 16 packed bytes,
; low nibble first, signed two's-complement [-7,7], zero-padded last block.

Procedure.i PmoStoragePrecision(Precision.s)
  ProcedureReturn Bool(FindString("|fp32|int8|fp16|bf16|int4|","|"+Precision+"|"))
EndProcedure

Procedure.q PmoStorageRound(Value.q, Shift.i)
  Protected base.q=Value>>Shift,tail.q=Value & ((1<<Shift)-1),half.q=1<<(Shift-1)
  If tail>half Or (tail=half And (base & 1)) : base+1 : EndIf
  ProcedureReturn base
EndProcedure

Procedure.i PmoStorageHalf(Bits.q)
  Protected sign.i=(Bits>>16)&$8000,exponent.i=((Bits>>23)&255)-127+15,mantissa.q=Bits & $7FFFFF
  If exponent<=0
    If exponent < -10 : ProcedureReturn sign : EndIf
    ProcedureReturn sign | PmoStorageRound(mantissa|$800000,14-exponent)
  EndIf
  ProcedureReturn sign | ((exponent<<10)+PmoStorageRound(mantissa,13))
EndProcedure

Procedure.i PmoStorageReduce(*Ir.PmoIrModel,Precision.s)
  Protected kind.i,index.q,bytes.q,bits.q,encoded.i,block.q,j.i,quant.i
  Protected *data,scale.f,magnitude.f,value.f,rounded.d,fraction.d,position.i
  NewMap used.i()
  NewMap eligible.i()
  Select Precision
    Case "fp16" : kind=10
    Case "bf16" : kind=16
    Case "int4" : kind=22
    Default : ProcedureReturn 1
  EndSelect
  ForEach *Ir\Outputs() : used(*Ir\Outputs())+1 : Next
  ForEach *Ir\Nodes()
    position=0
    ForEach *Ir\Nodes()\Node\Inputs()
      used(*Ir\Nodes()\Node\Inputs())+1
      If (position=1 And FindString("|MatMul|Gemm|Conv|ConvTranspose|LSTM|","|"+*Ir\Nodes()\Node\Operation+"|")) Or (position=2 And *Ir\Nodes()\Node\Operation="LSTM")
        eligible(*Ir\Nodes()\Node\Inputs())+1
      EndIf
      position+1
    Next
  Next
  ForEach *Ir\Constants()
    If *Ir\Constants()\ElementType<>1 Or *Ir\Constants()\Elements=0 Or eligible(*Ir\Constants()\Name)=0 Or used(*Ir\Constants()\Name)<>eligible(*Ir\Constants()\Name) : Continue : EndIf
    bytes=*Ir\Constants()\Elements*2
    If kind=22 : bytes=((*Ir\Constants()\Elements+31)/32)*20 : EndIf
    ; Small INT4 constants cost more with their scale; keep those lossless.
    If bytes>=*Ir\Constants()\Bytes : Continue : EndIf
    For index=0 To *Ir\Constants()\Elements-1
      bits=PeekL(*Ir\Constants()\Data+index*4)&$FFFFFFFF
      If (bits & $7F800000)=$7F800000 : ProcedureReturn PmoQuantFail("Reduced weight contains NaN or infinity: "+*Ir\Constants()\Name) : EndIf
      If kind=10 And Abs(PeekF(*Ir\Constants()\Data+index*4))>65504.0 : ProcedureReturn PmoQuantFail("FP16 weight exceeds finite range: "+*Ir\Constants()\Name+". Choose BF16, FP32, or another explicit precision.") : EndIf
      If kind=16 And (PmoStorageRound(bits & $7FFFFFFF,16) & $7F80)=$7F80 : ProcedureReturn PmoQuantFail("BF16 rounding overflows finite range: "+*Ir\Constants()\Name) : EndIf
    Next
    *data=AllocateMemory(bytes)
    If *data=0 : ProcedureReturn PmoQuantFail("Cannot allocate reduced weight storage.") : EndIf
    If kind=22
      For block=0 To (*Ir\Constants()\Elements+31)/32-1
        magnitude=0
        For j=0 To 31
          index=block*32+j
          If index<*Ir\Constants()\Elements
            value=Abs(PeekF(*Ir\Constants()\Data+index*4))
            If value>magnitude : magnitude=value : EndIf
          EndIf
        Next
        scale=magnitude/7.0
        If scale=0 : scale=1 : EndIf
        PokeF(*data+block*20,scale)
        For j=0 To 31
          index=block*32+j
          If index>=*Ir\Constants()\Elements : Break : EndIf
          value=PeekF(*Ir\Constants()\Data+index*4)/scale
          rounded=Round(Abs(value),#PB_Round_Down) : fraction=Abs(value)-rounded
          quant=rounded
          If fraction>0.5 Or (fraction=0.5 And (quant & 1)) : quant+1 : EndIf
          If value<0 : quant=-quant : EndIf
          If quant>7 : quant=7 : EndIf
          If quant< -7 : quant=-7 : EndIf
          PokeA(*data+block*20+4+j/2,PeekA(*data+block*20+4+j/2) | ((quant & 15)<<((j & 1)*4)))
        Next
      Next
    Else
      For index=0 To *Ir\Constants()\Elements-1
        bits=PeekL(*Ir\Constants()\Data+index*4)&$FFFFFFFF
        If kind=10 : encoded=PmoStorageHalf(bits) : Else : encoded=((bits>>16)&$8000) | PmoStorageRound(bits & $7FFFFFFF,16) : EndIf
        PokeW(*data+index*2,encoded)
      Next
    EndIf
    *Ir\Constants()\StorageData=*data : *Ir\Constants()\Bytes=bytes
    *Ir\Constants()\StorageKind=kind
    *Ir\Constants()\DecodedOffset=PmoIrAlign(*Ir\ArenaBytes)
    bytes=PmoIrAlign(*Ir\Constants()\Elements*4)
    *Ir\ArenaBytes=*Ir\Constants()\DecodedOffset+bytes
    *Ir\DecodedWeightBytes+bytes : *Ir\ReducedWeightCount+1
  Next
  ProcedureReturn 1
EndProcedure

Procedure PmoStorageEmit(File.i,Aarch64.i)
  Protected source.s=PeekS(?PmoStorageDecoderStart,?PmoStorageDecoderEnd-?PmoStorageDecoderStart,#PB_UTF8)
  If Aarch64 : source=ReplaceString(source,"PMO_STORAGE_FLOAT(q)","MathFFromInt(q)") : Else : source=ReplaceString(source,"PMO_STORAGE_FLOAT(q)","q") : EndIf
  WriteStringN(File,source,#PB_UTF8)
EndProcedure

DataSection
  PmoStorageDecoderStart:
  IncludeBinary "../../runtime/weight_storage.pmi"
  PmoStorageDecoderEnd:
EndDataSection
