; ======================================================================
; tensor_random_dynamic_windows.pbi - runtime-dimension wrappers for the ONNX
; random operators on Windows. The generator and its contract live in
; tensor_random.pmi; tensor_random_dynamic_portable.pmi is the portable twin
; of this file and differs only in how a dimension is read.
;
; Kind: bit 0 is the distribution (0 uniform, 1 normal). Bit 1 says the node
; has no dtype attribute, so ONNX takes the output type from the input, and
; only a FLOAT input is accepted.
; ======================================================================

Global Dim PmRandomDims.i(8)

Procedure.i DRandomFill(Y.i,Node.i,HasSeed.i,SeedBits.i,Kind.i,P1Bits.i,P2Bits.i)
  If PmRandomNodeBits(Dt(Y)\Data,Dt(Y)\Count,Node,HasSeed,SeedBits,Kind & 1,P1Bits,P2Bits)=0
    ProcedureReturn DFail("A random tensor has more than 4294967295 elements; the generator numbers elements with 32 bits. Check the shape feeding this node.")
  EndIf
  ProcedureReturn 1
EndProcedure

; RandomUniformLike / RandomNormalLike: the output takes the input's shape.
Procedure DRandomLike(Y.i,A.i,Node.i,HasSeed.i,SeedBits.i,Kind.i,P1Bits.i,P2Bits.i)
  If (Kind & 2)<>0 And Dt(A)\Kind<>1
    If (Kind & 1)=0
      DFail("RandomUniformLike has no dtype attribute, so its output would take the input's element type " + Str(Dt(A)\Kind) + "; only FLOAT (1) random tensors are generated. Add dtype=1 to the node.")
    Else
      DFail("RandomNormalLike has no dtype attribute, so its output would take the input's element type " + Str(Dt(A)\Kind) + "; only FLOAT (1) random tensors are generated. Add dtype=1 to the node.")
    EndIf
    ProcedureReturn
  EndIf
  If DLike(Y,A,1)=0 : ProcedureReturn : EndIf
  DRandomFill(Y,Node,HasSeed,SeedBits,Kind,P1Bits,P2Bits)
EndProcedure

; RandomUniform / RandomNormal: the shape attribute, copied into PmRandomDims
; by the generated lines before the call.
Procedure DRandomShape(Y.i,Rank.i,Node.i,HasSeed.i,SeedBits.i,Kind.i,P1Bits.i,P2Bits.i)
  If DAlloc(Y,1,Rank,@PmRandomDims(0))=0 : ProcedureReturn : EndIf
  DRandomFill(Y,Node,HasSeed,SeedBits,Kind,P1Bits,P2Bits)
EndProcedure

; Comparison mode: the caller supplied this node's tensor as a model input.
; A is the Like node's input (0 for the shape-attribute operators, whose
; shape is in PmRandomDims). The tensor must be FLOAT with the node's shape.
Procedure DRandomFed(Y.i,A.i,Rank.i)
  Protected i.i
  If Dt(Y)\Kind<>1
    DFail("A random node was compiled as a model input (--random-inputs) and its tensor was not supplied as FLOAT. Fill the input the manifest lists for this node before executing.")
    ProcedureReturn
  EndIf
  If A>0 : Rank=Dt(A)\Rank : EndIf
  If Dt(Y)\Rank<>Rank
    DFail("A random node compiled as a model input (--random-inputs) was supplied with rank " + Str(Dt(Y)\Rank) + " where the node produces rank " + Str(Rank) + ". Supply the tensor with the node's shape.")
    ProcedureReturn
  EndIf
  For i=0 To Rank-1
    If A>0
      If Dt(Y)\D[i]<>Dt(A)\D[i]
        DFail("A random node compiled as a model input (--random-inputs) was supplied with extent " + Str(Dt(Y)\D[i]) + " on axis " + Str(i) + " where the node produces " + Str(Dt(A)\D[i]) + ". Supply the tensor with the node's shape.")
        ProcedureReturn
      EndIf
    ElseIf Dt(Y)\D[i]<>PmRandomDims(i)
      DFail("A random node compiled as a model input (--random-inputs) was supplied with extent " + Str(Dt(Y)\D[i]) + " on axis " + Str(i) + " where the node produces " + Str(PmRandomDims(i)) + ". Supply the tensor with the node's shape.")
      ProcedureReturn
    EndIf
  Next
EndProcedure
