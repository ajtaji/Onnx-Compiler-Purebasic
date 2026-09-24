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

; Bernoulli: Y takes X's shape and Kind (the dtype attribute, or FLOAT when
; the node has none); X must be FLOAT.
Procedure DRandomBernoulli(Y.i,X.i,Node.i,HasSeed.i,SeedBits.i,Kind.i)
  If Dt(X)\Kind<>1
    DFail("Bernoulli takes FLOAT probabilities; its input has element type " + Str(Dt(X)\Kind) + ".")
    ProcedureReturn
  EndIf
  If DLike(Y,X,Kind)=0 : ProcedureReturn : EndIf
  If PmRandomBernoulli(Dt(X)\Data,Dt(Y)\Data,Dt(Y)\Count,Node,HasSeed,SeedBits,Kind)=0
    DFail("A random tensor has more than 4294967295 elements; the generator numbers elements with 32 bits. Check the shape feeding this node.")
  EndIf
EndProcedure

; Multinomial, first half: Y shaped [batch, Samples] of Kind and filled with
; the node's uniform draws as FLOAT; DOpMultinomialPick (tensor_dynamic_ops.pmi)
; turns them into class indices. Returns 0 after a refusal.
Procedure.i DRandomMultinomial(Y.i,X.i,Node.i,HasSeed.i,SeedBits.i,Samples.i,Kind.i)
  If Dt(X)\Kind<>1 Or Dt(X)\Rank<>2
    ProcedureReturn DFail("Multinomial takes a FLOAT input of shape [batch_size, class_size]; its input has element type " + Str(Dt(X)\Kind) + " and rank " + Str(Dt(X)\Rank) + ".")
  EndIf
  If Dt(X)\D[1]<1
    ProcedureReturn DFail("Multinomial's input has no classes (class_size 0).")
  EndIf
  PmRandomDims(0)=Dt(X)\D[0]
  PmRandomDims(1)=Samples
  If DAlloc(Y,Kind,2,@PmRandomDims(0))=0 : ProcedureReturn 0 : EndIf
  If PmRandomUniformDraws(Dt(Y)\Data,Dt(Y)\Count,Node,HasSeed,SeedBits)=0
    ProcedureReturn DFail("A random tensor has more than 4294967295 elements; the generator numbers elements with 32 bits. Check the shape feeding this node.")
  EndIf
  ProcedureReturn 1
EndProcedure

; Comparison mode for Bernoulli and Multinomial: the caller supplied the
; node's tensor. It must have the node's element type Kind and, for
; Bernoulli (Samples 0), X's shape; for Multinomial [X's first extent, Samples].
Procedure DRandomFedKind(Y.i,X.i,Samples.i,Kind.i)
  Protected i.i, ok.i
  If Dt(Y)\Kind<>Kind
    DFail("A random node compiled as a model input (--random-inputs) was supplied with element type " + Str(Dt(Y)\Kind) + " where the node produces " + Str(Kind) + ". Supply the tensor with the node's element type.")
    ProcedureReturn
  EndIf
  ok=1
  If Samples=0
    If Dt(Y)\Rank<>Dt(X)\Rank
      ok=0
    Else
      For i=0 To Dt(X)\Rank-1
        If Dt(Y)\D[i]<>Dt(X)\D[i] : ok=0 : EndIf
      Next
    EndIf
  ElseIf Dt(Y)\Rank<>2 Or Dt(X)\Rank<>2
    ok=0
  ElseIf Dt(Y)\D[0]<>Dt(X)\D[0] Or Dt(Y)\D[1]<>Samples
    ok=0
  EndIf
  If ok=0
    DFail("A random node compiled as a model input (--random-inputs) was supplied with a shape other than the one the node produces. Supply the tensor with the node's shape.")
  EndIf
EndProcedure
