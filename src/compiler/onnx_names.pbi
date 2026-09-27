; ============================================================================
; onnx_names.pbi - the names refusals speak in
; ----------------------------------------------------------------------------
; A refusal names what it refuses: an element type by its ONNX name and code
; ("FLOAT16 (10)", not "type 10"), and an operator that no path implements
; because of the kind of value it works on by that kind, with what to do
; instead. Nothing here reaches emitted source; only the sentences change.
; ============================================================================

; The name of an ONNX element type (TensorProto.DataType, onnx 1.22.0).
Procedure.s PmoTypeName(ElementType.i)
  Select ElementType
    Case 1 : ProcedureReturn "FLOAT"
    Case 2 : ProcedureReturn "UINT8"
    Case 3 : ProcedureReturn "INT8"
    Case 4 : ProcedureReturn "UINT16"
    Case 5 : ProcedureReturn "INT16"
    Case 6 : ProcedureReturn "INT32"
    Case 7 : ProcedureReturn "INT64"
    Case 8 : ProcedureReturn "STRING"
    Case 9 : ProcedureReturn "BOOL"
    Case 10 : ProcedureReturn "FLOAT16"
    Case 11 : ProcedureReturn "DOUBLE"
    Case 12 : ProcedureReturn "UINT32"
    Case 13 : ProcedureReturn "UINT64"
    Case 14 : ProcedureReturn "COMPLEX64"
    Case 15 : ProcedureReturn "COMPLEX128"
    Case 16 : ProcedureReturn "BFLOAT16"
    Case 17 : ProcedureReturn "FLOAT8E4M3FN"
    Case 18 : ProcedureReturn "FLOAT8E4M3FNUZ"
    Case 19 : ProcedureReturn "FLOAT8E5M2"
    Case 20 : ProcedureReturn "FLOAT8E5M2FNUZ"
    Case 21 : ProcedureReturn "UINT4"
    Case 22 : ProcedureReturn "INT4"
    Case 23 : ProcedureReturn "FLOAT4E2M1"
    Case 24 : ProcedureReturn "FLOAT8E8M0"
    Case 25 : ProcedureReturn "UINT2"
    Case 26 : ProcedureReturn "INT2"
  EndSelect
  ProcedureReturn "element type " + Str(ElementType)
EndProcedure

; "NAME (code)" for sentences; a code ONNX does not define stays a number.
Procedure.s PmoTypeLabel(ElementType.i)
  If ElementType < 1 Or ElementType > 26 : ProcedureReturn PmoTypeName(ElementType) : EndIf
  ProcedureReturn PmoTypeName(ElementType) + " (" + Str(ElementType) + ")"
EndProcedure

; Why an operator that no path implements cannot be compiled, for the
; operators whose reason is the kind of value they work on; "" for every
; other operator. The sentence follows the node's label ("ImageDecoder node
; (unnamed) ...").
Procedure.s PmoUnsupportedReason(Operation.s)
  Select Operation
    Case "StringNormalizer", "StringConcat", "StringSplit", "RegexFullMatch"
      ProcedureReturn "works on STRING tensors, which this compiler does not carry: it compiles numbers and booleans. Turn the text into numbers (token ids, for example) before the model."
    Case "ImageDecoder"
      ProcedureReturn "decodes compressed image bytes (JPEG, PNG, BMP, TIFF, WebP and others), and this compiler carries no image codec. Decode the image before the model and feed it the pixels."
    Case "Optional", "OptionalHasElement", "OptionalGetElement"
      ProcedureReturn "works on optional values, which this compiler does not carry. Give the model a plain tensor or sequence in place of the optional value."
  EndSelect
  ProcedureReturn ""
EndProcedure
