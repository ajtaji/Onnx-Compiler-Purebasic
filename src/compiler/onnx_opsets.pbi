; ============================================================================
; onnx_opsets.pbi - the ai.onnx opsets this compiler accepts, and each
; operator's ceiling
; ----------------------------------------------------------------------------
; A model imports one ai.onnx opset N. For each operator, the definition in
; effect is its newest version whose since_version is N or less
; (https://onnx.ai/onnx/operators/, each operator's version history). An
; operator's FLOOR (the oldest version whose definition is the one a kernel
; computes) lives with its lane (PmcOperatorFloor, PmoNsFloor, PmoOpsFloor).
; Its CEILING is here: the first version above the one implemented whose
; definition differs in more than the element types it admits.
;
; Opsets 21 to 27 were audited against onnx 1.22.0's schema history,
; operator by operator: attributes, inputs, outputs, arity and the text of
; every version above 20. Most of those versions only add element types
; (bfloat16, float16, int4, uint4, the float8 and float4 formats, int2,
; uint2), which every lane's element-type checks already refuse, or an
; attribute whose absence keeps the older behaviour, which the attribute
; checks refuse when present (DequantizeLinear-21 block_size, -23
; output_dtype; QuantizeLinear-21 block_size and output_dtype, -23
; precision; Range-27 stash_type), or one that acts only on a type this
; compiler does not carry (Cast-24 and CastLike-24 round_mode, for float8).
; Those keep the ceiling at the highest accepted opset. The others are
; listed below with the version that changed them. The gate
; opset_ceiling_check.py in the compiler repository's tools/onnx/tests/
; Diagnostics holds this table to onnx.defs, so a new ONNX release that
; changes an operator cannot be accepted by accident.
; ============================================================================

#PMO_OPSET_MAX = 27

; The version of Operation whose definition differs from the one implemented
; (the model's opset must stay below it), or 0 when every version up to
; #PMO_OPSET_MAX is the one implemented.
Procedure.i PmoOpsetChangedAt(Operation.s)
  Select Operation
    Case "GroupNormalization" : ProcedureReturn 21 ; scale and bias per channel, not per group
  EndSelect
  ProcedureReturn 0
EndProcedure

Procedure.s PmoOpsetChangeText(Operation.s)
  Select Operation
    Case "GroupNormalization" : ProcedureReturn "scales and shifts per channel, not per group"
  EndSelect
  ProcedureReturn "changed its definition"
EndProcedure

; "" when Operation, at the model's opset, is the definition implemented;
; otherwise the sentence that refuses the node.
Procedure.s PmoOpsetCeilingRefusal(Operation.s, Opset.i)
  Protected Changed.i = PmoOpsetChangedAt(Operation)
  If Opset > #PMO_OPSET_MAX
    ProcedureReturn "the model imports ai.onnx opset " + Str(Opset) + "; this compiler implements operator definitions through opset " + Str(#PMO_OPSET_MAX) + "."
  EndIf
  If Changed And Opset >= Changed
    ProcedureReturn "the model imports ai.onnx opset " + Str(Opset) + ", where " + Operation + " is " + Operation + "-" + Str(Changed) + " or later, which " +
                    PmoOpsetChangeText(Operation) + "; this compiler implements " + Operation + " through opset " + Str(Changed - 1) + "."
  EndIf
  ProcedureReturn ""
EndProcedure
