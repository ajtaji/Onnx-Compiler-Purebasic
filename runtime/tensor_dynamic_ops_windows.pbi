; ======================================================================
; tensor_dynamic_ops_windows.pbi - the Windows adapter of
; tensor_dynamic_ops.pmi: how a runtime-dimension tensor's extent is read
; in this dialect. Include tensor_dynamic_windows.pbi and tensor_ops.pmi
; first, tensor_dynamic_ops.pmi after.
; ======================================================================
Procedure.i DOpDim(Id.i, Axis.i)
  ProcedureReturn Dt(Id)\D[Axis]
EndProcedure
