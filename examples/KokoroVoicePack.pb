EnableExplicit

XIncludeFile "../src/compiler/kokoro_asset_pack.pbi"

Procedure Die(Message.s)
  PrintN("FAIL: " + Message)
  End 1
EndProcedure

OpenConsole()
If CountProgramParameters() < 2
  Die("usage: KokoroVoicePack [pack RAW OUTPUT [EXPECTED_SHA256] | verify PMVOICE]")
EndIf

Select LCase(ProgramParameter(0))
  Case "verify"
    If CountProgramParameters() <> 2 : Die("verify requires one PMVOICE path") : EndIf
    If PmoKokoroVoiceVerifyFile(ProgramParameter(1)) = 0 : Die(PmoKokoroVoicePackError) : EndIf
  Case "pack"
    If CountProgramParameters() < 3 Or CountProgramParameters() > 4 : Die("pack requires RAW, OUTPUT, and optional expected SHA-256") : EndIf
    Define expected.s
    If CountProgramParameters() = 4 : expected = ProgramParameter(3) : EndIf
    If PmoKokoroVoicePack(ProgramParameter(1), ProgramParameter(2), expected) = 0 : Die(PmoKokoroVoicePackError) : EndIf
  Default
    Die("unknown operation")
EndSelect

PrintN("PASS")
End 0
