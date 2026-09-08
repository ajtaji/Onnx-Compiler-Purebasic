; Local voice discovery for the generated resident reader. Data only; no build
; or download path. Validate the selected PMVOICE through KsAssets before use.
Structure KsVoiceEntry
  Path.s
  Label.s
EndStructure
Global NewList KsVoices.KsVoiceEntry()

Procedure.s KsVoiceLabel(Path.s)
  Protected name.s=GetFilePart(Path,#PB_FileSystem_NoExtension),prefix.s,label.s
  prefix=LCase(Left(name,3))
  Select prefix
    Case "af_" : label="American - Female"
    Case "am_" : label="American - Male"
    Case "bf_" : label="British - Female (US pronunciation)"
    Case "bm_" : label="British - Male (US pronunciation)"
  EndSelect
  If label<>""
    name=Mid(name,4)
    name=UCase(Left(name,1))+Mid(name,2)
    ProcedureReturn name+" - "+label
  EndIf
  If LCase(name)="voice" : ProcedureReturn "Default voice" : EndIf
  ProcedureReturn name+" - Custom voice"
EndProcedure

Procedure.i KsVoiceAdd(Path.s)
  Protected index.i
  If Path="" : ProcedureReturn -1 : EndIf
  ForEach KsVoices()
    If LCase(KsVoices()\Path)=LCase(Path) : ProcedureReturn index : EndIf
    index+1
  Next
  If AddElement(KsVoices())=0 : ProcedureReturn -1 : EndIf
  KsVoices()\Path=Path : KsVoices()\Label=KsVoiceLabel(Path)
  If FileSize(Path)<0 : KsVoices()\Label+" [missing file]" : EndIf
  ProcedureReturn index
EndProcedure

Procedure.s KsVoicePathAt(Index.i)
  If Index<0 Or Index>=ListSize(KsVoices()) : ProcedureReturn "" : EndIf
  If SelectElement(KsVoices(),Index) : ProcedureReturn KsVoices()\Path : EndIf
  ProcedureReturn ""
EndProcedure

; Keep ordering deterministic. Include a saved custom/missing path so startup
; never silently changes a voice. Missing selections get the normal load error.
Procedure.i KsVoiceDiscover(Folder.s,Selected.s)
  Protected dir.i,path.s
  NewList paths.s()
  ClearList(KsVoices())
  dir=ExamineDirectory(#PB_Any,Folder,"*.pmvoice")
  If dir
    While NextDirectoryEntry(dir)
      If DirectoryEntryType(dir)=#PB_DirectoryEntry_File
        AddElement(paths()) : paths()=Folder+DirectoryEntryName(dir)
      EndIf
    Wend
    FinishDirectory(dir)
  EndIf
  SortList(paths(),#PB_Sort_Ascending|#PB_Sort_NoCase)
  ForEach paths() : KsVoiceAdd(paths()) : Next
  ProcedureReturn KsVoiceAdd(Selected)
EndProcedure
