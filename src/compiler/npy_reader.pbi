; ============================================================================
; npy_reader.pbi - bounded NumPy .npy tensor reader for native shape tracing
; ----------------------------------------------------------------------------
; Supports the plain C-order numeric arrays accepted as ONNX graph inputs.
; Object arrays, big-endian payloads, structured dtypes and Fortran order are
; rejected rather than silently reinterpreted.
; ============================================================================

Structure PmoNpyArray
  Path.s
  Descriptor.s
  ElementType.i
  ElementBytes.i
  *Storage
  StorageBytes.q
  *Data
  DataBytes.q
  List Dims.q()
EndStructure

Global PmoNpyError.s

Procedure.i PmoNpyFail(Message.s)
  If PmoNpyError = "" : PmoNpyError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure PmoNpyFree(*Array.PmoNpyArray)
  If *Array = 0 : ProcedureReturn : EndIf
  If *Array\Storage : FreeMemory(*Array\Storage) : EndIf
  ClearStructure(*Array, PmoNpyArray)
  InitializeStructure(*Array, PmoNpyArray)
EndProcedure

Procedure.s PmoNpyHeaderString(Header.s, Key.s)
  Protected At.i = FindString(Header, "'" + Key + "'")
  Protected Colon.i
  Protected Quote.i
  Protected Finish.i
  If At = 0 : At = FindString(Header, Chr(34) + Key + Chr(34)) : EndIf
  If At = 0 : ProcedureReturn "" : EndIf
  Colon = FindString(Header, ":", At)
  If Colon = 0 : ProcedureReturn "" : EndIf
  For Quote = Colon + 1 To Len(Header)
    If Mid(Header, Quote, 1) = "'" Or Mid(Header, Quote, 1) = Chr(34) : Break : EndIf
  Next
  If Quote > Len(Header) : ProcedureReturn "" : EndIf
  Finish = FindString(Header, Mid(Header, Quote, 1), Quote + 1)
  If Finish = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn Mid(Header, Quote + 1, Finish - Quote - 1)
EndProcedure

Procedure.s PmoNpyHeaderValue(Header.s, Key.s)
  Protected At.i = FindString(Header, "'" + Key + "'")
  Protected Colon.i
  Protected Finish.i
  If At = 0 : At = FindString(Header, Chr(34) + Key + Chr(34)) : EndIf
  If At = 0 : ProcedureReturn "" : EndIf
  Colon = FindString(Header, ":", At)
  If Colon = 0 : ProcedureReturn "" : EndIf
  Finish = FindString(Header, ",", Colon + 1)
  If Finish = 0 : Finish = Len(Header) + 1 : EndIf
  ProcedureReturn Trim(Mid(Header, Colon + 1, Finish - Colon - 1))
EndProcedure

Procedure.s PmoNpyShapeText(Header.s)
  Protected At.i = FindString(Header, "'shape'")
  Protected OpenAt.i
  Protected CloseAt.i
  If At = 0 : At = FindString(Header, Chr(34) + "shape" + Chr(34)) : EndIf
  If At = 0 : ProcedureReturn "" : EndIf
  OpenAt = FindString(Header, "(", At)
  If OpenAt = 0 : ProcedureReturn "" : EndIf
  CloseAt = FindString(Header, ")", OpenAt + 1)
  If CloseAt = 0 : ProcedureReturn "" : EndIf
  ProcedureReturn Mid(Header, OpenAt + 1, CloseAt - OpenAt - 1)
EndProcedure

Procedure.i PmoNpySetType(*Array.PmoNpyArray, Descriptor.s)
  Protected Kind.s = Descriptor
  If Left(Kind, 1) = ">"
    ProcedureReturn PmoNpyFail("big-endian NumPy arrays are not supported")
  EndIf
  If Left(Kind, 1) = "<" Or Left(Kind, 1) = "=" Or Left(Kind, 1) = "|"
    Kind = Mid(Kind, 2)
  EndIf
  Select Kind
    Case "f4" : *Array\ElementType = 1  : *Array\ElementBytes = 4
    Case "u1" : *Array\ElementType = 2  : *Array\ElementBytes = 1
    Case "i1" : *Array\ElementType = 3  : *Array\ElementBytes = 1
    Case "u2" : *Array\ElementType = 4  : *Array\ElementBytes = 2
    Case "i2" : *Array\ElementType = 5  : *Array\ElementBytes = 2
    Case "i4" : *Array\ElementType = 6  : *Array\ElementBytes = 4
    Case "i8" : *Array\ElementType = 7  : *Array\ElementBytes = 8
    Case "b1" : *Array\ElementType = 9  : *Array\ElementBytes = 1
    Case "f8" : *Array\ElementType = 11 : *Array\ElementBytes = 8
    Case "u4" : *Array\ElementType = 12 : *Array\ElementBytes = 4
    Case "u8" : *Array\ElementType = 13 : *Array\ElementBytes = 8
    Default
      ProcedureReturn PmoNpyFail("unsupported NumPy dtype " + Descriptor)
  EndSelect
  ProcedureReturn #True
EndProcedure

Procedure.i PmoNpyLoad(Path.s, *Array.PmoNpyArray)
  Protected File.i
  Protected Bytes.q
  Protected ReadBytes.q
  Protected Major.i
  Protected Minor.i
  Protected HeaderBytes.q
  Protected HeaderOffset.i
  Protected Header.s
  Protected Shape.s
  Protected Piece.s
  Protected Count.i
  Protected Index.i
  Protected Extent.q
  Protected Elements.q = 1
  If *Array = 0 : ProcedureReturn PmoNpyFail("internal NumPy destination is null") : EndIf
  PmoNpyFree(*Array)
  PmoNpyError = ""
  Bytes = FileSize(Path)
  If Bytes < 10 : ProcedureReturn PmoNpyFail("NumPy input is missing or too short: " + Path) : EndIf
  *Array\Storage = AllocateMemory(Bytes)
  If *Array\Storage = 0 : ProcedureReturn PmoNpyFail("cannot allocate NumPy input") : EndIf
  *Array\StorageBytes = Bytes
  File = ReadFile(#PB_Any, Path, #PB_File_SharedRead)
  If File = 0 : PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("cannot open NumPy input: " + Path) : EndIf
  ReadBytes = ReadData(File, *Array\Storage, Bytes)
  CloseFile(File)
  If ReadBytes <> Bytes : PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("short read in NumPy input") : EndIf
  If PeekA(*Array\Storage) <> $93 Or PeekS(*Array\Storage + 1, 5, #PB_Ascii) <> "NUMPY"
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("NumPy magic is invalid")
  EndIf
  Major = PeekA(*Array\Storage + 6) & $FF
  Minor = PeekA(*Array\Storage + 7) & $FF
  If Major = 1
    HeaderBytes = PeekU(*Array\Storage + 8) & $FFFF
    HeaderOffset = 10
  ElseIf Major = 2 Or Major = 3
    HeaderBytes = PeekL(*Array\Storage + 8) & $FFFFFFFF
    HeaderOffset = 12
  Else
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("unsupported NumPy file version " + Str(Major) + "." + Str(Minor))
  EndIf
  If HeaderBytes < 2 Or HeaderOffset + HeaderBytes > Bytes
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("NumPy header exceeds the file")
  EndIf
  Header = PeekS(*Array\Storage + HeaderOffset, HeaderBytes, #PB_Ascii)
  *Array\Descriptor = PmoNpyHeaderString(Header, "descr")
  If *Array\Descriptor = "" Or PmoNpySetType(*Array, *Array\Descriptor) = 0
    PmoNpyFree(*Array) : ProcedureReturn #False
  EndIf
  If LCase(PmoNpyHeaderValue(Header, "fortran_order")) <> "false"
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("Fortran-order NumPy arrays are not supported")
  EndIf
  Shape = PmoNpyShapeText(Header)
  If Shape = "" And FindString(Header, "()") = 0
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("NumPy shape tuple is missing")
  EndIf
  If Trim(Shape) <> ""
    Count = CountString(Shape, ",") + 1
    For Index = 1 To Count
      Piece = Trim(StringField(Shape, Index, ","))
      If Piece = "" : Continue : EndIf
      If Left(Piece, 1) = "+" : Piece = Mid(Piece, 2) : EndIf
      If Piece = "" Or (Val(Piece) = 0 And Piece <> "0")
        PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("invalid NumPy shape extent")
      EndIf
      Extent = Val(Piece)
      If Extent < 0
        PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("negative NumPy shape extent")
      EndIf
      If Extent > 0 And Elements > $7FFFFFFFFFFFFFFF / Extent
        PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("NumPy element count overflows 64 bits")
      EndIf
      Elements * Extent
      AddElement(*Array\Dims()) : *Array\Dims() = Extent
    Next
  EndIf
  *Array\Data = *Array\Storage + HeaderOffset + HeaderBytes
  *Array\DataBytes = Bytes - HeaderOffset - HeaderBytes
  If Elements > $7FFFFFFFFFFFFFFF / *Array\ElementBytes Or
     Elements * *Array\ElementBytes <> *Array\DataBytes
    PmoNpyFree(*Array) : ProcedureReturn PmoNpyFail("NumPy payload size does not match dtype and shape")
  EndIf
  *Array\Path = GetPathPart(Path) + GetFilePart(Path)
  ProcedureReturn #True
EndProcedure
