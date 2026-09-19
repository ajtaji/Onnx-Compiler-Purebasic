; ============================================================================
; kokoro_dictionary_pack.pbi - native construction and verification of the
; checked Kokoro PMG2P pronunciation packs.
; ----------------------------------------------------------------------------
; The caller supplies the immutable upstream dictionary sources. This module
; performs no download, embeds no dictionary data, and contains no model or
; voice data. It builds two packs:
;
;   * the BASE pack "PMG2PUS\0", from the pinned Misaki US-English gold and
;     silver JSON lexicons;
;   * the EXTRA pack "PMG2PX\0\0", from the pinned CMUdict text, excluding
;     every word the base pack already answers directly or by regular
;     inflection.
;
; Both packs are byte-for-byte deterministic for a given pinned source set.
;
; ============================================================================
; PMG2P FORMAT SPECIFICATION
; ----------------------------------------------------------------------------
; A pack is a 256-byte fixed header followed by one payload. Every scalar is
; little-endian. Offsets in the header are absolute file offsets; offsets
; inside a slot record are relative to the start of their own blob.
;
; HEADER (256 bytes)
;   off  size  field
;     0     8  magic; "PMG2PUS\0" for base, "PMG2PX\0\0" for extra (NUL padded)
;     8     4  format version, always 1
;    12     4  header bytes, always 256
;    16     8  total file bytes
;    24     4  slot count; a power of two, the smallest one >= 2 * entries
;    28     4  entry count; the number of occupied slots
;    32     8  slot table offset, always 256
;    40     8  key blob offset = 256 + slot count * 16
;    48     8  key blob bytes
;    56     8  phoneme blob offset = key blob offset + key blob bytes
;    64     8  phoneme blob bytes
;    72     4  CRC-32 of the whole payload (slot table + key blob + phonemes)
;    76     4  zero pad
;    80    32  primary source SHA-256, raw bytes: Misaki gold (base) or
;              cmudict.dict (extra)
;   112    32  Misaki silver SHA-256 raw bytes for the base pack; ALL ZERO in
;              the extra pack, which has a single source
;   144    32  Kokoro vocabulary config.json SHA-256, raw bytes
;   176    40  upstream revision, ASCII, NUL padded (a 40-character git SHA-1)
;   216    40  reserved, all zero
;
; PAYLOAD
;   slot table   slot count records of 16 bytes, at file offset 256
;   key blob     the concatenated ASCII keys in canonical order
;   phoneme blob the concatenated UTF-8 pronunciations in canonical order
;
; SLOT RECORD (16 bytes)
;   off  size  field
;     0     4  FNV-1a hash of the key bytes; 0 means the slot is empty
;     4     4  key offset within the key blob
;     8     4  phoneme offset within the phoneme blob
;    12     2  key bytes, 1..63
;    14     2  phoneme bytes, >= 1
;   An empty slot is 16 zero bytes. No other field may be nonzero when the
;   hash is zero.
;
; HASHING
;   32-bit FNV-1a over the raw ASCII key bytes: value = 2166136261; for each
;   byte, value = ((value XOR byte) * 16777619) mod 2^32. A result of 0 is
;   remapped to 1 so that zero can mean "empty slot". Placement is linear
;   probing from (hash AND (slot count - 1)), stepping +1 with wraparound;
;   a lookup stops at the first empty slot.
;
; CANONICAL ORDER
;   Entries are sorted by their raw ASCII key bytes: unsigned byte compare,
;   and where one key is a prefix of the other, the shorter key sorts first.
;   Keys and pronunciations are appended to their blobs in exactly that order,
;   and slots are filled by probing in exactly that order, so the probe chains
;   and therefore every byte of the file are fully determined.
;
; STRING ENCODING
;   Keys are ASCII, 1..63 bytes, stored without a terminator.
;   Pronunciations are UTF-8, stored without a terminator, and every codepoint
;   must appear in the pinned 114-codepoint Kokoro vocabulary.
;
; PADDING AND CHECKSUM
;   There is no padding or alignment anywhere except the header's two reserved
;   runs (bytes 76..79 and 216..255), which are zero. The payload is a single
;   contiguous run and its CRC-32 (the standard reflected polynomial
;   $EDB88320, initial and final value $FFFFFFFF) is stored at header offset
;   72. There is no trailing checksum after the payload.
;
; BASE PACK SOURCE RULES
;   The gold and silver lexicons are JSON objects mapping a word to either a
;   pronunciation string or an object of part-of-speech variants. For an
;   object, the "DEFAULT" member is used when it is a string, otherwise the
;   first string-valued member in document order; if neither exists the word
;   is skipped. A word is also skipped when it is not ASCII, when it exceeds
;   63 bytes, or when its pronunciation is empty. Gold wins over silver: a
;   silver word is kept only when gold does not define it.
;
; EXTRA PACK SOURCE RULES
;   Each cmudict.dict line is truncated at the first "#", then split on
;   whitespace. The first field is the word and must match [a-z][a-z']{1,62}
;   exactly; anything else, including the "word(2)" variant spellings, is
;   skipped. A word is skipped when the base pack already answers it by exact
;   or lowercase lookup, or by the regular inflection rules implemented in
;   PmoKdpReaderInflect. The remaining ARPABET phone sequence is converted to
;   the Kokoro phoneme alphabet, with primary stress written U+02C8 and
;   secondary stress U+02CC before the vowel; unstressed AH becomes U+0259 and
;   unstressed ER becomes U+0259 U+0279. A fixed, reviewed list of technical
;   terms is added afterwards, again only where the base pack is silent.
; ============================================================================

XIncludeFile "kokoro_asset_pack.pbi"

#PMO_KDP_HEADER_BYTES = 256
#PMO_KDP_SLOT_BYTES = 16
#PMO_KDP_MAX_KEY_BYTES = 63
#PMO_KDP_KIND_BASE = 0
#PMO_KDP_KIND_EXTRA = 1
#PMO_KDP_VOCAB_COUNT = 114
#PMO_KDP_MAX_JSON_DEPTH = 64

#PMO_KDP_GOLD_SHA256$ = "a13911134e702c8fd1d79ea47cdc64f146aec1deabf4638a9a8cdd71da5b87a3"
#PMO_KDP_SILVER_SHA256$ = "a7d8629ff02614cd37837f44c41f0085f55bed4f0eb2c169964f0077bb102d6b"
#PMO_KDP_MISAKI_REVISION$ = "fba1236595f2d2bf21d414ba6e57d25256afada3"
#PMO_KDP_VOCAB_SHA256$ = "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"
#PMO_KDP_CMUDICT_SHA256$ = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
#PMO_KDP_CMUDICT_REVISION$ = "74790861f652b15e4ac49015a90074ad62a27690"

; The pinned deterministic contract. A self-consistent pack built from other
; sources is refused rather than quietly accepted as equivalent.
#PMO_KDP_BASE_FILE_BYTES = 12613954
#PMO_KDP_BASE_SLOTS = 524288
#PMO_KDP_BASE_ENTRIES = 183561
#PMO_KDP_BASE_KEY_BYTES = 1624349
#PMO_KDP_BASE_PHONEME_BYTES = 2600741
#PMO_KDP_BASE_PAYLOAD_CRC32 = $F0BE2D51
#PMO_KDP_EXTRA_FILE_BYTES = 5397245
#PMO_KDP_EXTRA_SLOTS = 262144
#PMO_KDP_EXTRA_ENTRIES = 66305
#PMO_KDP_EXTRA_KEY_BYTES = 468193
#PMO_KDP_EXTRA_PHONEME_BYTES = 734492
#PMO_KDP_EXTRA_PAYLOAD_CRC32 = $2641725D

Global PmoKokoroDictionaryPackError.s
Global PmoKokoroDictionaryReport.s

Structure PmoKdpArena
  *Data
  Bytes.q
  Capacity.q
EndStructure

Structure PmoKdpEntry
  KeyOffset.q
  PhonemeOffset.q
  KeyBytes.l
  PhonemeBytes.l
EndStructure

Structure PmoKdpTable
  Keys.PmoKdpArena
  Phonemes.PmoKdpArena
  *Entries
  Count.q
  Capacity.q
  *Index
  IndexSlots.q
EndStructure

Structure PmoKdpJson
  *Data
  Bytes.q
  Pos.q
  Source.s
EndStructure

Structure PmoKdpReader
  *Data
  Bytes.q
  Slots.q
  SlotOffset.q
  KeyOffset.q
  KeyBytes.q
  PhonemeOffset.q
  PhonemeBytes.q
EndStructure

; Stack-resident ASCII scratch. PureBasic string literals are UTF-16, so a
; literal's address is never a usable byte key; everything that needs bytes
; goes through one of these.
Structure PmoKdpWord
  Byte.a[80]
EndStructure

Structure PmoKdpCandidates
  Length.l[8]
  Tail.l[8]
EndStructure

Global Dim PmoKdpVocabCodepoint.l(#PMO_KDP_VOCAB_COUNT - 1)
Global Dim PmoKdpArpaName.s(38)
Global Dim PmoKdpArpaPhoneme.s(38)
Global PmoKdpTablesReady.i

Procedure.i PmoKdpFail(Message.s)
  If PmoKokoroDictionaryPackError = "" : PmoKokoroDictionaryPackError = Message : EndIf
  ProcedureReturn #False
EndProcedure

Procedure.s PmoKdpCode(Code.i)
  ProcedureReturn "PMG2P-" + RSet(Str(Code), 4, "0") + ": "
EndProcedure

; Writes Text as raw ASCII bytes and returns the byte count.
Procedure.i PmoKdpPutAscii(*Buffer, Text.s)
  Protected i.i
  Protected count.i = Len(Text)
  For i = 1 To count
    PokeA(*Buffer + i - 1, Asc(Mid(Text, i, 1)) & $FF)
  Next
  ProcedureReturn count
EndProcedure

; ---------------------------------------------------------------------------
; Pinned lookup tables. Built at first use so this file stays pure ASCII and
; carries no source-encoding dependency.
; ---------------------------------------------------------------------------
Procedure PmoKdpInitTables()
  Protected i.i
  If PmoKdpTablesReady : ProcedureReturn : EndIf
  Restore PmoKdpVocabData
  For i = 0 To #PMO_KDP_VOCAB_COUNT - 1
    Read.l PmoKdpVocabCodepoint(i)
  Next
  PmoKdpArpaName(0)  = "AA" : PmoKdpArpaPhoneme(0)  = Chr($0251)
  PmoKdpArpaName(1)  = "AE" : PmoKdpArpaPhoneme(1)  = Chr($00E6)
  PmoKdpArpaName(2)  = "AH" : PmoKdpArpaPhoneme(2)  = Chr($028C)
  PmoKdpArpaName(3)  = "AO" : PmoKdpArpaPhoneme(3)  = Chr($0254)
  PmoKdpArpaName(4)  = "AW" : PmoKdpArpaPhoneme(4)  = "W"
  PmoKdpArpaName(5)  = "AY" : PmoKdpArpaPhoneme(5)  = "I"
  PmoKdpArpaName(6)  = "B"  : PmoKdpArpaPhoneme(6)  = "b"
  PmoKdpArpaName(7)  = "CH" : PmoKdpArpaPhoneme(7)  = Chr($02A7)
  PmoKdpArpaName(8)  = "D"  : PmoKdpArpaPhoneme(8)  = "d"
  PmoKdpArpaName(9)  = "DH" : PmoKdpArpaPhoneme(9)  = Chr($00F0)
  PmoKdpArpaName(10) = "EH" : PmoKdpArpaPhoneme(10) = Chr($025B)
  PmoKdpArpaName(11) = "ER" : PmoKdpArpaPhoneme(11) = Chr($025C) + Chr($0279)
  PmoKdpArpaName(12) = "EY" : PmoKdpArpaPhoneme(12) = "A"
  PmoKdpArpaName(13) = "F"  : PmoKdpArpaPhoneme(13) = "f"
  PmoKdpArpaName(14) = "G"  : PmoKdpArpaPhoneme(14) = Chr($0261)
  PmoKdpArpaName(15) = "HH" : PmoKdpArpaPhoneme(15) = "h"
  PmoKdpArpaName(16) = "IH" : PmoKdpArpaPhoneme(16) = Chr($026A)
  PmoKdpArpaName(17) = "IY" : PmoKdpArpaPhoneme(17) = "i"
  PmoKdpArpaName(18) = "JH" : PmoKdpArpaPhoneme(18) = Chr($02A4)
  PmoKdpArpaName(19) = "K"  : PmoKdpArpaPhoneme(19) = "k"
  PmoKdpArpaName(20) = "L"  : PmoKdpArpaPhoneme(20) = "l"
  PmoKdpArpaName(21) = "M"  : PmoKdpArpaPhoneme(21) = "m"
  PmoKdpArpaName(22) = "N"  : PmoKdpArpaPhoneme(22) = "n"
  PmoKdpArpaName(23) = "NG" : PmoKdpArpaPhoneme(23) = Chr($014B)
  PmoKdpArpaName(24) = "OW" : PmoKdpArpaPhoneme(24) = "O"
  PmoKdpArpaName(25) = "OY" : PmoKdpArpaPhoneme(25) = "Y"
  PmoKdpArpaName(26) = "P"  : PmoKdpArpaPhoneme(26) = "p"
  PmoKdpArpaName(27) = "R"  : PmoKdpArpaPhoneme(27) = Chr($0279)
  PmoKdpArpaName(28) = "S"  : PmoKdpArpaPhoneme(28) = "s"
  PmoKdpArpaName(29) = "SH" : PmoKdpArpaPhoneme(29) = Chr($0283)
  PmoKdpArpaName(30) = "T"  : PmoKdpArpaPhoneme(30) = "t"
  PmoKdpArpaName(31) = "TH" : PmoKdpArpaPhoneme(31) = Chr($03B8)
  PmoKdpArpaName(32) = "UH" : PmoKdpArpaPhoneme(32) = Chr($028A)
  PmoKdpArpaName(33) = "UW" : PmoKdpArpaPhoneme(33) = "u"
  PmoKdpArpaName(34) = "V"  : PmoKdpArpaPhoneme(34) = "v"
  PmoKdpArpaName(35) = "W"  : PmoKdpArpaPhoneme(35) = "w"
  PmoKdpArpaName(36) = "Y"  : PmoKdpArpaPhoneme(36) = "j"
  PmoKdpArpaName(37) = "Z"  : PmoKdpArpaPhoneme(37) = "z"
  PmoKdpArpaName(38) = "ZH" : PmoKdpArpaPhoneme(38) = Chr($0292)
  PmoKdpTablesReady = #True
EndProcedure

Procedure.i PmoKdpVocabContains(Codepoint.i)
  Protected low.i, high.i, middle.i
  PmoKdpInitTables()
  high = #PMO_KDP_VOCAB_COUNT - 1
  While low <= high
    middle = (low + high) >> 1
    If PmoKdpVocabCodepoint(middle) = Codepoint
      ProcedureReturn #True
    ElseIf PmoKdpVocabCodepoint(middle) < Codepoint
      low = middle + 1
    Else
      high = middle - 1
    EndIf
  Wend
  ProcedureReturn #False
EndProcedure

; ---------------------------------------------------------------------------
; FNV-1a, exactly as the format specifies, including the 0 -> 1 remap.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpFnv1aRaw(*Data, Bytes.i)
  Protected i.i
  Protected value.i = 2166136261
  While i < Bytes
    value = ((value ! (PeekA(*Data + i) & $FF)) * 16777619) & $FFFFFFFF
    i + 1
  Wend
  ProcedureReturn value
EndProcedure

Procedure.i PmoKdpFnv1a(*Data, Bytes.i)
  Protected value.i = PmoKdpFnv1aRaw(*Data, Bytes)
  If value = 0 : ProcedureReturn 1 : EndIf
  ProcedureReturn value
EndProcedure

; ---------------------------------------------------------------------------
; Growable byte arenas.
; ---------------------------------------------------------------------------
Procedure PmoKdpArenaFree(*Arena.PmoKdpArena)
  If *Arena\Data : FreeMemory(*Arena\Data) : EndIf
  *Arena\Data = 0 : *Arena\Bytes = 0 : *Arena\Capacity = 0
EndProcedure

Procedure.i PmoKdpArenaReserve(*Arena.PmoKdpArena, Extra.q)
  Protected wanted.q, capacity.q
  Protected *grown
  wanted = *Arena\Bytes + Extra
  If wanted <= *Arena\Capacity : ProcedureReturn #True : EndIf
  capacity = *Arena\Capacity
  If capacity < 65536 : capacity = 65536 : EndIf
  While capacity < wanted
    capacity = capacity * 2
  Wend
  If *Arena\Data = 0
    *grown = AllocateMemory(capacity)
  Else
    *grown = ReAllocateMemory(*Arena\Data, capacity)
  EndIf
  If *grown = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2001) + "the dictionary builder could not grow a working buffer to " + Str(capacity) + " bytes, so the pack cannot be assembled. Check the free memory available to this 64-bit process before retrying.")
  EndIf
  *Arena\Data = *grown
  *Arena\Capacity = capacity
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpArenaPut(*Arena.PmoKdpArena, *Source, Bytes.q)
  If Bytes < 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2002) + "the dictionary builder was asked to append a negative number of bytes, which is an internal fault. Check the caller that produced the length.")
  EndIf
  If Bytes = 0 : ProcedureReturn #True : EndIf
  If PmoKdpArenaReserve(*Arena, Bytes) = 0 : ProcedureReturn #False : EndIf
  CopyMemory(*Source, *Arena\Data + *Arena\Bytes, Bytes)
  *Arena\Bytes = *Arena\Bytes + Bytes
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpArenaPutByte(*Arena.PmoKdpArena, Value.i)
  If PmoKdpArenaReserve(*Arena, 1) = 0 : ProcedureReturn #False : EndIf
  PokeA(*Arena\Data + *Arena\Bytes, Value & $FF)
  *Arena\Bytes = *Arena\Bytes + 1
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpArenaPutUtf8(*Arena.PmoKdpArena, Text.s)
  Protected bytes.i = StringByteLength(Text, #PB_UTF8)
  If bytes <= 0 : ProcedureReturn #True : EndIf
  If PmoKdpArenaReserve(*Arena, bytes + 1) = 0 : ProcedureReturn #False : EndIf
  PokeS(*Arena\Data + *Arena\Bytes, Text, -1, #PB_UTF8)
  *Arena\Bytes = *Arena\Bytes + bytes
  ProcedureReturn #True
EndProcedure

; Compares a raw UTF-8 byte run with the encoding of Text. PeekS() counts
; characters rather than bytes and stops at a NUL, so it can never be used to
; read a pronunciation out of a blob that has neither.
Procedure.i PmoKdpEqualsUtf8(*Data, Bytes.q, Text.s)
  Protected expected.PmoKdpArena
  Protected result.i = #False
  If PmoKdpArenaPutUtf8(@expected, Text)
    If expected\Bytes = Bytes
      If Bytes = 0 Or CompareMemory(*Data, expected\Data, Bytes) : result = #True : EndIf
    EndIf
  EndIf
  PmoKdpArenaFree(@expected)
  ProcedureReturn result
EndProcedure

; Appends one Unicode codepoint as UTF-8. Surrogates and values above
; U+10FFFF never reach here; the decoders refuse them first.
Procedure.i PmoKdpArenaPutCodepoint(*Arena.PmoKdpArena, Codepoint.i)
  If Codepoint < $80
    ProcedureReturn PmoKdpArenaPutByte(*Arena, Codepoint)
  ElseIf Codepoint < $800
    If PmoKdpArenaPutByte(*Arena, $C0 | (Codepoint >> 6)) = 0 : ProcedureReturn #False : EndIf
    ProcedureReturn PmoKdpArenaPutByte(*Arena, $80 | (Codepoint & $3F))
  ElseIf Codepoint < $10000
    If PmoKdpArenaPutByte(*Arena, $E0 | (Codepoint >> 12)) = 0 : ProcedureReturn #False : EndIf
    If PmoKdpArenaPutByte(*Arena, $80 | ((Codepoint >> 6) & $3F)) = 0 : ProcedureReturn #False : EndIf
    ProcedureReturn PmoKdpArenaPutByte(*Arena, $80 | (Codepoint & $3F))
  Else
    If PmoKdpArenaPutByte(*Arena, $F0 | (Codepoint >> 18)) = 0 : ProcedureReturn #False : EndIf
    If PmoKdpArenaPutByte(*Arena, $80 | ((Codepoint >> 12) & $3F)) = 0 : ProcedureReturn #False : EndIf
    If PmoKdpArenaPutByte(*Arena, $80 | ((Codepoint >> 6) & $3F)) = 0 : ProcedureReturn #False : EndIf
    ProcedureReturn PmoKdpArenaPutByte(*Arena, $80 | (Codepoint & $3F))
  EndIf
EndProcedure

; Decodes one UTF-8 sequence. Returns the codepoint, or -1 for any malformed,
; overlong, surrogate or out-of-range sequence. *NextIndex receives the offset
; of the following sequence.
Procedure.i PmoKdpUtf8Next(*Data, Index.q, Bytes.q, *NextIndex.Integer)
  Protected b0.i, b1.i, b2.i, b3.i, value.i
  If Index < 0 Or Index >= Bytes : ProcedureReturn -1 : EndIf
  b0 = PeekA(*Data + Index) & $FF
  If b0 < $80
    *NextIndex\i = Index + 1
    ProcedureReturn b0
  EndIf
  If b0 < $C2 : ProcedureReturn -1 : EndIf
  If b0 < $E0
    If Index + 1 >= Bytes : ProcedureReturn -1 : EndIf
    b1 = PeekA(*Data + Index + 1) & $FF
    If (b1 & $C0) <> $80 : ProcedureReturn -1 : EndIf
    *NextIndex\i = Index + 2
    ProcedureReturn ((b0 & $1F) << 6) | (b1 & $3F)
  EndIf
  If b0 < $F0
    If Index + 2 >= Bytes : ProcedureReturn -1 : EndIf
    b1 = PeekA(*Data + Index + 1) & $FF
    b2 = PeekA(*Data + Index + 2) & $FF
    If (b1 & $C0) <> $80 Or (b2 & $C0) <> $80 : ProcedureReturn -1 : EndIf
    value = ((b0 & $0F) << 12) | ((b1 & $3F) << 6) | (b2 & $3F)
    If value < $800 : ProcedureReturn -1 : EndIf
    If value >= $D800 And value <= $DFFF : ProcedureReturn -1 : EndIf
    *NextIndex\i = Index + 3
    ProcedureReturn value
  EndIf
  If b0 > $F4 : ProcedureReturn -1 : EndIf
  If Index + 3 >= Bytes : ProcedureReturn -1 : EndIf
  b1 = PeekA(*Data + Index + 1) & $FF
  b2 = PeekA(*Data + Index + 2) & $FF
  b3 = PeekA(*Data + Index + 3) & $FF
  If (b1 & $C0) <> $80 Or (b2 & $C0) <> $80 Or (b3 & $C0) <> $80 : ProcedureReturn -1 : EndIf
  value = ((b0 & 7) << 18) | ((b1 & $3F) << 12) | ((b2 & $3F) << 6) | (b3 & $3F)
  If value < $10000 Or value > $10FFFF : ProcedureReturn -1 : EndIf
  *NextIndex\i = Index + 4
  ProcedureReturn value
EndProcedure

; Returns the last codepoint of a UTF-8 run, or -1 when it is malformed.
Procedure.i PmoKdpUtf8Last(*Data, Bytes.q)
  Protected start.q
  Protected nextIndex.Integer
  Protected value.i
  If Bytes <= 0 : ProcedureReturn -1 : EndIf
  start = Bytes - 1
  While start > 0 And (PeekA(*Data + start) & $C0) = $80
    start - 1
  Wend
  value = PmoKdpUtf8Next(*Data, start, Bytes, @nextIndex)
  If value < 0 Or nextIndex\i <> Bytes : ProcedureReturn -1 : EndIf
  ProcedureReturn value
EndProcedure

; Every pronunciation byte run must be valid UTF-8 whose codepoints are all in
; the pinned Kokoro vocabulary. *BadCodepoint receives the first offender.
Procedure.i PmoKdpPhonemesValid(*Data, Bytes.q, *BadCodepoint.Integer)
  Protected index.q
  Protected nextIndex.Integer
  Protected value.i
  *BadCodepoint\i = -1
  If Bytes <= 0 : ProcedureReturn #False : EndIf
  While index < Bytes
    value = PmoKdpUtf8Next(*Data, index, Bytes, @nextIndex)
    If value < 0 : ProcedureReturn #False : EndIf
    If PmoKdpVocabContains(value) = 0
      *BadCodepoint\i = value
      ProcedureReturn #False
    EndIf
    index = nextIndex\i
  Wend
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; The working entry table: two blobs, an entry list, and a key index used for
; duplicate detection and for the gold-wins-over-silver merge.
; ---------------------------------------------------------------------------
Procedure PmoKdpTableFree(*Table.PmoKdpTable)
  PmoKdpArenaFree(@*Table\Keys)
  PmoKdpArenaFree(@*Table\Phonemes)
  If *Table\Entries : FreeMemory(*Table\Entries) : EndIf
  If *Table\Index : FreeMemory(*Table\Index) : EndIf
  *Table\Entries = 0 : *Table\Index = 0
  *Table\Count = 0 : *Table\Capacity = 0 : *Table\IndexSlots = 0
EndProcedure

Procedure.i PmoKdpTableInit(*Table.PmoKdpTable, ExpectedEntries.q)
  Protected slots.q = 1024
  FillMemory(*Table, SizeOf(PmoKdpTable), 0)
  While slots < ExpectedEntries * 4
    slots = slots * 2
  Wend
  *Table\Capacity = 4096
  *Table\Entries = AllocateMemory(*Table\Capacity * SizeOf(PmoKdpEntry))
  *Table\Index = AllocateMemory(slots * 4)
  If *Table\Entries = 0 Or *Table\Index = 0
    PmoKdpTableFree(*Table)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2003) + "the dictionary builder could not allocate its entry table for " + Str(ExpectedEntries) + " expected words. Check the free memory available to this 64-bit process.")
  EndIf
  FillMemory(*Table\Index, slots * 4, $FF)
  *Table\IndexSlots = slots
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpTableFind(*Table.PmoKdpTable, *Key, KeyBytes.i, Hashed.i)
  Protected slot.q, found.q
  Protected *entry.PmoKdpEntry
  slot = Hashed & (*Table\IndexSlots - 1)
  Repeat
    found = PeekL(*Table\Index + slot * 4)
    If found < 0 : ProcedureReturn -1 : EndIf
    *entry = *Table\Entries + found * SizeOf(PmoKdpEntry)
    If *entry\KeyBytes = KeyBytes And CompareMemory(*Table\Keys\Data + *entry\KeyOffset, *Key, KeyBytes)
      ProcedureReturn found
    EndIf
    slot = (slot + 1) & (*Table\IndexSlots - 1)
  ForEver
EndProcedure

Procedure.i PmoKdpIndexRehash(*Table.PmoKdpTable)
  Protected slots.q, i.q, slot.q
  Protected *grown
  Protected *entry.PmoKdpEntry
  slots = *Table\IndexSlots * 2
  *grown = ReAllocateMemory(*Table\Index, slots * 4)
  If *grown = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2004) + "the dictionary builder could not grow its key index to " + Str(slots) + " slots. Check the free memory available to this 64-bit process.")
  EndIf
  *Table\Index = *grown
  *Table\IndexSlots = slots
  FillMemory(*Table\Index, slots * 4, $FF)
  For i = 0 To *Table\Count - 1
    *entry = *Table\Entries + i * SizeOf(PmoKdpEntry)
    slot = PmoKdpFnv1a(*Table\Keys\Data + *entry\KeyOffset, *entry\KeyBytes) & (slots - 1)
    While PeekL(*Table\Index + slot * 4) >= 0
      slot = (slot + 1) & (slots - 1)
    Wend
    PokeL(*Table\Index + slot * 4, i)
  Next
  ProcedureReturn #True
EndProcedure

; OnDuplicate 0 refuses a repeated key; OnDuplicate 1 keeps the first one and
; reports success, which is what the gold-wins silver merge needs.
Procedure.i PmoKdpTableAdd(*Table.PmoKdpTable, *Key, KeyBytes.i, *Phonemes, PhonemeBytes.i, OnDuplicate.i, Origin.s)
  Protected hashed.i, slot.q, capacity.q
  Protected *grown
  Protected *entry.PmoKdpEntry
  If KeyBytes < 1 Or KeyBytes > #PMO_KDP_MAX_KEY_BYTES
    ProcedureReturn PmoKdpFail(PmoKdpCode(2005) + "a " + Origin + " key of " + Str(KeyBytes) + " bytes cannot be packed because a PMG2P key must be 1 to 63 bytes. Check that source entry's spelling.")
  EndIf
  If PhonemeBytes < 1 Or PhonemeBytes > 65535
    ProcedureReturn PmoKdpFail(PmoKdpCode(2006) + "a " + Origin + " pronunciation of " + Str(PhonemeBytes) + " bytes cannot be packed because a PMG2P pronunciation must be 1 to 65535 bytes. Check that source entry's pronunciation.")
  EndIf
  hashed = PmoKdpFnv1a(*Key, KeyBytes)
  If PmoKdpTableFind(*Table, *Key, KeyBytes, hashed) >= 0
    If OnDuplicate = 1 : ProcedureReturn #True : EndIf
    ProcedureReturn PmoKdpFail(PmoKdpCode(2007) + "the " + Origin + " defines the word " + Chr(34) + PeekS(*Key, KeyBytes, #PB_Ascii) + Chr(34) + " more than once, and a PMG2P pack cannot carry two pronunciations for one key. Check that source file for the repeated key.")
  EndIf
  If *Table\Count >= *Table\Capacity
    capacity = *Table\Capacity * 2
    *grown = ReAllocateMemory(*Table\Entries, capacity * SizeOf(PmoKdpEntry))
    If *grown = 0
      ProcedureReturn PmoKdpFail(PmoKdpCode(2008) + "the dictionary builder could not grow its entry list to " + Str(capacity) + " words. Check the free memory available to this 64-bit process.")
    EndIf
    *Table\Entries = *grown
    *Table\Capacity = capacity
  EndIf
  *entry = *Table\Entries + *Table\Count * SizeOf(PmoKdpEntry)
  *entry\KeyOffset = *Table\Keys\Bytes
  *entry\KeyBytes = KeyBytes
  *entry\PhonemeOffset = *Table\Phonemes\Bytes
  *entry\PhonemeBytes = PhonemeBytes
  If PmoKdpArenaPut(@*Table\Keys, *Key, KeyBytes) = 0 : ProcedureReturn #False : EndIf
  If PmoKdpArenaPut(@*Table\Phonemes, *Phonemes, PhonemeBytes) = 0 : ProcedureReturn #False : EndIf
  slot = hashed & (*Table\IndexSlots - 1)
  While PeekL(*Table\Index + slot * 4) >= 0
    slot = (slot + 1) & (*Table\IndexSlots - 1)
  Wend
  PokeL(*Table\Index + slot * 4, *Table\Count)
  *Table\Count = *Table\Count + 1
  If *Table\Count * 2 >= *Table\IndexSlots
    If PmoKdpIndexRehash(*Table) = 0 : ProcedureReturn #False : EndIf
  EndIf
  ProcedureReturn #True
EndProcedure

; Canonical order: unsigned byte compare, shorter key first on a prefix tie.
Procedure.i PmoKdpCompare(*Keys, *A.PmoKdpEntry, *B.PmoKdpEntry)
  Protected limit.i, i.i, left.i, right.i
  limit = *A\KeyBytes
  If *B\KeyBytes < limit : limit = *B\KeyBytes : EndIf
  While i < limit
    left = PeekA(*Keys + *A\KeyOffset + i) & $FF
    right = PeekA(*Keys + *B\KeyOffset + i) & $FF
    If left <> right : ProcedureReturn left - right : EndIf
    i + 1
  Wend
  ProcedureReturn *A\KeyBytes - *B\KeyBytes
EndProcedure

; Bottom-up merge sort: O(n log n) with no recursion and no worst case, so a
; pathological key distribution cannot turn the build into a stack overflow.
Procedure.i PmoKdpSort(*Table.PmoKdpTable)
  Protected n.q, width.q, i.q, left.q, middle.q, right.q, a.q, b.q, out.q
  Protected stride.i = SizeOf(PmoKdpEntry)
  Protected *aux, *source, *target, *swap
  n = *Table\Count
  If n < 2 : ProcedureReturn #True : EndIf
  *aux = AllocateMemory(n * stride)
  If *aux = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2009) + "the dictionary builder could not allocate the " + Str(n * stride) + " bytes its canonical sort needs. Check the free memory available to this 64-bit process.")
  EndIf
  *source = *Table\Entries
  *target = *aux
  width = 1
  While width < n
    i = 0
    While i < n
      left = i
      middle = i + width
      If middle > n : middle = n : EndIf
      right = i + width * 2
      If right > n : right = n : EndIf
      a = left : b = middle : out = left
      While a < middle Or b < right
        If a >= middle
          CopyMemory(*source + b * stride, *target + out * stride, stride) : b + 1
        ElseIf b >= right
          CopyMemory(*source + a * stride, *target + out * stride, stride) : a + 1
        ElseIf PmoKdpCompare(*Table\Keys\Data, *source + a * stride, *source + b * stride) <= 0
          CopyMemory(*source + a * stride, *target + out * stride, stride) : a + 1
        Else
          CopyMemory(*source + b * stride, *target + out * stride, stride) : b + 1
        EndIf
        out + 1
      Wend
      i = i + width * 2
    Wend
    *swap = *source : *source = *target : *target = *swap
    width = width * 2
  Wend
  If *source <> *Table\Entries
    CopyMemory(*source, *Table\Entries, n * stride)
  EndIf
  FreeMemory(*aux)
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; Pack verification. Structure is always checked; Pinned also checks the
; upstream identities and the deterministic size/count/CRC contract.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpHexEquals(*Data, Text.s)
  Protected i.i
  For i = 0 To Len(Text) / 2 - 1
    If (PeekA(*Data + i) & $FF) <> Val("$" + Mid(Text, i * 2 + 1, 2))
      ProcedureReturn #False
    EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpVerifyMemory(*Pack, Bytes.q, Kind.i, Pinned.i)
  Protected i.q, slot.q, occupied.q
  Protected magic.s, revision.s
  Protected slotCount.q, entryCount.q, slotOffset.q, keyOffset.q, keyBytes.q
  Protected phonemeOffset.q, phonemeBytes.q, crc.i
  Protected slotHash.i, recordKeyOffset.q, recordPhonemeOffset.q
  Protected recordKeyBytes.i, recordPhonemeBytes.i
  Protected bad.Integer
  Protected expectedFile.q, expectedSlots.q, expectedEntries.q
  Protected expectedKeyBytes.q, expectedPhonemeBytes.q, expectedCrc.i
  If *Pack = 0 Or Bytes < #PMO_KDP_HEADER_BYTES
    ProcedureReturn PmoKdpFail(PmoKdpCode(3001) + "the pronunciation pack is only " + Str(Bytes) + " bytes and cannot even hold its 256-byte header. Check that the file was written completely and is not a truncated copy.")
  EndIf
  magic = PeekS(*Pack, 8, #PB_Ascii)
  If Kind = #PMO_KDP_KIND_BASE
    If magic <> "PMG2PUS"
      ProcedureReturn PmoKdpFail(PmoKdpCode(3002) + "the pack magic is not " + Chr(34) + "PMG2PUS" + Chr(34) + ", so this file is not a base PMG2P pronunciation pack. Check that the base and extra pack paths have not been swapped.")
    EndIf
  Else
    If magic <> "PMG2PX"
      ProcedureReturn PmoKdpFail(PmoKdpCode(3003) + "the pack magic is not " + Chr(34) + "PMG2PX" + Chr(34) + ", so this file is not a supplementary PMG2P pronunciation pack. Check that the base and extra pack paths have not been swapped.")
    EndIf
  EndIf
  If (PeekL(*Pack + 8) & $FFFFFFFF) <> 1 Or (PeekL(*Pack + 12) & $FFFFFFFF) <> #PMO_KDP_HEADER_BYTES
    ProcedureReturn PmoKdpFail(PmoKdpCode(3004) + "the pack declares format version " + Str(PeekL(*Pack + 8) & $FFFFFFFF) + " with a " + Str(PeekL(*Pack + 12) & $FFFFFFFF) + "-byte header, and only version 1 with a 256-byte header is supported. Check that this pack was built by a current tool.")
  EndIf
  slotCount = PeekL(*Pack + 24) & $FFFFFFFF
  entryCount = PeekL(*Pack + 28) & $FFFFFFFF
  slotOffset = PeekQ(*Pack + 32)
  keyOffset = PeekQ(*Pack + 40)
  keyBytes = PeekQ(*Pack + 48)
  phonemeOffset = PeekQ(*Pack + 56)
  phonemeBytes = PeekQ(*Pack + 64)
  crc = PeekL(*Pack + 72) & $FFFFFFFF
  If PeekQ(*Pack + 16) <> Bytes
    ProcedureReturn PmoKdpFail(PmoKdpCode(3005) + "the pack header declares " + Str(PeekQ(*Pack + 16)) + " bytes but the file holds " + Str(Bytes) + ". Check that the file was not truncated or appended to after it was written.")
  EndIf
  If (PeekL(*Pack + 76) & $FFFFFFFF) <> 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(3006) + "the reserved header word at offset 76 is not zero, so this pack does not match the PMG2P header layout. Check that the file is a PMG2P pack and not another format with the same magic.")
  EndIf
  For i = 216 To #PMO_KDP_HEADER_BYTES - 1
    If (PeekA(*Pack + i) & $FF) <> 0
      ProcedureReturn PmoKdpFail(PmoKdpCode(3007) + "reserved header byte " + Str(i) + " is not zero, so this pack does not match the PMG2P header layout. Check that the file was not edited after it was written.")
    EndIf
  Next
  If slotCount < 2 Or (slotCount & (slotCount - 1)) <> 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(3008) + "the pack declares " + Str(slotCount) + " hash slots, which is not a power of two of at least 2, so its linear probing cannot be reproduced. Check that the pack was not edited after it was written.")
  EndIf
  If slotOffset <> #PMO_KDP_HEADER_BYTES
    ProcedureReturn PmoKdpFail(PmoKdpCode(3009) + "the slot table starts at offset " + Str(slotOffset) + " instead of 256. Check that the file is a PMG2P pack built by this tool.")
  EndIf
  If keyOffset <> slotOffset + slotCount * #PMO_KDP_SLOT_BYTES
    ProcedureReturn PmoKdpFail(PmoKdpCode(3010) + "the key blob starts at offset " + Str(keyOffset) + " but the slot table ends at " + Str(slotOffset + slotCount * #PMO_KDP_SLOT_BYTES) + ". Check the slot count in the header against the file size.")
  EndIf
  If phonemeOffset <> keyOffset + keyBytes Or phonemeOffset + phonemeBytes <> Bytes
    ProcedureReturn PmoKdpFail(PmoKdpCode(3011) + "the key and pronunciation blob extents do not tile the file exactly, so the pack is internally inconsistent. Check the key and phoneme byte counts in the header against the file size.")
  EndIf
  If entryCount < 1 Or entryCount * 2 > slotCount
    ProcedureReturn PmoKdpFail(PmoKdpCode(3012) + "the pack declares " + Str(entryCount) + " entries in " + Str(slotCount) + " slots, which breaks the required load factor of at most one half. Check the entry and slot counts in the header.")
  EndIf
  If PmoKavCrc32(*Pack + #PMO_KDP_HEADER_BYTES, Bytes - #PMO_KDP_HEADER_BYTES) <> crc
    ProcedureReturn PmoKdpFail(PmoKdpCode(3013) + "the pack payload CRC-32 does not match the value in its header, so the file is damaged. Check the storage or transfer the file came through and rebuild it from the pinned sources.")
  EndIf
  If Pinned
    If PmoKdpHexEquals(*Pack + 144, #PMO_KDP_VOCAB_SHA256$) = 0
      ProcedureReturn PmoKdpFail(PmoKdpCode(3014) + "the pack names a different Kokoro vocabulary than the pinned one, so its pronunciations may not tokenize. Check that the pack was built against the pinned Kokoro 82M v1.0 vocabulary.")
    EndIf
    If Kind = #PMO_KDP_KIND_BASE
      If PmoKdpHexEquals(*Pack + 80, #PMO_KDP_GOLD_SHA256$) = 0
        ProcedureReturn PmoKdpFail(PmoKdpCode(3015) + "the base pack names a different Misaki gold lexicon than the pinned " + #PMO_KDP_GOLD_SHA256$ + ". Check which us_gold.json the pack was built from.")
      EndIf
      If PmoKdpHexEquals(*Pack + 112, #PMO_KDP_SILVER_SHA256$) = 0
        ProcedureReturn PmoKdpFail(PmoKdpCode(3016) + "the base pack names a different Misaki silver lexicon than the pinned " + #PMO_KDP_SILVER_SHA256$ + ". Check which us_silver.json the pack was built from.")
      EndIf
      revision = #PMO_KDP_MISAKI_REVISION$
      expectedFile = #PMO_KDP_BASE_FILE_BYTES : expectedSlots = #PMO_KDP_BASE_SLOTS
      expectedEntries = #PMO_KDP_BASE_ENTRIES : expectedKeyBytes = #PMO_KDP_BASE_KEY_BYTES
      expectedPhonemeBytes = #PMO_KDP_BASE_PHONEME_BYTES : expectedCrc = #PMO_KDP_BASE_PAYLOAD_CRC32
    Else
      If PmoKdpHexEquals(*Pack + 80, #PMO_KDP_CMUDICT_SHA256$) = 0
        ProcedureReturn PmoKdpFail(PmoKdpCode(3017) + "the supplementary pack names a different CMUdict source than the pinned " + #PMO_KDP_CMUDICT_SHA256$ + ". Check which cmudict.dict the pack was built from.")
      EndIf
      For i = 112 To 143
        If (PeekA(*Pack + i) & $FF) <> 0
          ProcedureReturn PmoKdpFail(PmoKdpCode(3018) + "the supplementary pack carries a second source hash at header byte " + Str(i) + ", but it is built from one source and that field must be zero. Check that the file is a supplementary pack and not a base pack.")
        EndIf
      Next
      revision = #PMO_KDP_CMUDICT_REVISION$
      expectedFile = #PMO_KDP_EXTRA_FILE_BYTES : expectedSlots = #PMO_KDP_EXTRA_SLOTS
      expectedEntries = #PMO_KDP_EXTRA_ENTRIES : expectedKeyBytes = #PMO_KDP_EXTRA_KEY_BYTES
      expectedPhonemeBytes = #PMO_KDP_EXTRA_PHONEME_BYTES : expectedCrc = #PMO_KDP_EXTRA_PAYLOAD_CRC32
    EndIf
    If PeekS(*Pack + 176, 40, #PB_Ascii) <> revision
      ProcedureReturn PmoKdpFail(PmoKdpCode(3019) + "the pack names upstream revision " + Chr(34) + PeekS(*Pack + 176, 40, #PB_Ascii) + Chr(34) + " instead of the pinned " + revision + ". Check which upstream checkout the sources came from.")
    EndIf
    If Bytes <> expectedFile Or slotCount <> expectedSlots Or entryCount <> expectedEntries Or keyBytes <> expectedKeyBytes Or phonemeBytes <> expectedPhonemeBytes Or crc <> expectedCrc
      ProcedureReturn PmoKdpFail(PmoKdpCode(3020) + "the pack is self-consistent but does not match the pinned deterministic contract of " + Str(expectedFile) + " bytes, " + Str(expectedSlots) + " slots, " + Str(expectedEntries) + " entries, " + Str(expectedKeyBytes) + " key bytes, " + Str(expectedPhonemeBytes) + " pronunciation bytes and payload CRC-32 " + RSet(Hex(expectedCrc), 8, "0") + ". It holds " + Str(Bytes) + " bytes, " + Str(slotCount) + " slots, " + Str(entryCount) + " entries, " + Str(keyBytes) + " key bytes, " + Str(phonemeBytes) + " pronunciation bytes and payload CRC-32 " + RSet(Hex(crc), 8, "0") + ". Check that the sources are the pinned revisions and that no selection rule was changed.")
    EndIf
  EndIf
  For slot = 0 To slotCount - 1
    slotHash = PeekL(*Pack + slotOffset + slot * #PMO_KDP_SLOT_BYTES) & $FFFFFFFF
    recordKeyOffset = PeekL(*Pack + slotOffset + slot * #PMO_KDP_SLOT_BYTES + 4) & $FFFFFFFF
    recordPhonemeOffset = PeekL(*Pack + slotOffset + slot * #PMO_KDP_SLOT_BYTES + 8) & $FFFFFFFF
    recordKeyBytes = PeekW(*Pack + slotOffset + slot * #PMO_KDP_SLOT_BYTES + 12) & $FFFF
    recordPhonemeBytes = PeekW(*Pack + slotOffset + slot * #PMO_KDP_SLOT_BYTES + 14) & $FFFF
    If slotHash = 0
      If recordKeyOffset Or recordPhonemeOffset Or recordKeyBytes Or recordPhonemeBytes
        ProcedureReturn PmoKdpFail(PmoKdpCode(3021) + "empty hash slot " + Str(slot) + " carries nonzero record fields, so the slot table is corrupt. Check the storage the file came through and rebuild it from the pinned sources.")
      EndIf
      Continue
    EndIf
    occupied + 1
    If recordKeyBytes < 1 Or recordKeyBytes > #PMO_KDP_MAX_KEY_BYTES Or recordPhonemeBytes < 1
      ProcedureReturn PmoKdpFail(PmoKdpCode(3022) + "hash slot " + Str(slot) + " declares a " + Str(recordKeyBytes) + "-byte key and a " + Str(recordPhonemeBytes) + "-byte pronunciation, and both must be nonempty with a key of at most 63 bytes. Check the slot table for damage.")
    EndIf
    If recordKeyOffset + recordKeyBytes > keyBytes Or recordPhonemeOffset + recordPhonemeBytes > phonemeBytes
      ProcedureReturn PmoKdpFail(PmoKdpCode(3023) + "hash slot " + Str(slot) + " points outside its key or pronunciation blob. Check the slot table and the blob sizes in the header.")
    EndIf
    If PmoKdpFnv1a(*Pack + keyOffset + recordKeyOffset, recordKeyBytes) <> slotHash
      ProcedureReturn PmoKdpFail(PmoKdpCode(3024) + "hash slot " + Str(slot) + " stores a hash that its own key does not produce, so lookups through this pack would miss. Check the slot table for damage and rebuild from the pinned sources.")
    EndIf
    For i = 0 To recordKeyBytes - 1
      If (PeekA(*Pack + keyOffset + recordKeyOffset + i) & $FF) >= $80
        ProcedureReturn PmoKdpFail(PmoKdpCode(3025) + "the key in hash slot " + Str(slot) + " contains a non-ASCII byte at offset " + Str(i) + ", and PMG2P keys are ASCII. Check the key blob for damage.")
      EndIf
    Next
    If PmoKdpPhonemesValid(*Pack + phonemeOffset + recordPhonemeOffset, recordPhonemeBytes, @bad) = 0
      If bad\i >= 0
        ProcedureReturn PmoKdpFail(PmoKdpCode(3026) + "the pronunciation in hash slot " + Str(slot) + " uses U+" + RSet(Hex(bad\i), 4, "0") + ", which is outside the pinned Kokoro vocabulary. Check the pronunciation blob and the vocabulary the pack was built against.")
      EndIf
      ProcedureReturn PmoKdpFail(PmoKdpCode(3027) + "the pronunciation in hash slot " + Str(slot) + " is not valid UTF-8. Check the pronunciation blob for damage.")
    EndIf
  Next
  If occupied <> entryCount
    ProcedureReturn PmoKdpFail(PmoKdpCode(3028) + "the header declares " + Str(entryCount) + " entries but " + Str(occupied) + " hash slots are occupied. Check the slot table and the entry count in the header.")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpReadWholeFile(Path.s, *OutData.Integer, *OutBytes.Integer, What.s)
  Protected file.i, bytes.q, readBytes.q
  Protected *buffer
  *OutData\i = 0 : *OutBytes\i = 0
  bytes = FileSize(Path)
  If bytes < 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4001) + "the " + What + " " + Path + " is missing or is not a readable file. Check the path and that the file exists.")
  EndIf
  If bytes = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4002) + "the " + What + " " + Path + " is empty. Check that the file was copied completely.")
  EndIf
  *buffer = AllocateMemory(bytes)
  If *buffer = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4003) + "there is not enough memory to read the " + Str(bytes) + "-byte " + What + " " + Path + ". Check the free memory available to this 64-bit process.")
  EndIf
  file = ReadFile(#PB_Any, Path, #PB_File_SharedRead)
  If file
    readBytes = ReadData(file, *buffer, bytes)
    CloseFile(file)
  EndIf
  If readBytes <> bytes
    FreeMemory(*buffer)
    ProcedureReturn PmoKdpFail(PmoKdpCode(4004) + "only " + Str(readBytes) + " of " + Str(bytes) + " bytes of the " + What + " " + Path + " could be read. Check that no other program is holding the file open.")
  EndIf
  *OutData\i = *buffer
  *OutBytes\i = bytes
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpCheckSourceIdentity(*Data, Bytes.q, Expected.s, What.s)
  Protected actual.s = PmoKavMemorySha256(*Data, Bytes)
  If actual = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(4005) + "the SHA-256 of the " + What + " could not be computed, so its identity cannot be proved. Check that the PureBasic cipher library is available in this build.")
  EndIf
  If actual <> LCase(Expected)
    ProcedureReturn PmoKdpFail(PmoKdpCode(4006) + "the " + What + " has SHA-256 " + actual + " but the pinned revision requires " + LCase(Expected) + ". Check that you supplied the pinned upstream file and not a newer or re-saved copy.")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKokoroDictionaryVerifyFile(Path.s, Kind.i)
  Protected result.i
  Protected rawFile.Integer, bytes.Integer
  PmoKokoroDictionaryPackError = ""
  If PmoKdpReadWholeFile(Path, @rawFile, @bytes, "pronunciation pack") = 0 : ProcedureReturn #False : EndIf
  result = PmoKdpVerifyMemory(rawFile\i, bytes\i, Kind, #True)
  FreeMemory(rawFile\i)
  ProcedureReturn result
EndProcedure

; ---------------------------------------------------------------------------
; Emission and publication.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpEmit(*Table.PmoKdpTable, Kind.i, PrimarySha.s, SecondarySha.s, Revision.s, *OutPack.Integer, *OutBytes.Integer)
  Protected slots.q, i.q, slot.q, slotBytes.q, total.q, hashed.q
  Protected keyBase.q, phonemeBase.q, keyCursor.q, phonemeCursor.q
  Protected failed.i
  Protected *pack, *record
  Protected *entry.PmoKdpEntry
  *OutPack\i = 0 : *OutBytes\i = 0
  If *Table\Count < 1
    ProcedureReturn PmoKdpFail(PmoKdpCode(2010) + "no usable words survived the source rules, so there is nothing to pack. Check that the correct source files were supplied and that they are the pinned revisions.")
  EndIf
  slots = 1
  While slots < *Table\Count * 2
    slots = slots * 2
  Wend
  If slots > $40000000
    ProcedureReturn PmoKdpFail(PmoKdpCode(2011) + "the " + Str(*Table\Count) + " selected words need more than 2^30 hash slots, which the 32-bit slot count in the header cannot express. Check the source files for unexpected bulk.")
  EndIf
  slotBytes = slots * #PMO_KDP_SLOT_BYTES
  total = #PMO_KDP_HEADER_BYTES + slotBytes + *Table\Keys\Bytes + *Table\Phonemes\Bytes
  *pack = AllocateMemory(total)
  If *pack = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2012) + "there is not enough memory to assemble the " + Str(total) + "-byte pronunciation pack. Check the free memory available to this 64-bit process.")
  EndIf
  FillMemory(*pack, total, 0)
  ; The blobs are rewritten here in canonical order. The working arenas hold
  ; the words in the order they were parsed; only the sorted entry list knows
  ; the canonical order, and the file's bytes must follow it exactly.
  keyBase = #PMO_KDP_HEADER_BYTES + slotBytes
  phonemeBase = keyBase + *Table\Keys\Bytes
  For i = 0 To *Table\Count - 1
    *entry = *Table\Entries + i * SizeOf(PmoKdpEntry)
    If keyCursor > $FFFFFFFF Or phonemeCursor > $FFFFFFFF
      failed = #True
      Break
    EndIf
    hashed = PmoKdpFnv1a(*Table\Keys\Data + *entry\KeyOffset, *entry\KeyBytes)
    slot = hashed & (slots - 1)
    While (PeekL(*pack + #PMO_KDP_HEADER_BYTES + slot * #PMO_KDP_SLOT_BYTES) & $FFFFFFFF) <> 0
      slot = (slot + 1) & (slots - 1)
    Wend
    *record = *pack + #PMO_KDP_HEADER_BYTES + slot * #PMO_KDP_SLOT_BYTES
    PokeL(*record, hashed)
    PokeL(*record + 4, keyCursor)
    PokeL(*record + 8, phonemeCursor)
    PokeW(*record + 12, *entry\KeyBytes)
    PokeW(*record + 14, *entry\PhonemeBytes)
    CopyMemory(*Table\Keys\Data + *entry\KeyOffset, *pack + keyBase + keyCursor, *entry\KeyBytes)
    CopyMemory(*Table\Phonemes\Data + *entry\PhonemeOffset, *pack + phonemeBase + phonemeCursor, *entry\PhonemeBytes)
    keyCursor = keyCursor + *entry\KeyBytes
    phonemeCursor = phonemeCursor + *entry\PhonemeBytes
  Next
  If failed
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2013) + "a blob offset passed 4 GiB, which the 32-bit offsets in a PMG2P slot record cannot express. Check the source files for unexpected bulk.")
  EndIf
  If keyCursor <> *Table\Keys\Bytes Or phonemeCursor <> *Table\Phonemes\Bytes
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2027) + "the canonical blobs came to " + Str(keyCursor) + " key bytes and " + Str(phonemeCursor) + " pronunciation bytes while the working arenas hold " + Str(*Table\Keys\Bytes) + " and " + Str(*Table\Phonemes\Bytes) + ", which is an internal fault. Check the entry list against the arenas in the dictionary packing module.")
  EndIf
  If Kind = #PMO_KDP_KIND_BASE
    PmoKdpPutAscii(*pack, "PMG2PUS")
  Else
    PmoKdpPutAscii(*pack, "PMG2PX")
  EndIf
  PokeL(*pack + 8, 1)
  PokeL(*pack + 12, #PMO_KDP_HEADER_BYTES)
  PokeQ(*pack + 16, total)
  PokeL(*pack + 24, slots)
  PokeL(*pack + 28, *Table\Count)
  PokeQ(*pack + 32, #PMO_KDP_HEADER_BYTES)
  PokeQ(*pack + 40, #PMO_KDP_HEADER_BYTES + slotBytes)
  PokeQ(*pack + 48, *Table\Keys\Bytes)
  PokeQ(*pack + 56, #PMO_KDP_HEADER_BYTES + slotBytes + *Table\Keys\Bytes)
  PokeQ(*pack + 64, *Table\Phonemes\Bytes)
  PokeL(*pack + 72, PmoKavCrc32(*pack + #PMO_KDP_HEADER_BYTES, total - #PMO_KDP_HEADER_BYTES))
  If PmoKavHexToBytes(PrimarySha, *pack + 80, 32) = 0
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2014) + "the primary source SHA-256 could not be written into the header because it is not 64 hexadecimal digits. Check the pinned identity constants in this module.")
  EndIf
  If SecondarySha <> "" And PmoKavHexToBytes(SecondarySha, *pack + 112, 32) = 0
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2015) + "the secondary source SHA-256 could not be written into the header because it is not 64 hexadecimal digits. Check the pinned identity constants in this module.")
  EndIf
  If PmoKavHexToBytes(#PMO_KDP_VOCAB_SHA256$, *pack + 144, 32) = 0
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2016) + "the Kokoro vocabulary SHA-256 could not be written into the header because it is not 64 hexadecimal digits. Check the pinned identity constants in this module.")
  EndIf
  If Len(Revision) <> 40
    FreeMemory(*pack)
    ProcedureReturn PmoKdpFail(PmoKdpCode(2017) + "the upstream revision " + Chr(34) + Revision + Chr(34) + " is not the 40 characters the header reserves for it. Check the pinned revision constants in this module.")
  EndIf
  PmoKdpPutAscii(*pack + 176, Revision)
  *OutPack\i = *pack
  *OutBytes\i = total
  ProcedureReturn #True
EndProcedure

; Refuses an existing destination. Publishing is a rename of a fully written,
; closed and independently re-verified sibling temporary file.
Procedure.i PmoKdpPublish(*Pack, Bytes.q, Destination.s, Kind.i)
  Protected output.i, written.q
  Protected temporary.s
  Protected rawFile.Integer, readBytes.Integer
  Protected verified.i
  If Destination = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(4007) + "no output path was given, so the finished pronunciation pack has nowhere to go. Check the --output argument.")
  EndIf
  If FileSize(Destination) >= 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4008) + "the output " + Destination + " already exists and this tool never replaces a pronunciation pack. Check the path, then move or delete the existing file yourself.")
  EndIf
  temporary = Destination + ".tmp-" + Hex(ElapsedMilliseconds())
  If FileSize(temporary) >= 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4009) + "the temporary file " + temporary + " already exists, so packing cannot proceed safely. Check the output directory for leftover .tmp- files.")
  EndIf
  output = CreateFile(#PB_Any, temporary)
  If output = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4010) + "the temporary file " + temporary + " could not be created. Check that the output directory exists and is writable.")
  EndIf
  written = WriteData(output, *Pack, Bytes)
  CloseFile(output)
  If written <> Bytes
    DeleteFile(temporary)
    ProcedureReturn PmoKdpFail(PmoKdpCode(4011) + "only " + Str(written) + " of " + Str(Bytes) + " bytes reached " + temporary + ", so the pack was discarded. Check the free space on the output volume.")
  EndIf
  If PmoKdpReadWholeFile(temporary, @rawFile, @readBytes, "temporary pronunciation pack") = 0
    DeleteFile(temporary)
    ProcedureReturn #False
  EndIf
  verified = PmoKdpVerifyMemory(rawFile\i, readBytes\i, Kind, #True)
  FreeMemory(rawFile\i)
  If verified = 0
    DeleteFile(temporary)
    ProcedureReturn #False
  EndIf
  If FileSize(Destination) >= 0
    DeleteFile(temporary)
    ProcedureReturn PmoKdpFail(PmoKdpCode(4012) + "the output " + Destination + " appeared while the pack was being written, so it was not replaced. Check whether another build is writing to the same path.")
  EndIf
  If RenameFile(temporary, Destination) = 0
    DeleteFile(temporary)
    ProcedureReturn PmoKdpFail(PmoKdpCode(4013) + "the verified pack could not be renamed from " + temporary + " to " + Destination + ". Check that the output directory is writable and that no program is holding the destination open.")
  EndIf
  ProcedureReturn #True
EndProcedure

; ---------------------------------------------------------------------------
; A JSON reader scoped to the pinned Misaki lexicons: an object whose members
; are pronunciation strings or part-of-speech objects. It refuses malformed
; input loudly rather than skipping past it.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpJsonError(*Json.PmoKdpJson, Code.i, Meaning.s, FirstCheck.s)
  ProcedureReturn PmoKdpFail(PmoKdpCode(Code) + Meaning + " at byte offset " + Str(*Json\Pos) + " of " + *Json\Source + ". " + FirstCheck)
EndProcedure

Procedure PmoKdpJsonSkipSpace(*Json.PmoKdpJson)
  Protected b.i
  While *Json\Pos < *Json\Bytes
    b = PeekA(*Json\Data + *Json\Pos) & $FF
    If b <> 32 And b <> 9 And b <> 10 And b <> 13 : Break : EndIf
    *Json\Pos = *Json\Pos + 1
  Wend
EndProcedure

Procedure.i PmoKdpJsonPeek(*Json.PmoKdpJson)
  If *Json\Pos >= *Json\Bytes : ProcedureReturn -1 : EndIf
  ProcedureReturn PeekA(*Json\Data + *Json\Pos) & $FF
EndProcedure

Procedure.i PmoKdpJsonHexDigit(*Json.PmoKdpJson)
  Protected b.i = PmoKdpJsonPeek(*Json)
  If b >= 48 And b <= 57 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn b - 48 : EndIf
  If b >= 97 And b <= 102 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn b - 87 : EndIf
  If b >= 65 And b <= 70 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn b - 55 : EndIf
  ProcedureReturn -1
EndProcedure

Procedure.i PmoKdpJsonUnit(*Json.PmoKdpJson, *Value.Integer)
  Protected i.i, digit.i
  Protected value.i
  For i = 0 To 3
    digit = PmoKdpJsonHexDigit(*Json)
    If digit < 0
      ProcedureReturn PmoKdpJsonError(*Json, 1001, "a unicode escape is not followed by four hexadecimal digits", "Check that source file for a truncated escape sequence.")
    EndIf
    value = (value << 4) | digit
  Next
  *Value\i = value
  ProcedureReturn #True
EndProcedure

; Parses one JSON string into *Out, which is reset first. The result is
; well-formed UTF-8: escapes are decoded and raw sequences are validated and
; re-encoded, so a malformed source cannot reach a pack.
Procedure.i PmoKdpJsonString(*Json.PmoKdpJson, *Out.PmoKdpArena)
  Protected b.i, codepoint.i, low.i
  Protected nextIndex.Integer
  Protected unit.Integer
  *Out\Bytes = 0
  If PmoKdpJsonPeek(*Json) <> 34
    ProcedureReturn PmoKdpJsonError(*Json, 1002, "a JSON string was expected", "Check that source file around this offset; the Misaki lexicons are objects of quoted words.")
  EndIf
  *Json\Pos = *Json\Pos + 1
  Repeat
    If *Json\Pos >= *Json\Bytes
      ProcedureReturn PmoKdpJsonError(*Json, 1003, "the file ends inside a JSON string", "Check that the source file was copied completely.")
    EndIf
    b = PeekA(*Json\Data + *Json\Pos) & $FF
    If b = 34
      *Json\Pos = *Json\Pos + 1
      ProcedureReturn #True
    EndIf
    If b = 92
      *Json\Pos = *Json\Pos + 1
      b = PmoKdpJsonPeek(*Json)
      *Json\Pos = *Json\Pos + 1
      Select b
        Case 34 : codepoint = 34
        Case 92 : codepoint = 92
        Case 47 : codepoint = 47
        Case 98 : codepoint = 8
        Case 102 : codepoint = 12
        Case 110 : codepoint = 10
        Case 114 : codepoint = 13
        Case 116 : codepoint = 9
        Case 117
          If PmoKdpJsonUnit(*Json, @unit) = 0 : ProcedureReturn #False : EndIf
          codepoint = unit\i
          If codepoint >= $D800 And codepoint <= $DBFF
            If PmoKdpJsonPeek(*Json) <> 92
              ProcedureReturn PmoKdpJsonError(*Json, 1004, "a high surrogate escape is not followed by its low surrogate", "Check that source file; a lone surrogate cannot be encoded as UTF-8.")
            EndIf
            *Json\Pos = *Json\Pos + 1
            If PmoKdpJsonPeek(*Json) <> 117
              ProcedureReturn PmoKdpJsonError(*Json, 1005, "a high surrogate escape is not followed by a low surrogate escape", "Check that source file; a lone surrogate cannot be encoded as UTF-8.")
            EndIf
            *Json\Pos = *Json\Pos + 1
            If PmoKdpJsonUnit(*Json, @unit) = 0 : ProcedureReturn #False : EndIf
            low = unit\i
            If low < $DC00 Or low > $DFFF
              ProcedureReturn PmoKdpJsonError(*Json, 1006, "a high surrogate escape is paired with U+" + RSet(Hex(low), 4, "0") + ", which is not a low surrogate", "Check that source file for a broken surrogate pair.")
            EndIf
            codepoint = $10000 + ((codepoint - $D800) << 10) + (low - $DC00)
          ElseIf codepoint >= $DC00 And codepoint <= $DFFF
            ProcedureReturn PmoKdpJsonError(*Json, 1007, "a lone low surrogate escape U+" + RSet(Hex(codepoint), 4, "0") + " cannot be encoded as UTF-8", "Check that source file for a broken surrogate pair.")
          EndIf
        Default
          ProcedureReturn PmoKdpJsonError(*Json, 1008, "an unknown JSON escape was found", "Check that source file; JSON allows only the quote, solidus, reverse solidus, b, f, n, r, t and u escapes.")
      EndSelect
      If PmoKdpArenaPutCodepoint(*Out, codepoint) = 0 : ProcedureReturn #False : EndIf
      Continue
    EndIf
    If b < 32
      ProcedureReturn PmoKdpJsonError(*Json, 1009, "an unescaped control character U+" + RSet(Hex(b), 4, "0") + " appears inside a JSON string", "Check that source file for a raw newline or tab inside quotes.")
    EndIf
    If b < $80
      If PmoKdpArenaPutByte(*Out, b) = 0 : ProcedureReturn #False : EndIf
      *Json\Pos = *Json\Pos + 1
      Continue
    EndIf
    codepoint = PmoKdpUtf8Next(*Json\Data, *Json\Pos, *Json\Bytes, @nextIndex)
    If codepoint < 0
      ProcedureReturn PmoKdpJsonError(*Json, 1010, "a malformed, overlong, surrogate or out-of-range UTF-8 sequence appears inside a JSON string", "Check that the source file really is UTF-8 and was not re-saved in a code page.")
    EndIf
    If PmoKdpArenaPutCodepoint(*Out, codepoint) = 0 : ProcedureReturn #False : EndIf
    *Json\Pos = nextIndex\i
  ForEver
EndProcedure

Declare.i PmoKdpJsonSkipValue(*Json.PmoKdpJson, Depth.i, *Scratch.PmoKdpArena)

Procedure.i PmoKdpJsonSkipLiteral(*Json.PmoKdpJson, Text.s)
  Protected i.i
  For i = 1 To Len(Text)
    If PmoKdpJsonPeek(*Json) <> Asc(Mid(Text, i, 1))
      ProcedureReturn PmoKdpJsonError(*Json, 1011, "a JSON value begins like " + Text + " but does not spell it", "Check that source file around this offset.")
    EndIf
    *Json\Pos = *Json\Pos + 1
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpJsonSkipNumber(*Json.PmoKdpJson)
  Protected b.i, digits.i
  b = PmoKdpJsonPeek(*Json)
  If b = 45 : *Json\Pos = *Json\Pos + 1 : EndIf
  While PmoKdpJsonPeek(*Json) >= 48 And PmoKdpJsonPeek(*Json) <= 57
    *Json\Pos = *Json\Pos + 1 : digits + 1
  Wend
  If digits = 0
    ProcedureReturn PmoKdpJsonError(*Json, 1012, "a JSON value is neither a string, object, array, number, true, false nor null", "Check that source file around this offset.")
  EndIf
  If PmoKdpJsonPeek(*Json) = 46
    *Json\Pos = *Json\Pos + 1 : digits = 0
    While PmoKdpJsonPeek(*Json) >= 48 And PmoKdpJsonPeek(*Json) <= 57
      *Json\Pos = *Json\Pos + 1 : digits + 1
    Wend
    If digits = 0
      ProcedureReturn PmoKdpJsonError(*Json, 1013, "a JSON number has a decimal point with no digits after it", "Check that source file around this offset.")
    EndIf
  EndIf
  b = PmoKdpJsonPeek(*Json)
  If b = 101 Or b = 69
    *Json\Pos = *Json\Pos + 1
    b = PmoKdpJsonPeek(*Json)
    If b = 43 Or b = 45 : *Json\Pos = *Json\Pos + 1 : EndIf
    digits = 0
    While PmoKdpJsonPeek(*Json) >= 48 And PmoKdpJsonPeek(*Json) <= 57
      *Json\Pos = *Json\Pos + 1 : digits + 1
    Wend
    If digits = 0
      ProcedureReturn PmoKdpJsonError(*Json, 1014, "a JSON number has an exponent with no digits", "Check that source file around this offset.")
    EndIf
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpJsonSkipValue(*Json.PmoKdpJson, Depth.i, *Scratch.PmoKdpArena)
  Protected b.i
  If Depth > #PMO_KDP_MAX_JSON_DEPTH
    ProcedureReturn PmoKdpJsonError(*Json, 1015, "JSON nesting passed " + Str(#PMO_KDP_MAX_JSON_DEPTH) + " levels", "Check that the source file is a Misaki lexicon and not an unrelated document.")
  EndIf
  PmoKdpJsonSkipSpace(*Json)
  b = PmoKdpJsonPeek(*Json)
  Select b
    Case 34
      ProcedureReturn PmoKdpJsonString(*Json, *Scratch)
    Case 123
      *Json\Pos = *Json\Pos + 1
      PmoKdpJsonSkipSpace(*Json)
      If PmoKdpJsonPeek(*Json) = 125 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn #True : EndIf
      Repeat
        PmoKdpJsonSkipSpace(*Json)
        If PmoKdpJsonString(*Json, *Scratch) = 0 : ProcedureReturn #False : EndIf
        PmoKdpJsonSkipSpace(*Json)
        If PmoKdpJsonPeek(*Json) <> 58
          ProcedureReturn PmoKdpJsonError(*Json, 1016, "a JSON object member name is not followed by a colon", "Check that source file around this offset.")
        EndIf
        *Json\Pos = *Json\Pos + 1
        If PmoKdpJsonSkipValue(*Json, Depth + 1, *Scratch) = 0 : ProcedureReturn #False : EndIf
        PmoKdpJsonSkipSpace(*Json)
        b = PmoKdpJsonPeek(*Json)
        If b = 44 : *Json\Pos = *Json\Pos + 1 : Continue : EndIf
        If b = 125 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn #True : EndIf
        ProcedureReturn PmoKdpJsonError(*Json, 1017, "a JSON object member is not followed by a comma or a closing brace", "Check that source file around this offset.")
      ForEver
    Case 91
      *Json\Pos = *Json\Pos + 1
      PmoKdpJsonSkipSpace(*Json)
      If PmoKdpJsonPeek(*Json) = 93 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn #True : EndIf
      Repeat
        If PmoKdpJsonSkipValue(*Json, Depth + 1, *Scratch) = 0 : ProcedureReturn #False : EndIf
        PmoKdpJsonSkipSpace(*Json)
        b = PmoKdpJsonPeek(*Json)
        If b = 44 : *Json\Pos = *Json\Pos + 1 : Continue : EndIf
        If b = 93 : *Json\Pos = *Json\Pos + 1 : ProcedureReturn #True : EndIf
        ProcedureReturn PmoKdpJsonError(*Json, 1018, "a JSON array element is not followed by a comma or a closing bracket", "Check that source file around this offset.")
      ForEver
    Case 116
      ProcedureReturn PmoKdpJsonSkipLiteral(*Json, "true")
    Case 102
      ProcedureReturn PmoKdpJsonSkipLiteral(*Json, "false")
    Case 110
      ProcedureReturn PmoKdpJsonSkipLiteral(*Json, "null")
    Default
      ProcedureReturn PmoKdpJsonSkipNumber(*Json)
  EndSelect
EndProcedure

; Resolves a part-of-speech object: "DEFAULT" when it is a string, otherwise
; the first string-valued member in document order. *Resolved is cleared when
; the object holds no string at all.
Procedure.i PmoKdpJsonVariantObject(*Json.PmoKdpJson, *Out.PmoKdpArena, *Member.PmoKdpArena, *Scratch.PmoKdpArena, *Resolved.Integer)
  Protected b.i, haveDefault.i, haveFirst.i, isDefault.i
  Protected firstOffset.q, firstBytes.q
  Protected defaultOffset.q, defaultBytes.q
  Protected pool.PmoKdpArena
  Protected result.i = #False
  *Resolved\i = #False
  *Out\Bytes = 0
  *Json\Pos = *Json\Pos + 1
  PmoKdpJsonSkipSpace(*Json)
  If PmoKdpJsonPeek(*Json) = 125
    *Json\Pos = *Json\Pos + 1
    ProcedureReturn #True
  EndIf
  Repeat
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonString(*Json, *Member) = 0 : Break : EndIf
    isDefault = #False
    If *Member\Bytes = 7 And PeekS(*Member\Data, 7, #PB_Ascii) = "DEFAULT" : isDefault = #True : EndIf
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonPeek(*Json) <> 58
      PmoKdpJsonError(*Json, 1019, "a part-of-speech member name is not followed by a colon", "Check that source file around this offset.")
      Break
    EndIf
    *Json\Pos = *Json\Pos + 1
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonPeek(*Json) = 34
      If PmoKdpJsonString(*Json, *Scratch) = 0 : Break : EndIf
      If isDefault
        defaultOffset = pool\Bytes : defaultBytes = *Scratch\Bytes
        If PmoKdpArenaPut(@pool, *Scratch\Data, *Scratch\Bytes) = 0 : Break : EndIf
        haveDefault = #True
      ElseIf haveFirst = #False
        firstOffset = pool\Bytes : firstBytes = *Scratch\Bytes
        If PmoKdpArenaPut(@pool, *Scratch\Data, *Scratch\Bytes) = 0 : Break : EndIf
        haveFirst = #True
      EndIf
    Else
      If PmoKdpJsonSkipValue(*Json, 1, *Scratch) = 0 : Break : EndIf
    EndIf
    PmoKdpJsonSkipSpace(*Json)
    b = PmoKdpJsonPeek(*Json)
    If b = 44 : *Json\Pos = *Json\Pos + 1 : Continue : EndIf
    If b = 125
      *Json\Pos = *Json\Pos + 1
      result = #True
      Break
    EndIf
    PmoKdpJsonError(*Json, 1020, "a part-of-speech member is not followed by a comma or a closing brace", "Check that source file around this offset.")
    Break
  ForEver
  If result
    If haveDefault
      If PmoKdpArenaPut(*Out, pool\Data + defaultOffset, defaultBytes) = 0 : result = #False : EndIf
      *Resolved\i = #True
    ElseIf haveFirst
      If PmoKdpArenaPut(*Out, pool\Data + firstOffset, firstBytes) = 0 : result = #False : EndIf
      *Resolved\i = #True
    EndIf
  EndIf
  PmoKdpArenaFree(@pool)
  ProcedureReturn result
EndProcedure

Procedure.i PmoKdpJsonLoad(*Json.PmoKdpJson, *Table.PmoKdpTable, Origin.s)
  Protected b.i, i.i, keyIsAscii.i, resolved.i
  Protected key.PmoKdpArena, phonemes.PmoKdpArena, member.PmoKdpArena, scratch.PmoKdpArena
  Protected variant.Integer
  Protected bad.Integer
  Protected result.i = #False
  Protected running.i = #True
  PmoKdpJsonSkipSpace(*Json)
  If PmoKdpJsonPeek(*Json) <> 123
    PmoKdpJsonError(*Json, 1021, "the file does not begin with a JSON object", "Check that this is the pinned Misaki lexicon and that it carries no byte-order mark.")
    running = #False
  EndIf
  If running
    *Json\Pos = *Json\Pos + 1
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonPeek(*Json) = 125
      *Json\Pos = *Json\Pos + 1
      result = #True
      running = #False
    EndIf
  EndIf
  While running
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonString(*Json, @key) = 0 : Break : EndIf
    PmoKdpJsonSkipSpace(*Json)
    If PmoKdpJsonPeek(*Json) <> 58
      PmoKdpJsonError(*Json, 1022, "a lexicon word is not followed by a colon", "Check that source file around this offset.")
      Break
    EndIf
    *Json\Pos = *Json\Pos + 1
    PmoKdpJsonSkipSpace(*Json)
    resolved = #False
    phonemes\Bytes = 0
    b = PmoKdpJsonPeek(*Json)
    If b = 34
      If PmoKdpJsonString(*Json, @phonemes) = 0 : Break : EndIf
      resolved = #True
    ElseIf b = 123
      If PmoKdpJsonVariantObject(*Json, @phonemes, @member, @scratch, @variant) = 0 : Break : EndIf
      resolved = variant\i
    Else
      If PmoKdpJsonSkipValue(*Json, 1, @scratch) = 0 : Break : EndIf
    EndIf
    If key\Bytes = 0
      PmoKdpJsonError(*Json, 1023, "the lexicon defines an empty word", "Check that source file for a key that is just two quotation marks.")
      Break
    EndIf
    keyIsAscii = #True
    For i = 0 To key\Bytes - 1
      If (PeekA(key\Data + i) & $FF) >= $80 : keyIsAscii = #False : Break : EndIf
    Next
    ; A non-ASCII or oversized word, or one with no usable pronunciation, is
    ; outside this pack's contract and is dropped rather than refused: the
    ; pinned lexicons carry such rows deliberately.
    If resolved And phonemes\Bytes > 0 And keyIsAscii And key\Bytes <= #PMO_KDP_MAX_KEY_BYTES
      If PmoKdpPhonemesValid(phonemes\Data, phonemes\Bytes, @bad) = 0
        If bad\i >= 0
          PmoKdpJsonError(*Json, 1024, "the pronunciation of " + Chr(34) + PeekS(key\Data, key\Bytes, #PB_Ascii) + Chr(34) + " uses U+" + RSet(Hex(bad\i), 4, "0") + ", which is outside the pinned Kokoro vocabulary", "Check that the lexicon is the pinned revision for this vocabulary.")
        Else
          PmoKdpJsonError(*Json, 1025, "the pronunciation of " + Chr(34) + PeekS(key\Data, key\Bytes, #PB_Ascii) + Chr(34) + " is not valid UTF-8", "Check that the source file really is UTF-8.")
        EndIf
        Break
      EndIf
      If PmoKdpTableAdd(*Table, key\Data, key\Bytes, phonemes\Data, phonemes\Bytes, 0, Origin) = 0 : Break : EndIf
    EndIf
    PmoKdpJsonSkipSpace(*Json)
    b = PmoKdpJsonPeek(*Json)
    If b = 44 : *Json\Pos = *Json\Pos + 1 : Continue : EndIf
    If b = 125
      *Json\Pos = *Json\Pos + 1
      PmoKdpJsonSkipSpace(*Json)
      If *Json\Pos <> *Json\Bytes
        PmoKdpJsonError(*Json, 1026, "trailing text follows the closing brace of the lexicon", "Check that source file for appended content.")
        Break
      EndIf
      result = #True
      Break
    EndIf
    PmoKdpJsonError(*Json, 1027, "a lexicon entry is not followed by a comma or a closing brace", "Check that source file around this offset.")
    Break
  Wend
  PmoKdpArenaFree(@key)
  PmoKdpArenaFree(@phonemes)
  PmoKdpArenaFree(@member)
  PmoKdpArenaFree(@scratch)
  ProcedureReturn result
EndProcedure

Procedure.i PmoKdpLoadLexicon(Path.s, *Table.PmoKdpTable, ExpectedSha.s, Origin.s, ExpectedEntries.q)
  Protected json.PmoKdpJson
  Protected rawFile.Integer, bytes.Integer
  Protected result.i
  If PmoKdpReadWholeFile(Path, @rawFile, @bytes, Origin) = 0 : ProcedureReturn #False : EndIf
  If PmoKdpCheckSourceIdentity(rawFile\i, bytes\i, ExpectedSha, Origin + " " + Path) = 0
    FreeMemory(rawFile\i)
    ProcedureReturn #False
  EndIf
  If PmoKdpTableInit(*Table, ExpectedEntries) = 0
    FreeMemory(rawFile\i)
    ProcedureReturn #False
  EndIf
  json\Data = rawFile\i : json\Bytes = bytes\i : json\Pos = 0 : json\Source = Path
  result = PmoKdpJsonLoad(@json, *Table, Origin)
  FreeMemory(rawFile\i)
  If result = 0 : PmoKdpTableFree(*Table) : EndIf
  ProcedureReturn result
EndProcedure

; ---------------------------------------------------------------------------
; The base pack: pinned Misaki gold and silver, gold winning every collision.
; ---------------------------------------------------------------------------
Procedure.i PmoKokoroDictionaryPackBase(GoldPath.s, SilverPath.s, Destination.s)
  Protected i.q
  Protected gold.PmoKdpTable, silver.PmoKdpTable
  Protected *entry.PmoKdpEntry
  Protected pack.Integer, packBytes.Integer
  Protected haveGold.i, haveSilver.i, havePack.i, merged.i
  Protected result.i = #False
  PmoKokoroDictionaryPackError = ""
  PmoKokoroDictionaryReport = ""
  If GoldPath = "" Or SilverPath = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(2018) + "both the Misaki gold and silver lexicon paths are required to build a base pronunciation pack. Check the source argument and the --silver argument.")
  EndIf
  If Destination = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(2019) + "no output path was given for the base pronunciation pack. Check the --output argument.")
  EndIf
  If FileSize(Destination) >= 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4014) + "the output " + Destination + " already exists and this tool never replaces a pronunciation pack. Check the path, then move or delete the existing file yourself.")
  EndIf
  If PmoKdpLoadLexicon(GoldPath, @gold, #PMO_KDP_GOLD_SHA256$, "Misaki gold lexicon", 131072)
    haveGold = #True
    If PmoKdpLoadLexicon(SilverPath, @silver, #PMO_KDP_SILVER_SHA256$, "Misaki silver lexicon", 131072)
      haveSilver = #True
      merged = #True
      For i = 0 To silver\Count - 1
        *entry = silver\Entries + i * SizeOf(PmoKdpEntry)
        If PmoKdpTableAdd(@gold, silver\Keys\Data + *entry\KeyOffset, *entry\KeyBytes, silver\Phonemes\Data + *entry\PhonemeOffset, *entry\PhonemeBytes, 1, "Misaki silver lexicon") = 0
          merged = #False
          Break
        EndIf
      Next
      PmoKdpTableFree(@silver)
      haveSilver = #False
      If merged
        If PmoKdpSort(@gold)
          If PmoKdpEmit(@gold, #PMO_KDP_KIND_BASE, #PMO_KDP_GOLD_SHA256$, #PMO_KDP_SILVER_SHA256$, #PMO_KDP_MISAKI_REVISION$, @pack, @packBytes)
            havePack = #True
            If PmoKdpPublish(pack\i, packBytes\i, Destination, #PMO_KDP_KIND_BASE)
              PmoKokoroDictionaryReport = Str(gold\Count) + " entries, " + Str(PeekL(pack\i + 24) & $FFFFFFFF) + " slots, " + Str(gold\Keys\Bytes) + " key bytes, " + Str(gold\Phonemes\Bytes) + " pronunciation bytes, payload CRC-32 " + RSet(Hex(PeekL(pack\i + 72) & $FFFFFFFF), 8, "0") + ", " + Str(packBytes\i) + " file bytes"
              result = #True
            EndIf
          EndIf
        EndIf
      EndIf
    EndIf
  EndIf
  If havePack : FreeMemory(pack\i) : EndIf
  If haveSilver : PmoKdpTableFree(@silver) : EndIf
  If haveGold : PmoKdpTableFree(@gold) : EndIf
  ProcedureReturn result
EndProcedure

; ---------------------------------------------------------------------------
; A validated reader over a finished base pack, used by the extra builder to
; ask exactly what the target asks at run time.
; ---------------------------------------------------------------------------
Procedure PmoKdpReaderClose(*Reader.PmoKdpReader)
  If *Reader\Data : FreeMemory(*Reader\Data) : EndIf
  *Reader\Data = 0 : *Reader\Bytes = 0
EndProcedure

Procedure.i PmoKdpReaderBind(*Reader.PmoKdpReader, *Pack, Bytes.q)
  FillMemory(*Reader, SizeOf(PmoKdpReader), 0)
  *Reader\Data = *Pack
  *Reader\Bytes = Bytes
  *Reader\Slots = PeekL(*Pack + 24) & $FFFFFFFF
  *Reader\SlotOffset = PeekQ(*Pack + 32)
  *Reader\KeyOffset = PeekQ(*Pack + 40)
  *Reader\KeyBytes = PeekQ(*Pack + 48)
  *Reader\PhonemeOffset = PeekQ(*Pack + 56)
  *Reader\PhonemeBytes = PeekQ(*Pack + 64)
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpReaderOpen(*Reader.PmoKdpReader, Path.s)
  Protected rawFile.Integer, bytes.Integer
  FillMemory(*Reader, SizeOf(PmoKdpReader), 0)
  If PmoKdpReadWholeFile(Path, @rawFile, @bytes, "base pronunciation pack") = 0 : ProcedureReturn #False : EndIf
  If PmoKdpVerifyMemory(rawFile\i, bytes\i, #PMO_KDP_KIND_BASE, #True) = 0
    FreeMemory(rawFile\i)
    ProcedureReturn #False
  EndIf
  ProcedureReturn PmoKdpReaderBind(*Reader, rawFile\i, bytes\i)
EndProcedure

; Exact-spelling probe. Returns #True and fills *OutData / *OutBytes with a
; pointer into the pack's own phoneme blob.
Procedure.i PmoKdpReaderOne(*Reader.PmoKdpReader, *Key, KeyBytes.i, *OutData.Integer, *OutBytes.Integer)
  Protected hashed.i, slot.q, probe.q
  Protected slotHash.i, recordKeyOffset.q, recordPhonemeOffset.q
  Protected recordKeyBytes.i, recordPhonemeBytes.i
  Protected *record
  *OutData\i = 0 : *OutBytes\i = 0
  If KeyBytes < 1 Or KeyBytes > #PMO_KDP_MAX_KEY_BYTES : ProcedureReturn #False : EndIf
  hashed = PmoKdpFnv1a(*Key, KeyBytes)
  slot = hashed & (*Reader\Slots - 1)
  For probe = 0 To *Reader\Slots - 1
    *record = *Reader\Data + *Reader\SlotOffset + slot * #PMO_KDP_SLOT_BYTES
    slotHash = PeekL(*record) & $FFFFFFFF
    If slotHash = 0 : ProcedureReturn #False : EndIf
    recordKeyOffset = PeekL(*record + 4) & $FFFFFFFF
    recordPhonemeOffset = PeekL(*record + 8) & $FFFFFFFF
    recordKeyBytes = PeekW(*record + 12) & $FFFF
    recordPhonemeBytes = PeekW(*record + 14) & $FFFF
    If slotHash = hashed And recordKeyBytes = KeyBytes And CompareMemory(*Reader\Data + *Reader\KeyOffset + recordKeyOffset, *Key, KeyBytes)
      *OutData\i = *Reader\Data + *Reader\PhonemeOffset + recordPhonemeOffset
      *OutBytes\i = recordPhonemeBytes
      ProcedureReturn #True
    EndIf
    slot = (slot + 1) & (*Reader\Slots - 1)
  Next
  ProcedureReturn #False
EndProcedure

Procedure.i PmoKdpLowerByte(Value.i)
  If Value >= 65 And Value <= 90 : ProcedureReturn Value + 32 : EndIf
  ProcedureReturn Value
EndProcedure

; Exact spelling first, then ASCII lowercase, matching the target's lookup.
Procedure.i PmoKdpReaderLookup(*Reader.PmoKdpReader, *Key, KeyBytes.i, *OutData.Integer, *OutBytes.Integer)
  Protected i.i, b.i, differs.i
  Protected lowered.PmoKdpWord
  If PmoKdpReaderOne(*Reader, *Key, KeyBytes, *OutData, *OutBytes) : ProcedureReturn #True : EndIf
  If KeyBytes < 1 Or KeyBytes > #PMO_KDP_MAX_KEY_BYTES : ProcedureReturn #False : EndIf
  For i = 0 To KeyBytes - 1
    b = PeekA(*Key + i) & $FF
    If b >= 65 And b <= 90 : b + 32 : differs = #True : EndIf
    lowered\Byte[i] = b
  Next
  If differs = #False : ProcedureReturn #False : EndIf
  ProcedureReturn PmoKdpReaderOne(*Reader, @lowered\Byte[0], KeyBytes, *OutData, *OutBytes)
EndProcedure

Procedure.i PmoKdpIsDoubler(Letter.i)
  Select Letter
    Case 'b', 'c', 'd', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x', 'z'
      ProcedureReturn #True
  EndSelect
  ProcedureReturn #False
EndProcedure

; Dictionary-backed regular endings, exactly as the target frontend applies
; them: plural or third person -s, past -ed and progressive -ing, over a small
; ordered candidate list. Writes the inflected pronunciation into *Out.
; Tail 0 appends nothing to a candidate, 1 appends "y", 2 appends "e".
Procedure.i PmoKdpReaderInflect(*Reader.PmoKdpReader, *Key, KeyBytes.i, *Out.PmoKdpArena)
  Protected n.i, i.i, count.i, kind.i, baseBytes.i, candidateBytes.i, last.i
  Protected e1.i, e2.i, e3.i
  Protected lowered.PmoKdpWord
  Protected candidate.PmoKdpWord
  Protected list.PmoKdpCandidates
  Protected found.Integer, foundBytes.Integer
  Protected suffix.s
  *Out\Bytes = 0
  n = KeyBytes
  If n < 3 Or n > #PMO_KDP_MAX_KEY_BYTES : ProcedureReturn #False : EndIf
  For i = 0 To n - 1
    If (PeekA(*Key + i) & $FF) >= $80 : ProcedureReturn #False : EndIf
    lowered\Byte[i] = PmoKdpLowerByte(PeekA(*Key + i) & $FF)
  Next
  e1 = lowered\Byte[n - 1]
  e2 = lowered\Byte[n - 2]
  e3 = lowered\Byte[n - 3]
  count = 0
  baseBytes = 0
  If e1 = 's'
    kind = 0
    If e2 <> 's' And e2 <> 39
      list\Length[count] = n - 1 : list\Tail[count] = 0 : count + 1
    EndIf
    If e2 = 39 Or (n > 4 And e2 = 'e' And e3 <> 'i')
      list\Length[count] = n - 2 : list\Tail[count] = 0 : count + 1
    EndIf
    If n > 4 And e2 = 'e' And e3 = 'i'
      list\Length[count] = n - 3 : list\Tail[count] = 1 : count + 1
    EndIf
  ElseIf n >= 4 And e2 = 'e' And e1 = 'd'
    kind = 1
    list\Length[count] = n - 1 : list\Tail[count] = 0 : count + 1
    If n > 4 And e3 <> 'e'
      list\Length[count] = n - 2 : list\Tail[count] = 0 : count + 1
    EndIf
    If n > 4 And e3 = 'i'
      list\Length[count] = n - 3 : list\Tail[count] = 1 : count + 1
    EndIf
    baseBytes = n - 2
  ElseIf n >= 5 And e3 = 'i' And e2 = 'n' And e1 = 'g'
    kind = 2
    If n > 5
      list\Length[count] = n - 3 : list\Tail[count] = 0 : count + 1
    EndIf
    list\Length[count] = n - 3 : list\Tail[count] = 2 : count + 1
    baseBytes = n - 3
  Else
    ProcedureReturn #False
  EndIf
  If baseBytes > 2 And lowered\Byte[baseBytes - 1] = lowered\Byte[baseBytes - 2] And PmoKdpIsDoubler(lowered\Byte[baseBytes - 1])
    list\Length[count] = baseBytes - 1 : list\Tail[count] = 0 : count + 1
  EndIf
  For i = 0 To count - 1
    candidateBytes = list\Length[i]
    If candidateBytes < 0 : Continue : EndIf
    If candidateBytes > 0 : CopyMemory(*Key, @candidate\Byte[0], candidateBytes) : EndIf
    If list\Tail[i] = 1
      candidate\Byte[candidateBytes] = 'y' : candidateBytes + 1
    ElseIf list\Tail[i] = 2
      candidate\Byte[candidateBytes] = 'e' : candidateBytes + 1
    EndIf
    If candidateBytes < 2 : Continue : EndIf
    If PmoKdpReaderLookup(*Reader, @candidate\Byte[0], candidateBytes, @found, @foundBytes) = 0 : Continue : EndIf
    last = PmoKdpUtf8Last(found\i, foundBytes\i)
    If last < 0
      ProcedureReturn PmoKdpFail(PmoKdpCode(2020) + "a base pack pronunciation ends in a malformed UTF-8 sequence, so its inflected form cannot be derived. Check the base pack with the verify command.")
    EndIf
    Select kind
      Case 2
        suffix = Chr($026A) + Chr($014B)
      Case 0
        Select last
          Case 'p', 't', 'k', 'f', $03B8
            suffix = "s"
          Case 's', 'z', $0283, $0292, $02A7, $02A4
            suffix = Chr($1D7B) + "z"
          Default
            suffix = "z"
        EndSelect
      Default
        Select last
          Case 't', 'd'
            suffix = Chr($1D7B) + "d"
          Case 'p', 'k', 'f', $03B8, $0283, 's', $02A7
            suffix = "t"
          Default
            suffix = "d"
        EndSelect
    EndSelect
    If PmoKdpArenaPut(*Out, found\i, foundBytes\i) = 0 : ProcedureReturn #False : EndIf
    If PmoKdpArenaPutUtf8(*Out, suffix) = 0 : ProcedureReturn #False : EndIf
    ProcedureReturn #True
  Next
  ProcedureReturn #False
EndProcedure

Procedure.i PmoKdpReaderKnows(*Reader.PmoKdpReader, *Key, KeyBytes.i, *Scratch.PmoKdpArena)
  Protected found.Integer, foundBytes.Integer
  If PmoKdpReaderLookup(*Reader, *Key, KeyBytes, @found, @foundBytes) : ProcedureReturn #True : EndIf
  *Scratch\Bytes = 0
  ProcedureReturn PmoKdpReaderInflect(*Reader, *Key, KeyBytes, *Scratch)
EndProcedure

; ---------------------------------------------------------------------------
; ARPABET to the Kokoro phoneme alphabet.
; ---------------------------------------------------------------------------
Procedure.s PmoKdpArpabetPhone(Name.s, Stress.s, *Ok.Integer)
  Protected i.i
  *Ok\i = #False
  PmoKdpInitTables()
  For i = 0 To 38
    If PmoKdpArpaName(i) = Name
      *Ok\i = #True
      If Stress = "0"
        If Name = "AH" : ProcedureReturn Chr($0259) : EndIf
        If Name = "ER" : ProcedureReturn Chr($0259) + Chr($0279) : EndIf
        ProcedureReturn PmoKdpArpaPhoneme(i)
      EndIf
      If Stress = "1" : ProcedureReturn Chr($02C8) + PmoKdpArpaPhoneme(i) : EndIf
      If Stress = "2" : ProcedureReturn Chr($02CC) + PmoKdpArpaPhoneme(i) : EndIf
      ProcedureReturn PmoKdpArpaPhoneme(i)
    EndIf
  Next
  ProcedureReturn ""
EndProcedure

; ---------------------------------------------------------------------------
; The extra pack: pinned CMUdict, minus everything the base pack already
; answers, plus a fixed list of reviewed technical terms.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpCmuWordAcceptable(*Key, KeyBytes.i)
  Protected i.i, b.i
  If KeyBytes < 2 Or KeyBytes > #PMO_KDP_MAX_KEY_BYTES : ProcedureReturn #False : EndIf
  b = PeekA(*Key) & $FF
  If b < 'a' Or b > 'z' : ProcedureReturn #False : EndIf
  For i = 1 To KeyBytes - 1
    b = PeekA(*Key + i) & $FF
    If (b < 'a' Or b > 'z') And b <> 39 : ProcedureReturn #False : EndIf
  Next
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKdpAddTermBytes(*Table.PmoKdpTable, *Reader.PmoKdpReader, Word.s, *Spoken, SpokenBytes.i, *Scratch.PmoKdpArena)
  Protected key.PmoKdpWord
  Protected keyBytes.i
  Protected bad.Integer
  keyBytes = PmoKdpPutAscii(@key\Byte[0], Word)
  If PmoKdpReaderKnows(*Reader, @key\Byte[0], keyBytes, *Scratch)
    ProcedureReturn #True
  EndIf
  If PmoKokoroDictionaryPackError <> "" : ProcedureReturn #False : EndIf
  If PmoKdpPhonemesValid(*Spoken, SpokenBytes, @bad) = 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(2021) + "the reviewed technical term " + Chr(34) + Word + Chr(34) + " has a pronunciation that is not valid UTF-8 inside the pinned Kokoro vocabulary. Check the reviewed additions table in this module.")
  EndIf
  ProcedureReturn PmoKdpTableAdd(*Table, @key\Byte[0], keyBytes, *Spoken, SpokenBytes, 0, "reviewed additions table")
EndProcedure

Procedure.i PmoKdpAddTerm(*Table.PmoKdpTable, *Reader.PmoKdpReader, Word.s, Spoken.s, *Scratch.PmoKdpArena)
  Protected phonemes.PmoKdpArena
  Protected result.i = #False
  If PmoKdpArenaPutUtf8(@phonemes, Spoken)
    result = PmoKdpAddTermBytes(*Table, *Reader, Word, phonemes\Data, phonemes\Bytes, *Scratch)
  EndIf
  PmoKdpArenaFree(@phonemes)
  ProcedureReturn result
EndProcedure

Procedure.i PmoKokoroDictionaryPackExtra(CmudictPath.s, BasePath.s, Destination.s)
  Protected position.q, lineStart.q, lineEnd.q, cut.q, cursor.q
  Protected wordOffset.q, fieldStart.q
  Protected b.i, wordBytes.i, clearKeyBytes.i
  Protected phoneOk.Integer
  Protected spoken.s, phone.s, stress.s
  Protected clearKey.PmoKdpWord
  Protected reader.PmoKdpReader
  Protected extra.PmoKdpTable
  Protected scratch.PmoKdpArena, phonemes.PmoKdpArena
  Protected rawFile.Integer, bytes.Integer
  Protected pack.Integer, packBytes.Integer
  Protected clear.Integer, clearBytes.Integer
  Protected bad.Integer
  Protected haveReader.i, haveTable.i, haveData.i, havePack.i
  Protected scanned.i, running.i
  Protected result.i = #False
  PmoKokoroDictionaryPackError = ""
  PmoKokoroDictionaryReport = ""
  If CmudictPath = "" Or BasePath = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(2022) + "both the cmudict.dict path and the finished base pack path are required to build a supplementary pronunciation pack. Check the source argument and the --base argument.")
  EndIf
  If Destination = ""
    ProcedureReturn PmoKdpFail(PmoKdpCode(2023) + "no output path was given for the supplementary pronunciation pack. Check the --output argument.")
  EndIf
  If FileSize(Destination) >= 0
    ProcedureReturn PmoKdpFail(PmoKdpCode(4015) + "the output " + Destination + " already exists and this tool never replaces a pronunciation pack. Check the path, then move or delete the existing file yourself.")
  EndIf
  running = #True
  If PmoKdpReaderOpen(@reader, BasePath) = 0 : running = #False : EndIf
  If running
    haveReader = #True
    If PmoKdpReadWholeFile(CmudictPath, @rawFile, @bytes, "CMUdict source") = 0 : running = #False : EndIf
  EndIf
  If running
    haveData = #True
    If PmoKdpCheckSourceIdentity(rawFile\i, bytes\i, #PMO_KDP_CMUDICT_SHA256$, "CMUdict source " + CmudictPath) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpTableInit(@extra, 131072) = 0 : running = #False : EndIf
  EndIf
  If running
    haveTable = #True
    scanned = #True
    position = 0
    While position < bytes\i
      lineStart = position
      lineEnd = lineStart
      While lineEnd < bytes\i
        b = PeekA(rawFile\i + lineEnd) & $FF
        If b = 10 Or b = 13 : Break : EndIf
        lineEnd + 1
      Wend
      position = lineEnd + 1
      If lineEnd < bytes\i And (PeekA(rawFile\i + lineEnd) & $FF) = 13
        If position < bytes\i And (PeekA(rawFile\i + position) & $FF) = 10 : position + 1 : EndIf
      EndIf
      cut = lineStart
      While cut < lineEnd
        If (PeekA(rawFile\i + cut) & $FF) = 35 : Break : EndIf
        cut + 1
      Wend
      cursor = lineStart
      While cursor < cut And (PeekA(rawFile\i + cursor) & $FF) <= 32
        cursor + 1
      Wend
      If cursor >= cut : Continue : EndIf
      wordOffset = cursor
      While cursor < cut And (PeekA(rawFile\i + cursor) & $FF) > 32
        cursor + 1
      Wend
      wordBytes = cursor - wordOffset
      If PmoKdpCmuWordAcceptable(rawFile\i + wordOffset, wordBytes) = 0 : Continue : EndIf
      If PmoKdpReaderKnows(@reader, rawFile\i + wordOffset, wordBytes, @scratch)
        Continue
      EndIf
      If PmoKokoroDictionaryPackError <> "" : scanned = #False : Break : EndIf
      spoken = ""
      While cursor < cut
        While cursor < cut And (PeekA(rawFile\i + cursor) & $FF) <= 32
          cursor + 1
        Wend
        If cursor >= cut : Break : EndIf
        fieldStart = cursor
        While cursor < cut And (PeekA(rawFile\i + cursor) & $FF) > 32
          cursor + 1
        Wend
        phone = PeekS(rawFile\i + fieldStart, cursor - fieldStart, #PB_Ascii)
        stress = Right(phone, 1)
        If stress >= "0" And stress <= "9"
          phone = Left(phone, Len(phone) - 1)
        Else
          stress = ""
        EndIf
        spoken = spoken + PmoKdpArpabetPhone(phone, stress, @phoneOk)
        If phoneOk\i = 0
          PmoKdpFail(PmoKdpCode(2024) + "the CMUdict entry for " + Chr(34) + PeekS(rawFile\i + wordOffset, wordBytes, #PB_Ascii) + Chr(34) + " uses the phone " + Chr(34) + phone + Chr(34) + ", which is not one of the 39 ARPABET phones this converter knows. Check that cmudict.dict is the pinned revision.")
          scanned = #False
          Break
        EndIf
      Wend
      If scanned = #False : Break : EndIf
      phonemes\Bytes = 0
      If PmoKdpArenaPutUtf8(@phonemes, spoken) = 0 : scanned = #False : Break : EndIf
      If PmoKdpPhonemesValid(phonemes\Data, phonemes\Bytes, @bad) = 0
        PmoKdpFail(PmoKdpCode(2025) + "the CMUdict entry for " + Chr(34) + PeekS(rawFile\i + wordOffset, wordBytes, #PB_Ascii) + Chr(34) + " converts to a pronunciation that is empty or outside the pinned Kokoro vocabulary. Check that cmudict.dict is the pinned revision.")
        scanned = #False
        Break
      EndIf
      If PmoKdpTableAdd(@extra, rawFile\i + wordOffset, wordBytes, phonemes\Data, phonemes\Bytes, 0, "CMUdict source") = 0
        scanned = #False
        Break
      EndIf
    Wend
    If scanned = #False : running = #False : EndIf
  EndIf
  ; Reviewed specialist additions. They only ever fill gaps in the
  ; authoritative table, never override it.
  If running
    If PmoKdpAddTerm(@extra, @reader, "microcontroller", "m" + Chr($02CC) + "Ik" + Chr($0279) + "Ok" + Chr($0259) + "nt" + Chr($0279) + Chr($02C8) + "Ol" + Chr($0259) + Chr($0279), @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "microcontrollers", "m" + Chr($02CC) + "Ik" + Chr($0279) + "Ok" + Chr($0259) + "nt" + Chr($0279) + Chr($02C8) + "Ol" + Chr($0259) + Chr($0279) + "z", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "gigahertz", Chr($0261) + Chr($02C8) + Chr($026A) + Chr($0261) + Chr($0259) + "h" + Chr($02CC) + Chr($025C) + Chr($0279) + "ts", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "megabit", "m" + Chr($02C8) + Chr($025B) + Chr($0261) + Chr($0259) + "b" + Chr($02CC) + Chr($026A) + "t", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "megabits", "m" + Chr($02C8) + Chr($025B) + Chr($0261) + Chr($0259) + "b" + Chr($02CC) + Chr($026A) + "ts", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "mbit", "m" + Chr($02C8) + Chr($025B) + Chr($0261) + Chr($0259) + "b" + Chr($02CC) + Chr($026A) + "t", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "mbits", "m" + Chr($02C8) + Chr($025B) + Chr($0261) + Chr($0259) + "b" + Chr($02CC) + Chr($026A) + "ts", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    clearKeyBytes = PmoKdpPutAscii(@clearKey\Byte[0], "clear")
    If PmoKdpReaderLookup(@reader, @clearKey\Byte[0], clearKeyBytes, @clear, @clearBytes) = 0
      PmoKdpFail(PmoKdpCode(2026) + "the base pack does not contain the word " + Chr(34) + "clear" + Chr(34) + ", which the reviewed abbreviation " + Chr(34) + "clr" + Chr(34) + " is defined from. Check that the base pack was built from the pinned Misaki lexicons.")
      running = #False
    EndIf
  EndIf
  If running
    If PmoKdpAddTermBytes(@extra, @reader, "clr", clear\i, clearBytes\i, @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "bitwise", "b" + Chr($02C8) + Chr($026A) + "tw" + Chr($02CC) + "Iz", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "bootloader", "b" + Chr($02C8) + "utl" + Chr($02CC) + "Od" + Chr($0259) + Chr($0279), @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "bootloaders", "b" + Chr($02C8) + "utl" + Chr($02CC) + "Od" + Chr($0259) + Chr($0279) + "z", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpAddTerm(@extra, @reader, "multicore", "m" + Chr($02CC) + Chr($028C) + "ltik" + Chr($02C8) + Chr($0254) + Chr($0279), @scratch) = 0 : running = #False : EndIf
  EndIf
  ; The speech model's own name. CMUdict has no entry for it, so without this
  ; line the reader spells it out one letter at a time. ARPABET K OW0 K OW1
  ; R OW0, stress on the second syllable. The pinned contract above (66,305
  ; entries, CRC-32 2641725D) counts this entry: a table without it builds the
  ; previous 66,304-entry pack, which the verifier and the board both refuse.
  If running
    If PmoKdpAddTerm(@extra, @reader, "kokoro", "kOk" + Chr($02C8) + "O" + Chr($0279) + "O", @scratch) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpSort(@extra) = 0 : running = #False : EndIf
  EndIf
  If running
    If PmoKdpEmit(@extra, #PMO_KDP_KIND_EXTRA, #PMO_KDP_CMUDICT_SHA256$, "", #PMO_KDP_CMUDICT_REVISION$, @pack, @packBytes) = 0
      running = #False
    Else
      havePack = #True
    EndIf
  EndIf
  If running
    If PmoKdpPublish(pack\i, packBytes\i, Destination, #PMO_KDP_KIND_EXTRA)
      PmoKokoroDictionaryReport = Str(extra\Count) + " entries, " + Str(PeekL(pack\i + 24) & $FFFFFFFF) + " slots, " + Str(extra\Keys\Bytes) + " key bytes, " + Str(extra\Phonemes\Bytes) + " pronunciation bytes, payload CRC-32 " + RSet(Hex(PeekL(pack\i + 72) & $FFFFFFFF), 8, "0") + ", " + Str(packBytes\i) + " file bytes"
      result = #True
    EndIf
  EndIf
  PmoKdpArenaFree(@scratch)
  PmoKdpArenaFree(@phonemes)
  If havePack : FreeMemory(pack\i) : EndIf
  If haveTable : PmoKdpTableFree(@extra) : EndIf
  If haveData : FreeMemory(rawFile\i) : EndIf
  If haveReader : PmoKdpReaderClose(@reader) : EndIf
  ProcedureReturn result
EndProcedure

; ---------------------------------------------------------------------------
; Core self-test. No file, no network, no dictionary data: it proves the
; hash, the checksum, the UTF-8 and JSON readers, the canonical order, the
; emitted layout, the inflection rules and the ARPABET conversion, and that
; the named rejections really do refuse.
; ---------------------------------------------------------------------------
Procedure.i PmoKdpSelfTestFail(Check.s)
  ProcedureReturn PmoKdpFail(PmoKdpCode(5001) + "the pronunciation-pack self-test failed its " + Check + " check, so this build cannot be trusted to produce a correct pack. Check the most recent change to the dictionary packing module.")
EndProcedure

Procedure.i PmoKdpSelfTestJson(Text.s, *Table.PmoKdpTable, ExpectedEntries.q)
  Protected json.PmoKdpJson
  Protected bytes.i = StringByteLength(Text, #PB_UTF8)
  Protected result.i
  Protected *buffer = AllocateMemory(bytes + 1)
  If *buffer = 0 : ProcedureReturn #False : EndIf
  PokeS(*buffer, Text, -1, #PB_UTF8)
  If PmoKdpTableInit(*Table, ExpectedEntries) = 0
    FreeMemory(*buffer)
    ProcedureReturn #False
  EndIf
  json\Data = *buffer : json\Bytes = bytes : json\Pos = 0 : json\Source = "the self-test document"
  result = PmoKdpJsonLoad(@json, *Table, "self-test lexicon")
  FreeMemory(*buffer)
  If result = 0 : PmoKdpTableFree(*Table) : EndIf
  ProcedureReturn result
EndProcedure

Procedure.i PmoKdpSelfTestRefuses(Document.s, *Table.PmoKdpTable, Check.s)
  If PmoKdpSelfTestJson(Document, *Table, 16) <> 0
    PmoKdpTableFree(*Table)
    PmoKokoroDictionaryPackError = ""
    ProcedureReturn PmoKdpSelfTestFail(Check)
  EndIf
  If PmoKokoroDictionaryPackError = ""
    ProcedureReturn PmoKdpSelfTestFail(Check + " message")
  EndIf
  PmoKokoroDictionaryPackError = ""
  ProcedureReturn #True
EndProcedure

Procedure.i PmoKokoroDictionarySelfTest()
  Protected table.PmoKdpTable
  Protected reader.PmoKdpReader
  Protected out.PmoKdpArena
  Protected probe.PmoKdpWord
  Protected zero.PmoKdpWord
  Protected document.s
  Protected quote.s = Chr(34)
  Protected backslash.s = Chr(92)
  Protected pack.Integer, packBytes.Integer
  Protected found.Integer, foundBytes.Integer
  Protected ready.Integer
  Protected *entry.PmoKdpEntry
  Protected result.i = #False
  Protected running.i = #True
  Protected havePack.i, haveTable.i
  PmoKokoroDictionaryPackError = ""
  PmoKdpInitTables()

  ; 1. FNV-1a against the published 32-bit vectors, and the 0 -> 1 remap.
  PmoKdpPutAscii(@probe\Byte[0], "a")
  If PmoKdpFnv1aRaw(@probe\Byte[0], 0) <> 2166136261 : ProcedureReturn PmoKdpSelfTestFail("FNV-1a offset basis") : EndIf
  If PmoKdpFnv1a(@probe\Byte[0], 1) <> $E40C292C : ProcedureReturn PmoKdpSelfTestFail("FNV-1a single byte") : EndIf
  PmoKdpPutAscii(@probe\Byte[0], "foobar")
  If PmoKdpFnv1a(@probe\Byte[0], 6) <> $BF9CF968 : ProcedureReturn PmoKdpSelfTestFail("FNV-1a multi byte") : EndIf
  zero\Byte[0] = $CC : zero\Byte[1] = $24 : zero\Byte[2] = $31 : zero\Byte[3] = $C4 : zero\Byte[4] = $00
  If PmoKdpFnv1aRaw(@zero\Byte[0], 5) <> 0 : ProcedureReturn PmoKdpSelfTestFail("FNV-1a zero preimage") : EndIf
  If PmoKdpFnv1a(@zero\Byte[0], 5) <> 1 : ProcedureReturn PmoKdpSelfTestFail("FNV-1a empty-slot remap") : EndIf

  ; 2. CRC-32 against its published check value.
  PmoKdpPutAscii(@probe\Byte[0], "123456789")
  If PmoKavCrc32(@probe\Byte[0], 9) <> $CBF43926 : ProcedureReturn PmoKdpSelfTestFail("payload CRC-32") : EndIf

  ; 3. The vocabulary really is the pinned set.
  If PmoKdpVocabContains($0251) = 0 Or PmoKdpVocabContains(32) = 0 Or PmoKdpVocabContains($AB67) = 0
    ProcedureReturn PmoKdpSelfTestFail("vocabulary membership")
  EndIf
  If PmoKdpVocabContains($0041) = 0 Or PmoKdpVocabContains($0042) <> 0 Or PmoKdpVocabContains(0) <> 0
    ProcedureReturn PmoKdpSelfTestFail("vocabulary exclusion")
  EndIf

  ; 4. A lexicon that exercises a plain string, a DEFAULT variant object, a
  ;    first-string fallback, an unusable null value and a unicode escape.
  document = "{" + quote + "dog" + quote + ":" + quote + "d" + Chr($0254) + Chr($0261) + quote + "," +
             quote + "cat" + quote + ":{" + quote + "NOUN" + quote + ":null," + quote + "DEFAULT" + quote + ":" + quote + "kat" + quote + "}," +
             quote + "bus" + quote + ":{" + quote + "VERB" + quote + ":" + quote + "b" + Chr($028C) + "s" + quote + "}," +
             quote + "ant" + quote + ":null," +
             quote + "a" + quote + ":" + quote + backslash + "u0251" + quote + "}"
  If PmoKdpSelfTestJson(document, @table, 64) = 0 : ProcedureReturn #False : EndIf
  haveTable = #True
  If table\Count <> 4 : PmoKdpSelfTestFail("lexicon selection") : running = #False : EndIf
  If running
    If PmoKdpSort(@table) = 0 : running = #False : EndIf
  EndIf
  If running
    *entry = table\Entries
    If *entry\KeyBytes <> 1 Or PeekS(table\Keys\Data + *entry\KeyOffset, 1, #PB_Ascii) <> "a"
      PmoKdpSelfTestFail("canonical order") : running = #False
    ElseIf *entry\PhonemeBytes <> 2
      PmoKdpSelfTestFail("unicode escape decoding") : running = #False
    EndIf
  EndIf
  If running
    *entry = table\Entries + 3 * SizeOf(PmoKdpEntry)
    If PeekS(table\Keys\Data + *entry\KeyOffset, *entry\KeyBytes, #PB_Ascii) <> "dog"
      PmoKdpSelfTestFail("canonical order tail") : running = #False
    EndIf
  EndIf

  ; 5. Emit that lexicon and verify the layout it produces.
  If running
    If PmoKdpEmit(@table, #PMO_KDP_KIND_EXTRA, #PMO_KDP_CMUDICT_SHA256$, "", #PMO_KDP_CMUDICT_REVISION$, @pack, @packBytes) = 0
      running = #False
    Else
      havePack = #True
    EndIf
  EndIf
  If running
    If PmoKdpVerifyMemory(pack\i, packBytes\i, #PMO_KDP_KIND_EXTRA, #False) = 0 : running = #False : EndIf
  EndIf
  If running
    If (PeekL(pack\i + 24) & $FFFFFFFF) <> 8
      PmoKdpSelfTestFail("slot count rounding") : running = #False
    ElseIf PeekQ(pack\i + 40) <> #PMO_KDP_HEADER_BYTES + 8 * #PMO_KDP_SLOT_BYTES
      PmoKdpSelfTestFail("key blob placement") : running = #False
    EndIf
  EndIf

  ; 6. Read the pack back through the same probe the target uses.
  If running
    PmoKdpReaderBind(@reader, pack\i, packBytes\i)
    PmoKdpPutAscii(@probe\Byte[0], "cat")
    If PmoKdpReaderOne(@reader, @probe\Byte[0], 3, @found, @foundBytes) = 0 Or PmoKdpEqualsUtf8(found\i, foundBytes\i, "kat") = 0
      PmoKdpSelfTestFail("exact lookup") : running = #False
    EndIf
  EndIf
  If running
    PmoKdpPutAscii(@probe\Byte[0], "ant")
    If PmoKdpReaderOne(@reader, @probe\Byte[0], 3, @found, @foundBytes) <> 0
      PmoKdpSelfTestFail("absent-word lookup") : running = #False
    EndIf
  EndIf
  If running
    PmoKdpPutAscii(@probe\Byte[0], "CAT")
    If PmoKdpReaderLookup(@reader, @probe\Byte[0], 3, @found, @foundBytes) = 0
      PmoKdpSelfTestFail("lowercase fallback") : running = #False
    EndIf
  EndIf

  ; 7. The regular inflection rules, including the voiced/voiceless split.
  If running
    out\Bytes = 0
    PmoKdpPutAscii(@probe\Byte[0], "cats")
    If PmoKdpReaderInflect(@reader, @probe\Byte[0], 4, @out) = 0 Or PmoKdpEqualsUtf8(out\Data, out\Bytes, "kats") = 0
      PmoKdpSelfTestFail("voiceless plural") : running = #False
    EndIf
  EndIf
  If running
    out\Bytes = 0
    PmoKdpPutAscii(@probe\Byte[0], "dogs")
    If PmoKdpReaderInflect(@reader, @probe\Byte[0], 4, @out) = 0 Or PmoKdpEqualsUtf8(out\Data, out\Bytes, "d" + Chr($0254) + Chr($0261) + "z") = 0
      PmoKdpSelfTestFail("voiced plural") : running = #False
    EndIf
  EndIf
  If running
    out\Bytes = 0
    PmoKdpPutAscii(@probe\Byte[0], "buses")
    If PmoKdpReaderInflect(@reader, @probe\Byte[0], 5, @out) = 0 Or PmoKdpEqualsUtf8(out\Data, out\Bytes, "b" + Chr($028C) + "s" + Chr($1D7B) + "z") = 0
      PmoKdpSelfTestFail("sibilant plural") : running = #False
    EndIf
  EndIf
  If running
    out\Bytes = 0
    PmoKdpPutAscii(@probe\Byte[0], "catting")
    If PmoKdpReaderInflect(@reader, @probe\Byte[0], 7, @out) = 0 Or PmoKdpEqualsUtf8(out\Data, out\Bytes, "kat" + Chr($026A) + Chr($014B)) = 0
      PmoKdpSelfTestFail("doubled progressive") : running = #False
    EndIf
  EndIf
  If running
    out\Bytes = 0
    PmoKdpPutAscii(@probe\Byte[0], "xyzzys")
    If PmoKdpReaderInflect(@reader, @probe\Byte[0], 6, @out) <> 0
      PmoKdpSelfTestFail("unknown stem refusal") : running = #False
    EndIf
  EndIf

  ; 8. ARPABET conversion, stress marks and the unstressed special cases.
  If running
    If PmoKdpArpabetPhone("B", "", @ready) <> "b" Or ready\i = 0 : PmoKdpSelfTestFail("ARPABET consonant") : running = #False : EndIf
  EndIf
  If running
    If PmoKdpArpabetPhone("AE", "1", @ready) <> Chr($02C8) + Chr($00E6) : PmoKdpSelfTestFail("ARPABET primary stress") : running = #False : EndIf
  EndIf
  If running
    If PmoKdpArpabetPhone("AA", "2", @ready) <> Chr($02CC) + Chr($0251) : PmoKdpSelfTestFail("ARPABET secondary stress") : running = #False : EndIf
  EndIf
  If running
    If PmoKdpArpabetPhone("AH", "0", @ready) <> Chr($0259) : PmoKdpSelfTestFail("ARPABET unstressed AH") : running = #False : EndIf
  EndIf
  If running
    If PmoKdpArpabetPhone("ER", "0", @ready) <> Chr($0259) + Chr($0279) : PmoKdpSelfTestFail("ARPABET unstressed ER") : running = #False : EndIf
  EndIf
  If running
    PmoKdpArpabetPhone("QQ", "", @ready)
    If ready\i <> 0 : PmoKdpSelfTestFail("unknown ARPABET phone refusal") : running = #False : EndIf
  EndIf

  ; 9. The named rejections. Each must refuse, and each must say why.
  If running
    PmoKdpTableFree(@table) : haveTable = #False
    PmoKokoroDictionaryPackError = ""
    document = "{" + quote + "dog" + quote + ":" + quote + "d" + quote + "," + quote + "cat" + quote + "}"
    If PmoKdpSelfTestRefuses(document, @table, "malformed-line rejection") = 0 : running = #False : EndIf
  EndIf
  If running
    document = "{" + quote + "dog" + quote + ":" + quote + "d" + quote + "," + quote + "dog" + quote + ":" + quote + "t" + quote + "}"
    If PmoKdpSelfTestRefuses(document, @table, "duplicate-key rejection") = 0 : running = #False : EndIf
  EndIf
  If running
    document = "{" + quote + "dog" + quote + ":" + quote + "B" + quote + "}"
    If PmoKdpSelfTestRefuses(document, @table, "out-of-vocabulary rejection") = 0 : running = #False : EndIf
  EndIf
  If running
    document = "{" + quote + "dog" + quote + ":" + quote + "d" + backslash + "ud800" + quote + "}"
    If PmoKdpSelfTestRefuses(document, @table, "lone-surrogate rejection") = 0 : running = #False : EndIf
  EndIf
  If running
    PmoKokoroDictionaryPackError = ""
    result = #True
  EndIf
  PmoKdpArenaFree(@out)
  If havePack : FreeMemory(pack\i) : EndIf
  If haveTable : PmoKdpTableFree(@table) : EndIf
  ProcedureReturn result
EndProcedure

DataSection
  PmoKdpVocabData:
  Data.l 32, 33, 34, 40, 41, 44, 46, 58, 59, 63, 65, 73
  Data.l 79, 81, 83, 84, 87, 89, 97, 98, 99, 100, 101, 102
  Data.l 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115
  Data.l 116, 117, 118, 119, 120, 121, 122, 230, 231, 240, 248, 331
  Data.l 339, 592, 593, 594, 596, 597, 598, 601, 602, 603, 604, 607
  Data.l 609, 611, 612, 613, 616, 618, 623, 624, 626, 627, 628, 632
  Data.l 633, 635, 637, 638, 641, 642, 643, 648, 650, 651, 652, 654
  Data.l 658, 660, 669, 675, 676, 677, 678, 679, 680, 688, 690, 712
  Data.l 716, 720, 771, 946, 952, 967, 7498, 7517, 7547, 8212, 8220, 8221
  Data.l 8230, 8594, 8595, 8599, 8600, 43879
EndDataSection
