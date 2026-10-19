"""Prepare a resident Kitten reply session using Anvil's Bluetooth stack."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

SUPPORTED_SOURCE_SHA256 = "f12075e4c0ee0d832ff858724cc21d56bcc5cf51c22bed174b52c2e53407805c"


def procedure(text, name):
    match = re.search(r"Procedure(?:\.\w+)? " + name + r"\(.*?\nEndProcedure", text, re.S)
    if not match:
        raise ValueError("Required generated procedure missing: " + name)
    return match[0]


def rewrite_model(model, out):
    original = model.read_text()
    source = original
    flat = []
    for match in re.finditer(r"Procedure PmModelStage1_\d+\(\)\n(.*?)\nEndProcedure", original, re.S):
        flat.append(re.sub(r"^  If DRejectProgressReentry\(\)\s+ProcedureReturn\s+EndIf\s*", "", match[1]))
    flat = "\n".join(flat)
    excitation = flat[flat.index("  DNode=1349\n"):flat.index("  DNode=1459\n")]
    assert 'DConcat(2179,"2178,2173",1)' in excitation
    cut = source.index("  DNode=1348\n")
    stage = source.rfind("Procedure PmModelStage1_", 0, cut)
    if not source[stage:].startswith("Procedure PmModelStage1_16()"):
        raise ValueError("The supported vocoder stage boundary changed.")
    source = source[:cut] + "EndProcedure\nProcedure KittenWindowTailFirst()\n" + source[cut:]

    def remove_excitation(match):
        body = match[0]
        for node in range(1349, 1459):
            body = re.sub(r"  DNode=" + str(node) + r"\n.*?(?=  DNode=|\nEndProcedure)", "", body, flags=re.S)
        return body

    source = re.sub(r"Procedure (?:KittenWindowTailFirst|PmModelStage1_\d+)\(\)\n.*?\nEndProcedure", remove_excitation, source, flags=re.S)
    tail_start = source.index("Procedure KittenWindowTailFirst()")
    tail_end = source.index("Procedure PmModelStage1()")
    source = source[:tail_start] + source[tail_start:tail_end].replace("  DRelease(1948)", "  ; Shared immutable style remains live across windows.") + source[tail_end:]
    old = procedure(source, "PmModelStage1")
    cut = old.index("  PmModelStage1_16()") + len("  PmModelStage1_16()")
    front = old[:cut].replace("Procedure PmModelStage1()", "Procedure KittenWindowFrontend()") + "\nEndProcedure"
    tail = "Procedure KittenWindowTail()\n  KittenWindowTailFirst()\n" + old[cut:]
    fragment = Path(__file__).with_name("KittenReplyWindows.pi4.in").read_text()
    boundary = fragment.index("Procedure PmModelStage1()")
    replacement = fragment[:boundary] + front + "\n" + tail + "\nProcedure KittenPrepareExcitation()\n" + excitation + "\nEndProcedure\n" + fragment[boundary:]
    source = source.replace(old, replacement, 1)
    needle = "  DCast(1701,1700,7)"
    if source.count(needle) != 1:
        raise ValueError("The supported duration output changed.")
    member = chr(92)
    source = source.replace(needle, needle + f"\n  If Dt(1701){member}Data<>0 And Dt(1701){member}Count>0\n    If PeekI(Dt(1701){member}Data)>8 : PokeI(Dt(1701){member}Data,8) : EndIf\n  EndIf")
    source = source.replace('"kitten.runtime' + member, '"' + (model.parent / "kitten.runtime").as_posix() + "/")
    out.write_text(source, encoding="utf-8")


def rewrite_reader(source, model, prepared=False):
    source = source.replace('@MODEL_SOURCE@', model.as_posix())
    playback = Path(__file__).with_name("KittenReplyPlayback.pi4.in").read_text()
    pump = procedure(playback, "BtPump")
    source = source.replace(procedure(source, "BtPump"), pump)
    globals_end = playback.index("Procedure.i ReplyPublish")
    source = source.replace("Declare BtPump()", playback[:globals_end] + "Declare BtPump()")
    if prepared:
        cache = Path(__file__).with_name("KittenReplyCache.pi4.in").read_text()
        source = source.replace("Procedure BtReport()", cache + "\nProcedure BtReport()")
    else:
        source = source.replace("Procedure.i KtChunk(k.i)", procedure(playback, "ReplyPublish") + "\nProcedure.i KtChunk(k.i)")
        start = source.index("  keep=count-#KT_TRIM", source.index("Procedure.i KtChunk"))
        end = source.index("\nEndProcedure", start)
        source = source[:start] + Path(__file__).with_name("KittenReplyChunkTail.pi4.in").read_text().rstrip() + source[end:]
    api = Path(__file__).with_name("KittenReplySession.pi4.in").read_text()
    if prepared:
        old = procedure(api, "ReplyStart")
        new = old.replace("ReplyDone=0", "ReplyDone=1")
        new = new.replace("If KtChunk(index)<>0 Or BtFault", "If ReplyCacheSubmit(index)=0 Or BtFault")
        new = new.replace("KittenWindowFirstTicks", "0")
        api = api.replace(old, new)
        api = api.replace("  *KittenWindowPublish=@ReplyPublish\n", "")
        api = api.replace("  If BtStart()=0 :", "  For status=1 To limit\n    If ReplyCachePrepare(status)=0 : ProcedureReturn KtFinish(600+status) : EndIf\n  Next\n  KtWaveUsed=0 : KtResultSamples=0 : KtResultRequests=0\n  If BtStart()=0 :")
    api_end = api.index("; Bounded demonstration loop")
    source = source.replace("Procedure.i Main()", api[:api_end] + "\nProcedure.i Main()")
    source = source.replace("Procedure.i Main()\n", "Procedure.i Main()\n  Protected sessionStarted.i\n")
    start = source.index("  status=KtChunk(1) :", source.index("Procedure.i Main()"))
    end = source.index("  BtStage=4", start)
    main = api[api.index("\n", api_end)+1:]
    source = source[:start] + main + source[end:]
    source = source.replace("  KtResultTotalTicks=KtTicks()-started", "  KtResultTotalTicks=KtTicks()-sessionStarted")
    source = source.replace("PeekI($52000008)<2", "PeekI($52000008)<1")
    source = source.replace("If limit<2 Or limit>PeekI($52000008)", "If limit<1 Or limit>128 Or limit>PeekI($52000008)")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anvil", "model", "inputs", "firmware", "private-config", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepared", action="store_true", help="Generate complete instructions once before connecting; timed requests then play cached Pi audio")
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != SUPPORTED_SOURCE_SHA256:
        parser.error("The reply adapter requires the hardware-validated Kitten 0.8 INT8 source. Revalidate the adapter when the emitted graph changes.")
    data = args.inputs.read_bytes()
    if len(data) < 64 or data[:8] != b"KTBK0001" or not 1 <= struct.unpack_from("<Q", data, 8)[0] <= 128:
        parser.error("Supply a prepared pack of 1..128 short replies.")
    command = [sys.executable, str(Path(__file__).with_name("prepare_reader.py"))]
    for name in ("anvil", "model", "firmware", "private-config", "out"):
        command += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    command += ["--book", str(args.inputs)]
    subprocess.run(command, check=True)
    out = args.out.resolve()
    model = args.model.resolve()
    if not args.prepared:
        model = out / "KittenReplyModel.pi4"
        rewrite_model(args.model.resolve(), model)
    source = Path(__file__).with_name("KittenBluetoothBook.pi4.in").read_text().replace("@ANVIL_ROOT@", args.anvil.resolve().as_posix())
    reader = out / "KittenBookReader.pi4"
    reader.write_text(rewrite_reader(source, model, args.prepared), encoding="utf-8")
    reader.replace(out / "KittenReplySession.pi4")
    manifest_path = out / "book.run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "kitten-prepared-directions" if args.prepared else "kitten-warm-replies"
    manifest["payload"]["file"] = "KittenReplySession.img.pmf"
    manifest["results"] += [dict(name="replies", address="0x58004000", bytes=8192), dict(name="preparation" if args.prepared else "publication", address="0x58C01000", bytes=4096)]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path.replace(out / "reply.run.json")
    print("Prepared resident Kitten with a persistent A2DP connection and " + ("cached directions." if args.prepared else "fresh streamed replies."))


if __name__ == "__main__":
    main()
