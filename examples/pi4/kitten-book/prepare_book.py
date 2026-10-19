"""Prepare Kitten 0.8.1 text inputs; synthesis runs on the Pi, not this host."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="Folder containing voices.npz and config.json")
    parser.add_argument("--frontend", required=True, type=Path, help="Extracted kittentts 0.8.1 folder containing onnx_model.py and preprocess.py")
    parser.add_argument("--text", required=True, type=Path, help="UTF-8 plain text")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--voice", default="expr-voice-4-f")
    parser.add_argument("--max-chars", default=100, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Read only the first N chunks; zero reads all")
    args = parser.parse_args()
    if not 40 <= args.max_chars <= 400 or args.limit < 0:
        parser.error("Use 40..400 characters and a nonnegative chunk limit.")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("Output exists; choose another name.")
    import numpy as np
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    source = args.frontend / "onnx_model.py"
    names = {"TextCleaner", "basic_english_tokenize", "ensure_punctuation", "chunk_text"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    scope = {}
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in body} != names:
        raise ValueError("The released front end does not contain the expected definitions.")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), scope)
    spec = importlib.util.spec_from_file_location("kitten_preprocess", args.frontend / "preprocess.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    os.environ["ESPEAK_DATA_PATH"] = str(espeakng_loader.get_data_path())
    backend = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
    text = args.text.read_text(encoding="utf-8-sig")
    chunks = scope["chunk_text"](pre.TextPreprocessor(remove_punctuation=False)(text), max_len=args.max_chars)
    if not chunks or args.limit > len(chunks):
        raise ValueError("No text chunks, or limit exceeds the chunk count.")
    config = json.loads((args.assets / "config.json").read_text())
    with np.load(args.assets / "voices.npz") as voices:
        table = voices[args.voice].astype("<f4")
    speed = np.float32(config["speed_priors"][args.voice])
    cleaner = scope["TextCleaner"]()
    pack = bytearray(64 + len(chunks) * 32)
    pack[:8] = b"KTBK0001"
    struct.pack_into("<QQQQ", pack, 8, len(chunks), 0, 64, args.limit)
    records = []
    for k, chunk in enumerate(chunks):
        phones = backend.phonemize([chunk])[0]
        joined = " ".join(scope["basic_english_tokenize"](phones))
        ids = np.array([0] + cleaner(joined) + [10, 0], dtype="<i8")
        if not 3 <= len(ids) <= 512:
            raise ValueError(f"Chunk {k+1} has {len(ids)} tokens; choose shorter chunks.")
        row = min(len(chunk), table.shape[0] - 1)
        pack += bytes((-len(pack)) % 16)
        tokens_at = len(pack)
        pack += ids.tobytes()
        pack += bytes((-len(pack)) % 16)
        style_at = len(pack)
        pack += table[row].tobytes()
        struct.pack_into("<QQQfI", pack, 64+k*32, tokens_at, len(ids), style_at, speed, 0)
        records.append({"text": chunk, "tokens": len(ids), "style_row": row,
                        "dropped_symbols": sorted(set(c for c in joined if c not in cleaner.word_index_dictionary))})
    if len(pack) > 0x01000000:
        raise ValueError("Pack exceeds the 16 MiB input slot; divide the book into parts.")
    struct.pack_into("<Q", pack, 16, len(pack))
    provenance = {"format": "Kitten book inputs 1", "frontend_version": "0.8.1",
                  "voice": args.voice, "speed": float(speed), "max_chars": args.max_chars,
                  "limit": args.limit, "pack_sha256": hashlib.sha256(pack).hexdigest(),
                  "inputs": {str(f): sha(f) for f in [args.text, source, args.frontend/"preprocess.py",
                            args.assets/"config.json", args.assets/"voices.npz"]}, "chunks": records}
    args.out.write_bytes(pack)
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {len(chunks)} chunks, {len(pack)} bytes; speech is generated on the Pi.")


if __name__ == "__main__":
    main()
