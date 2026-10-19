"""Materialize the Pi reader and a bounded Anvil board-run manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anvil", "model", "book", "firmware", "private-config", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    for item in (args.model, args.book, args.firmware, args.private_config, args.model.with_suffix(".pmw")):
        if not item.is_file():
            parser.error(f"Required file missing: {item}")
    if len(args.private_config.read_bytes()) != 80:
        parser.error("The private speaker configuration must be 80 bytes.")
    if args.out.exists():
        parser.error("Output folder exists; choose a new folder.")
    template = Path(__file__).with_name("KittenBluetoothBook.pi4.in").read_text()
    text = template.replace("@ANVIL_ROOT@", args.anvil.resolve().as_posix()).replace("@MODEL_SOURCE@", args.model.resolve().as_posix())
    args.out.mkdir(parents=True)
    args.out.joinpath("KittenBookReader.pi4").write_text(text, encoding="utf-8")
    assets = []
    for name, file, address in [("weights", args.model.with_suffix(".pmw"), "0x44000000"),
                                 ("book", args.book, "0x52000000"),
                                 ("bluetooth-firmware", args.firmware, "0x53000000"),
                                 ("private-speaker-config", args.private_config, "0x53010000")]:
        assets.append(dict(name=name, file=str(file.resolve()), address=address,
                           sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    manifest = {"format": "anvil-board-model-run 1", "name": "kitten-book", "base": ".", "out": "runs",
                "payload": {"file": "KittenBookReader.img.pmf", "stage": "0x03000000", "expect_x0": "0", "timeout_seconds": 240},
                "deadman_seconds": 15, "assets": assets,
                "results": [{"name": "playback", "address": "0x58001000", "bytes": 192},
                            {"name": "chunks", "address": "0x58002000", "bytes": 4096}]}
    args.out.joinpath("book.run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared reader source and manifest. Build the payload with PureMetal Forge before running.")


if __name__ == "__main__":
    main()
