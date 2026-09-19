#!/usr/bin/env python3
"""f32_to_wav.py - turn the waveform read off the board into a WAV you can play.

A DEVELOPER TOOL FOR THE LAST STEP OF THE PI 4 RECIPE. Building the ONNX
compiler, generating a model with it and building that model never need Python
or this file. It touches no board: it reads a file kokoro_readback.py already
proved came off one.

The board returns raw little-endian FLOAT32 samples, mono, 24 kHz. A media
player wants a RIFF file. This writes two:

    <wav>          16-bit PCM, the one to listen to
    <wav stem>.f32.wav
                   32-bit IEEE float, a lossless copy of exactly the samples
                   the board returned, for anyone re-checking the numbers

and <wav>.json beside them, with the sizes, hashes and sample statistics.

THE CONVERSION, STATED RATHER THAN ASSUMED. A sample is clamped to [-1, +1],
multiplied by 32767 and rounded half away from zero. Every clamped sample is
COUNTED and reported: a waveform that clipped is a finding about the model's
output, not a detail of the file format, so it is never silent. The requested
sentence clamped none.

  python f32_to_wav.py --raw run/kokoro-sentence.f32 --wav run/kokoro-sentence.wav

--expect-sha256 refuses a raw file whose digest is not the one given, for
converting a file whose identity is already known. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def riff(fmt_chunk: bytes, data: bytes) -> bytes:
    size = 4 + 8 + len(fmt_chunk) + 8 + len(data)
    return (b"RIFF" + struct.pack("<I", size) + b"WAVEfmt " +
            struct.pack("<I", len(fmt_chunk)) + fmt_chunk +
            b"data" + struct.pack("<I", len(data)) + data)


def float_wav(samples: list[float], rate: int) -> bytes:
    # WAVE_FORMAT_IEEE_FLOAT (3), mono, 32-bit.
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    return riff(fmt, struct.pack("<%df" % len(samples), *samples))


def pcm16_wav(samples: list[float], rate: int) -> tuple[bytes, int]:
    # WAVE_FORMAT_PCM (1), mono, 16-bit.
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    clamped = 0
    out = bytearray()
    for value in samples:
        if value > 1.0:
            value, clamped = 1.0, clamped + 1
        elif value < -1.0:
            value, clamped = -1.0, clamped + 1
        scaled = value * 32767.0
        whole = int(scaled + 0.5) if scaled >= 0 else int(scaled - 0.5)
        out += struct.pack("<h", whole)
    return riff(fmt, bytes(out)), clamped


class ConvertError(RuntimeError):
    """A refusal, always a full sentence."""


def convert(raw_path: Path, wav_path: Path, rate: int, text: str,
            expect_sha256: str | None) -> dict:
    float_path = wav_path.with_suffix(".f32.wav")
    sidecar = wav_path.with_name(wav_path.name + ".json")
    for path in (wav_path, float_path, sidecar):
        if path.exists():
            raise ConvertError(
                "Error 1: %s already exists and this tool never replaces a "
                "file. Move it, or give --wav a new name." % path)
    if not raw_path.is_file():
        raise ConvertError(
            "Error 2: there is no raw waveform at %s. It is the file "
            "kokoro_readback.py wave writes (run/kokoro-sentence.f32 by "
            "default)." % raw_path)
    raw = raw_path.read_bytes()
    raw_sha = digest(raw)
    if expect_sha256 and raw_sha != expect_sha256.lower():
        raise ConvertError(
            "Error 3: %s has sha256 %s, not the %s that was expected. Nothing "
            "was written." % (raw_path, raw_sha, expect_sha256.lower()))
    if not raw or len(raw) % 4:
        raise ConvertError(
            "Error 4: %s is %d bytes, which is not a whole, non-zero number of "
            "FLOAT32 samples. The readback was truncated or this is not a "
            "waveform. Nothing was written." % (raw_path, len(raw)))
    samples = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if any(value != value or value in (float("inf"), float("-inf")) for value in samples):
        raise ConvertError(
            "Error 5: %s holds a NaN or infinity. The payload refuses to copy "
            "such a waveform, so this file did not come from a clean run. "
            "Nothing was written." % raw_path)
    pcm, clamped = pcm16_wav(samples, rate)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(pcm)
    floats = float_wav(samples, rate)
    float_path.write_bytes(floats)
    record = {
        "what_this_is": "A waveform returned by the resident Kokoro payload on "
                        "a Raspberry Pi 4, written as a playable WAV.",
        "text": text,
        "source_raw": raw_path.name,
        "source_raw_bytes": len(raw),
        "source_raw_sha256": raw_sha,
        "sample_count": len(samples),
        "sample_rate_hz": rate,
        "seconds": len(samples) / float(rate),
        "channels": 1,
        "minimum_sample": min(samples),
        "maximum_sample": max(samples),
        "clamped_samples": clamped,
        "pcm16": {"file": wav_path.name, "bytes": len(pcm), "sha256": digest(pcm)},
        "float32": {"file": float_path.name, "bytes": len(floats), "sha256": digest(floats)},
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--raw", type=Path, default=here / "run" / "kokoro-sentence.f32",
                        help="the raw FLOAT32 samples (default: %(default)s)")
    parser.add_argument("--wav", type=Path, default=here / "run" / "kokoro-sentence.wav",
                        help="the 16-bit WAV to write (default: %(default)s)")
    parser.add_argument("--rate", type=int, default=24000,
                        help="samples per second (default: %(default)s, the model's)")
    parser.add_argument("--text", default="", help="the sentence, for the record")
    parser.add_argument("--expect-sha256", default=None,
                        help="refuse a raw file with any other SHA-256")
    args = parser.parse_args(argv)
    try:
        record = convert(args.raw, args.wav, args.rate, args.text, args.expect_sha256)
    except ConvertError as error:
        print("!! %s" % error)
        return 2
    print("%s: %d samples, %.3f s at %d Hz, %d clamped"
          % (args.wav, record["sample_count"], record["seconds"], args.rate,
             record["clamped_samples"]))
    print("  pcm16   %d bytes  sha256 %s" % (record["pcm16"]["bytes"], record["pcm16"]["sha256"]))
    print("  float32 %d bytes  sha256 %s" % (record["float32"]["bytes"], record["float32"]["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
