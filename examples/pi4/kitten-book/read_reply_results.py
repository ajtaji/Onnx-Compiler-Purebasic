"""Decode a bounded warm-reply run without confusing cached and fresh latency."""
from pathlib import Path
import argparse
import json
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    player = struct.unpack("<24q", (args.run / "playback.bin").read_bytes())
    replies = (args.run / "replies.bin").read_bytes()
    chunks = (args.run / "chunks.bin").read_bytes()
    prepared_path = args.run / "preparation.bin"
    prepared = prepared_path.read_bytes() if prepared_path.exists() else None
    report = dict(mode="prepared" if prepared else "fresh", cores=player[22],
                  bluetooth_fault=player[2], receive_errors=player[6], late_events=player[10],
                  max_pump_gap_ms=player[13]*1000/player[21], replies=[])
    for index in range(min(player[23], 128)):
        value = struct.unpack_from("<8q", replies, index*64)
        chunk = struct.unpack_from("<4q", chunks, index*32)
        start, packet, scheduled, hz, complete, first_window, underruns, active = value
        item = dict(index=index+1, packet_seconds=(packet-start)/hz,
                    first_voice_stream_seconds=(scheduled-start)/hz,
                    scheduled_audio_seconds=(active-start)/hz if active else (scheduled-start)/hz,
                    completed_seconds=(complete-start)/hz, audio_seconds=chunk[2]/24000,
                    speech_queue_underrun_frames=underruns)
        if prepared:
            item["preparation_seconds"] = struct.unpack_from("<q", prepared, index*32)[0]/hz
        else:
            item["model_seconds"] = chunk[1]/hz
            item["first_window_seconds"] = first_window/hz
        report["replies"].append(item)
    report["timing_definition"] = "Stream scheduling and packet timestamps; acoustic speaker latency is not measured. Prepared synthesis precedes the timed request."
    text = json.dumps(report, indent=2)
    (args.run / "reply-summary.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
