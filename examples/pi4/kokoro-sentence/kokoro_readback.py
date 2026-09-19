#!/usr/bin/env python3
"""kokoro_readback.py - take a Kokoro sentence run's result off an Anvil board.

A DEVELOPER TOOL FOR ONE STEP OF THE PI 4 RECIPE. Building the ONNX compiler,
generating a model with it and building that model never need Python or this
file. It exists because the waveform a board run makes stays in the board's
memory, and something on the host has to fetch it and prove it is the same
bytes.

IT USES THE ANVIL REPOSITORY'S OWN HOST CODE, BY PATH, NOT A COPY OF IT:

    <anvil>/tools/board_run.py       NetConsole - the UDP console client and
                                     its echo-anchored command reader
    <anvil>/tools/anvil_readback.py  read_range - the one verified reader for
                                     the monitor's `readback` command

`--anvil` names the Anvil checkout. The default is a folder called Anvil beside
this repository's checkout, which is the layout the recipe uses. Neither file
needs anything outside Python's standard library.

EVERY COMMAND THIS SENDS IS READ-ONLY: `coretest status`, `version`,
`sha256sum` and `readback`. It never uploads, writes memory or starts a payload.
`readback` needs Anvil build 151 or later.

THREE COMMANDS

  cores   print the monitor's four-core dispatch counter. Run it just before
          and just after the payload: the difference is how many times the
          convolutions were split across the cores (84 for the sentence).

            python kokoro_readback.py cores --console-ip 192.168.137.1

  status  read the payload's KTRC trace and its result block, and say whether
          the run finished and what it returned. This is the answer to use
          when board_run.py says the payload never returned: the trace is
          written by the payload itself, at every node, so it does not depend
          on the console having heard the return line. Exit 0 only for a
          finished run with status 0.

            python kokoro_readback.py status --console-ip 192.168.137.1

  wave    read the waveform. The board hashes the range, the bytes come back
          through `readback` (whose length and crc32 are checked per request),
          the board hashes the range again, and all three SHA-256 digests must
          agree. The XOR of the sample words must also equal the hash the
          payload computed on the board before it copied them. Only then are
          <out>, <out>.json and <out>.transcript.txt written.

            python kokoro_readback.py wave --console-ip 192.168.137.1

--sym is the symbol file the compiler wrote beside the payload image. It is
how the result block is found: the addresses of the payload's globals are
the build's, not constants of this tool. Default: build/kokoro-sentence.img.sym
beside this file.

  python kokoro_readback.py --self-test --anvil <anvil>
      runs the decoding and verification paths against a scripted console
      that answers the way the monitor does. No board is contacted.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import struct
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_ANVIL = REPOSITORY.parent / "Anvil"
DEFAULT_SYM = HERE / "build" / "kokoro-sentence.img.sym"
DEFAULT_OUT = HERE / "run" / "kokoro-sentence.f32"

# Where KokoroSentence.pi4 puts things. They are #KR_* constants in that file.
TRACE_ADDRESS = 0x57E00000
TRACE_BYTES = 256
TRACE_MAGIC = 0x4352544B          # "KTRC"
TRACE_FINAL_STAGE = 9
WAVE_ADDRESS = 0x57400000         # #KR_COPY3_BASE
WAVE_SLOT_BYTES = 2097152         # #KR_COPY_STRIDE
SAMPLE_RATE = 24000
DEFAULT_TICK_HZ = 54000000

# The result-block globals this reads, by symbol, with their widths. A .l
# global is 4 bytes and a .i global is 8 on this 64-bit target.
RESULT_FIELDS = [
    ("krresultstatus", 4, "status"),
    ("krresultrequests", 4, "requests"),
    ("krresulthz", 8, "tick_hz"),
    ("krresulttotalticks", 8, "total_ticks"),
    ("krresultservicestatus", 8, "service_status"),
    ("krparallelcores", 8, "parallel_cores_reported"),
    ("krresultmodelbinderror", 8, "model_bind_error"),
    ("krresulttextbinderror", 8, "text_bind_error"),
    ("krresulttexterror3", 8, "text_error"),
    ("krresultg2perror3", 8, "g2p_error"),
    ("krresultg2perrorat3", 8, "g2p_error_at"),
    ("krresultderror3", 8, "derror"),
    ("krresulttokens3", 8, "tokens"),
    ("krresultsamples3", 8, "samples"),
    ("krresultbytes3", 8, "bytes"),
    ("krresultticks3", 8, "request_ticks"),
    ("krresulthash3", 4, "sample_xor"),
]

STATUS_MEANING = {
    0: "a clean run",
    1: "the generic timer reported no frequency",
    2: "the emitted model expects a different weight-pack size",
    3: "the model refused its weights or arena - check the weight pack at $40000000",
    4: "the text adapter refused a pronunciation pack or the voice - check the three assets",
    5: "there was no usable service table in x0 - build the payload with --wants-services",
    6: "the model runtime refused the progress callback",
    7: "the payload was not at EL2 or EL3, or the MMU could not be built from that state",
    8: "the page tables could not be built",
    9: "the caches did not turn on",
    90: "working memory was still in use after close",
    91: "the model was still bound after close",
    92: "the text adapter still held an asset after close",
    93: "the weights changed during the run",
    94: "the stack ran into its canary",
    95: "the MMU was still on at exit",
    301: "the render returned no tensor - see text_error, g2p_error and derror",
    311: "the output tensor id was out of range",
    312: "the output was not a non-empty FLOAT tensor",
    313: "the sentence is too long for the 2 MiB copy slot",
    314: "a NaN or infinity was among the samples, so nothing was copied",
    320: "the token count differs from #KR_EXPECT_TOKENS3; the samples were still copied",
    321: "the run took longer than #KR_TIMEOUT_SECONDS",
}


class RunError(RuntimeError):
    """A refusal, always a full sentence."""


def load_anvil(anvil: Path):
    """Import the Anvil repository's console client and readback reader."""
    tools = anvil / "tools"
    for name in ("board_run.py", "anvil_readback.py"):
        if not (tools / name).is_file():
            raise RunError(
                "Error 1: there is no %s under %s, so this tool has no console "
                "client to use. Check --anvil: it must name a checkout of the "
                "Anvil repository (the default is a folder called Anvil beside "
                "this repository's checkout)." % (name, tools))
    sys.path.insert(0, str(tools))
    import board_run  # noqa: E402
    import anvil_readback  # noqa: E402
    return board_run, anvil_readback


def read_symbols(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise RunError(
            "Error 2: the symbol file %s does not exist. It is written beside "
            "the image when the payload is compiled (kokoro-sentence.img.sym); "
            "give its path with --sym." % path)
    symbols = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name, sep, value = line.partition("=")
        if sep and value.strip().lstrip("-").isdigit():
            symbols[name.strip().lower()] = int(value.strip())
    return symbols


def result_addresses(symbols: dict[str, int]) -> list[tuple[str, int, int, str]]:
    fields = []
    for symbol, width, key in RESULT_FIELDS:
        address = symbols.get("global_" + symbol)
        if address is None:
            raise RunError(
                "Error 3: the symbol file has no global_%s, so it is not the "
                "symbol file of KokoroSentence.pi4. Give --sym the .sym written "
                "beside the payload that ran." % symbol)
        fields.append((symbol, address, width, key))
    return fields


def read_memory(readback, console, address: int, count: int, evidence) -> bytes:
    try:
        data, _stats = readback.read_range(console, address, count,
                                           evidence=evidence, progress=None)
    except readback.ReadbackError as error:
        raise RunError(
            "Error 4: %d bytes at %08X could not be read back: %s If the board "
            "answered 'unknown command', its monitor is older than build 151, "
            "which is the first with readback." % (count, address, error))
    return data


def read_result(readback, console, symbols, evidence) -> dict:
    fields = result_addresses(symbols)
    low = min(address for _s, address, _w, _k in fields)
    high = max(address + width for _s, address, width, _k in fields)
    block = read_memory(readback, console, low, high - low, evidence)
    result = {}
    for _symbol, address, width, key in fields:
        raw = block[address - low: address - low + width]
        result[key] = struct.unpack("<I" if width == 4 else "<Q", raw)[0]
    for key in ("status",):
        if result[key] >= 0x80000000:
            result[key] -= 1 << 32
    return result


def decode_trace(raw: bytes) -> dict:
    magic, version, stage, counter = struct.unpack_from("<IIII", raw, 0)
    names = ("phase", "node", "ticks_now", "ticks_since_start", "derror",
             "dcancel", "service_status", "result_status", "requests")
    values = struct.unpack_from("<9q", raw, 16)
    trace = {"magic_ok": magic == TRACE_MAGIC, "version": version,
             "stage": stage, "writes": counter}
    trace.update(dict(zip(names, values)))
    return trace


def board_sha256(console, address: int, count: int) -> str:
    line = "sha256sum %X %X" % (address, count)
    reply = console.command(line, 900.0).replace("\r", "")
    extent = re.search(r"sha256sum over ([0-9]+) bytes, ([0-9A-Fa-f]+) \.\. ([0-9A-Fa-f]+)", reply)
    digest = re.search(r"The sha256sum is ([0-9a-fA-F]{64})", reply)
    if extent is None or digest is None:
        raise RunError(
            "Error 5: the board's reply to '%s' does not carry both the range "
            "it hashed and the digest, so it cannot be used as a witness. The "
            "board said: %s" % (line, reply.strip()[-400:]))
    got, first, last = int(extent.group(1)), int(extent.group(2), 16), int(extent.group(3), 16)
    if got != count or first != address or last != address + count - 1:
        raise RunError(
            "Error 6: the board hashed %d bytes, %08X .. %08X, but %d bytes at "
            "%08X were asked for. No digest was accepted."
            % (got, first, last, count, address))
    return digest.group(1).lower()


def dispatch_count(console) -> tuple[int, str]:
    reply = console.command("coretest status", 30.0).replace("\r", "")
    found = re.search(r"coretest started=([0-9]+) .*?nonce=([0-9]+) error=([0-9]+)", reply)
    if found is None:
        raise RunError(
            "Error 7: the board's reply to 'coretest status' has no nonce, so "
            "the four-core dispatch count cannot be read. The board said: %s"
            % reply.strip()[-300:])
    return int(found.group(2)), found.group(0)


def monitor_build(console) -> str:
    reply = console.command("version", 30.0).replace("\r", "")
    found = re.search(r"^build ([0-9]+) date ([0-9]+) time ([0-9]+)", reply, re.M)
    return found.group(0) if found else "unknown (the version reply had no build line)"


def command_cores(args, board_run, readback, console) -> int:
    nonce, line = dispatch_count(console)
    print(line)
    print("dispatch counter: %d. Read it again after the run; the difference "
          "is how many times the payload split a convolution across the cores."
          % nonce)
    return 0


def command_status(args, board_run, readback, console, evidence) -> int:
    symbols = read_symbols(args.sym)
    build = monitor_build(console)
    trace = decode_trace(read_memory(readback, console, TRACE_ADDRESS, TRACE_BYTES, evidence))
    result = read_result(readback, console, symbols, evidence)
    nonce, core_line = dispatch_count(console)
    hz = result["tick_hz"] or DEFAULT_TICK_HZ
    report = {"monitor": build, "trace": trace, "result": result,
              "coretest": core_line}
    print("monitor   %s" % build)
    if not trace["magic_ok"]:
        print("trace     no KTRC record at %08X - the payload has not run since "
              "this board was reset, or something else overwrote it" % TRACE_ADDRESS)
        finished = False
    else:
        finished = trace["stage"] == TRACE_FINAL_STAGE
        print("trace     stage %d at node %d, %d writes, %.3f s since the model "
              "began binding%s" % (trace["stage"], trace["node"], trace["writes"],
                                   trace["ticks_since_start"] / hz,
                                   "" if finished else " - NOT FINISHED"))
    status = result["status"]
    print("status    %d, %s" % (status, STATUS_MEANING.get(status, "a code KokoroSentence.pi4 does not define")))
    print("request   %d tokens, %d samples (%.3f s of audio), %.3f s to render"
          % (result["tokens"], result["samples"], result["samples"] / SAMPLE_RATE,
             result["request_ticks"] / hz))
    print("samples   XOR of the sample words on the board: %08X" % result["sample_xor"])
    print("cores     %s" % core_line)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if (finished and status == 0) else 1


def command_wave(args, board_run, readback, console, evidence) -> int:
    out = args.out
    sidecar = out.with_name(out.name + ".json")
    transcript = out.with_name(out.name + ".transcript.txt")
    for path in (out, sidecar, transcript):
        if path.exists():
            raise RunError(
                "Error 8: %s already exists and this tool never replaces "
                "evidence. Move it, or give --out a new name." % path)
    symbols = read_symbols(args.sym)
    build = monitor_build(console)
    result = read_result(readback, console, symbols, evidence)
    count = args.bytes if args.bytes else result["bytes"]
    if count <= 0 or count % 4 or count > WAVE_SLOT_BYTES:
        raise RunError(
            "Error 9: the result block says the waveform is %d bytes, which is "
            "not a whole number of samples inside the 2 MiB copy slot. Run the "
            "status command first: the payload did not finish, or --sym is not "
            "the symbol file of the payload that ran." % count)
    print("reading %d samples (%d bytes) at %08X from a board on %s"
          % (count // 4, count, args.address, build))
    before = board_sha256(console, args.address, count)
    print("  board sha256 before  %s" % before)
    started = time.monotonic()
    data = read_memory(readback, console, args.address, count, evidence)
    seconds = time.monotonic() - started
    after = board_sha256(console, args.address, count)
    print("  board sha256 after   %s" % after)
    host = hashlib.sha256(data).hexdigest()
    print("  host  sha256         %s   (%d bytes in %.1f s)" % (host, len(data), seconds))
    if before != after:
        raise RunError(
            "Error 10: the board hashed the range to %s before the read and %s "
            "after it, so the bytes changed while they were being read. Nothing "
            "was saved. Check that no payload is running and no other session "
            "is using the board." % (before, after))
    if host != before:
        raise RunError(
            "Error 11: the board hashes the range to %s and the bytes read back "
            "hash to %s, so they are not the same bytes. Nothing was saved."
            % (before, host))
    xor = 0
    for (word,) in struct.iter_unpack("<I", data):
        xor ^= word
    if not args.bytes and xor != result["sample_xor"]:
        raise RunError(
            "Error 12: the XOR of the sample words read back is %08X, and the "
            "payload computed %08X over the same samples on the board before it "
            "copied them. The range holds different samples than the run "
            "produced. Nothing was saved." % (xor, result["sample_xor"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    evidence_record = {
        "what_this_is": "Raw little-endian FLOAT32 samples at %d Hz, mono, read "
                        "off the board with readback." % SAMPLE_RATE,
        "monitor": build,
        "address": "%08X" % args.address,
        "bytes": count,
        "samples": count // 4,
        "seconds_of_audio": count / 4 / SAMPLE_RATE,
        "sha256": host,
        "board_sha256_before": before,
        "board_sha256_after": after,
        "sample_xor": "%08X" % xor,
        "payload_sample_xor": "%08X" % result["sample_xor"],
        "payload_result": result,
        "read_seconds": round(seconds, 3),
    }
    sidecar.write_text(json.dumps(evidence_record, indent=2) + "\n", encoding="utf-8")
    transcript.write_text("".join(getattr(console, "transcript", [])), encoding="utf-8")
    print("PASS: %s" % out)
    return 0


# ---------------------------------------------------------------------------
#  SELF TEST: a console that answers the way the monitor does.
# ---------------------------------------------------------------------------
class ScriptedMonitor:
    """Duck-types board_run.NetConsole for the calls this tool makes."""

    def __init__(self, memory: dict[int, bytes], nonce: int = 84):
        self.memory = memory
        self.nonce = nonce
        self.pending: list[str] = []
        self.transcript: list[str] = []
        self.change_after_read = False
        self.reads = 0

    def peek(self, address: int, count: int) -> bytes:
        for base, blob in self.memory.items():
            if base <= address and address + count <= base + len(blob):
                return blob[address - base: address - base + count]
        return bytes(count)

    def settle(self, quiet: float = 0.6, cap: float = 8.0) -> str:
        return ""

    def send(self, line: str) -> None:
        self.transcript.append(">>> %s\n" % line)
        words = line.split()
        if words[:1] == ["readback"]:
            address, count = int(words[1], 16), int(words[2], 16)
            data = self.peek(address, count)
            if address == WAVE_ADDRESS:
                self.reads += 1
            text = "pmf> %s\nreadback %X %d bytes base64\n" % (line, address, count)
            for offset in range(0, len(data), 48):
                text += base64.b64encode(data[offset:offset + 48]).decode() + "\n"
            text += "readback end %d bytes crc32 %08X\npmf> " % (count, zlib.crc32(data) & 0xFFFFFFFF)
            # A realistic reply arrives in pieces.
            self.pending.extend(text[i:i + 700] for i in range(0, len(text), 700))

    def recv_some(self) -> str:
        return self.pending.pop(0) if self.pending else ""

    def command(self, line: str, seconds: float = 8.0) -> str:
        words = line.split()
        if words[0] == "sha256sum":
            address, count = int(words[1], 16), int(words[2], 16)
            data = self.peek(address, count)
            if self.change_after_read and self.reads:
                data = bytes([data[0] ^ 1]) + data[1:]
            return ("%s\nsha256sum over %d bytes, %08X .. %08X ...\nThe sha256sum is %s\npmf> "
                    % (line, count, address, address + count - 1, hashlib.sha256(data).hexdigest()))
        if line == "version":
            return "version\nbuild 151 date 20260916 time 105849\npmf> "
        if line == "coretest status":
            return ("coretest status\ncoretest started=1 stopped=0 runs=0 nonce=%d error=0 core=0\npmf> "
                    % self.nonce)
        return "%s\nunknown command\npmf> " % line


def self_test(args) -> int:
    import tempfile
    _board_run, readback = load_anvil(args.anvil)
    failures = []

    def check(name, condition):
        print("  %-60s %s" % (name, "ok" if condition else "FAILED"))
        if not condition:
            failures.append(name)

    samples = struct.pack("<3000f", *[((i * 37) % 200 - 100) / 250.0 for i in range(3000)])
    xor = 0
    for (word,) in struct.iter_unpack("<I", samples):
        xor ^= word
    base = 0x01000000
    block = bytearray(0x800)
    symbols = {}
    for index, (symbol, width, key) in enumerate(RESULT_FIELDS):
        address = base + index * 16
        symbols["global_" + symbol] = address
        value = {"status": 0, "requests": 1, "tick_hz": DEFAULT_TICK_HZ,
                 "tokens": 295, "samples": 3000, "bytes": 12000,
                 "request_ticks": 81 * DEFAULT_TICK_HZ, "sample_xor": xor}.get(key, 0)
        struct.pack_into("<I" if width == 4 else "<Q", block, address - base, value)
    trace = bytearray(TRACE_BYTES)
    struct.pack_into("<IIII", trace, 0, TRACE_MAGIC, 1, TRACE_FINAL_STAGE, 9862)
    struct.pack_into("<9q", trace, 16, 4, 2462, 47300000000, 8540801559, 0, 0, 0, 0, 1)
    memory = {base: bytes(block), TRACE_ADDRESS: bytes(trace), WAVE_ADDRESS: samples}

    with tempfile.TemporaryDirectory() as folder:
        sym = Path(folder) / "p.img.sym"
        sym.write_text("".join("%s=%d\n" % item for item in symbols.items()), encoding="utf-8")
        ns = argparse.Namespace(sym=sym, json=None, bytes=0, address=WAVE_ADDRESS,
                                out=Path(folder) / "run" / "w.f32")
        monitor = ScriptedMonitor(memory)
        evidence = lambda line, reply: None
        check("status of a finished clean run exits 0",
              command_status(ns, None, readback, monitor, evidence) == 0)
        monitor = ScriptedMonitor(memory)
        check("wave writes the samples when all hashes agree",
              command_wave(ns, None, readback, monitor, evidence) == 0
              and ns.out.read_bytes() == samples)
        try:
            command_wave(ns, None, readback, ScriptedMonitor(memory), evidence)
            check("wave refuses to replace existing evidence", False)
        except RunError as error:
            check("wave refuses to replace existing evidence", "Error 8" in str(error))
        ns.out = Path(folder) / "run" / "w2.f32"
        moving = ScriptedMonitor(memory)
        moving.change_after_read = True
        try:
            command_wave(ns, None, readback, moving, evidence)
            check("wave refuses a range that changed during the read", False)
        except RunError as error:
            check("wave refuses a range that changed during the read",
                  "Error 10" in str(error) and not ns.out.exists())
        wrong = dict(memory)
        struct.pack_into("<I", block, symbols["global_krresulthash3"] - base, xor ^ 1)
        wrong[base] = bytes(block)
        try:
            command_wave(ns, None, readback, ScriptedMonitor(wrong), evidence)
            check("wave refuses samples whose XOR is not the payload's", False)
        except RunError as error:
            check("wave refuses samples whose XOR is not the payload's",
                  "Error 12" in str(error) and not ns.out.exists())
        unfinished = dict(memory)
        stale = bytearray(trace)
        struct.pack_into("<I", stale, 8, 6)
        unfinished[TRACE_ADDRESS] = bytes(stale)
        check("status of an unfinished run exits 1",
              command_status(ns, None, readback, ScriptedMonitor(unfinished), evidence) == 1)
        bad_sym = Path(folder) / "other.sym"
        bad_sym.write_text("global_something=16\n", encoding="utf-8")
        try:
            read_result(readback, ScriptedMonitor(memory), read_symbols(bad_sym), evidence)
            check("a symbol file from another program is refused", False)
        except RunError as error:
            check("a symbol file from another program is refused", "Error 3" in str(error))
    if failures:
        print("SELF-TEST FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        return 1
    print("SELF-TEST PASSED. No board was contacted.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("cores", "status", "wave"))
    parser.add_argument("--console-ip", help="the board's network console address")
    parser.add_argument("--console-port", type=int, default=5555)
    parser.add_argument("--anvil", type=Path, default=DEFAULT_ANVIL,
                        help="the Anvil checkout (default: %(default)s)")
    parser.add_argument("--sym", type=Path, default=DEFAULT_SYM,
                        help="the payload's symbol file (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="wave: where the samples go (default: %(default)s)")
    parser.add_argument("--address", type=lambda text: int(text, 0), default=WAVE_ADDRESS,
                        help="wave: the copy slot (default: 0x%(default)X)")
    parser.add_argument("--bytes", type=lambda text: int(text, 0), default=0,
                        help="wave: read this many bytes instead of the count in the "
                             "result block; the XOR check is then skipped")
    parser.add_argument("--json", type=Path, help="status: also write the report here")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        if args.self_test:
            return self_test(args)
        if args.command is None or not args.console_ip:
            parser.error("give a command (cores, status or wave) and --console-ip")
        board_run, readback = load_anvil(args.anvil)
        console = board_run.NetConsole(args.console_ip, args.console_port)
        evidence = lambda line, reply: None
        try:
            if not console.at_prompt(15.0):
                raise RunError(
                    "Error 13: nothing answered with a prompt on the Anvil console "
                    "at %s:%d within 15 s. Check that the board is powered, that "
                    "its address is right (its own `net` command prints it), and "
                    "that no payload is still running." % (args.console_ip, args.console_port))
            if args.command == "cores":
                return command_cores(args, board_run, readback, console)
            if args.command == "status":
                return command_status(args, board_run, readback, console, evidence)
            return command_wave(args, board_run, readback, console, evidence)
        finally:
            console.close()
    except RunError as error:
        print("!! %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
