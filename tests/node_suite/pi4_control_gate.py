#!/usr/bin/env python3
"""Runs If, Loop and sequence cases on the Pi 4 target, in an A64 interpreter.

DEVELOPER CHECK ONLY, in the node_suite family. No board is touched.

For each case folder (a model.onnx with test_data_set_0, as written by
targeted_control.py or taken from the ONNX node-test corpus) this:

  1. runs the compiler with --target pi4;
  2. writes a returning payload that binds the generated model to a static arena,
     fills its inputs from input_*.pb (tensors with DAlloc, sequences with
     DSeqReset/DSeqAppendShape), runs PmModelExecute once, and serialises every
     output into a RAM buffer in the node-suite result format;
  3. builds it with the PureMetal compiler for pi4 and executes the image in the
     independent A64 interpreter;
  4. compares every output with output_*.pb exactly as node_suite.py does.

Usage (repository root):
  py -3.12 tests/node_suite/pi4_control_gate.py --forge PureMetalForge.exe --cli CLI.exe --interp a64_interp.py CASE_DIR ...
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import node_suite as ns  # noqa: E402

LOAD = 0x00500000
BSS = 0x00600000
STACK = 0x00C00000
RETURN = 0xDEADBEE0
ARENA_BYTES = 3 * 1024 * 1024
OUT_BYTES = 1024 * 1024
STEP_CAP = 2_000_000_000


def byte_lines(label: str, values: bytes) -> str:
    lines = [label + ":"]
    if not values:
        lines.append("  Data.a 0")
    for offset in range(0, len(values), 16):
        lines.append("  Data.a " + ",".join(str(v) for v in values[offset:offset + 16]))
    return "\n".join(lines)


def runner_source(prefix: Path, inputs: list, weights: bytes, outputs: int) -> str:
    fill = []
    data = []
    for index, (kind, elem, arrays) in enumerate(inputs):
        fill.append("  id = PmModelInput(%d)" % index)
        fill.append("  If id = 0 : ProcedureReturn 10 : EndIf")
        if kind == "sequence":
            fill.append("  If DSeqReset(id, %d) = 0 : ProcedureReturn 11 : EndIf" % elem)
        for item, arr in enumerate(arrays):
            label = "GateIn%d_%d" % (index, item)
            raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
            for axis in range(8):
                fill.append("  dims(%d) = %d" % (axis, arr.shape[axis] if axis < arr.ndim else 0))
            if kind == "sequence":
                fill.append("  target = DSeqAppendShape(id, %d, %d, @dims(0))" % (elem, arr.ndim))
                fill.append("  If target = 0 : ProcedureReturn 12 : EndIf")
            else:
                fill.append("  If DAlloc(id, %d, %d, @dims(0)) = 0 : ProcedureReturn 13 : EndIf" % (elem, arr.ndim))
                fill.append("  target = Dt(id)\\Data")
            if raw:
                fill.append("  DCopy(?%s, target, %d)" % (label, len(raw)))
            data.append(byte_lines(label, raw))
    fill_text = "\n".join(fill)
    data_text = "\n".join(data)
    return f'''\
; Generated returning payload for one control-flow case. RAM only.
XIncludeFile "{prefix.name}.pi4"

#GATE_ARENA_BYTES = {ARENA_BYTES}
#GATE_OUT_BYTES = {OUT_BYTES}

Global Dim GateArena.a(#GATE_ARENA_BYTES + 15)
Global Dim GateOut.a(#GATE_OUT_BYTES)
Global GateStatus.i
Global GateUsed.i
Global GateError.i
Global GatePeak.i

; The result buffer is written a byte at a time: an element of three BOOL
; bytes leaves the next field off a 4-byte boundary, and with the MMU off an
; unaligned 4-byte store is an alignment fault on the Pi 4.
Procedure GatePutByte(value.i)
  If GateUsed < #GATE_OUT_BYTES
    PokeA(@GateOut(0) + GateUsed, value & 255)
  EndIf
  GateUsed = GateUsed + 1
EndProcedure

Procedure GatePutL(value.i)
  Define i.i
  i = 0
  While i < 4
    GatePutByte(value >> (i * 8))
    i = i + 1
  Wend
EndProcedure

Procedure GatePutQ(value.i)
  Define i.i
  i = 0
  While i < 8
    GatePutByte(value >> (i * 8))
    i = i + 1
  Wend
EndProcedure

Procedure GatePutBytes(address.i, count.i)
  Define i.i
  i = 0
  While i < count
    If GateUsed < #GATE_OUT_BYTES
      PokeA(@GateOut(0) + GateUsed, PeekA(address + i))
    EndIf
    GateUsed = GateUsed + 1
    i = i + 1
  Wend
EndProcedure

Procedure.i GateRun()
  Define id.i
  Define target.i
  Define index.i
  Define item.i
  Define axis.i
  Define *arena.i
  Protected Dim dims.i(8)
  *arena = (@GateArena(0) + 15) & -16
  If PmModelBindMemory(?GateWeights, #PMO_WEIGHT_FILE_BYTES, *arena, #GATE_ARENA_BYTES) = 0
    ProcedureReturn 1
  EndIf
  PmModelResetRequest()
{fill_text}
  If PmModelExecute() = 0
    ProcedureReturn 2
  EndIf
  GatePutL($524E4D50)
  GatePutL(1)
  GatePutL({outputs})
  index = 0
  While index < {outputs}
    id = PmModelOutput(index)
    If Dt(id)\\Kind = #PMD_KIND_SEQUENCE
      GatePutL($80000000 | DSeqElementKind(id))
      GatePutL(DSeqCount(id))
      item = 0
      While item < DSeqCount(id)
        GatePutL(DSeqItemRank(id, item))
        axis = 0
        While axis < DSeqItemRank(id, item)
          GatePutQ(DSeqItemDim(id, item, axis))
          axis = axis + 1
        Wend
        GatePutQ(DSeqItemBytes(id, item))
        GatePutBytes(DSeqItemData(id, item), DSeqItemBytes(id, item))
        item = item + 1
      Wend
    Else
      GatePutL(Dt(id)\\Kind)
      GatePutL(Dt(id)\\Rank)
      axis = 0
      While axis < Dt(id)\\Rank
        GatePutQ(DDims(id * 8 + axis))
        axis = axis + 1
      Wend
      GatePutQ(Dt(id)\\Bytes)
      GatePutBytes(Dt(id)\\Data, Dt(id)\\Bytes)
    EndIf
    index = index + 1
  Wend
  GatePeak = DHeapPeak
  PmModelResetRequest()
  If DHeapUsed <> 0
    ProcedureReturn 3
  EndIf
  PmModelClose()
  ProcedureReturn 0
EndProcedure

Procedure.i Main()
  GateStatus = -1
  GateStatus = GateRun()
  GateError = DError
  ProcedureReturn GateStatus
EndProcedure

DataSection
{byte_lines("GateWeights", weights)}
{data_text}
EndDataSection
'''


def read_symbols(path: Path) -> dict[str, int]:
    symbols: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            symbols[name.lower()] = int(value, 0)
    return symbols


def load_interpreter(path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("control_gate_a64", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def c_string(cpu, address: int) -> str:
    out = bytearray()
    while address and len(out) < 400:
        b = cpu.raw_load(address, 1)
        if b == 0:
            break
        out.append(b)
        address += 1
    return out.decode("ascii", errors="replace")


def run_case(case: Path, args, a64) -> tuple[bool, str]:
    work = args.work / case.name
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "m"
    proc = subprocess.run([str(args.cli), "--compile", str(case / "model.onnx"), "--output", str(prefix), "--target", "pi4"],
                          capture_output=True, text=True, cwd=str(work))
    if "PASS: reusable source emitted" not in proc.stdout:
        return False, "the compiler did not emit pi4 source: " + ns.one_line(proc.stdout + proc.stderr)
    model = onnx.load(str(case / "model.onnx"), load_external_data=False)
    ds = case / "test_data_set_0"
    inputs = []
    for i, vi in enumerate(model.graph.input):
        kind, proto, value = ns.load_value(ds / ("input_%d.pb" % i), vi.type)
        if kind == "sequence":
            inputs.append(("sequence", vi.type.sequence_type.elem_type.tensor_type.elem_type, value))
        else:
            inputs.append(("tensor", proto.data_type, [value]))
    source = runner_source(prefix, inputs, prefix.with_suffix(".pmw").read_bytes(), len(model.graph.output))
    runner = work / "gate.pi4"
    runner.write_text(source, encoding="utf-8", newline="\n")
    image = work / "gate.img"
    build = subprocess.run([str(args.forge), "--compile", str(runner), "-t", "pi4", "--load-addr", hex(LOAD),
                            "--bss-addr", hex(BSS), "--stack-addr", hex(STACK), "--entry-returns", "-o", str(image)],
                           capture_output=True, text=True, cwd=str(work))
    if "pmfc: OK" not in build.stdout + build.stderr:
        lines = [l for l in (build.stdout + build.stderr).splitlines() if "ERROR" in l or "Error" in l]
        return False, "the pi4 payload did not build: " + ns.one_line(" ".join(lines) or (build.stdout + build.stderr)[-800:])
    symbols = read_symbols(Path(str(image) + ".sym"))
    cpu = a64.A64()
    for offset, value in enumerate(image.read_bytes()):
        cpu.memory[LOAD + offset] = value
    cpu.enable_system_registers(preset={0xD51BE000: 54_000_000, 0xD51BE020: 0})
    cpu.pc = LOAD + symbols["main"]
    cpu.sp = STACK
    cpu.x[30] = RETURN
    steps = 0
    started = time.time()
    while cpu.pc != RETURN:
        cpu.step()
        steps += 1
        if steps > STEP_CAP:
            return False, "exceeded %d interpreted instructions" % STEP_CAP
    status = cpu.raw_load(symbols["global_gatestatus"], 8)
    if status != 0:
        return False, "payload status %d: %s" % (status, c_string(cpu, cpu.raw_load(symbols["global_gateerror"], 8)))
    used = cpu.raw_load(symbols["global_gateused"], 8)
    if used > OUT_BYTES:
        return False, "outputs need %d bytes; the result buffer holds %d" % (used, OUT_BYTES)
    base = symbols["global_gateout"]
    data = bytes(cpu.raw_load(base + i, 1) for i in range(used))
    results = ns.parse_result(data)
    outs = [ns.load_value(ds / ("output_%d.pb" % i), vi.type) for i, vi in enumerate(model.graph.output)]
    rtol, atol = ns.DEFAULT_RTOL, ns.DEFAULT_ATOL
    for i, ((kind, proto, expected), (elem, dims, raw)) in enumerate(zip(outs, results)):
        if kind == "sequence":
            if not elem & ns.SEQUENCE_FLAG or len(raw) != len(expected):
                return False, "output %d: sequence shape or kind differs" % i
            for j, ((edims, eraw), earr, eproto) in enumerate(zip(raw, expected, proto.tensor_values)):
                outcome, detail = ns.compare(earr, eproto.data_type, elem & ~ns.SEQUENCE_FLAG, edims, eraw, rtol, atol, i)
                if outcome != "PASS":
                    return False, "sequence element %d: %s" % (j, detail)
            continue
        outcome, detail = ns.compare(expected, proto.data_type, elem, dims, raw, rtol, atol, i)
        if outcome != "PASS":
            return False, detail
    peak = cpu.raw_load(symbols["global_gatepeak"], 8)
    return True, "%d instructions, arena peak %d bytes, %.0f s" % (steps, peak, time.time() - started)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", required=True, type=Path)
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--interp", required=True, type=Path, help="the independent A64 instruction interpreter (a64_interp.py)")
    ap.add_argument("--work", type=Path, default=HERE / "work" / "control_pi4")
    ap.add_argument("cases", nargs="+", type=Path)
    args = ap.parse_args()
    args.work = args.work.resolve()
    a64 = load_interpreter(args.interp)
    bad = 0
    for case in args.cases:
        ok, detail = run_case(case.resolve(), args, a64)
        bad += 0 if ok else 1
        print("%-4s %-45s %s" % ("PASS" if ok else "FAIL", case.name, detail), flush=True)
    print("pi4 control gate: %d of %d pass" % (len(args.cases) - bad, len(args.cases)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
