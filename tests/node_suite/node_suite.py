#!/usr/bin/env python3
"""ONNX backend node-test coverage harness for the Windows target.

DEVELOPER CHECK ONLY. Building the compiler and using it need no Python; this
script is a measuring instrument for people changing the compiler, in the same
family as the other developer gates under tests/. It turns "how much of ONNX
does the compiler handle" into a number.

For every official ONNX backend node test (onnx/backend/test/data/node) whose
model imports ai.onnx at opset 27 or lower, it:

  1. classifies the case by the operators its graph contains;
  2. runs the compiler (built fresh from this checkout's src/) with
     --compile ... --target windows into a scratch folder;
  3. if source was emitted, builds ONE generic driver (node_driver_windows.pbi)
     around it and runs it with the case's input_*.pb tensors;
  4. compares every output with output_*.pb: shape and element type must match,
     integer and bool outputs exactly, floating outputs with the tolerance the
     onnx backend test loader assigns (rtol 1e-3, atol 1e-7:
     onnx/backend/test/loader/__init__.py lines 31-32, overridable per case by
     data.json, applied by np.testing.assert_allclose in
     onnx/backend/test/runner/__init__.py assert_similar_outputs).

Each case ends in exactly one outcome: PASS, FAIL_NUMERIC, FAIL_SHAPE,
FAIL_DTYPE, REFUSED, BUILD_ERROR, RUN_ERROR or HARNESS_ERROR.

The corpus is not in this repository. It ships inside the onnx Python package;
the harness locates it through the interpreter running this script and refuses,
in a sentence, to score a corpus whose onnx version or content hash differs
from the pin below.

Usage (from the repository root, with an interpreter that has onnx installed):
  python tests/node_suite/node_suite.py --pbcompiler PATH\\pbcompiler.exe
  python tests/node_suite/node_suite.py --pbcompiler ... --cases "test_add*"
  python tests/node_suite/node_suite.py --pbcompiler ... --self-check
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper
except ImportError:  # pragma: no cover - the refusal is the point
    sys.exit("node_suite: the onnx Python package is not importable by this interpreter. "
             "Install onnx==1.22.0 (pip install onnx==1.22.0) and run the harness with that interpreter.")

# ---------------------------------------------------------------------------
# The pin. A different corpus is a different measurement; the harness refuses
# rather than publish a number against an unknown set of cases.
# ---------------------------------------------------------------------------
PINNED_ONNX_VERSION = "1.22.0"
PINNED_CORPUS_SHA256 = "4955443be1453848b40f58dd92d8e709907cb73a0141cf17d0a9346cb863ce6b"
MAX_OPSET = 27
DEFAULT_RTOL = 1e-3   # onnx/backend/test/loader/__init__.py:31
DEFAULT_ATOL = 1e-7   # onnx/backend/test/loader/__init__.py:32

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIVER = HERE / "node_driver_windows.pbi"

OUTCOMES = ["PASS", "FAIL_NUMERIC", "FAIL_SHAPE", "FAIL_DTYPE", "REFUSED",
            "BUILD_ERROR", "RUN_ERROR", "HARNESS_ERROR"]
FAILING = {"FAIL_NUMERIC", "FAIL_SHAPE", "FAIL_DTYPE", "BUILD_ERROR", "RUN_ERROR"}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def corpus_dir() -> Path:
    return Path(onnx.__file__).resolve().parent / "backend" / "test" / "data" / "node"


def corpus_hash(node_dir: Path) -> tuple[str, int]:
    """sha256 over the sorted case list and every file's relative path and sha256."""
    outer = hashlib.sha256()
    cases = sorted(p.name for p in node_dir.iterdir() if p.is_dir())
    for case in cases:
        outer.update(("case " + case + "\n").encode())
        base = node_dir / case
        files = sorted((f for f in base.rglob("*") if f.is_file()), key=lambda f: f.relative_to(base).as_posix())
        for f in files:
            outer.update((f.relative_to(base).as_posix() + " " + hashlib.sha256(f.read_bytes()).hexdigest() + "\n").encode())
    return outer.hexdigest(), len(cases)


def ai_onnx_opset(model: onnx.ModelProto) -> int | None:
    version = None
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            version = o.version
    return version


def graph_ops(graph: onnx.GraphProto, into: set[str]) -> None:
    for node in graph.node:
        into.add(node.op_type if node.domain in ("", "ai.onnx") else node.domain + "." + node.op_type)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                graph_ops(attr.g, into)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    graph_ops(g, into)


# ---------------------------------------------------------------------------
# What the compiler says it supports, read from its own validators so the list
# can never drift from the source.
# ---------------------------------------------------------------------------
def supported_operators(source_root: Path = ROOT) -> tuple[set[str], set[str]]:
    compile_src = (source_root / "src" / "compiler" / "onnx_compile.pbi").read_text(encoding="utf-8", errors="replace")
    m = re.search(r'Procedure\.i PmoCompileSupportedOp\(.*?FindString\("\|([^"]+)\|"', compile_src, re.S)
    if not m:
        sys.exit("node_suite: cannot find PmoCompileSupportedOp's operator list in src/compiler/onnx_compile.pbi; "
                 "the harness reads the supported set from that source and will not guess it.")
    fixed = set(filter(None, m.group(1).split("|")))
    # The operators that complete the set (onnx_emit_ops.pbi) are listed once,
    # by PmoOpsOwns, for both paths.
    ops_path = source_root / "src" / "compiler" / "onnx_emit_ops.pbi"
    owned: set[str] = set()
    if ops_path.exists():
        ops_src = ops_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'Procedure\.i PmoOpsOwns\(.*?FindString\((.*?), "\|" \+ Operation', ops_src, re.S)
        if m:
            owned = set(filter(None, "".join(re.findall(r'"([^"]*)"', m.group(1))).split("|")))
    fixed |= owned
    dyn_src =(source_root / "src" / "compiler" / "onnx_dynamic_emit.pbi").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Procedure\.s PmdCall\(.*?EndProcedure", dyn_src, re.S)
    if not m:
        sys.exit("node_suite: cannot find PmdCall in src/compiler/onnx_dynamic_emit.pbi; "
                 "the harness reads the runtime-dimension operator list from it.")
    dynamic = set()
    for line in m.group(0).splitlines():
        s = line.strip()
        if s.startswith("Case ") and '"' in s:
            dynamic.update(re.findall(r'"([A-Za-z]+)"', s.split(":")[0]))
    return fixed, dynamic | owned


_SNAKE_OPS: list[tuple[str, str]] = []
_SNAKE_LOCK = threading.Lock()


def subject_operator(case: str, ops: list[str], refusal: str) -> str:
    """The operator a case is about: the one its refusal names, else the operator whose snake_case
    name the case is named after (node tests are test_<operator>_...), else its only operator."""
    named = [op for op in ops if refusal and re.search(r"\b%s\b" % re.escape(op), refusal)]
    if named:
        return named[0]
    with _SNAKE_LOCK:
        if not _SNAKE_OPS:
            names = {s.name for s in onnx.defs.get_all_schemas_with_history() if s.domain in ("", "ai.onnx")}
            for name in names:
                snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()
                _SNAKE_OPS.append((snake, name))
            _SNAKE_OPS.sort(key=lambda t: (-len(t[0]), t[0]))
    body = case[5:] if case.startswith("test_") else case
    for snake, op in _SNAKE_OPS:
        if body == snake or body.startswith(snake + "_"):
            return op
    return ops[0] if len(ops) == 1 else "(unattributed)"


def reason_key(detail: str) -> str:
    """Folds node numbers and tensor names so one refusal reason groups as one line."""
    t = re.sub(r"\bnode \d+\b", "node N", detail)
    t = re.sub(r"\b(folded tensor|tensor|output|input) (\S+) (has|uses)\b", r"\1 <name> \3", t)
    return t


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------
def quiet_crashes() -> None:
    """A crashing child must end as an exit code, never as a dialog that waits forever."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


def run_process(args: list[str], cwd: Path, timeout: float, env: dict | None = None) -> tuple[int | None, str]:
    """Returns (exit code or None on timeout, combined output). Kills the whole tree on timeout."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, creationflags=flags)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out = b""
        return None, out.decode("utf-8", errors="replace")


def exit_text(code: int) -> str:
    if code < 0:
        code &= 0xFFFFFFFF
    if code >= 0x80000000:
        return "exit 0x%08X" % code
    return "exit %d" % code


def one_line(text: str, limit: int = 400) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + " ..."


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------
def load_value(path: Path, value_type: onnx.TypeProto):
    """Loads a test_data .pb the way the onnx runner does: by the graph's declared type."""
    data = path.read_bytes()
    kind = value_type.WhichOneof("value")
    if kind == "tensor_type" or kind is None:
        t = onnx.TensorProto()
        t.ParseFromString(data)
        return "tensor", t, numpy_helper.to_array(t)
    if kind == "sequence_type":
        s = onnx.SequenceProto()
        s.ParseFromString(data)
        if s.elem_type == onnx.SequenceProto.TENSOR or not s.tensor_values:
            return "sequence", s, [numpy_helper.to_array(t) for t in s.tensor_values]
        return "sequence", s, None
    if kind == "optional_type":
        o = onnx.OptionalProto()
        o.ParseFromString(data)
        return "optional", o, None
    return kind, None, None


# A sequence value travels as SEQUENCE_FLAG | element type, the element count,
# then per element: rank, dims, byte count, bytes (a tensor record without its
# own element type, since a sequence holds one element type).
SEQUENCE_FLAG = 0x80000000


def tensor_record(arr: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
    return (struct.pack("<I", arr.ndim) + b"".join(struct.pack("<q", d) for d in arr.shape)
            + struct.pack("<q", len(raw)) + raw)


def request_bytes(inputs: list[tuple], output_bytes: list[int]) -> bytes:
    """inputs: (element type, array) for a tensor, (element type, [arrays], "sequence") for a sequence."""
    out = bytearray(b"PMNH" + struct.pack("<II", 1, len(inputs)))
    for item in inputs:
        if len(item) == 3:
            elem, arrays, _ = item
            out += struct.pack("<II", SEQUENCE_FLAG | elem, len(arrays))
            for arr in arrays:
                out += tensor_record(arr)
            continue
        elem, arr = item
        out += struct.pack("<I", elem) + tensor_record(arr)
    out += struct.pack("<I", len(output_bytes))
    out += b"".join(struct.pack("<q", b) for b in output_bytes)
    return bytes(out)


def parse_result(data: bytes) -> list[tuple[int, list[int], bytes]]:
    if data[:4] != b"PMNR":
        raise ValueError("result file does not start with PMNR")
    version, count = struct.unpack_from("<II", data, 4)
    pos = 12
    outputs = []
    for _ in range(count):
        (elem,) = struct.unpack_from("<I", data, pos)
        if elem & SEQUENCE_FLAG:
            (items,) = struct.unpack_from("<I", data, pos + 4)
            pos += 8
            elements = []
            for _ in range(items):
                (rank,) = struct.unpack_from("<I", data, pos)
                pos += 4
                dims = list(struct.unpack_from("<%dq" % rank, data, pos)) if rank else []
                pos += 8 * rank
                (n,) = struct.unpack_from("<q", data, pos)
                pos += 8
                elements.append((dims, data[pos:pos + n]))
                pos += n
            outputs.append((elem, None, elements))
            continue
        (rank,) = struct.unpack_from("<I", data, pos + 4)
        pos += 8
        dims = list(struct.unpack_from("<%dq" % rank, data, pos)) if rank else []
        pos += 8 * rank
        (n,) = struct.unpack_from("<q", data, pos)
        pos += 8
        outputs.append((elem, dims, data[pos:pos + n]))
        pos += n
    if pos != len(data):
        raise ValueError("result file has %d trailing bytes" % (len(data) - pos))
    return outputs


def elem_name(elem: int) -> str:
    try:
        return onnx.TensorProto.DataType.Name(elem)
    except ValueError:
        return "type %d" % elem


# Operators whose values the ONNX specification leaves to the implementation:
# a case that uses one is scored on shape and element type (and, where the
# expected output holds only 0 and 1, on the program's doing the same); the
# values themselves are this compiler's specified generator, not the
# reference's numpy draws.
RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial"}


def compare(expected: np.ndarray, expected_elem: int, elem: int, dims: list[int], raw: bytes,
            rtol: float, atol: float, index: int, drawn: bool = False) -> tuple[str, str]:
    exp_shape = list(expected.shape)
    if dims != exp_shape:
        return "FAIL_SHAPE", "output %d: shape %s, expected %s" % (index, dims, exp_shape)
    if elem != expected_elem:
        return "FAIL_DTYPE", "output %d: element type %s, expected %s" % (index, elem_name(elem), elem_name(expected_elem))
    np_dtype = helper.tensor_dtype_to_np_dtype(elem)
    count = int(np.prod(exp_shape, dtype=np.int64))
    if len(raw) != count * np.dtype(np_dtype).itemsize:
        return "FAIL_SHAPE", "output %d: %d bytes for shape %s of %s" % (index, len(raw), exp_shape, elem_name(elem))
    actual = np.frombuffer(raw, dtype=np.dtype(np_dtype).newbyteorder("<")).reshape(exp_shape)
    if drawn:
        e01 = np.isin(expected.astype(np.float64), (0.0, 1.0)).all()
        if e01 and not np.isin(actual.astype(np.float64), (0.0, 1.0)).all():
            return "FAIL_NUMERIC", "output %d: a random operator's 0/1 output holds other values" % index
        return "PASS", "values drawn by this compiler's specified generator; shape and element type checked"
    if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(expected.dtype, np.complexfloating):
        a = actual.astype(np.float64)
        e = expected.astype(np.float64)
        close = np.isclose(a, e, rtol=rtol, atol=atol, equal_nan=True)
        if close.all():
            return "PASS", ""
        diff = np.where(close, 0.0, np.abs(a - e))
        diff = np.where(np.isnan(diff), np.inf, diff)
        worst = np.unravel_index(int(np.argmax(diff)), diff.shape) if diff.ndim else ()
        bad = int((~close).sum())
        return "FAIL_NUMERIC", ("output %d: %d of %d elements outside rtol %g atol %g; max-abs %.6g at %s "
                                "(got %r, expected %r)" % (index, bad, close.size, rtol, atol, float(diff[worst]),
                                                          list(int(i) for i in worst), float(a[worst]), float(e[worst])))
    a = actual.view(np.uint8) if actual.dtype == np.bool_ else actual
    e = expected.view(np.uint8) if expected.dtype == np.bool_ else expected
    same = a == e
    if same.all():
        return "PASS", ""
    first = np.unravel_index(int(np.argmin(same)), same.shape) if same.ndim else ()
    return "FAIL_NUMERIC", ("output %d: %d of %d elements differ (exact comparison for %s); first at %s "
                            "(got %r, expected %r)" % (index, int((~same).sum()), same.size, elem_name(elem),
                                                       list(int(i) for i in first), a[first].item(), e[first].item()))


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------
class Tools:
    def __init__(self, cli: Path, pbcompiler: Path, compile_timeout: float, build_timeout: float, run_timeout: float):
        self.cli = cli
        self.pbcompiler = pbcompiler
        self.compile_timeout = compile_timeout
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout


def run_case(case_dir: Path, name: str, work: Path, tools: Tools, supported: set[str], keep: bool) -> dict:
    record = {"case": name, "opset": None, "ops": [], "unsupported_ops": [], "path": None,
              "outcome": "HARNESS_ERROR", "detail": ""}
    scratch = work / name
    try:
        model = onnx.load(str(case_dir / "model.onnx"), load_external_data=False)
        record["opset"] = ai_onnx_opset(model)
        ops: set[str] = set()
        graph_ops(model.graph, ops)
        record["ops"] = sorted(ops)
        record["unsupported_ops"] = sorted(ops - supported)
        rtol, atol = DEFAULT_RTOL, DEFAULT_ATOL
        if (case_dir / "data.json").exists():
            meta = json.loads((case_dir / "data.json").read_text())
            rtol, atol = meta.get("rtol", rtol), meta.get("atol", atol)
        initializer_names = {i.name for i in model.graph.initializer}
        graph_inputs = [i for i in model.graph.input if i.name not in initializer_names]
        data_sets = sorted(p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("test_data_set"))
        if not data_sets:
            record["detail"] = "the case has no test_data_set folder"
            return record
        loaded = []
        for ds in data_sets:
            n_in = len(list(ds.glob("input_*.pb")))
            n_out = len(list(ds.glob("output_*.pb")))
            ins = [load_value(ds / ("input_%d.pb" % i), model.graph.input[i].type) for i in range(n_in)]
            outs = [load_value(ds / ("output_%d.pb" % i), model.graph.output[i].type) for i in range(n_out)]
            loaded.append((ds.name, ins, outs))
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
    except Exception as exc:  # the harness could not prepare the case
        record["detail"] = "%s: %s" % (type(exc).__name__, one_line(str(exc)))
        return record

    try:
        outcome, detail = _compile_build_run(case_dir, record, model, graph_inputs, loaded, scratch, tools, rtol, atol)
    except Exception as exc:
        outcome, detail = "HARNESS_ERROR", "%s: %s" % (type(exc).__name__, one_line(str(exc)))
    # No machine path reaches a published record.
    for path, label in ((scratch, "<case>"), (work, "<work>"), (ROOT, "<repo>")):
        for spelling in (str(path), path.as_posix()):
            detail = detail.replace(spelling + "\\", "").replace(spelling + "/", "").replace(spelling, label)
    record["outcome"], record["detail"] = outcome, detail
    if outcome == "REFUSED":
        record["named_ops"] = [op for op in record["ops"] if re.search(r"\b%s\b" % re.escape(op), detail)]
    record["subject"] = subject_operator(name, record["ops"], detail if outcome == "REFUSED" else "")
    if not keep and outcome in ("PASS", "REFUSED"):
        shutil.rmtree(scratch, ignore_errors=True)
    return record


def _compile_build_run(case_dir, record, model, graph_inputs, loaded, scratch: Path, tools: Tools, rtol, atol):
    # (b) the compiler
    prefix = scratch / "m"
    code, out = run_process([str(tools.cli), "--compile", str(case_dir / "model.onnx"), "--output", str(prefix),
                             "--target", "windows"], scratch, tools.compile_timeout)
    (scratch / "compile.log").write_text(out, encoding="utf-8")
    if code is None:
        return "BUILD_ERROR", "the compiler did not finish within %g s" % tools.compile_timeout
    marker = "ONNX COMPILER ERROR:"
    if code != 0:
        if marker in out:
            return "REFUSED", one_line(out[out.index(marker) + len(marker):])
        return "BUILD_ERROR", "the compiler ended with %s and no refusal sentence: %s" % (exit_text(code), one_line(out))
    source, weights, manifest_path = prefix.with_suffix(".pb"), prefix.with_suffix(".pmw"), prefix.with_suffix(".json")
    for f in (source, weights, manifest_path):
        if not f.exists():
            return "BUILD_ERROR", "the compiler reported success but did not write %s" % f.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    dynamic = manifest.get("shape_contract") == "runtime-dimensions"
    record["path"] = "runtime-dimensions" if dynamic else "fixed-shape"

    # (c) one generic driver around the emitted source
    stub = scratch / "case.pb"
    stub.write_text('XIncludeFile "m.pb"\nXIncludeFile "%s"\n' % DRIVER.as_posix().replace("/", "\\"), encoding="utf-8")
    exe = scratch / "case.exe"
    env = dict(os.environ)
    env["TMP"] = env["TEMP"] = str(scratch)  # parallel builds never share a temporary file
    code, out = run_process([str(tools.pbcompiler), str(stub), "/CONSOLE", "/THREAD", "/OPTIMIZER", "/OUTPUT", str(exe)],
                            scratch, tools.build_timeout, env)
    (scratch / "build.log").write_text(out, encoding="utf-8")
    if code is None:
        return "BUILD_ERROR", "the generated program did not build within %g s" % tools.build_timeout
    if code != 0 or not exe.exists():
        errors = [l for l in out.splitlines() if "rror" in l or "Line" in l]
        return "BUILD_ERROR", "the generated program did not build (%s): %s" % (exit_text(code), one_line(" ".join(errors) or out))

    # manifest-declared contract for the fixed-shape API
    if not dynamic:
        m_in, m_out = manifest.get("inputs", []), manifest.get("outputs", [])
        expected_outputs = len(model.graph.output)
        if len(m_out) != expected_outputs:
            return "FAIL_SHAPE", "the compiled model declares %d outputs; the case has %d" % (len(m_out), expected_outputs)
        if len(m_in) != len(graph_inputs):
            return "FAIL_SHAPE", "the compiled model declares %d inputs; the model has %d" % (len(m_in), len(graph_inputs))

    note = ""
    for ds_name, ins, outs in loaded:
        inputs = []
        for i, (kind, proto, arr) in enumerate(ins):
            if kind == "sequence" and dynamic and arr is not None:
                elem_type = model.graph.input[i].type.sequence_type.elem_type
                if elem_type.WhichOneof("value") != "tensor_type":
                    return "FAIL_DTYPE", "%s input %d is a sequence of non-tensor values; the compiled program carries sequences of tensors only" % (ds_name, i)
                inputs.append((elem_type.tensor_type.elem_type, arr, "sequence"))
                continue
            if kind != "tensor":
                return "FAIL_DTYPE", "%s input %d is a %s value; the compiled program takes tensors only" % (ds_name, i, kind)
            if arr.dtype == object:
                return "FAIL_DTYPE", "%s input %d is a string tensor; the compiled program cannot carry it" % (ds_name, i)
            if not dynamic:
                decl = manifest["inputs"][i]
                if decl["element_type"] != proto.data_type:
                    return "FAIL_DTYPE", "input %d: the compiled model declares %s, the case supplies %s" % (
                        i, elem_name(decl["element_type"]), elem_name(proto.data_type))
                if list(decl["shape"]) != list(arr.shape):
                    return "FAIL_SHAPE", "input %d: the compiled model declares shape %s, the case supplies %s" % (
                        i, decl["shape"], list(arr.shape))
            inputs.append((proto.data_type, arr))
        out_bytes = [0] * len(outs) if dynamic else [int(o["bytes"]) for o in manifest["outputs"]]
        req, res = scratch / (ds_name + ".request.bin"), scratch / (ds_name + ".result.bin")
        req.write_bytes(request_bytes(inputs, out_bytes))
        if res.exists():
            res.unlink()
        code, out = run_process([str(exe), str(req), str(res), str(weights)], scratch, tools.run_timeout)
        (scratch / (ds_name + ".run.log")).write_text(out, encoding="utf-8")
        if code is None:
            return "RUN_ERROR", "%s: the program did not finish within %g s" % (ds_name, tools.run_timeout)
        if code != 0:
            return "RUN_ERROR", "%s: the program ended with %s: %s" % (ds_name, exit_text(code), one_line(out) or "(no output)")
        if not res.exists():
            return "RUN_ERROR", "%s: the program exited 0 without writing its result" % ds_name
        results = parse_result(res.read_bytes())
        if len(results) != len(outs):
            return "FAIL_SHAPE", "%s: %d outputs returned, %d expected" % (ds_name, len(results), len(outs))
        for i, ((kind, proto, expected), (elem, dims, raw)) in enumerate(zip(outs, results)):
            if kind == "sequence" and expected is not None:
                if not elem & SEQUENCE_FLAG:
                    return "FAIL_DTYPE", "%s output %d: expected a sequence value, the program returned a tensor" % (ds_name, i)
                if len(raw) != len(expected):
                    return "FAIL_SHAPE", "%s output %d: a sequence of %d elements, expected %d" % (ds_name, i, len(raw), len(expected))
                for j, ((edims, eraw), earr, eproto) in enumerate(zip(raw, expected, proto.tensor_values)):
                    outcome, detail = compare(earr, eproto.data_type, elem & ~SEQUENCE_FLAG, edims, eraw, rtol, atol, i)
                    if outcome != "PASS":
                        return outcome, ("%s " % ds_name if len(loaded) > 1 else "") + "sequence element %d: " % j + detail
                continue
            if elem & SEQUENCE_FLAG:
                return "FAIL_DTYPE", "%s output %d: expected a %s value, the program returned a sequence" % (ds_name, i, kind)
            if kind != "tensor":
                return "FAIL_DTYPE", "%s output %d: expected a %s value, the program returned a tensor" % (ds_name, i, kind)
            if not dynamic:
                decl = manifest["outputs"][i]
                elem, dims = decl["element_type"], list(decl["shape"])
            outcome, detail = compare(expected, proto.data_type, elem, dims, raw, rtol, atol, i,
                                      drawn=bool(RANDOM_OPS & set(record["ops"])))
            if outcome != "PASS":
                return outcome, ("%s " % ds_name if len(loaded) > 1 else "") + detail
            note = detail or note
    return "PASS", note


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def summarize(records: list[dict], header: dict, supported: set[str]) -> str:
    total = len(records)
    counts = {o: 0 for o in OUTCOMES}
    for r in records:
        counts[r["outcome"]] += 1
    attempted = total - counts["REFUSED"] - counts["HARNESS_ERROR"]
    lines = []
    w = lines.append
    w("# ONNX node-test coverage - %s" % header["date"])
    w("")
    w("Developer measurement, not part of building or using the compiler. Produced by "
      "`tests/node_suite/node_suite.py`; every case with its operators, outcome and detail is in "
      "`%s`." % header["json_name"])
    w("")
    w("| | |")
    w("|---|---|")
    w("| onnx package (corpus) | %s |" % header["onnx_version"])
    w("| corpus sha256 | `%s` |" % header["corpus_sha256"])
    w("| cases in corpus | %d |" % header["corpus_cases"])
    w("| scored (ai.onnx opset <= %d) | %d |" % (MAX_OPSET, total))
    w("| excluded: opset above %d | %d |" % (MAX_OPSET, header["excluded_opset"]))
    w("| excluded: no ai.onnx import | %d |" % header["excluded_no_ai_onnx"])
    w("| compiler source | `%s` |" % header["compiler_commit"])
    w("| target | windows |")
    w("| operators the validators list | %d |" % len(supported))
    w("| tolerance | rtol %g, atol %g for floating outputs; exact for integer and bool |" % (DEFAULT_RTOL, DEFAULT_ATOL))
    w("")
    w("## Headline")
    w("")
    w("**PASS %d of %d attempted (%.1f%%); PASS %d of %d scored (%.1f%%).** Attempted means the compiler "
      "emitted source; a REFUSED case was declined with a sentence and is a coverage gap, not a wrong answer."
      % (counts["PASS"], attempted, 100.0 * counts["PASS"] / max(attempted, 1),
         counts["PASS"], total, 100.0 * counts["PASS"] / max(total, 1)))
    w("")
    w("| outcome | cases |")
    w("|---|---:|")
    for o in OUTCOMES:
        w("| %s | %d |" % (o, counts[o]))
    w("")

    per_op: dict[str, dict] = {}
    for r in records:
        for op in r["ops"]:
            d = per_op.setdefault(op, {"cases": 0, "pass": 0, "fail": 0, "refused": 0})
            d["cases"] += 1
            if r["outcome"] == "PASS":
                d["pass"] += 1
            elif r["outcome"] == "REFUSED":
                d["refused"] += 1
            elif r["outcome"] in FAILING:
                d["fail"] += 1
    ranked = sorted(per_op.items(), key=lambda kv: (-(kv[1]["fail"] + kv[1]["refused"]), kv[0]))
    w("## Operators with the most failing or refused cases")
    w("")
    w("| operator | listed | cases | pass | fail | refused |")
    w("|---|---|---:|---:|---:|---:|")
    for op, d in ranked[:10]:
        w("| %s | %s | %d | %d | %d | %d |" % (op, "yes" if op in supported else "no", d["cases"], d["pass"], d["fail"], d["refused"]))
    w("")

    w("A case counts for every operator it contains, so the function-expanded node tests (built from Constant, "
      "Shape, Reshape and arithmetic) weigh heavily here. The same ranking by the one operator each case is about:")
    w("")
    by_subject: dict[str, list[int]] = {}
    for r in records:
        s = by_subject.setdefault(r["subject"], [0, 0, 0, 0])
        s[0] += 1
        s[1] += r["outcome"] == "PASS"
        s[2] += r["outcome"] in FAILING
        s[3] += r["outcome"] == "REFUSED"
    w("| operator the case is about | listed | cases | pass | fail | refused |")
    w("|---|---|---:|---:|---:|---:|")
    for op, s in sorted(by_subject.items(), key=lambda kv: (-(kv[1][2] + kv[1][3]), kv[0]))[:10]:
        w("| %s | %s | %d | %d | %d | %d |" % (op, "yes" if op in supported else "no", s[0], s[1], s[2], s[3]))
    w("")

    fails = [r for r in records if r["outcome"] in FAILING or r["outcome"] == "HARNESS_ERROR"]
    w("## Every failure (%d)" % len(fails))
    w("")
    w("A failure on a case whose operators are all listed is a defect in a supported operator, not a coverage gap.")
    w("")
    if fails:
        w("| case | outcome | operators | detail |")
        w("|---|---|---|---|")
        for r in fails:
            w("| %s | %s | %s | %s |" % (r["case"], r["outcome"], " ".join(r["ops"]), r["detail"].replace("|", "\\|")))
    else:
        w("None.")
    w("")

    w("## Refusals grouped by operator")
    w("")
    w("Grouped by the operator the case is about: the one the refusal sentence names, otherwise the operator the "
      "case is named after. Node numbers and tensor names are folded so one reason is one line; `(no operator "
      "named)` marks a sentence that names none of the case's operators.")
    w("")
    groups: dict[str, dict[str, list[str]]] = {}
    for r in records:
        if r["outcome"] != "REFUSED":
            continue
        reason = reason_key(r["detail"]) + ("" if r.get("named_ops") else " (no operator named)")
        groups.setdefault(r["subject"], {}).setdefault(reason, []).append(r["case"])
    for op in sorted(groups, key=lambda k: (-sum(len(v) for v in groups[k].values()), k)):
        n = sum(len(v) for v in groups[op].values())
        w("### %s (%d, %s)" % (op, n, "listed" if op in supported else "not listed"))
        w("")
        for reason, cases in sorted(groups[op].items(), key=lambda kv: (-len(kv[1]), kv[0])):
            shown = ", ".join(cases[:6]) + (", ..." if len(cases) > 6 else "")
            w("- %d x \"%s\" - %s" % (len(cases), reason, shown))
        w("")

    gaps = [r for r in records if r["outcome"] == "REFUSED" and not r["unsupported_ops"]]
    w("## Coverage gaps inside the listed operators (%d refused cases)" % len(gaps))
    w("")
    w("Every operator in these cases is on the validators' list; the refusal is for an attribute, element type, "
      "opset or shape form.")
    w("")
    by_reason: dict[str, list[str]] = {}
    for r in gaps:
        by_reason.setdefault(reason_key(r["detail"]), []).append(r["case"])
    w("| cases | reason | examples |")
    w("|---:|---|---|")
    for reason, cases in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shown = ", ".join(cases[:5]) + (", ..." if len(cases) > 5 else "")
        w("| %d | %s | %s |" % (len(cases), reason.replace("|", "\\|"), shown))
    w("")

    blocking: dict[str, list[int]] = {}
    for r in records:
        if r["outcome"] != "REFUSED":
            continue
        for op in r["unsupported_ops"]:
            b = blocking.setdefault(op, [0, 0])
            b[0] += 1
            if len(r["unsupported_ops"]) == 1:
                b[1] += 1
    w("## Operators not listed, by the refused cases that contain them")
    w("")
    w("`alone` counts the refused cases in which this is the only operator missing from the list, so adding it "
      "is necessary and may be sufficient for them. Planning input, not a promise: a case can still be refused "
      "for an attribute or type form.")
    w("")
    w("| operator | refused cases containing it | alone |")
    w("|---|---:|---:|")
    for op, (anywhere, alone) in sorted(blocking.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0])):
        w("| %s | %d | %d |" % (op, anywhere, alone))
    w("")

    w("## Per-operator table")
    w("")
    w("A case counts once for every distinct operator it contains.")
    w("")
    w("| operator | listed | cases | pass | fail | refused |")
    w("|---|---|---:|---:|---:|---:|")
    for op in sorted(per_op):
        d = per_op[op]
        w("| %s | %s | %d | %d | %d | %d |" % (op, "yes" if op in supported else "no", d["cases"], d["pass"], d["fail"], d["refused"]))
    w("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "runtime"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return head + (" + uncommitted src/runtime changes" if dirty else "")
    except OSError:
        return "unknown"


def export_commit(rev: str, work: Path) -> tuple[Path, str]:
    """Extracts src/ and runtime/ of one commit, so uncommitted edits in the tree are not measured."""
    import io
    import tarfile
    full = subprocess.run(["git", "rev-parse", "--short", rev], cwd=ROOT, capture_output=True, text=True)
    if full.returncode != 0:
        sys.exit("node_suite: --from-commit %s names no commit in this repository." % rev)
    tar = subprocess.run(["git", "archive", "--format=tar", rev, "src", "runtime"], cwd=ROOT, capture_output=True)
    if tar.returncode != 0:
        sys.exit("node_suite: git archive of %s failed: %s" % (rev, one_line(tar.stderr.decode(errors="replace"))))
    dest = work / "_source"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as archive:
        archive.extractall(dest, filter="data")
    return dest, full.stdout.strip()


def build_cli(pbcompiler: Path, work: Path, timeout: float, source_root: Path = ROOT) -> Path:
    out_dir = work / "_compiler"
    out_dir.mkdir(parents=True, exist_ok=True)
    cli = out_dir / "PureMetalOnnxCompilerCLI.exe"
    if cli.exists():
        cli.unlink()
    env = dict(os.environ)
    env["TMP"] = env["TEMP"] = str(out_dir)
    code, out = run_process([str(pbcompiler), str(source_root / "src" / "PureMetalOnnxCompiler.pb"), "/CONSOLE", "/THREAD",
                             "/OPTIMIZER", "/OUTPUT", str(cli)], source_root, timeout, env)
    if code != 0 or not cli.exists():
        sys.exit("node_suite: building the compiler from src/PureMetalOnnxCompiler.pb failed (%s): %s"
                 % ("timeout" if code is None else exit_text(code), one_line(out)))
    return cli


def check_corpus(node_dir: Path, allow_unpinned: bool) -> tuple[str, int]:
    if onnx.__version__ != PINNED_ONNX_VERSION and not allow_unpinned:
        sys.exit("node_suite: this interpreter has onnx %s but the harness is pinned to onnx %s. Install "
                 "onnx==%s (pip install onnx==%s) so the score is measured against the same cases."
                 % (onnx.__version__, PINNED_ONNX_VERSION, PINNED_ONNX_VERSION, PINNED_ONNX_VERSION))
    if not node_dir.is_dir():
        sys.exit("node_suite: the node-test corpus is missing: %s does not exist. The onnx package normally installs "
                 "it; reinstall onnx==%s." % (node_dir, PINNED_ONNX_VERSION))
    digest, count = corpus_hash(node_dir)
    if digest != PINNED_CORPUS_SHA256 and not allow_unpinned:
        sys.exit("node_suite: the node-test corpus at %s hashes to %s, not the pinned %s. The installed test data "
                 "differs from the pinned onnx %s release, so a score against it would not be comparable; reinstall "
                 "onnx==%s, or pass --allow-unpinned for a private look that is never published."
                 % (node_dir, digest, PINNED_CORPUS_SHA256, PINNED_ONNX_VERSION, PINNED_ONNX_VERSION))
    return digest, count


def select_cases(node_dir: Path, patterns: list[str]) -> tuple[list[str], int, int]:
    chosen, over, no_ai = [], 0, 0
    for case in sorted(p.name for p in node_dir.iterdir() if p.is_dir()):
        if patterns and not any(fnmatch.fnmatch(case, p) for p in patterns):
            continue
        model = onnx.load(str(node_dir / case / "model.onnx"), load_external_data=False)
        v = ai_onnx_opset(model)
        if v is None:
            no_ai += 1
        elif v > MAX_OPSET:
            over += 1
        else:
            chosen.append(case)
    return chosen, over, no_ai


def run_all(names: list[str], node_dir: Path, work: Path, tools: Tools, supported: set[str], jobs: int, keep: bool) -> list[dict]:
    records: list[dict] = []
    lock = threading.Lock()
    done = [0]
    started = time.time()

    def one(name: str) -> dict:
        r = run_case(node_dir / name, name, work, tools, supported, keep)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(names):
                print("  %d/%d cases, %.0f s" % (done[0], len(names), time.time() - started), flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for r in pool.map(one, names):
            records.append(r)
    return sorted(records, key=lambda r: r["case"])


# ---------------------------------------------------------------------------
# Self-check: prove the harness can say no
# ---------------------------------------------------------------------------
def write_case(folder: Path, model: onnx.ModelProto, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    ds = folder / "test_data_set_0"
    ds.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(folder / "model.onnx"))
    for i, a in enumerate(inputs):
        (ds / ("input_%d.pb" % i)).write_bytes(numpy_helper.from_array(a).SerializeToString())
    for i, a in enumerate(outputs):
        (ds / ("output_%d.pb" % i)).write_bytes(numpy_helper.from_array(a).SerializeToString())


def self_check(node_dir: Path, work: Path, tools: Tools, supported: set[str]) -> int:
    base = work / "_selfcheck"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    results = []

    def check(label: str, ok: bool, detail: str) -> None:
        results.append(ok)
        print("%s  %s: %s" % ("PASS" if ok else "FAIL", label, detail), flush=True)

    known = "test_add"
    control = run_case(node_dir / known, known, base / "run", tools, supported, keep=True)
    check("control: %s from the corpus" % known, control["outcome"] == "PASS",
          "%s %s" % (control["outcome"], control["detail"]))

    # (i) one expected element nudged beyond tolerance
    mutated = base / "corpus" / "mutated_expected"
    shutil.copytree(node_dir / known, mutated)
    t = onnx.TensorProto()
    t.ParseFromString((mutated / "test_data_set_0" / "output_0.pb").read_bytes())
    arr = numpy_helper.to_array(t).copy()
    flat = arr.reshape(-1)
    nudge = 10 * (DEFAULT_ATOL + DEFAULT_RTOL * abs(float(flat[7])))
    flat[7] += nudge
    (mutated / "test_data_set_0" / "output_0.pb").write_bytes(numpy_helper.from_array(arr, t.name).SerializeToString())
    r = run_case(mutated, "mutated_expected", base / "run", tools, supported, keep=True)
    where = list(int(i) for i in np.unravel_index(7, arr.shape))
    check("(i) expected element 7 nudged by %.3g (10x tolerance)" % nudge,
          r["outcome"] == "FAIL_NUMERIC" and str(where) in r["detail"], "%s %s" % (r["outcome"], r["detail"]))

    # (ii) two same-shape outputs in swapped order
    rng = np.random.default_rng(20260916)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    y = rng.standard_normal((3, 4)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["sum"]), helper.make_node("Sub", ["x", "y"], ["difference"])],
        "two_outputs",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4]),
         helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("sum", onnx.TensorProto.FLOAT, [3, 4]),
         helper.make_tensor_value_info("difference", onnx.TensorProto.FLOAT, [3, 4])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])
    m.ir_version = 7
    write_case(base / "corpus" / "two_outputs", m, [x, y], [x + y, x - y])
    write_case(base / "corpus" / "two_outputs_swapped", m, [x, y], [x - y, x + y])
    r = run_case(base / "corpus" / "two_outputs", "two_outputs", base / "run", tools, supported, keep=True)
    check("(ii) control: two outputs in graph order", r["outcome"] == "PASS", "%s %s" % (r["outcome"], r["detail"]))
    r = run_case(base / "corpus" / "two_outputs_swapped", "two_outputs_swapped", base / "run", tools, supported, keep=True)
    check("(ii) the same expected outputs swapped", r["outcome"] == "FAIL_NUMERIC" and r["detail"].startswith("output 0"),
          "%s %s" % (r["outcome"], r["detail"]))

    # (iii) an operator outside the validators' list
    outside = "Erf"
    if outside in supported:
        check("(iii) operator outside the list", False, "%s is now listed; choose another operator for this check" % outside)
    else:
        g = helper.make_graph([helper.make_node(outside, ["x"], ["y"])], "outside",
                              [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4])],
                              [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 4])])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 7
        write_case(base / "corpus" / "outside_operator", m, [x], [x])
        r = run_case(base / "corpus" / "outside_operator", "outside_operator", base / "run", tools, supported, keep=True)
        check("(iii) a model using %s" % outside, r["outcome"] == "REFUSED" and r.get("named_ops") == [outside],
              "%s %s (named: %s)" % (r["outcome"], r["detail"], r.get("named_ops")))

    # (iv) the known-good case twice, from nothing each time
    a = run_case(node_dir / known, known, base / "twice_a", tools, supported, keep=True)
    b = run_case(node_dir / known, known, base / "twice_b", tools, supported, keep=True)
    ra = (base / "twice_a" / known / "test_data_set_0.result.bin").read_bytes() if a["outcome"] == "PASS" else b""
    rb = (base / "twice_b" / known / "test_data_set_0.result.bin").read_bytes() if b["outcome"] == "PASS" else b""
    check("(iv) %s built and run twice" % known, a == b and a["outcome"] == "PASS" and ra == rb and len(ra) > 12,
          "records %s, result bytes %s (%d bytes, sha256 %s)" % ("identical" if a == b else "DIFFER",
                                                                 "identical" if ra == rb else "DIFFER", len(ra),
                                                                 hashlib.sha256(ra).hexdigest()[:16]))
    print("self-check: %s (%d of %d checks)" % ("PASS" if all(results) else "FAIL", sum(results), len(results)))
    return 0 if all(results) else 1


def compact_json(document: dict) -> str:
    """One case per line: small enough to commit, and a later run diffs line by line."""
    head = {k: v for k, v in document.items() if k != "cases"}
    body = json.dumps(head, indent=1)[:-2]
    rows = ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in document["cases"])
    return body + ',\n "cases": [\n' + rows + "\n ]\n}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pbcompiler", required=True, type=Path, help="the x64 build compiler (docs/BUILD.md names it)")
    ap.add_argument("--cli", type=Path, help="use this compiler CLI instead of building one from src/")
    ap.add_argument("--from-commit", metavar="REV", help="build the compiler from this commit's src/ and runtime/ "
                    "instead of the working tree; published scores use this")
    ap.add_argument("--cases", nargs="*", default=[], help="glob patterns of case names; default is every case")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--work", type=Path, default=HERE / "work", help="scratch folder (ignored by git)")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--keep", action="store_true", help="keep the scratch folders of passing and refused cases too")
    ap.add_argument("--no-report", action="store_true", help="print the outcomes, write no results files")
    ap.add_argument("--self-check", action="store_true", help="prove the harness reports failures, then stop")
    ap.add_argument("--allow-unpinned", action="store_true", help="score a different corpus; never publish the result")
    ap.add_argument("--compile-timeout", type=float, default=120)
    ap.add_argument("--build-timeout", type=float, default=300)
    ap.add_argument("--run-timeout", type=float, default=60)
    ap.add_argument("--print-corpus-hash", action="store_true")
    args = ap.parse_args()
    quiet_crashes()

    node_dir = corpus_dir()
    if args.print_corpus_hash:
        digest, count = corpus_hash(node_dir)
        print("onnx %s corpus %s: %d cases, sha256 %s" % (onnx.__version__, node_dir, count, digest))
        return 0
    digest, corpus_cases = check_corpus(node_dir, args.allow_unpinned)
    if not args.pbcompiler.is_file():
        sys.exit("node_suite: the build compiler %s does not exist. Pass --pbcompiler with the x64 compiler "
                 "described in docs/BUILD.md." % args.pbcompiler)
    args.work, args.results = args.work.resolve(), args.results.resolve()  # the compiler runs in each case folder
    args.work.mkdir(parents=True, exist_ok=True)
    if args.from_commit:
        source_root, commit = export_commit(args.from_commit, args.work)
    else:
        source_root, commit = ROOT, git_head()
    if args.cli:
        commit = "prebuilt CLI %s" % args.cli.name
    fixed, dynamic = supported_operators(source_root)
    supported = fixed | dynamic
    cli = args.cli if args.cli else build_cli(args.pbcompiler, args.work, args.build_timeout, source_root)
    tools = Tools(cli, args.pbcompiler, args.compile_timeout, args.build_timeout, args.run_timeout)
    print("onnx %s, corpus sha256 %s (%d cases); compiler source %s; %d listed operators; %d jobs"
          % (onnx.__version__, digest, corpus_cases, commit, len(supported), args.jobs), flush=True)

    if args.self_check:
        return self_check(node_dir, args.work, tools, supported)

    names, over, no_ai = select_cases(node_dir, args.cases)
    print("scoring %d cases (excluded: %d above opset %d, %d without an ai.onnx import)" % (len(names), over, MAX_OPSET, no_ai), flush=True)
    started = time.time()
    records = run_all(names, node_dir, args.work / "cases", tools, supported, args.jobs, args.keep)
    elapsed = time.time() - started
    counts = {o: sum(1 for r in records if r["outcome"] == o) for o in OUTCOMES}
    attempted = len(records) - counts["REFUSED"] - counts["HARNESS_ERROR"]
    print("PASS %d of %d attempted, %d of %d scored; %s; %.0f s" % (
        counts["PASS"], attempted, counts["PASS"], len(records),
        ", ".join("%s %d" % (o, counts[o]) for o in OUTCOMES), elapsed))
    if args.no_report or args.cases:
        for r in records:
            if r["outcome"] != "PASS":
                print("  %-60s %-13s %s" % (r["case"], r["outcome"], r["detail"]))
        return 0 if counts["HARNESS_ERROR"] == 0 else 2
    if counts["HARNESS_ERROR"]:
        for r in records:
            if r["outcome"] == "HARNESS_ERROR":
                print("  HARNESS_ERROR %s: %s" % (r["case"], r["detail"]))
        sys.exit("node_suite: %d cases ended in HARNESS_ERROR; the harness itself could not prepare them, so no "
                 "score was written. Fix the harness first." % counts["HARNESS_ERROR"])
    stem = "%s-onnx-%s" % (args.date, onnx.__version__)
    header = {"date": args.date, "onnx_version": onnx.__version__, "corpus_sha256": digest,
              "corpus_cases": corpus_cases, "excluded_opset": over, "excluded_no_ai_onnx": no_ai,
              "compiler_commit": commit, "json_name": stem + ".json"}
    args.results.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "ONNX node-test coverage", "version": 1, "target": "windows",
        "onnx_version": onnx.__version__, "corpus_sha256": digest, "corpus_cases": corpus_cases,
        "max_opset": MAX_OPSET, "scored_cases": len(records), "excluded_opset_above_max": over,
        "excluded_no_ai_onnx_import": no_ai, "compiler_source": commit,
        "tolerance": {"rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL, "source": "onnx/backend/test/loader/__init__.py:31-32"},
        "listed_operators": sorted(supported),
        "outcomes": counts, "pass_of_attempted": [counts["PASS"], attempted], "pass_of_scored": [counts["PASS"], len(records)],
        "cases": records,
    }
    (args.results / (stem + ".json")).write_text(compact_json(document), encoding="utf-8")
    (args.results / (stem + ".md")).write_text(summarize(records, header, supported), encoding="utf-8")
    print("wrote %s and %s" % (args.results / (stem + ".json"), args.results / (stem + ".md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
