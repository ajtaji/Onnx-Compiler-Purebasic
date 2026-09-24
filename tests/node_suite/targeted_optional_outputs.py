#!/usr/bin/env python3
"""Nodes whose optional outputs are absent (forum 988).

DEVELOPER CHECK ONLY, in the node_suite family.

An ONNX node may leave an optional output out: it lists fewer outputs, or it
names an output position with the empty string. The compiler must then pass
the null address for that position where the kernel takes none, and give the
kernel memory for it where the kernel still writes it (LSTM keeps its running
state in Y_h and Y_c, and builds Y_h from Y). Forum 988: an LSTM whose only
output is Y was emitted as "PmOnnxLstm\\YH = " followed by nothing, and the
generated program did not build.

Every case below leaves out at least one optional output, in both spellings,
and is built with every extent declared (fixed-shape path) and, where that path
implements the form, with a symbolic leading extent (runtime-dimension path).
Expected outputs come from onnx.reference.ReferenceEvaluator, cross-checked
against onnxruntime when it runs the model (onnxruntime alone where the
reference evaluator does not compute the output: its LSTM has no Y_c). Every
case must PASS, except one that must be REFUSED with the sentence it names.

--mutants rebuilds the compiler from a copy of this repository's source with
the forum 988 correction taken out again, once per path, runs that path's
cases with it, and requires that the gate goes red.

Usage (repository root):
  py -3.12 tests/node_suite/targeted_optional_outputs.py --pbcompiler PATH [--cli PATH] [--mutants]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import node_suite as ns  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

RNG = np.random.default_rng(988)
F, I64 = TensorProto.FLOAT, TensorProto.INT64

# The correction a mutant takes out again: (label, file, text, replacement,
# the path whose cases must then fail).
MUTANTS = [
    ("fixed-absent-unnamed", "src/compiler/onnx_compile.pbi",
     "  PmoCompileNameAbsentOutputs(@Model, #True)\n", "", "fixed"),
    ("dynamic-absent-unnamed", "src/compiler/onnx_dynamic_emit.pbi",
     "  PmoCompileNameAbsentOutputs(@model,#False)\n", "", "dynamic"),
]


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), opset=20, dynamic=True, refuse=None):
        self.name, self.nodes, self.inputs, self.outputs, self.feeds = name, nodes, inputs, outputs, feeds
        self.initializers, self.opset, self.dynamic, self.refuse = list(initializers), opset, dynamic, refuse


def model_for(case: Case, symbolic: bool) -> onnx.ModelProto:
    def shape(s, prefix):
        if not symbolic:
            return s
        return ["%s_%d" % (prefix, i) if i == 0 else d for i, d in enumerate(s)]
    ins = [helper.make_tensor_value_info(n, e, shape(s, "S")) for n, e, s in case.inputs]
    outs = [helper.make_tensor_value_info(n, e, shape(s, n) if symbolic else s) for n, e, s in case.outputs]
    g = helper.make_graph(case.nodes, case.name, ins, outs, case.initializers)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", case.opset)])
    m.ir_version = 8
    return m


def expected(case: Case, m: onnx.ModelProto):
    # The reference evaluator's LSTM loses track of the outputs after an
    # empty-named one, so it evaluates a copy in which every left-out output
    # has a name; the graph outputs it is asked for are the same tensors.
    named = onnx.ModelProto()
    named.CopyFrom(m)
    for n, node in enumerate(named.graph.node):
        for i, out in enumerate(node.output):
            if out == "":
                node.output[i] = "__absent_%d_%d" % (n, i)
    try:
        got = [np.asarray(x) for x in ReferenceEvaluator(named).run([o.name for o in m.graph.output], case.feeds)]
    except RuntimeError as exc:
        # Its LSTM computes Y and Y_h only; onnxruntime decides Y_c.
        if ort is None:
            raise SystemExit("%s: the reference evaluator cannot compute it (%s) and onnxruntime is not installed"
                             % (case.name, str(exc)[:80]))
        options = ort.SessionOptions()
        options.log_severity_level = 4
        print("note: %s: the reference evaluator cannot compute every output; onnxruntime decides" % case.name)
        return [np.asarray(x) for x in ort.InferenceSession(m.SerializeToString(), options,
                                                            providers=["CPUExecutionProvider"]).run(None, case.feeds)]
    if ort is not None:
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 4
            other = ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, case.feeds)
        except Exception as exc:
            print("note: %s: onnxruntime cannot run it (%s); the reference alone decides" % (case.name, str(exc)[-100:]))
            other = []
        for i, (a, b) in enumerate(zip(got, other)):
            if a.shape != np.asarray(b).shape or not np.allclose(a, b, rtol=1e-4, atol=1e-5):
                raise SystemExit("%s output %d: onnxruntime and the reference evaluator disagree" % (case.name, i))
    return got


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def cases() -> list[Case]:
    c: list[Case] = []
    N = helper.make_node
    T, B, I, H = 3, 1, 2, 2
    x = f32(T, B, I)
    lw, lr, lb = f32(1, 4 * H, I), f32(1, 4 * H, H), f32(1, 8 * H)
    y, yh = (T, 1, B, H), (1, B, H)
    base = [init("W", lw), init("R", lr)]
    # The model forum 988 was reproduced with: X, W, R in and Y out, nothing else.
    c.append(Case("lstm_y_only", [N("LSTM", ["X", "W", "R"], ["Y"], hidden_size=H)],
                  [("X", F, [T, B, I])], [("Y", F, list(y))], {"X": x}, base, opset=14))
    c.append(Case("lstm_y_only_with_bias", [N("LSTM", ["X", "W", "R", "B"], ["Y"], hidden_size=H)],
                  [("X", F, [T, B, I])], [("Y", F, list(y))], {"X": x}, base + [init("B", lb)], opset=14))
    c.append(Case("lstm_yh_only", [N("LSTM", ["X", "W", "R"], ["", "Y_h"], hidden_size=H)],
                  [("X", F, [T, B, I])], [("Y_h", F, list(yh))], {"X": x}, base, opset=14))
    c.append(Case("lstm_y_and_yc", [N("LSTM", ["X", "W", "R"], ["Y", "", "Y_c"], hidden_size=H)],
                  [("X", F, [T, B, I])], [("Y", F, list(y)), ("Y_c", F, list(yh))], {"X": x}, base, opset=14))
    c.append(Case("lstm_y_and_yh", [N("LSTM", ["X", "W", "R"], ["Y", "Y_h"], hidden_size=H)],
                  [("X", F, [T, B, I])], [("Y", F, list(y)), ("Y_h", F, list(yh))], {"X": x}, base, opset=14))
    # BatchNormalization-15 in training mode with running_mean and running_var
    # left out: the operator's own shape inference (and onnxruntime) requires
    # all three outputs there, so the compiler refuses it with a sentence.
    bx = f32(2, 3, 4)
    bn = [init("scale", f32(3)), init("bias", f32(3)), init("mean", f32(3)), init("var", np.abs(f32(3)) + 0.5)]
    c.append(Case("batchnorm_training_y_only",
                  [N("BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"], training_mode=1, epsilon=1e-5)],
                  [("X", F, [2, 3, 4])], [("Y", F, [2, 3, 4])], {"X": bx}, bn, opset=15, dynamic=False,
                  refuse="training_mode = 1 without output running_mean"))
    # LayerNormalization with Mean and InvStdDev left out, and with Mean only left out.
    lx = f32(2, 5)
    ln = [init("s", f32(5)), init("b", f32(5))]
    c.append(Case("layernorm_y_only", [N("LayerNormalization", ["X", "s", "b"], ["Y"], axis=-1)],
                  [("X", F, [2, 5])], [("Y", F, [2, 5])], {"X": lx}, ln, opset=17))
    c.append(Case("layernorm_y_and_invstd", [N("LayerNormalization", ["X", "s", "b"], ["Y", "", "InvStdDev"], axis=-1)],
                  [("X", F, [2, 5])], [("Y", F, [2, 5]), ("InvStdDev", F, [2, 1])], {"X": lx}, ln, opset=17, dynamic=False))
    return c


def write(folder: Path, case: Case, symbolic: bool):
    m = model_for(case, symbolic)
    onnx.checker.check_model(m)
    outs = [] if case.refuse else expected(case, m)
    ns.write_case(folder, m, [case.feeds[n] for n, _, _ in case.inputs], outs)


def run(cli: Path, pbcompiler: Path, corpus: Path, names: list[str], work: Path, jobs: int) -> list[dict]:
    tools = ns.Tools(cli, pbcompiler, 120, 300, 60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda n: ns.run_case(corpus / n, n, work, tools, set(), False), names))


def main() -> int:
    ap = argparse.ArgumentParser(description="absent optional outputs (forum 988)")
    ap.add_argument("--pbcompiler", required=True, type=Path)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--work", type=Path, default=HERE / "work" / "optional_outputs")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    ns.quiet_crashes()
    corpus = args.work / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    names, fixed, refusals = [], [], {}
    for case in cases():
        for variant, symbolic, wanted in (("fixed", False, True), ("dynamic", True, case.dynamic)):
            if wanted:
                name = "%s_%s" % (case.name, variant)
                write(corpus / name, case, symbolic)
                names.append(name)
                if case.refuse:
                    refusals[name] = case.refuse
                elif not symbolic:
                    fixed.append(name)
    cli = args.cli or ns.build_cli(args.pbcompiler, args.work, 600)
    bad = 0
    for r in sorted(run(cli, args.pbcompiler, corpus, names, args.work / "cases", args.jobs), key=lambda r: r["case"]):
        if r["case"] in refusals:
            ok = r["outcome"] == "REFUSED" and refusals[r["case"]] in r["detail"]
        else:
            ok = r["outcome"] == "PASS"
        bad += 0 if ok else 1
        print("%-4s %-40s %-12s %-18s %s" % ("ok" if ok else "BAD", r["case"], r["outcome"], r.get("path") or "", r["detail"][:200]))
    print("optional outputs: %d of %d as expected" % (len(names) - bad, len(names)))
    if args.mutants:
        killed = 0
        for label, rel, text, replacement, path in MUTANTS:
            tree = args.work / ("mutant-" + label)
            shutil.rmtree(tree, ignore_errors=True)
            for part in ("src", "runtime", "Build.pb"):
                source = ROOT / part
                (shutil.copytree if source.is_dir() else shutil.copy2)(source, tree / part)
            target = tree / rel
            body = target.read_text(encoding="utf-8")
            if body.count(text) != 1:
                raise SystemExit("mutant %s: the text to take out is not in %s exactly once" % (label, rel))
            target.write_text(body.replace(text, replacement), encoding="utf-8")
            mcli = ns.build_cli(args.pbcompiler, tree / "build", 600, tree)
            chosen = [n for n in names if n.endswith("_" + path) and n not in refusals]
            records = run(mcli, args.pbcompiler, corpus, chosen, tree / "cases", args.jobs)
            failed = [r["case"] for r in records if r["outcome"] != "PASS"]
            killed += 1 if failed else 0
            print("%s mutant %s: %d of %d %s cases fail (%s)" % ("KILLED  " if failed else "SURVIVED", label,
                                                                 len(failed), len(chosen), path, ", ".join(failed[:4])))
        print("mutants: %d of %d killed" % (killed, len(MUTANTS)))
        bad += len(MUTANTS) - killed
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
