#!/usr/bin/env python3
"""Node-test coverage for InstanceNormalization, TopK, ScatterElements, ReduceMax,
ReduceProd, Not, Identity and Pad beyond what the corpus scores at opset <= 20.

DEVELOPER CHECK ONLY, in the node_suite family.

Two groups of cases:

  r20_<case>   Every official node test for these operators that onnx 1.22.0
               publishes only above opset 20 (InstanceNormalization-22, TopK-24,
               Pad-25, Identity-25). Those later definitions add element types
               and nothing else, so the same model is re-imported at opset 20 and
               scored against the official output_*.pb files unchanged. A case the
               opset-20 schema rejects is skipped with a line saying so.

  <name>_fixed / <name>_dynamic
               Targeted forms the node tests do not reach, and the exact forms the
               Kitten TTS nano graph uses, built once with every extent declared
               (fixed-shape path) and once with a symbolic leading extent
               (runtime-dimension path). Expected outputs come from
               onnx.reference.ReferenceEvaluator, cross-checked against
               onnxruntime when it runs the model.

Cases whose name starts with refuse_ must end REFUSED with a sentence containing
the given text; everything else must PASS.

Usage (repository root):
  py -3.12 tests/node_suite/targeted_norm_small.py --pbcompiler PATH [--cli PATH] [--only GLOB ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import node_suite as ns  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

RNG = np.random.default_rng(20260916)
F, I64, I32, B = TensorProto.FLOAT, TensorProto.INT64, TensorProto.INT32, TensorProto.BOOL
REEXPORT = ("test_instancenorm_*", "test_top_k*", "test_edge_pad", "test_reflect_pad", "test_wrap_pad",
            "test_constant_pad*", "test_identity*")


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), opset=20, dynamic=True, fixed=True,
                 refuse=None, symbolic_axes=(0,), oracle=None):
        self.name, self.nodes, self.inputs, self.outputs, self.feeds = name, nodes, inputs, outputs, feeds
        self.initializers, self.opset, self.dynamic, self.fixed, self.refuse = list(initializers), opset, dynamic, fixed, refuse
        self.symbolic_axes, self.oracle = symbolic_axes, oracle


def pad_oracle(pads, mode):
    """Pad-19 with negative pads: crop first, then pad the cropped data (the reference
    evaluator hands negative widths to numpy.pad, which refuses them)."""
    def run(feeds):
        x = feeds["x"]
        rank = x.ndim
        begins, ends = pads[:rank], pads[rank:]
        crop = tuple(slice(max(-b, 0), x.shape[i] - max(-e, 0)) for i, (b, e) in enumerate(zip(begins, ends)))
        widths = [(max(b, 0), max(e, 0)) for b, e in zip(begins, ends)]
        return [np.pad(x[crop], widths, mode=mode)]
    return run


def model_for(case: Case, symbolic: bool) -> onnx.ModelProto:
    def shape(s):
        if not symbolic or not s:
            return s
        return ["S%d" % i if i in case.symbolic_axes else d for i, d in enumerate(s)]
    ins = [helper.make_tensor_value_info(n, e, shape(s)) for n, e, s in case.inputs]
    outs = [helper.make_tensor_value_info(n, e, (["%s_%d" % (n, i) for i in range(len(s))] if symbolic else s))
            for n, e, s in case.outputs]
    g = helper.make_graph(case.nodes, case.name, ins, outs, case.initializers)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", case.opset)])
    m.ir_version = 9
    return m


def expected(case: Case, m: onnx.ModelProto):
    source = case.oracle(case.feeds) if case.oracle else ReferenceEvaluator(m).run(None, case.feeds)
    got = [np.asarray(x) for x in source]
    if ort is not None:
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 4
            sess = ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
            other = sess.run(None, case.feeds)
        except Exception as exc:
            print("note: %s: onnxruntime cannot run it (%s); the reference alone decides" % (case.name, str(exc)[-100:]))
            other = []
        for i, (a, b) in enumerate(zip(got, other)):
            b = np.asarray(b)
            if a.shape != b.shape or not np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=1e-4, atol=1e-5, equal_nan=True):
                print("note: %s output %d: onnxruntime differs from the reference evaluator; the reference decides\n  ref %r\n  ort %r"
                      % (case.name, i, a.ravel()[:12], b.ravel()[:12]))
    return got


def write(folder: Path, case: Case, symbolic: bool):
    m = model_for(case, symbolic)
    onnx.checker.check_model(m)
    outs = [] if case.refuse else expected(case, m)
    ns.write_case(folder, m, [case.feeds[n] for n, _, _ in case.inputs], outs)


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def cases() -> list[Case]:
    c: list[Case] = []
    N = helper.make_node

    # ---- InstanceNormalization -------------------------------------------------
    c.append(Case("instnorm_kitten_rank3", [N("InstanceNormalization", ["x", "s", "b"], ["y"], epsilon=9.999999747378752e-06)],
                  [("x", F, [1, 16, 37])], [("y", F, [1, 16, 37])], {"x": f32(1, 16, 37) * 3 + 1},
                  [init("s", f32(16)), init("b", f32(16))], symbolic_axes=(0, 2)))
    c.append(Case("instnorm_rank4_tails", [N("InstanceNormalization", ["x", "s", "b"], ["y"], epsilon=0.01)],
                  [("x", F, [2, 3, 5, 3]), ("s", F, [3]), ("b", F, [3])], [("y", F, [2, 3, 5, 3])],
                  {"x": f32(2, 3, 5, 3), "s": f32(3), "b": f32(3)}))
    for width in (1, 2, 3, 4, 5, 8, 9, 130):
        c.append(Case("instnorm_width%d" % width, [N("InstanceNormalization", ["x", "s", "b"], ["y"])],
                      [("x", F, [2, 2, width])], [("y", F, [2, 2, width])], {"x": f32(2, 2, width) * 100},
                      [init("s", f32(2)), init("b", f32(2))], symbolic_axes=(2,)))

    # ---- TopK ------------------------------------------------------------------
    x = np.array([5, 1, 5, 3, 7, 3, 5, 0, 7], dtype=np.int64)
    c.append(Case("topk_kitten_int64_full_sort", [N("TopK", ["x", "k"], ["v", "i"], sorted=1, axis=-1, largest=1)],
                  [("x", I64, [9]), ("k", I64, [1])], [("v", I64, [9]), ("i", I64, [9])],
                  {"x": x, "k": np.array([9], dtype=np.int64)}))
    c.append(Case("topk_ties_smallest", [N("TopK", ["x", "k"], ["v", "i"], axis=0, largest=0)],
                  [("x", I64, [9]), ("k", I64, [1])], [("v", I64, [4]), ("i", I64, [4])],
                  {"x": x, "k": np.array([4], dtype=np.int64)}))
    xf = np.round(f32(4, 6, 3), 1)
    c.append(Case("topk_float_rank3_axis_neg2", [N("TopK", ["x", "k"], ["v", "i"], axis=-2)],
                  [("x", F, [4, 6, 3]), ("k", I64, [1])], [("v", F, [4, 3, 3]), ("i", I64, [4, 3, 3])],
                  {"x": xf, "k": np.array([3], dtype=np.int64)}))
    c.append(Case("topk_int32_constant_k", [N("TopK", ["x", "k"], ["v", "i"], axis=1, largest=0)],
                  [("x", I32, [3, 7])], [("v", I32, [3, 2]), ("i", I64, [3, 2])],
                  {"x": RNG.integers(-3, 3, (3, 7)).astype(np.int32)}, [init("k", np.array([2], dtype=np.int64))], fixed=False))
    c.append(Case("topk_float_constant_k", [N("TopK", ["x", "k"], ["v", "i"], axis=1)],
                  [("x", F, [3, 7])], [("v", F, [3, 4]), ("i", I64, [3, 4])],
                  {"x": np.round(f32(3, 7), 0)}, [init("k", np.array([4], dtype=np.int64))]))
    c.append(Case("topk_k_zero", [N("TopK", ["x", "k"], ["v", "i"])],
                  [("x", F, [2, 5]), ("k", I64, [1])], [("v", F, [2, 0]), ("i", I64, [2, 0])],
                  {"x": f32(2, 5), "k": np.array([0], dtype=np.int64)}))

    # ---- ScatterElements ---------------------------------------------------------
    perm = np.array([3, 0, 5, 1, 4, 2], dtype=np.int64)
    c.append(Case("scatter_kitten_int64_axis0", [N("ScatterElements", ["d", "i", "u"], ["y"], axis=0, reduction="none")],
                  [("d", I64, [6]), ("i", I64, [6]), ("u", I64, [6])], [("y", I64, [6])],
                  {"d": np.zeros(6, np.int64), "i": perm, "u": np.arange(6, dtype=np.int64)}))
    for red in ("add", "mul", "max", "min"):
        c.append(Case("scatter_float_rank3_axis1_%s" % red, [N("ScatterElements", ["d", "i", "u"], ["y"], axis=1, reduction=red)],
                      [("d", F, [2, 4, 3]), ("i", I32, [2, 3, 3]), ("u", F, [2, 3, 3])], [("y", F, [2, 4, 3])],
                      {"d": f32(2, 4, 3), "i": RNG.integers(-4, 4, (2, 3, 3)).astype(np.int32), "u": f32(2, 3, 3)},
                      symbolic_axes=(0,), fixed=False))
        c.append(Case("scatter_int64_axis_neg1_%s" % red, [N("ScatterElements", ["d", "i", "u"], ["y"], axis=-1, reduction=red)],
                      [("d", I64, [3, 5]), ("i", I64, [3, 4]), ("u", I64, [3, 4])], [("y", I64, [3, 5])],
                      {"d": RNG.integers(-9, 9, (3, 5)), "i": RNG.integers(-5, 5, (3, 4)), "u": RNG.integers(-9, 9, (3, 4))}))
    for red in ("none", "max", "min"):
        c.append(Case("scatter_bool_%s" % red, [N("ScatterElements", ["d", "i", "u"], ["y"], axis=0, reduction=red)],
                      [("d", B, [4, 2]), ("i", I64, [3, 2]), ("u", B, [3, 2])], [("y", B, [4, 2])],
                      {"d": RNG.integers(0, 2, (4, 2)).astype(bool), "i": np.array([[0, 1], [0, 3], [2, 1]], np.int64),
                       "u": RNG.integers(0, 2, (3, 2)).astype(bool)}))
    c.append(Case("refuse_scatter_reduction_sum", [N("ScatterElements", ["d", "i", "u"], ["y"], reduction="sum")],
                  [("d", F, [4]), ("i", I64, [2]), ("u", F, [2])], [("y", F, [4])],
                  {"d": f32(4), "i": np.array([0, 1], np.int64), "u": f32(2)}, refuse="reduction = sum"))

    # ---- ReduceMax / ReduceProd ---------------------------------------------------
    c.append(Case("reducemax_kitten_int64_all", [N("ReduceMax", ["x"], ["y"], keepdims=0, noop_with_empty_axes=0)],
                  [("x", I64, [7])], [("y", I64, [])], {"x": RNG.integers(-100, 100, 7)}))
    c.append(Case("reduceprod_kitten_shape", [N("ReduceProd", ["x"], ["y"], keepdims=0, noop_with_empty_axes=0)],
                  [("x", I64, [3])], [("y", I64, [])], {"x": np.array([2, 5, 7], np.int64)}, dynamic=False))
    for op in ("ReduceMax", "ReduceProd"):
        low = op.lower()
        c.append(Case("%s_axes_0_2_keep" % low, [N(op, ["x", "a"], ["y"], keepdims=1)],
                      [("x", F, [2, 3, 4, 2])], [("y", F, [1, 3, 1, 2])], {"x": f32(2, 3, 4, 2)},
                      [init("a", np.array([0, -2], np.int64))]))
        c.append(Case("%s_axes_1_3_drop" % low, [N(op, ["x", "a"], ["y"], keepdims=0)],
                      [("x", F, [2, 3, 4, 2])], [("y", F, [2, 4])], {"x": f32(2, 3, 4, 2)},
                      [init("a", np.array([1, 3], np.int64))]))
        c.append(Case("%s_noop_empty_axes" % low, [N(op, ["x"], ["y"], keepdims=1, noop_with_empty_axes=1)],
                      [("x", F, [2, 3])], [("y", F, [2, 3])], {"x": f32(2, 3)}))
        c.append(Case("%s_noop_axes_input_empty" % low, [N(op, ["x", "a"], ["y"], keepdims=0, noop_with_empty_axes=1)],
                      [("x", F, [2, 3]), ("a", I64, [0])], [("y", F, [2, 3])],
                      {"x": f32(2, 3), "a": np.array([], np.int64)}, fixed=False))
        c.append(Case("%s_opset13_attribute_axes" % low, [N(op, ["x"], ["y"], axes=[1, -1], keepdims=0)],
                      [("x", F, [2, 3, 4])], [("y", F, [2])], {"x": f32(2, 3, 4)}, opset=13))
        c.append(Case("%s_int32" % low, [N(op, ["x"], ["y"], axes=[0], keepdims=1)],
                      [("x", I32, [4, 3])], [("y", I32, [1, 3])], {"x": RNG.integers(-5, 6, (4, 3)).astype(np.int32)},
                      opset=13, fixed=False))
    c.append(Case("reducemax_nan_and_inf", [N("ReduceMax", ["x"], ["y"], axes=[1], keepdims=0)],
                  [("x", F, [3, 3])], [("y", F, [3])],
                  {"x": np.array([[1, np.nan, 2], [-np.inf, -np.inf, -np.inf], [0, -0.0, 3]], np.float32)}, opset=13))

    # ---- Not / Identity -----------------------------------------------------------
    c.append(Case("not_kitten_bool_rank2", [N("Not", ["x"], ["y"])], [("x", B, [5, 4])], [("y", B, [5, 4])],
                  {"x": RNG.integers(0, 2, (5, 4)).astype(bool)}))
    for elem, arr in ((F, f32(2, 3)), (I64, RNG.integers(-9, 9, (2, 3))), (B, RNG.integers(0, 2, (2, 3)).astype(bool)),
                      (I32, RNG.integers(-9, 9, (2, 3)).astype(np.int32))):
        c.append(Case("identity_%s" % TensorProto.DataType.Name(elem).lower(), [N("Identity", ["x"], ["y"])],
                      [("x", elem, [2, 3])], [("y", elem, [2, 3])], {"x": arr}, fixed=(elem != I32)))

    # ---- Pad ------------------------------------------------------------------
    c.append(Case("pad_kitten_edge_rank2", [N("Pad", ["x", "p"], ["y"], mode="edge")],
                  [("x", F, [3, 12])], [("y", F, [3, 32])], {"x": f32(3, 12)},
                  [init("p", np.array([0, 10, 0, 10], np.int64))]))
    c.append(Case("pad_kitten_reflect_rank3", [N("Pad", ["x", "p"], ["y"], mode="reflect")],
                  [("x", F, [1, 5, 9])], [("y", F, [1, 5, 10])], {"x": f32(1, 5, 9)},
                  [init("p", np.array([0, 0, 1, 0, 0, 0], np.int64))], symbolic_axes=(2,)))
    for mode in ("constant", "reflect", "edge", "wrap"):
        c.append(Case("pad_%s_every_axis" % mode, [N("Pad", ["x", "p"], ["y"], mode=mode)],
                      [("x", F, [2, 3, 4])], [("y", F, [5, 4, 11])], {"x": f32(2, 3, 4)},
                      [init("p", np.array([1, 0, 3, 2, 1, 4], np.int64))]))
        c.append(Case("pad_%s_negative" % mode, [N("Pad", ["x", "p"], ["y"], mode=mode)],
                      [("x", F, [4, 6])], [("y", F, [3, 7])], {"x": f32(4, 6)},
                      [init("p", np.array([-1, 2, 0, -1], np.int64))], oracle=pad_oracle([-1, 2, 0, -1], mode)))
    c.append(Case("pad_wrap_longer_than_axis", [N("Pad", ["x", "p"], ["y"], mode="wrap")],
                  [("x", I64, [2, 3])], [("y", I64, [11, 13])], {"x": RNG.integers(-9, 9, (2, 3))},
                  [init("p", np.array([4, 5, 5, 5], np.int64))]))
    c.append(Case("pad_constant_axes_value", [N("Pad", ["x", "p", "v", "a"], ["y"], mode="constant")],
                  [("x", F, [2, 3, 4]), ("v", F, [])], [("y", F, [2, 3, 7])], {"x": f32(2, 3, 4), "v": np.array(2.5, np.float32)},
                  [init("p", np.array([1, 2], np.int64)), init("a", np.array([-1], np.int64))]))
    c.append(Case("pad_bool_constant", [N("Pad", ["x", "p"], ["y"])],
                  [("x", B, [2, 2])], [("y", B, [4, 3])], {"x": np.ones((2, 2), bool)},
                  [init("p", np.array([1, 0, 1, 1], np.int64))]))
    c.append(Case("refuse_pad_wrap_opset18", [N("Pad", ["x", "p"], ["y"], mode="wrap")],
                  [("x", F, [2, 3])], [("y", F, [2, 5])], {"x": f32(2, 3)},
                  [init("p", np.array([0, 1, 0, 1], np.int64))], opset=18, refuse="mode = wrap"))
    return c


def reexported(node_dir: Path, corpus: Path) -> list[tuple[str, str | None]]:
    out = []
    for case_dir in sorted(p for p in node_dir.iterdir() if p.is_dir()):
        if not any(fnmatch.fnmatch(case_dir.name, g) for g in REEXPORT):
            continue
        m = onnx.load(str(case_dir / "model.onnx"), load_external_data=False)
        if ns.ai_onnx_opset(m) is None or ns.ai_onnx_opset(m) <= 20:
            continue
        for o in m.opset_import:
            if o.domain in ("", "ai.onnx"):
                o.version = 20
        try:
            onnx.checker.check_model(m, full_check=True)
        except Exception as exc:
            print("skip r20_%s: the opset-20 schema rejects it (%s)" % (case_dir.name, str(exc).splitlines()[0][:120]))
            continue
        name = "r20_" + case_dir.name
        dest = corpus / name
        shutil.copytree(case_dir, dest)
        onnx.save(m, str(dest / "model.onnx"))
        refuse = None
        if case_dir.name in ("test_top_k_uint64",):
            refuse = "element type UINT64"
        # Identity over an optional type is still refused. Identity over a
        # sequence used to be refused too, until the sequence operators
        # landed (SequenceEmpty .. ConcatFromSequence); it passes since then.
        if case_dir.name in ("test_identity_opt",):
            refuse = ""
        out.append((name, refuse))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="lane cases for normalisation and small kernels")
    ap.add_argument("--pbcompiler", required=True, type=Path)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--work", type=Path, default=HERE / "work" / "norm_small")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    ns.quiet_crashes()
    corpus = args.work / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    todo = []
    for name, refuse in reexported(ns.corpus_dir(), corpus):
        if not args.only or any(fnmatch.fnmatch(name, p) for p in args.only):
            todo.append((name, refuse))
    for case in cases():
        for variant, symbolic, wanted in (("fixed", False, case.fixed), ("dynamic", True, case.dynamic)):
            name = "%s_%s" % (case.name, variant)
            if not wanted or (args.only and not any(fnmatch.fnmatch(name, p) for p in args.only)):
                continue
            write(corpus / name, case, symbolic)
            todo.append((name, case.refuse))
    cli = args.cli or ns.build_cli(args.pbcompiler, args.work, 600)
    tools = ns.Tools(cli, args.pbcompiler, 120, 300, 60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda t: (t[1], ns.run_case(corpus / t[0], t[0], args.work / "cases", tools, set(), False)), todo))
    bad = 0
    for refuse, r in sorted(records, key=lambda t: t[1]["case"]):
        if refuse is not None:
            ok = r["outcome"] == "REFUSED" and refuse in r["detail"]
        else:
            ok = r["outcome"] == "PASS"
        bad += 0 if ok else 1
        print("%-4s %-58s %-12s %-18s %s" % ("ok" if ok else "BAD", r["case"], r["outcome"], r.get("path") or "", r["detail"][:260]))
    print("norm_small: %d of %d as expected" % (len(records) - bad, len(records)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
