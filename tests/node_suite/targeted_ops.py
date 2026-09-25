#!/usr/bin/env python3
"""Node-test coverage for the operators that complete the ONNX operator set
(src/compiler/onnx_emit_ops.pbi, runtime/tensor_ops.pmi).

DEVELOPER CHECK ONLY, in the node_suite family.

Two groups of cases:

  r20_<case>   Every official node test for these operators that onnx 1.22.0
               publishes only above opset 20 (MaxPool-22, AveragePool-22, ...).
               Those later definitions add element types and nothing else, so
               the same model is re-imported at opset 20 and scored against the
               official output_*.pb files unchanged. A case the opset-20 schema
               rejects is skipped with a line saying so.

  <name>_fixed / <name>_dynamic
               Targeted forms, built once with every extent declared
               (fixed-shape path) and once with a symbolic leading extent
               (runtime-dimension path). Expected outputs come from
               onnx.reference.ReferenceEvaluator, cross-checked against
               onnxruntime when it runs the model; a disagreement beyond the
               node tests' tolerance stops the generator.

Cases whose name starts with refuse_ must end REFUSED with a sentence containing
the given text; everything else must PASS.

Usage (repository root):
  py -3.12 tests/node_suite/targeted_ops.py --pbcompiler PATH [--cli PATH] [--only GLOB ...]
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

# the random cases' expected values are the generator contract written out below
ns.RANDOM_VALUES_EXPECTED = True

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

RNG = np.random.default_rng(20260924)
F, I64, I32, B = TensorProto.FLOAT, TensorProto.INT64, TensorProto.INT32, TensorProto.BOOL
OPS = ("erf", "reciprocal", "ceil", "sign", "softplus", "softsign", "elu", "selu", "celu", "hardsigmoid", "hardswish",
       "mish", "gelu", "thresholdedrelu", "shrink", "isnan", "isinf", "min", "max", "sum", "mean", "mod", "prelu", "or",
       "xor", "reduce_min", "reduce_l1", "reduce_l2", "reduce_sum_square", "reduce_log_sum", "argmax", "argmin",
       "logsoftmax", "maxpool", "averagepool", "lppool", "globalmaxpool", "globalaveragepool", "globallppool", "split",
       "tile", "depthtospace", "spacetodepth", "tril", "triu", "gather_elements", "gathernd", "onehot", "einsum",
       "dropout", "castlike", "size",
       # second group (GroupNormalization is left out: its official cases are
       # GroupNormalization-21, whose per-channel scale differs from 18's)
       "tan", "asin", "acos", "sinh", "cosh", "asinh", "acosh", "atanh", "bitwise_not", "bitwise_and", "bitwise_or",
       "bitwise_xor", "hardmax", "lpnormalization", "l1normalization", "l2normalization", "mvn", "lrn", "eyelike", "det",
       "compress", "reversesequence", "upsample", "rnn", "simple_rnn", "gru", "nonmaxsuppression", "roialign",
       # after the second group
       "gridsample", "quantizelinear", "dequantizelinear", "qlinearmatmul",
       "col2im", "center_crop_pad", "maxunpool", "affine_grid", "deform_conv")

# re-imported official cases of a form that is refused, and the sentence that
# must refuse them
R20_REFUSED = {"r20_test_gru_batchwise": "attribute layout = 1 (batch first) is not implemented",
               "r20_test_simple_rnn_batchwise": "attribute layout = 1 (batch first) is not implemented"}


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), opset=20, dynamic=True, fixed=True,
                 refuse=None, symbolic_axes=(0,), oracle=None, infer=True):
        self.name, self.nodes, self.inputs, self.outputs, self.feeds = name, nodes, inputs, outputs, feeds
        self.initializers, self.opset, self.dynamic, self.fixed, self.refuse = list(initializers), opset, dynamic, fixed, refuse
        # infer=False: the intermediates carry no value_info, as in a
        # function-expanded graph
        self.symbolic_axes, self.oracle, self.infer = symbolic_axes, oracle, infer


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
    m.ir_version = 8
    if len(case.nodes) > 1 and not symbolic and case.infer:
        # the intermediate values' shapes, as an exporter writes them
        m = onnx.shape_inference.infer_shapes(m)
    return m


def expected(case: Case, m: onnx.ModelProto):
    if case.oracle == "ort":
        # forms the reference evaluator does not implement (RNN and GRU with
        # sequence lengths or two directions, NonMaxSuppression's options):
        # onnxruntime alone decides
        options = ort.SessionOptions()
        options.log_severity_level = 4
        return [np.asarray(v) for v in ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, case.feeds)]
    if isinstance(case.oracle, tuple) and case.oracle[0] == "own":
        # a random operator: its values are this compiler's specified
        # generator, which neither ONNX Runtime nor the reference reproduces
        return [np.asarray(v) for v in case.oracle[1](case.feeds)]
    if case.oracle == "ref":
        # a form ONNX Runtime computes otherwise than the reference and the
        # official test data (MaxUnpool with output_shape): the reference alone
        return [np.asarray(x) for x in ReferenceEvaluator(m).run(None, case.feeds)]
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
            # two independent computations agree to a relative 1e-3, and near
            # zero to 1e-5 of the output's largest magnitude (an FFT's zeros)
            scale = float(np.nanmax(np.abs(a.astype(np.float64)))) if a.size and np.issubdtype(a.dtype, np.floating) else 0.0
            if a.shape != b.shape or not np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=1e-3,
                                                      atol=max(1e-6, 1e-5 * (scale if np.isfinite(scale) else 0.0)), equal_nan=True):
                raise SystemExit("%s output %d: onnxruntime and the reference evaluator disagree\n  ref %r\n  ort %r"
                                 % (case.name, i, a.ravel()[:12], b.ravel()[:12]))
    return got


def write(folder: Path, case: Case, symbolic: bool):
    m = model_for(case, symbolic)
    if not any(n.op_type == "GroupNormalization" for n in m.graph.node):
        onnx.checker.check_model(m)
    outs = [] if case.refuse else expected(case, m)
    ns.write_case(folder, m, [case.feeds[n] for n, _, _ in case.inputs], outs)


def f32(*shape, scale=1.0):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


# ---------------------------------------------------------------------------
# The random operators' generator, as runtime/tensor_random.pmi specifies it
# (docs/VALIDATION.md, "Random operators"): Threefry-2x32-20, the node key
# from the seed attribute or the model seed, element i of request 0, and
# the uniform u = (w0 >> 8) * 2^-24. Bernoulli and Multinomial draw from it.
# ---------------------------------------------------------------------------
def _threefry(k0, k1, c0, c1):
    m = 0xFFFFFFFF
    rot = (13, 15, 26, 6, 17, 29, 16, 24)
    ks = (k0 & m, k1 & m, (0x1BD11BDA ^ k0 ^ k1) & m)
    x0, x1 = (c0 + ks[0]) & m, (c1 + ks[1]) & m
    for block in range(5):
        for step in range(4):
            x0 = (x0 + x1) & m
            r = rot[(block % 2) * 4 + step]
            x1 = (((x1 << r) | (x1 >> (32 - r))) & m) ^ x0
        s = block + 1
        x0 = (x0 + ks[s % 3]) & m
        x1 = (x1 + ks[(s + 1) % 3] + s) & m
    return x0, x1


assert _threefry(0, 0, 0, 0) == (0x6B200159, 0x99BA4EFE)
assert _threefry(0x13198A2E, 0x03707344, 0x243F6A88, 0x85A308D3) == (0xC4923A9C, 0x483DF7A0)


def uniform_draws(count, node, seed=None):
    if seed is None:
        n0, n1 = _threefry(0, 0, node, 0)
    else:
        n0, n1 = _threefry(int(np.array([seed], np.float32).view(np.uint32)[0]), 0, node, 1)
    return np.array([((_threefry(n0, n1, i, 0)[0] >> 8) & 0xFFFFFF) for i in range(count)], np.float32) * np.float32(2.0 ** -24)


def bernoulli_oracle(node, seed, dtype):
    def run(feeds):
        p = next(iter(feeds.values())) if node == 0 else np.abs(next(iter(feeds.values())))
        u = uniform_draws(p.size, node, seed).reshape(p.shape)
        return [(u < p).astype(dtype)]
    return ("own", run)


def multinomial_oracle(node, seed, samples, dtype):
    def run(feeds):
        x = next(iter(feeds.values())).astype(np.float64)
        u = uniform_draws(x.shape[0] * samples, node, seed).reshape(x.shape[0], samples).astype(np.float64)
        w = np.exp(x - x.max(axis=1, keepdims=True))
        c = np.cumsum(w, axis=1)
        t = u * c[:, -1:]
        # the kernel works in binary32: keep every draw well away from a class boundary
        gap = np.min(np.abs(t[:, :, None] - c[:, None, :]) / c[:, -1:, None])
        assert gap > 1e-4, "a draw lies within %g of a class boundary; choose other data" % gap
        return [np.array([[int(np.searchsorted(c[b], t[b, s], side="right")) for s in range(samples)] for b in range(x.shape[0])], dtype)]
    return ("own", run)


def cases() -> list[Case]:
    c: list[Case] = []
    N = helper.make_node
    x = f32(3, 4, 5, scale=2.0)

    def unary(name, op, attrs=None, opset=20, feed=None, out=F, elem=F):
        v = x if feed is None else feed
        c.append(Case(name, [N(op, ["x"], ["y"], **(attrs or {}))], [("x", elem, list(v.shape))],
                      [("y", out, list(v.shape))], {"x": v}, opset=opset))

    unary("erf", "Erf")
    unary("reciprocal", "Reciprocal", feed=np.where(np.abs(x) < 0.1, 0.5, x).astype(np.float32))
    unary("ceil", "Ceil", feed=(x * 3).astype(np.float32))
    unary("sign_float", "Sign", feed=np.where(np.abs(x) < 0.3, 0, x).astype(np.float32))
    unary("sign_int64", "Sign", feed=RNG.integers(-4, 5, (3, 4)).astype(np.int64), out=I64, elem=I64)
    unary("softplus", "Softplus", feed=(x * 5).astype(np.float32), opset=13)
    unary("softsign", "Softsign")
    unary("elu", "Elu", {"alpha": 0.7})
    unary("selu_default", "Selu")
    unary("selu_attrs", "Selu", {"alpha": 2.0, "gamma": 0.5})
    unary("celu", "Celu", {"alpha": 0.5}, opset=13)
    unary("hardsigmoid", "HardSigmoid", {"alpha": 0.3, "beta": 0.4})
    unary("hardswish", "HardSwish", feed=(x * 2).astype(np.float32))
    unary("mish", "Mish", opset=18)
    unary("gelu_none", "Gelu")
    unary("gelu_tanh", "Gelu", {"approximate": "tanh"})
    unary("thresholdedrelu", "ThresholdedRelu", {"alpha": 0.5})
    unary("shrink", "Shrink", {"bias": 0.2, "lambd": 0.6})
    special = x.copy()
    special.flat[[1, 7, 13]] = [np.nan, np.inf, -np.inf]
    unary("isnan", "IsNaN", feed=special, out=B)
    unary("isinf", "IsInf", feed=special, out=B)
    unary("isinf_positive_only", "IsInf", {"detect_negative": 0}, feed=special, out=B)
    c.append(Case("refuse_gelu_mode", [N("Gelu", ["x"], ["y"], approximate="fast")], [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])],
                  {"x": x}, refuse="approximate = fast"))

    # variadic and binary
    a, b3, d = f32(3, 4, 5), f32(4, 1), f32(5)
    c.append(Case("min_three_broadcast", [N("Min", ["a", "b", "d"], ["y"])], [("a", F, [3, 4, 5]), ("b", F, [4, 1]), ("d", F, [5])],
                  [("y", F, [3, 4, 5])], {"a": a, "b": b3, "d": d}))
    c.append(Case("max_two", [N("Max", ["a", "b"], ["y"])], [("a", F, [3, 4, 5]), ("b", F, [4, 1])], [("y", F, [3, 4, 5])],
                  {"a": a, "b": b3}))
    ia, ib = RNG.integers(-50, 50, (3, 4)).astype(np.int64), RNG.integers(-50, 50, (4,)).astype(np.int64)
    c.append(Case("max_int64", [N("Max", ["a", "b"], ["y"])], [("a", I64, [3, 4]), ("b", I64, [4])], [("y", I64, [3, 4])],
                  {"a": ia, "b": ib}))
    c.append(Case("sum_one", [N("Sum", ["a"], ["y"])], [("a", F, [3, 4, 5])], [("y", F, [3, 4, 5])], {"a": a}))
    c.append(Case("sum_three", [N("Sum", ["a", "b", "d"], ["y"])], [("a", F, [3, 4, 5]), ("b", F, [4, 1]), ("d", F, [5])],
                  [("y", F, [3, 4, 5])], {"a": a, "b": b3, "d": d}))
    c.append(Case("mean_two", [N("Mean", ["a", "d"], ["y"])], [("a", F, [3, 4, 5]), ("d", F, [5])], [("y", F, [3, 4, 5])],
                  {"a": a, "d": d}))
    ib2 = np.where(ib == 0, 7, ib).astype(np.int64)
    c.append(Case("mod_int64", [N("Mod", ["a", "b"], ["y"])], [("a", I64, [3, 4]), ("b", I64, [4])], [("y", I64, [3, 4])],
                  {"a": ia, "b": ib2}))
    c.append(Case("mod_int64_fmod", [N("Mod", ["a", "b"], ["y"], fmod=1)], [("a", I64, [3, 4]), ("b", I64, [4])], [("y", I64, [3, 4])],
                  {"a": ia, "b": ib2}))
    fa, fb = f32(3, 4, scale=10), np.where(np.abs(f32(4, scale=3)) < 0.2, 1.5, f32(4, scale=3)).astype(np.float32)
    c.append(Case("mod_float_fmod", [N("Mod", ["a", "b"], ["y"], fmod=1)], [("a", F, [3, 4]), ("b", F, [4])], [("y", F, [3, 4])],
                  {"a": fa, "b": fb}))
    c.append(Case("refuse_mod_float_no_fmod", [N("Mod", ["a", "b"], ["y"])], [("a", F, [3, 4]), ("b", F, [4])], [("y", F, [3, 4])],
                  {"a": fa, "b": fb}, refuse="fmod = 1"))
    slope = f32(4, 1, 1)
    xp = f32(2, 4, 3, 3)
    c.append(Case("prelu_channel", [N("PRelu", ["x", "s"], ["y"])], [("x", F, [2, 4, 3, 3])], [("y", F, [2, 4, 3, 3])], {"x": xp},
                  [init("s", slope)]))
    ba, bb = RNG.integers(0, 2, (3, 4)).astype(bool), RNG.integers(0, 2, (4,)).astype(bool)
    c.append(Case("or", [N("Or", ["a", "b"], ["y"])], [("a", B, [3, 4]), ("b", B, [4])], [("y", B, [3, 4])], {"a": ba, "b": bb}))
    c.append(Case("xor", [N("Xor", ["a", "b"], ["y"])], [("a", B, [3, 4]), ("b", B, [4])], [("y", B, [3, 4])], {"a": ba, "b": bb}))

    # reductions: the axes attribute (opset 13) and the axes input (opset 18)
    r = f32(3, 4, 5)
    pos = (np.abs(r) + 0.1).astype(np.float32)
    for op in ("ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"):
        v = pos if op == "ReduceLogSum" else r
        snake = "reduce_" + op[6:].lower()
        c.append(Case(snake + "_attr_keep", [N(op, ["x"], ["y"], axes=[1], keepdims=1)], [("x", F, [3, 4, 5])], [("y", F, [3, 1, 5])],
                      {"x": v}, opset=13))
        c.append(Case(snake + "_attr_drop_two", [N(op, ["x"], ["y"], axes=[0, 2], keepdims=0)], [("x", F, [3, 4, 5])], [("y", F, [4])],
                      {"x": v}, opset=13, dynamic=False))
        c.append(Case(snake + "_input_axes", [N(op, ["x", "ax"], ["y"], keepdims=0)], [("x", F, [3, 4, 5])], [("y", F, [3, 4])],
                      {"x": v}, [init("ax", np.array([-1], np.int64))], opset=18))
        c.append(Case(snake + "_all", [N(op, ["x"], ["y"], keepdims=1)], [("x", F, [3, 4, 5])], [("y", F, [1, 1, 1])],
                      {"x": v}, opset=18))
    c.append(Case("reduce_min_noop", [N("ReduceMin", ["x"], ["y"], noop_with_empty_axes=1)], [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])],
                  {"x": r}, opset=18))
    # ReduceSum and ReduceMean on the runtime-dimension path beyond one final
    # axis (the general form its dynamic runtime gained with this lane)
    c.append(Case("reduce_sum_axes_0_2", [N("ReduceSum", ["x", "ax"], ["y"], keepdims=0)], [("x", F, [3, 4, 5])], [("y", F, [4])],
                  {"x": r}, [init("ax", np.array([0, 2], np.int64))], opset=13, fixed=False))
    c.append(Case("reduce_mean_all", [N("ReduceMean", ["x"], ["y"], keepdims=1)], [("x", F, [3, 4, 5])], [("y", F, [1, 1, 1])],
                  {"x": r}, opset=18, fixed=False))
    c.append(Case("reduce_sum_noop_empty", [N("ReduceSum", ["x"], ["y"], noop_with_empty_axes=1)], [("x", F, [3, 4, 5])],
                  [("y", F, [3, 4, 5])], {"x": r}, opset=13, fixed=False))
    ri = RNG.integers(-9, 9, (3, 4)).astype(np.int64)
    for op in ("ReduceMin", "ReduceL1", "ReduceSumSquare"):
        c.append(Case("reduce_%s_int64" % op[6:].lower(), [N(op, ["x"], ["y"], axes=[1], keepdims=0)], [("x", I64, [3, 4])],
                      [("y", I64, [3])], {"x": ri}, opset=13))

    # ArgMax / ArgMin / LogSoftmax
    t = RNG.integers(-3, 4, (3, 5, 4)).astype(np.float32)
    for op in ("ArgMax", "ArgMin"):
        c.append(Case(op.lower() + "_axis1", [N(op, ["x"], ["y"], axis=1)], [("x", F, [3, 5, 4])], [("y", I64, [3, 1, 4])], {"x": t}))
        c.append(Case(op.lower() + "_last_nokeep", [N(op, ["x"], ["y"], axis=-1, keepdims=0, select_last_index=1)],
                      [("x", F, [3, 5, 4])], [("y", I64, [3, 5])], {"x": t}))
    c.append(Case("argmax_int64", [N("ArgMax", ["x"], ["y"], axis=0)], [("x", I64, [3, 4])], [("y", I64, [1, 4])], {"x": ri}))
    ls = f32(2, 5, 3, scale=3)
    c.append(Case("logsoftmax_axis1", [N("LogSoftmax", ["x"], ["y"], axis=1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])], {"x": ls}))
    c.append(Case("logsoftmax_default", [N("LogSoftmax", ["x"], ["y"])], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])], {"x": ls}))
    c.append(Case("logsoftmax_opset11_last", [N("LogSoftmax", ["x"], ["y"], axis=-1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])],
                  {"x": ls}, opset=11))
    c.append(Case("refuse_logsoftmax_opset11_axis1", [N("LogSoftmax", ["x"], ["y"], axis=1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])],
                  {"x": ls}, opset=11, refuse="LogSoftmax"))

    # pooling
    img = f32(2, 3, 9, 8)
    def pool(name, op, attrs, out_hw, opset=20, outputs=1, feed=img):
        outs = [("y", F, [feed.shape[0], feed.shape[1]] + out_hw)]
        names = ["y"]
        if outputs == 2:
            outs.append(("i", I64, [feed.shape[0], feed.shape[1]] + out_hw))
            names.append("i")
        c.append(Case(name, [N(op, ["x"], names, **attrs)], [("x", F, list(feed.shape))], outs, {"x": feed}, opset=opset))
    pool("maxpool_3x3_s2_pads", "MaxPool", {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]}, [5, 4])
    pool("maxpool_ceil", "MaxPool", {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1}, [4, 4])
    pool("maxpool_dilation", "MaxPool", {"kernel_shape": [2, 2], "dilations": [2, 2]}, [7, 6])
    pool("maxpool_same_upper", "MaxPool", {"kernel_shape": [3, 2], "strides": [2, 2], "auto_pad": "SAME_UPPER"}, [5, 4])
    pool("maxpool_indices", "MaxPool", {"kernel_shape": [2, 2], "strides": [2, 2]}, [4, 4], outputs=2)
    pool("maxpool_indices_column_major", "MaxPool", {"kernel_shape": [3, 3], "strides": [2, 2], "storage_order": 1}, [4, 3], outputs=2)
    pool("averagepool_exclude_pad", "AveragePool", {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]}, [5, 4])
    pool("averagepool_include_pad", "AveragePool", {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "count_include_pad": 1}, [5, 4])
    pool("averagepool_same_lower", "AveragePool", {"kernel_shape": [3, 3], "auto_pad": "SAME_LOWER"}, [9, 8])
    pool("averagepool_ceil", "AveragePool", {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1}, [5, 4])
    pool("averagepool_dilation", "AveragePool", {"kernel_shape": [2, 2], "dilations": [2, 1]}, [7, 7], opset=19)
    pool("lppool_p2", "LpPool", {"kernel_shape": [2, 3], "strides": [1, 2]}, [8, 3])
    pool("lppool_p3", "LpPool", {"kernel_shape": [2, 2], "p": 3}, [8, 7])
    pool("globalmaxpool", "GlobalMaxPool", {}, [1, 1])
    pool("globalaveragepool", "GlobalAveragePool", {}, [1, 1])
    # the reference evaluator has no GlobalLpPool; the specification's formula decides
    c.append(Case("globallppool", [N("GlobalLpPool", ["x"], ["y"], p=2)], [("x", F, list(img.shape))], [("y", F, [2, 3, 1, 1])],
                  {"x": img}, oracle=lambda feeds: [np.sqrt(np.sum(feeds["x"].astype(np.float64) ** 2, axis=(2, 3), keepdims=True)).astype(np.float32)]))
    seq = f32(2, 3, 11)
    pool("maxpool_1d", "MaxPool", {"kernel_shape": [3], "strides": [2]}, [5], feed=seq)
    vol = f32(1, 2, 4, 5, 6)
    pool("averagepool_3d", "AveragePool", {"kernel_shape": [2, 2, 2], "strides": [1, 2, 2]}, [3, 2, 3], feed=vol)

    # data movement
    s = f32(2, 6, 3)
    c.append(Case("split_input_sizes", [N("Split", ["x", "sz"], ["a", "b", "c"], axis=1)], [("x", F, [2, 6, 3])],
                  [("a", F, [2, 1, 3]), ("b", F, [2, 2, 3]), ("c", F, [2, 3, 3])], {"x": s}, [init("sz", np.array([1, 2, 3], np.int64))],
                  opset=13))
    c.append(Case("split_attr_sizes", [N("Split", ["x"], ["a", "b"], axis=-1, split=[1, 2])], [("x", F, [2, 6, 3])],
                  [("a", F, [2, 6, 1]), ("b", F, [2, 6, 2])], {"x": s}, opset=11))
    c.append(Case("split_equal", [N("Split", ["x"], ["a", "b", "c"], axis=1)], [("x", F, [2, 6, 3])],
                  [("a", F, [2, 2, 3]), ("b", F, [2, 2, 3]), ("c", F, [2, 2, 3])], {"x": s}, opset=13))
    c.append(Case("split_num_outputs_uneven", [N("Split", ["x"], ["a", "b", "c", "d"], axis=1, num_outputs=4)], [("x", F, [2, 6, 3])],
                  [("a", F, [2, 2, 3]), ("b", F, [2, 2, 3]), ("c", F, [2, 2, 3]), ("d", F, [2, 0, 3])], {"x": s}, opset=18,
                  oracle=lambda feeds: np.split(feeds["x"], [2, 4, 6], axis=1)))
    c.append(Case("tile", [N("Tile", ["x", "r"], ["y"])], [("x", F, [2, 3, 4])], [("y", F, [4, 3, 12])], {"x": f32(2, 3, 4)},
                  [init("r", np.array([2, 1, 3], np.int64))]))
    c.append(Case("tile_int64", [N("Tile", ["x", "r"], ["y"])], [("x", I64, [3, 4])], [("y", I64, [6, 4])], {"x": ri},
                  [init("r", np.array([2, 1], np.int64))]))
    dts = f32(1, 8, 2, 3)
    c.append(Case("depthtospace_dcr", [N("DepthToSpace", ["x"], ["y"], blocksize=2)], [("x", F, [1, 8, 2, 3])], [("y", F, [1, 2, 4, 6])], {"x": dts}))
    c.append(Case("depthtospace_crd", [N("DepthToSpace", ["x"], ["y"], blocksize=2, mode="CRD")], [("x", F, [1, 8, 2, 3])], [("y", F, [1, 2, 4, 6])], {"x": dts}))
    std = f32(2, 3, 4, 6)
    c.append(Case("spacetodepth", [N("SpaceToDepth", ["x"], ["y"], blocksize=2)], [("x", F, [2, 3, 4, 6])], [("y", F, [2, 12, 2, 3])], {"x": std}))
    tr = f32(2, 4, 5)
    c.append(Case("triu", [N("Trilu", ["x"], ["y"])], [("x", F, [2, 4, 5])], [("y", F, [2, 4, 5])], {"x": tr}, opset=14))
    c.append(Case("tril_k", [N("Trilu", ["x", "k"], ["y"], upper=0)], [("x", F, [2, 4, 5]), ("k", I64, [])], [("y", F, [2, 4, 5])],
                  {"x": tr, "k": np.array(-1, np.int64)}, opset=14))
    c.append(Case("triu_k_initializer", [N("Trilu", ["x", "k"], ["y"])], [("x", F, [2, 4, 5])], [("y", F, [2, 4, 5])], {"x": tr},
                  [init("k", np.array(2, np.int64))], opset=14))
    ge_data, ge_idx = f32(3, 4), RNG.integers(-4, 4, (3, 2)).astype(np.int64)
    c.append(Case("gather_elements", [N("GatherElements", ["d", "i"], ["y"], axis=1)], [("d", F, [3, 4]), ("i", I64, [3, 2])],
                  [("y", F, [3, 2])], {"d": ge_data, "i": ge_idx}))
    gd = f32(2, 3, 4)
    c.append(Case("gathernd", [N("GatherND", ["d", "i"], ["y"])], [("d", F, [2, 3, 4]), ("i", I64, [2, 2])], [("y", F, [2, 4])],
                  {"d": gd, "i": np.array([[1, -1], [0, 2]], np.int64)}, symbolic_axes=(1,)))
    c.append(Case("gathernd_batch_dims", [N("GatherND", ["d", "i"], ["y"], batch_dims=1)], [("d", F, [2, 3, 4]), ("i", I64, [2, 1, 2])],
                  [("y", F, [2, 1])], {"d": gd, "i": np.array([[[2, 3]], [[0, 1]]], np.int64)}))
    oh_idx = np.array([[0, 3, -1], [2, 1, -4]], np.int64)
    c.append(Case("onehot_last", [N("OneHot", ["i", "depth", "v"], ["y"])], [("i", I64, [2, 3])], [("y", F, [2, 3, 4])], {"i": oh_idx},
                  [init("depth", np.array(4, np.int64)), init("v", np.array([-1.0, 2.5], np.float32))], opset=11))
    c.append(Case("onehot_axis1_int64", [N("OneHot", ["i", "depth", "v"], ["y"], axis=1)], [("i", I64, [2, 3])], [("y", I64, [2, 5, 3])],
                  {"i": oh_idx}, [init("depth", np.array([5], np.int64)), init("v", np.array([0, 7], np.int64))], opset=11,
                  # depth as a one-element rank-1 tensor, which the specification allows and the
                  # reference evaluator does not take; onnxruntime cross-checks this oracle
                  oracle=lambda feeds: [np.moveaxis(np.where(np.arange(5)[:, None, None] == np.where(feeds["i"] < 0, feeds["i"] + 5, feeds["i"])[None], 7, 0).astype(np.int64), 0, 1)]))
    ea, eb = f32(3, 4), f32(4, 5)
    c.append(Case("einsum_matmul", [N("Einsum", ["a", "b"], ["y"], equation="ij,jk->ik")], [("a", F, [3, 4]), ("b", F, [4, 5])],
                  [("y", F, [3, 5])], {"a": ea, "b": eb}))
    c.append(Case("einsum_implicit", [N("Einsum", ["a", "b"], ["y"], equation="ij,jk")], [("a", F, [3, 4]), ("b", F, [4, 5])],
                  [("y", F, [3, 5])], {"a": ea, "b": eb}))
    bq, bk = f32(2, 2, 3, 4), f32(2, 2, 5, 4)
    c.append(Case("einsum_attention", [N("Einsum", ["q", "k"], ["y"], equation="bhqd,bhkd->bhqk")], [("q", F, [2, 2, 3, 4]), ("k", F, [2, 2, 5, 4])],
                  [("y", F, [2, 2, 3, 5])], {"q": bq, "k": bk}))
    sq = f32(4, 4)
    c.append(Case("einsum_diagonal", [N("Einsum", ["a"], ["y"], equation="ii->i")], [("a", F, [4, 4])], [("y", F, [4])], {"a": sq}))
    c.append(Case("einsum_transpose", [N("Einsum", ["a"], ["y"], equation="ij->ji")], [("a", F, [3, 4])], [("y", F, [4, 3])], {"a": ea}))
    c.append(Case("einsum_ellipsis", [N("Einsum", ["a", "b"], ["y"], equation="...ij,...jk->...ik")], [("a", F, [2, 3, 4]), ("b", F, [2, 4, 5])],
                  [("y", F, [2, 3, 5])], {"a": f32(2, 3, 4), "b": f32(2, 4, 5)}, dynamic=False))
    c.append(Case("refuse_einsum_ellipsis_dynamic", [N("Einsum", ["a", "b"], ["y"], equation="...ij,...jk->...ik")],
                  [("a", F, [2, 3, 4]), ("b", F, [2, 4, 5])], [("y", F, [2, 3, 5])], {"a": f32(2, 3, 4), "b": f32(2, 4, 5)},
                  fixed=False, refuse="ellipsis"))
    dr = f32(3, 4)
    c.append(Case("dropout_inference", [N("Dropout", ["x"], ["y", "m"])], [("x", F, [3, 4])], [("y", F, [3, 4]), ("m", B, [3, 4])], {"x": dr}, opset=13))
    c.append(Case("dropout_training_false", [N("Dropout", ["x", "r", "t"], ["y"])], [("x", F, [3, 4])], [("y", F, [3, 4])], {"x": dr},
                  [init("r", np.array(0.3, np.float32)), init("t", np.array(False))], opset=13))
    c.append(Case("refuse_dropout_training", [N("Dropout", ["x", "r", "t"], ["y"])], [("x", F, [3, 4])], [("y", F, [3, 4])], {"x": dr},
                  [init("r", np.array(0.3, np.float32)), init("t", np.array(True))], opset=13, refuse="training"))
    c.append(Case("castlike", [N("CastLike", ["x", "t"], ["y"])], [("x", F, [3, 4]), ("t", I64, [1])], [("y", I64, [3, 4])],
                  {"x": (dr * 10).astype(np.float32), "t": np.array([0], np.int64)}, opset=15))
    c.append(Case("size", [N("Size", ["x"], ["y"])], [("x", F, [3, 4])], [("y", I64, [])], {"x": dr}))

    # ---- second group ------------------------------------------------------
    u = RNG.uniform(-3, 3, (3, 4, 5)).astype(np.float32)
    unary("tan", "Tan", feed=u)
    unary("asin", "Asin", feed=RNG.uniform(-1, 1, (3, 4, 5)).astype(np.float32))
    unary("acos", "Acos", feed=RNG.uniform(-1, 1, (3, 4, 5)).astype(np.float32))
    unary("sinh", "Sinh", feed=(u * 2).astype(np.float32))
    unary("cosh", "Cosh", feed=(u * 2).astype(np.float32))
    unary("asinh", "Asinh", feed=(u * 3).astype(np.float32))
    unary("acosh", "Acosh", feed=RNG.uniform(1, 6, (3, 4, 5)).astype(np.float32))
    unary("atanh", "Atanh", feed=RNG.uniform(-0.95, 0.95, (3, 4, 5)).astype(np.float32))
    bi = RNG.integers(-2 ** 40, 2 ** 40, (3, 4)).astype(np.int64)
    bj = RNG.integers(-2 ** 40, 2 ** 40, (4,)).astype(np.int64)
    unary("bitwise_not", "BitwiseNot", feed=bi, out=I64, elem=I64, opset=18)
    for op in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
        c.append(Case(op.lower(), [N(op, ["a", "b"], ["y"])], [("a", I64, [3, 4]), ("b", I64, [4])], [("y", I64, [3, 4])],
                      {"a": bi, "b": bj}, opset=18))
    # the same with values the 32-bit targets' INT64 contract carries, so the
    # Pico and Pico 2 run them too (the forms above are refused there at the
    # request: their inputs leave the 32-bit range)
    si = (bi >> 10).astype(np.int64)
    sj = (bj >> 10).astype(np.int64)
    unary("bitwise_not_int32_range", "BitwiseNot", feed=si, out=I64, elem=I64, opset=18)
    for op in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
        c.append(Case(op.lower() + "_int32_range", [N(op, ["a", "b"], ["y"])], [("a", I64, [3, 4]), ("b", I64, [4])],
                      [("y", I64, [3, 4])], {"a": si, "b": sj}, opset=18))
    b32 = RNG.integers(-2 ** 30, 2 ** 30, (3, 4)).astype(np.int32)
    c.append(Case("bitwise_and_int32", [N("BitwiseAnd", ["a", "b"], ["y"])], [("a", I32, [3, 4]), ("b", I32, [3, 4])],
                  [("y", I32, [3, 4])], {"a": b32, "b": b32[::-1].copy()}, opset=18, fixed=False))
    hm = RNG.integers(-3, 4, (2, 5, 3)).astype(np.float32)
    c.append(Case("hardmax_default", [N("Hardmax", ["x"], ["y"])], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])], {"x": hm}, opset=13))
    c.append(Case("hardmax_axis1", [N("Hardmax", ["x"], ["y"], axis=1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])], {"x": hm}, opset=13))
    c.append(Case("hardmax_opset11_last", [N("Hardmax", ["x"], ["y"], axis=-1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])], {"x": hm}, opset=11))
    c.append(Case("refuse_hardmax_opset11_axis1", [N("Hardmax", ["x"], ["y"], axis=1)], [("x", F, [2, 5, 3])], [("y", F, [2, 5, 3])],
                  {"x": hm}, opset=11, refuse="Hardmax"))
    lp = f32(3, 4, 5)
    lp[1, 2] = 0.0
    def lp_oracle(axis, p):
        def run(feeds):
            x = feeds["x"].astype(np.float64)
            n = np.sum(np.abs(x), axis=axis, keepdims=True) if p == 1 else np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
            return [np.where(n == 0, 0, x / np.where(n == 0, 1, n)).astype(np.float32)]
        return run
    c.append(Case("lpnormalization_p1_axis0", [N("LpNormalization", ["x"], ["y"], axis=0, p=1)], [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])],
                  {"x": lp}, oracle=lp_oracle(0, 1)))
    c.append(Case("lpnormalization_p2_default", [N("LpNormalization", ["x"], ["y"])], [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])],
                  {"x": lp}, oracle=lp_oracle(-1, 2)))
    mv = (f32(2, 3, 4, 5) * 2 + 1).astype(np.float32)
    c.append(Case("mvn_default_axes", [N("MeanVarianceNormalization", ["x"], ["y"])], [("x", F, [2, 3, 4, 5])], [("y", F, [2, 3, 4, 5])],
                  {"x": mv}, opset=13))
    c.append(Case("mvn_axes_2_3", [N("MeanVarianceNormalization", ["x"], ["y"], axes=[2, 3])], [("x", F, [2, 3, 4, 5])],
                  [("y", F, [2, 3, 4, 5])], {"x": mv}, opset=13))
    lr = f32(2, 5, 3, 3)
    def lrn_oracle(size, alpha, beta, bias):
        def run(feeds):
            x = feeds["x"].astype(np.float64)
            sq = np.zeros_like(x)
            for ch in range(x.shape[1]):
                lo, hi = max(0, ch - (size - 1) // 2), min(x.shape[1], ch + int(np.ceil((size - 1) / 2)) + 1)
                sq[:, ch] = np.sum(x[:, lo:hi] ** 2, axis=1)
            return [(x / (bias + alpha / size * sq) ** beta).astype(np.float32)]
        return run
    c.append(Case("lrn_size3", [N("LRN", ["x"], ["y"], size=3, alpha=0.01, beta=0.75, bias=1.5)], [("x", F, [2, 5, 3, 3])],
                  [("y", F, [2, 5, 3, 3])], {"x": lr}, oracle=lrn_oracle(3, 0.01, 0.75, 1.5)))
    c.append(Case("lrn_size4_defaults", [N("LRN", ["x"], ["y"], size=4)], [("x", F, [2, 5, 3, 3])], [("y", F, [2, 5, 3, 3])], {"x": lr},
                  oracle=lrn_oracle(4, 0.0001, 0.75, 1.0)))
    gx = f32(2, 6, 4)
    gs, gb = f32(3), f32(3)
    def gn_oracle(feeds):
        x = feeds["x"].astype(np.float64).reshape(2, 3, -1)
        m = x.mean(axis=2, keepdims=True)
        v = ((x - m) ** 2).mean(axis=2, keepdims=True)
        y = (x - m) / np.sqrt(v + 1e-4) * gs.reshape(1, 3, 1) + gb.reshape(1, 3, 1)
        return [y.reshape(2, 6, 4).astype(np.float32)]
    # GroupNormalization-18 is marked deprecated in onnx 1.22 (GroupNormalization-21
    # replaced it), so the checker refuses the model; the specification's formula decides
    c.append(Case("groupnormalization", [N("GroupNormalization", ["x", "s", "b"], ["y"], num_groups=3, epsilon=1e-4)], [("x", F, [2, 6, 4])],
                  [("y", F, [2, 6, 4])], {"x": gx}, [init("s", gs), init("b", gb)], opset=18, oracle=gn_oracle))
    c.append(Case("eyelike_float_k1", [N("EyeLike", ["x"], ["y"], k=1, dtype=F)], [("x", I64, [3, 5])], [("y", F, [3, 5])],
                  {"x": np.zeros((3, 5), np.int64)}))
    c.append(Case("eyelike_int64_kneg", [N("EyeLike", ["x"], ["y"], k=-1)], [("x", I64, [4, 4])], [("y", I64, [4, 4])],
                  {"x": np.zeros((4, 4), np.int64)}))
    c.append(Case("det_batch", [N("Det", ["x"], ["y"])], [("x", F, [3, 3, 3])], [("y", F, [3])], {"x": f32(3, 3, 3)}))
    cx = f32(3, 5)
    c.append(Case("compress_axis1_constant", [N("Compress", ["x", "c"], ["y"], axis=1)], [("x", F, [3, 5])], [("y", F, [3, 3])],
                  {"x": cx}, [init("c", np.array([True, False, True, True], bool))]))
    c.append(Case("compress_flat_constant", [N("Compress", ["x", "c"], ["y"])], [("x", F, [3, 5])], [("y", F, [2])],
                  {"x": cx}, [init("c", np.array([False, True, False, False, True], bool))]))
    c.append(Case("compress_runtime_condition", [N("Compress", ["x", "c"], ["y"], axis=0)], [("x", F, [3, 5]), ("c", B, [3])],
                  [("y", F, [2, 5])], {"x": cx, "c": np.array([True, False, True])}, fixed=False))
    rs = f32(4, 3, 2)
    c.append(Case("reversesequence_time_first", [N("ReverseSequence", ["x", "l"], ["y"], time_axis=0, batch_axis=1)],
                  [("x", F, [4, 3, 2]), ("l", I64, [3])], [("y", F, [4, 3, 2])], {"x": rs, "l": np.array([4, 1, 3], np.int64)},
                  symbolic_axes=(2,)))
    c.append(Case("reversesequence_batch_first", [N("ReverseSequence", ["x", "l"], ["y"], time_axis=1, batch_axis=0)],
                  [("x", F, [3, 4, 2]), ("l", I64, [3])], [("y", F, [3, 4, 2])], {"x": f32(3, 4, 2), "l": np.array([2, 4, 1], np.int64)},
                  symbolic_axes=(2,)))
    up = f32(1, 2, 3, 4)
    c.append(Case("upsample_nearest_opset9", [N("Upsample", ["x", "s"], ["y"], mode="nearest")], [("x", F, [1, 2, 3, 4])],
                  [("y", F, [1, 2, 6, 12])], {"x": up}, [init("s", np.array([1, 1, 2, 3], np.float32))], opset=9, dynamic=False))
    def bilinear_oracle(sh, sw):
        def run(feeds):
            x = feeds["x"]
            n, ch, h, w = x.shape
            oh, ow = int(h * sh), int(w * sw)
            y = np.empty((n, ch, oh, ow), np.float32)
            for i in range(oh):
                iy = min(i / sh, h - 1)
                y1 = int(iy); y2 = min(y1 + 1, h - 1); dy1 = iy - y1; dy2 = 1 - dy1
                for j in range(ow):
                    ix = min(j / sw, w - 1)
                    x1 = int(ix); x2 = min(x1 + 1, w - 1); dx1 = ix - x1; dx2 = 1 - dx1
                    y[:, :, i, j] = dx2 * dy2 * x[:, :, y1, x1] + dx1 * dy2 * x[:, :, y1, x2] + dx2 * dy1 * x[:, :, y2, x1] + dx1 * dy1 * x[:, :, y2, x2]
            return [y]
        return run
    c.append(Case("upsample_linear_opset9", [N("Upsample", ["x", "s"], ["y"], mode="linear")], [("x", F, [1, 2, 3, 4])],
                  [("y", F, [1, 2, 6, 8])], {"x": up}, [init("s", np.array([1, 1, 2, 2], np.float32))], opset=9, dynamic=False,
                  oracle=bilinear_oracle(2, 2)))
    c.append(Case("upsample_attr_opset7", [N("Upsample", ["x"], ["y"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0])], [("x", F, [1, 2, 3, 4])],
                  [("y", F, [1, 2, 6, 8])], {"x": up}, opset=7, dynamic=False,
                  oracle=lambda feeds: [np.repeat(np.repeat(feeds["x"], 2, axis=2), 2, axis=3)]))
    c.append(Case("refuse_upsample_runtime_dimensions", [N("Upsample", ["x", "s"], ["y"], mode="nearest")], [("x", F, [1, 2, 3, 4])],
                  [("y", F, [1, 2, 6, 12])], {"x": up}, [init("s", np.array([1, 1, 2, 3], np.float32))], opset=9, fixed=False,
                  refuse="Upsample is implemented by fixed-shape emission only"))
    T, Bt, I, H = 5, 3, 4, 3
    xs = f32(T, Bt, I)
    def rnn_case(name, op, G, D, attrs, bias=False, lens=None, h0=False, outs=("Y", "Y_h"), oracle="ort"):
        inits = [init("W", (f32(D, G * H, I) * 0.5).astype(np.float32)), init("R", (f32(D, G * H, H) * 0.5).astype(np.float32))]
        ins = ["x", "W", "R"]
        if bias or lens is not None or h0:
            ins.append("B" if bias else "")
            if bias:
                inits.append(init("B", f32(D, 2 * G * H)))
        if lens is not None or h0:
            ins.append("L" if lens is not None else "")
            if lens is not None:
                inits.append(init("L", np.array(lens, np.int32)))
        if h0:
            ins.append("H0")
            inits.append(init("H0", f32(D, Bt, H)))
        shapes = {"Y": [T, D, Bt, H], "Y_h": [D, Bt, H]}
        c.append(Case(name, [N(op, ins, list(outs), hidden_size=H, **attrs)], [("x", F, [T, Bt, I])],
                      [(o, F, shapes[o]) for o in outs if o], {"x": xs}, inits, opset=14, oracle=oracle, symbolic_axes=(0,)))
    rnn_case("rnn_forward", "RNN", 1, 1, {}, oracle=None)
    rnn_case("rnn_bidirectional_bias_lens_h0", "RNN", 1, 2, {"direction": "bidirectional"}, bias=True, lens=[5, 2, 4], h0=True)
    rnn_case("rnn_relu_reverse", "RNN", 1, 1, {"direction": "reverse", "activations": ["Relu"]}, bias=True)
    rnn_case("gru_forward", "GRU", 3, 1, {}, oracle=None)
    rnn_case("gru_linear_before_reset", "GRU", 3, 1, {"linear_before_reset": 1}, bias=True, oracle=None)
    rnn_case("gru_bidirectional_lens_h0", "GRU", 3, 2, {"direction": "bidirectional"}, bias=True, lens=[5, 3, 1], h0=True)
    rnn_case("gru_y_h_only_clip", "GRU", 3, 1, {"clip": 0.5, "direction": "reverse"}, bias=True, outs=("", "Y_h"))
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]]], np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.1, 0.8, 0.7, 0.2, 0.9, 0.95]]], np.float32)
    for center, name in ((0, "corners"), (1, "center")):
        bx = boxes if center == 0 else np.array([[[0.5, 0.5, 1.0, 1.0], [0.5, 0.6, 1.0, 1.0], [0.5, 0.4, 1.0, 1.0], [0.5, 10.5, 1.0, 1.0],
                                                   [0.5, 10.6, 1.0, 1.0], [0.5, 100.5, 1.0, 1.0]]], np.float32)
        c.append(Case("nms_" + name, [N("NonMaxSuppression", ["b", "s", "m", "i", "t"], ["y"], center_point_box=center)],
                      [("b", F, [1, 6, 4]), ("s", F, [1, 2, 6])], [("y", I64, [None, 3])], {"b": bx, "s": scores},
                      [init("m", np.array([3], np.int64)), init("i", np.array([0.5], np.float32)), init("t", np.array([0.2], np.float32))],
                      opset=11, symbolic_axes=()))
    c.append(Case("nms_no_threshold", [N("NonMaxSuppression", ["b", "s", "m"], ["y"])], [("b", F, [1, 6, 4]), ("s", F, [1, 2, 6])],
                  [("y", I64, [None, 3])], {"b": boxes, "s": scores}, [init("m", np.array([2], np.int64))], opset=11, symbolic_axes=(), oracle="ort"))
    rx = f32(2, 3, 6, 7)
    rois = np.array([[0.5, 0.2, 4.0, 5.5], [-1.0, -0.5, 7.5, 6.5], [2.0, 2.0, 2.2, 2.6]], np.float32)
    ri2 = np.array([0, 1, 1], np.int64)
    for mode in ("avg", "max"):
        c.append(Case("roialign_" + mode, [N("RoiAlign", ["x", "r", "i"], ["y"], mode=mode, output_height=3, output_width=4, sampling_ratio=2)],
                      [("x", F, [2, 3, 6, 7]), ("r", F, [3, 4]), ("i", I64, [3])], [("y", F, [3, 3, 3, 4])], {"x": rx, "r": rois, "i": ri2},
                      opset=16, symbolic_axes=(0,)))
    c.append(Case("roialign_output_half_pixel_adaptive", [N("RoiAlign", ["x", "r", "i"], ["y"], output_height=2, output_width=2, spatial_scale=0.5)],
                  [("x", F, [2, 3, 6, 7]), ("r", F, [3, 4]), ("i", I64, [3])], [("y", F, [3, 3, 2, 2])], {"x": rx, "r": rois * 2, "i": ri2},
                  opset=10, dynamic=False, oracle=roialign_legacy))

    # ---- GridSample ---------------------------------------------------------
    def gs_case(name, r, mode, pad, align, opset=20, oracle=None, refuse=None, fixed=True):
        dims = [3, 4, 5][:r] if r <= 3 else [2, 3, 4, 3][:r]
        outd = [[7], [3, 4], [2, 3, 2], [2, 2, 2, 2]][r - 1]
        x = f32(2, 3, *dims)
        g = RNG.uniform(-1.4, 1.4, (2, *outd, r)).astype(np.float32)
        attrs = {"mode": mode, "padding_mode": pad, "align_corners": align}
        full = "refuse_gridsample_" + name[7:] if name.startswith("refuse_") else "gridsample_" + name
        c.append(Case(full, [N("GridSample", ["x", "g"], ["y"], **attrs)],
                      [("x", F, [2, 3, *dims]), ("g", F, [2, *outd, r])], [("y", F, [2, 3, *outd])], {"x": x, "g": g},
                      opset=opset, oracle=oracle, refuse=refuse, fixed=fixed))

    for mode in ("nearest", "linear", "cubic"):
        for pad in ("zeros", "border", "reflection"):
            for align in (0, 1):
                # cubic with border padding: ONNX Runtime and PyTorch clamp each
                # tap; the reference evaluator clamps the coordinate as well
                gs_case("%s_%s_align%d" % (mode, pad, align), 2, mode, pad, align,
                        oracle="ort" if (mode, pad) == ("cubic", "border") else None)
    for mode, pad, align in (("linear", "zeros", 0), ("linear", "reflection", 1), ("nearest", "border", 0), ("nearest", "zeros", 1)):
        gs_case("volumetric_%s_%s_align%d" % (mode, pad, align), 3, mode, pad, align)
    gs_case("1d_linear_zeros", 1, "linear", "zeros", 0)
    gs_case("1d_nearest_reflection", 1, "nearest", "reflection", 1)

    def gs16(mode20, pad, align):
        def run(feeds):
            node = helper.make_node("GridSample", ["x", "g"], ["y"], mode=mode20, padding_mode=pad, align_corners=align)
            g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, None),
                                                helper.make_tensor_value_info("g", TensorProto.FLOAT, None)],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
            return ReferenceEvaluator(helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])).run(None, feeds)
        return run
    # GridSample-16 names its modes bilinear and bicubic; the reference
    # evaluator implements only -20's names, so it runs the -20 node
    for mode16, mode20, pad, align in (("bilinear", "linear", "zeros", 0), ("bilinear", "linear", "border", 1),
                                       ("nearest", "nearest", "reflection", 0), ("bicubic", "cubic", "zeros", 1)):
        gs_case("opset16_%s_%s_align%d" % (mode16, pad, align), 2, mode16, pad, align, opset=16, oracle=gs16(mode20, pad, align))
    gs_case("opset16_default_modes", 2, "bilinear", "zeros", 0, opset=16, oracle=gs16("linear", "zeros", 0))
    gs_case("refuse_opset16_linear", 2, "linear", "zeros", 0, opset=16, refuse="is not a GridSample-16 mode")
    gs_case("refuse_cubic_volumetric", 3, "cubic", "zeros", 0, refuse="cubic interpolation is implemented for a 4-D input")
    gs_case("refuse_four_spatial_axes", 4, "linear", "zeros", 0, refuse="one to three spatial axes")

    # ---- Scatter (opset 9 and 10): ScatterElements without reduction ---------
    def scatter_oracle(axis):
        def run(feeds):
            y = feeds["d"].copy()
            for pos in np.ndindex(*feeds["i"].shape):
                at = list(pos)
                at[axis] = int(feeds["i"][pos])
                y[tuple(at)] = feeds["u"][pos]
            return [y]
        return run
    sd = f32(4, 5)
    si = np.array([[1, 0, 3], [3, 2, 0]], np.int64)
    c.append(Case("scatter_axis1_opset10", [N("Scatter", ["d", "i", "u"], ["y"], axis=1)],
                  [("d", F, [4, 5]), ("i", I64, [2, 3]), ("u", F, [2, 3])], [("y", F, [4, 5])],
                  {"d": sd, "i": si, "u": f32(2, 3)}, opset=10, oracle=scatter_oracle(1)))
    sdi = RNG.integers(-50, 50, (3, 4)).astype(np.int64)
    si2 = np.array([[2, 0, 1, 1]], np.int32)
    c.append(Case("scatter_default_axis_int64_opset9", [N("Scatter", ["d", "i", "u"], ["y"])],
                  [("d", I64, [3, 4]), ("i", I32, [1, 4]), ("u", I64, [1, 4])], [("y", I64, [3, 4])],
                  {"d": sdi, "i": si2, "u": RNG.integers(-9, 9, (1, 4)).astype(np.int64)}, opset=9, oracle=scatter_oracle(0)))

    # ---- the quantized operators and the UINT8, INT8 and INT32 tensors -------
    U8, S8 = TensorProto.UINT8, TensorProto.INT8

    def at_opset(node_op, ins, attrs, opset, consts):
        # the reference evaluator implements DequantizeLinear from opset 19;
        # the same node there (the definition is unchanged for these types)
        def run(feeds):
            node = helper.make_node(node_op, [n for n, _ in ins], ["y"], **attrs)
            g = helper.make_graph([node], "g", [helper.make_tensor_value_info(n, t, None) for n, t in ins],
                                  [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)])
            return ReferenceEvaluator(helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])).run(None, {**feeds, **consts})
        return run

    qx = (RNG.standard_normal((3, 4, 5)) * 40).astype(np.float32)
    qx.flat[:6] = np.array([2.5, -3.5, 0.5, -0.5, 1000.0, -1000.0], np.float32)
    c.append(Case("quantize_per_tensor_uint8", [N("QuantizeLinear", ["x", "s", "z"], ["y"])], [("x", F, [3, 4, 5])],
                  [("y", U8, [3, 4, 5])], {"x": qx}, [init("s", np.array(0.5, np.float32)), init("z", np.array(128, np.uint8))], opset=10))
    c.append(Case("quantize_no_zero_point", [N("QuantizeLinear", ["x", "s"], ["y"])], [("x", F, [3, 4, 5])],
                  [("y", U8, [3, 4, 5])], {"x": np.abs(qx)}, [init("s", np.array(2.0, np.float32))], opset=13))
    c.append(Case("quantize_per_axis_int8", [N("QuantizeLinear", ["x", "s", "z"], ["y"], axis=1)], [("x", F, [3, 4, 5])],
                  [("y", S8, [3, 4, 5])], {"x": qx},
                  [init("s", np.array([0.25, 1.0, 3.0, 0.5], np.float32)), init("z", np.array([-10, 0, 5, 100], np.int8))], opset=13))
    c.append(Case("quantize_opset19_saturate", [N("QuantizeLinear", ["x", "s", "z"], ["y"], axis=-1, saturate=1)], [("x", F, [3, 4, 5])],
                  [("y", S8, [3, 4, 5])], {"x": qx},
                  [init("s", np.array([0.3, 0.6, 0.9, 1.2, 1.5], np.float32)), init("z", np.array([1, 2, 3, 4, 5], np.int8))], opset=19))
    dq = RNG.integers(0, 256, (3, 4, 5)).astype(np.uint8)
    c.append(Case("dequantize_per_tensor_uint8", [N("DequantizeLinear", ["x", "s", "z"], ["y"])], [("x", U8, [3, 4, 5])],
                  [("y", F, [3, 4, 5])], {"x": dq}, [init("s", np.array(0.02, np.float32)), init("z", np.array(100, np.uint8))], opset=10,
                  oracle=at_opset("DequantizeLinear", [("x", U8), ("s", F), ("z", U8)], {}, 19,
                                  {"s": np.array(0.02, np.float32), "z": np.array(100, np.uint8)})))
    dqi = RNG.integers(-128, 128, (3, 4, 5)).astype(np.int8)
    dqs, dqz = np.array([0.5, 0.25, 2.0], np.float32), np.array([-3, 0, 7], np.int8)
    c.append(Case("dequantize_per_axis_int8", [N("DequantizeLinear", ["x", "s", "z"], ["y"], axis=0)], [("x", S8, [3, 4, 5])],
                  [("y", F, [3, 4, 5])], {"x": dqi}, [init("s", dqs), init("z", dqz)], opset=13,
                  oracle=at_opset("DequantizeLinear", [("x", S8), ("s", F), ("z", S8)], {"axis": 0}, 19, {"s": dqs, "z": dqz})))
    dq32 = RNG.integers(-2 ** 30, 2 ** 30, (3, 4, 5)).astype(np.int32)
    c.append(Case("dequantize_int32", [N("DequantizeLinear", ["x", "s"], ["y"])], [("x", I32, [3, 4, 5])],
                  [("y", F, [3, 4, 5])], {"x": dq32}, [init("s", np.array(1e-4, np.float32))], opset=13,
                  oracle=at_opset("DequantizeLinear", [("x", I32), ("s", F)], {}, 19, {"s": np.array(1e-4, np.float32)})))
    dqx = (RNG.standard_normal((4, 6)) * 3).astype(np.float32)
    c.append(Case("dynamic_quantize", [N("DynamicQuantizeLinear", ["x"], ["y", "ys", "yz"])], [("x", F, [4, 6])],
                  [("y", U8, [4, 6]), ("ys", F, []), ("yz", U8, [])], {"x": dqx}, opset=11))
    c.append(Case("dynamic_quantize_positive", [N("DynamicQuantizeLinear", ["x"], ["y", "ys", "yz"])], [("x", F, [4, 6])],
                  [("y", U8, [4, 6]), ("ys", F, []), ("yz", U8, [])], {"x": np.abs(dqx) + 1}, opset=11))
    ma = RNG.integers(0, 256, (2, 3, 4)).astype(np.uint8)
    mb = RNG.integers(0, 256, (4, 5)).astype(np.uint8)
    c.append(Case("matmulinteger_uint8", [N("MatMulInteger", ["a", "b", "az", "bz"], ["y"])], [("a", U8, [2, 3, 4]), ("b", U8, [4, 5])],
                  [("y", I32, [2, 3, 5])], {"a": ma, "b": mb}, [init("az", np.array(12, np.uint8)), init("bz", np.array(130, np.uint8))], opset=10))
    mbi = RNG.integers(-128, 128, (2, 4, 5)).astype(np.int8)
    c.append(Case("matmulinteger_per_column_batch", [N("MatMulInteger", ["a", "b", "az", "bz"], ["y"])],
                  [("a", U8, [3, 4]), ("b", S8, [2, 4, 5])], [("y", I32, [2, 3, 5])], {"a": ma[0], "b": mbi},
                  [init("az", np.array([3], np.uint8)), init("bz", np.array([1, -2, 3, -4, 5], np.int8))], opset=10))
    c.append(Case("matmulinteger_no_zero_points_1d", [N("MatMulInteger", ["a", "b"], ["y"])], [("a", S8, [4]), ("b", S8, [4, 5])],
                  [("y", I32, [5])], {"a": RNG.integers(-128, 128, 4).astype(np.int8), "b": mbi[0]}, opset=10))
    c.append(Case("qlinearmatmul_uint8", [N("QLinearMatMul", ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"])],
                  [("a", U8, [2, 3, 4]), ("b", U8, [4, 5])], [("y", U8, [2, 3, 5])], {"a": ma, "b": mb},
                  [init("as", np.array(0.02, np.float32)), init("az", np.array(120, np.uint8)), init("bs", np.array(0.03, np.float32)),
                   init("bz", np.array(130, np.uint8)), init("ys", np.array(0.9, np.float32)), init("yz", np.array(110, np.uint8))], opset=10))
    c.append(Case("qlinearmatmul_int8_per_column", [N("QLinearMatMul", ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"])],
                  [("a", S8, [3, 4]), ("b", S8, [4, 5])], [("y", S8, [3, 5])],
                  {"a": RNG.integers(-128, 128, (3, 4)).astype(np.int8), "b": mbi[1]},
                  [init("as", np.array(0.05, np.float32)), init("az", np.array(-2, np.int8)),
                   init("bs", np.array([0.01, 0.02, 0.03, 0.04, 0.05], np.float32)), init("bz", np.array([0, 1, 2, 3, 4], np.int8)),
                   init("ys", np.array(0.7, np.float32)), init("yz", np.array(-5, np.int8))], opset=10))
    cx = RNG.integers(0, 256, (1, 4, 5, 6)).astype(np.uint8)
    cw = RNG.integers(0, 256, (6, 2, 3, 3)).astype(np.uint8)
    c.append(Case("convinteger_group_pads", [N("ConvInteger", ["x", "w", "xz", "wz"], ["y"], group=2, pads=[1, 0, 1, 2], strides=[1, 2])],
                  [("x", U8, [1, 4, 5, 6]), ("w", U8, [6, 2, 3, 3])], [("y", I32, [1, 6, 5, 3])], {"x": cx, "w": cw},
                  [init("xz", np.array(7, np.uint8)), init("wz", np.array(9, np.uint8))], opset=10))
    c.append(Case("convinteger_same_upper", [N("ConvInteger", ["x", "w"], ["y"], auto_pad="SAME_UPPER", strides=[2, 2])],
                  [("x", U8, [1, 4, 5, 6]), ("w", U8, [3, 4, 2, 3])], [("y", I32, [1, 3, 3, 3])], {"x": cx, "w": cw[:3, :, :2, :].repeat(2, axis=1)},
                  opset=10))
    c.append(Case("qlinearconv_bias_per_channel", [N("QLinearConv", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"], ["y"],
                                                       pads=[1, 1, 1, 1], dilations=[1, 2])],
                  [("x", U8, [1, 4, 5, 6]), ("w", U8, [3, 4, 3, 2])], [("y", U8, [1, 3, 5, 6])], {"x": cx, "w": cw[:3].reshape(3, 2, 3, 3)[:, :, :, :2].repeat(2, axis=1)},
                  [init("xs", np.array(0.03, np.float32)), init("xz", np.array(128, np.uint8)),
                   init("ws", np.array([0.004, 0.006, 0.008], np.float32)), init("wz", np.array([120, 128, 136], np.uint8)),
                   init("ys", np.array(0.2, np.float32)), init("yz", np.array(100, np.uint8)), init("b", np.array([-300, 0, 500], np.int32))],
                  opset=10))
    c.append(Case("qlinearconv_int8_1d", [N("QLinearConv", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"], ["y"], strides=[2])],
                  [("x", S8, [2, 3, 9]), ("w", S8, [4, 3, 3])], [("y", S8, [2, 4, 4])],
                  {"x": RNG.integers(-128, 128, (2, 3, 9)).astype(np.int8), "w": RNG.integers(-128, 128, (4, 3, 3)).astype(np.int8)},
                  [init("xs", np.array(0.05, np.float32)), init("xz", np.array(3, np.int8)), init("ws", np.array(0.01, np.float32)),
                   init("wz", np.array(-1, np.int8)), init("ys", np.array(0.5, np.float32)), init("yz", np.array(0, np.int8))], opset=10))
    # Cast to and from UINT8, INT8 and INT32
    for name, src, dst, feed in (("cast_uint8_to_float", U8, F, dq[0]), ("cast_int8_to_int64", S8, I64, dqi[0]),
                                 ("cast_int32_to_float", I32, F, dq32[0]), ("cast_float_to_int32", F, I32, (qx[0] * 3).astype(np.float32)),
                                 ("cast_int64_to_uint8", I64, U8, RNG.integers(-1000, 1000, (4, 5)).astype(np.int64)),
                                 ("cast_bool_to_int8", B, S8, RNG.integers(0, 2, (4, 5)).astype(bool)),
                                 ("cast_uint8_to_int8", U8, S8, dq[1])):
        c.append(Case(name, [N("Cast", ["x"], ["y"], to=dst)], [("x", src, list(feed.shape))], [("y", dst, list(feed.shape))],
                      {"x": feed}, opset=13))
    c.append(Case("refuse_quantize_int32_input", [N("QuantizeLinear", ["x", "s", "z"], ["y"])], [("x", I32, [4])], [("y", U8, [4])],
                  {"x": np.arange(4, dtype=np.int32)}, [init("s", np.array(0.5, np.float32)), init("z", np.array(0, np.uint8))], opset=13,
                  refuse="INT32"))
    c.append(Case("refuse_matmulinteger_per_row_zero_point", [N("MatMulInteger", ["a", "b", "az"], ["y"])],
                  [("a", U8, [3, 4]), ("b", U8, [4, 5])], [("y", I32, [3, 5])], {"a": ma[0], "b": mb},
                  [init("az", np.array([1, 2, 3], np.uint8))], opset=10, symbolic_axes=(), dynamic=False,
                  refuse="per-row zero point is not implemented"))

    # ---- group C: windows, DFT, MelWeightMatrix and the losses ---------------
    # a window's size is an initializer; a FLOAT input x is added to the result
    # so the program has an input to bind
    for op in ("HannWindow", "HammingWindow", "BlackmanWindow"):
        for periodic in (0, 1):
            c.append(Case("%s_periodic%d" % (op.lower(), periodic), [N(op, ["n"], ["w"], periodic=periodic), N("Add", ["w", "x"], ["y"])],
                          [("x", F, [10])], [("y", F, [10])], {"x": np.zeros(10, np.float32)}, [init("n", np.array(10, np.int64))],
                          opset=17))
    for fixed_path, text in ((True, "uses unsupported runtime type 11"), (False, "output_datatype DOUBLE is not implemented")):
        c.append(Case("refuse_hannwindow_double" + ("" if fixed_path else "_runtime"),
                      [N("HannWindow", ["n"], ["w"], output_datatype=11), N("Cast", ["w"], ["w32"], to=F), N("Add", ["w32", "x"], ["y"])],
                      [("x", F, [10])], [("y", F, [10])], {"x": np.zeros(10, np.float32)}, [init("n", np.array(10, np.int64))], opset=17,
                      fixed=fixed_path, dynamic=not fixed_path, refuse=text))
    dx1 = f32(2, 6, 3, 1)
    dx2 = f32(2, 6, 3, 2)
    for name, x, n, inv, ones, oshape in (("real", dx1, 6, 0, 0, [2, 6, 3, 2]), ("real_onesided", dx1, 6, 0, 1, [2, 4, 3, 2]),
                                          ("complex_padded", dx2, 8, 0, 0, [2, 8, 3, 2]), ("complex_cropped", dx2, 4, 0, 0, [2, 4, 3, 2]),
                                          ("inverse", dx2, 6, 1, 0, [2, 6, 3, 2]), ("inverse_onesided", dx2, 10, 1, 1, [2, 10, 3, 1])):
        c.append(Case("dft_opset17_" + name, [N("DFT", ["x", "n"], ["y"], axis=1, inverse=inv, onesided=ones)],
                      [("x", F, list(x.shape))], [("y", F, oshape)], {"x": x}, [init("n", np.array(n, np.int64))], opset=17))
    c.append(Case("dft_opset20_axis_input", [N("DFT", ["x", "", "a"], ["y"])], [("x", F, [2, 6, 3, 2])], [("y", F, [2, 6, 3, 2])],
                  {"x": dx2}, [init("a", np.array(1, np.int64))], opset=20))
    c.append(Case("dft_opset20_default_axis", [N("DFT", ["x"], ["y"], onesided=1)], [("x", F, [2, 5, 7, 1])], [("y", F, [2, 5, 4, 2])],
                  {"x": f32(2, 5, 7, 1)}, opset=20))

    def mel_oracle(nb, dl, sr, lo, hi, then=None):
        def run(feeds):
            node = helper.make_node("MelWeightMatrix", ["a", "b", "c", "d", "e"], ["m"])
            g = helper.make_graph([node], "g", [], [helper.make_tensor_value_info("m", TensorProto.FLOAT, None)],
                                  [init("a", np.array(nb, np.int64)), init("b", np.array(dl, np.int64)), init("c", np.array(sr, np.int64)),
                                   init("d", np.array(lo, np.float32)), init("e", np.array(hi, np.float32))])
            m = ReferenceEvaluator(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])).run(None, {})[0]
            return [m * feeds["x"]] if then else [m + feeds["x"]]
        return run
    mel_inits = [init("a", np.array(8, np.int64)), init("b", np.array(64, np.int64)), init("c", np.array(16000, np.int64)),
                 init("d", np.array(20.0, np.float32)), init("e", np.array(8000.0, np.float32))]
    # (a product with Mul: MatMul's Windows kernel sums in another order than
    # the bare-metal one, which the target gate's bit identity would flag)
    c.append(Case("melweightmatrix_mul", [N("MelWeightMatrix", ["a", "b", "c", "d", "e"], ["m"]), N("Mul", ["m", "x"], ["y"])],
                  [("x", F, [1, 8])], [("y", F, [33, 8])], {"x": np.abs(f32(1, 8)) + 1}, mel_inits, opset=17,
                  oracle=mel_oracle(8, 64, 16000, 20.0, 8000.0, then=True)))
    c.append(Case("melweightmatrix_add", [N("MelWeightMatrix", ["a", "b", "c", "d", "e"], ["m"]), N("Add", ["m", "x"], ["y"])],
                  [("x", F, [33, 8])], [("y", F, [33, 8])], {"x": np.zeros((33, 8), np.float32)}, mel_inits, opset=17,
                  oracle=mel_oracle(8, 64, 16000, 20.0, 8000.0)))
    c.append(Case("refuse_melweightmatrix_runtime_input", [N("MelWeightMatrix", ["a", "b", "c", "d", "hi"], ["m"]), N("Add", ["m", "x"], ["y"])],
                  [("hi", F, []), ("x", F, [33, 8])], [("y", F, [33, 8])], {"hi": np.array(8000.0, np.float32), "x": np.zeros((33, 8), np.float32)},
                  mel_inits[:4], opset=17, symbolic_axes=(1,), refuse="MelWeightMatrix is implemented for five initializer inputs"))
    lx = f32(3, 5, 4, scale=2.0)
    lt = RNG.integers(0, 5, (3, 4)).astype(np.int64)
    lt[0, 0] = 2
    lw = RNG.uniform(0.5, 2.0, 5).astype(np.float32)
    for red in ("none", "sum", "mean"):
        for weighted in (0, 1):
            ins = ["x", "t"] + (["w"] if weighted else [])
            inits = [init("w", lw)] if weighted else []
            oshape = [3, 4] if red == "none" else []
            c.append(Case("nllloss_%s_w%d" % (red, weighted), [N("NegativeLogLikelihoodLoss", ins, ["y"], reduction=red, ignore_index=2)],
                          [("x", F, [3, 5, 4]), ("t", I64, [3, 4])], [("y", F, oshape)], {"x": lx, "t": lt}, inits, opset=13))
            c.append(Case("sce_%s_w%d" % (red, weighted), [N("SoftmaxCrossEntropyLoss", ins, ["y", "lp"], reduction=red)],
                          [("x", F, [3, 5, 4]), ("t", I64, [3, 4])], [("y", F, oshape), ("lp", F, [3, 5, 4])], {"x": lx, "t": lt}, inits,
                          opset=13))
    c.append(Case("sce_2d_int32_target_no_log_prob", [N("SoftmaxCrossEntropyLoss", ["x", "t"], ["y"])],
                  [("x", F, [6, 5]), ("t", I32, [6])], [("y", F, [])], {"x": f32(6, 5), "t": RNG.integers(0, 5, 6).astype(np.int32)}, opset=13))
    c.append(Case("nllloss_4d", [N("NegativeLogLikelihoodLoss", ["x", "t"], ["y"], reduction="sum")],
                  [("x", F, [2, 3, 2, 2]), ("t", I64, [2, 2, 2])], [("y", F, [])],
                  {"x": f32(2, 3, 2, 2), "t": RNG.integers(0, 3, (2, 2, 2)).astype(np.int64)}, opset=13))

    # ---- group C, second batch -----------------------------------------------
    for name, image, block, attrs, L in (("2d", [5, 6], [2, 3], {"pads": [1, 0, 0, 1], "strides": [1, 2], "dilations": [2, 1]}, 12),
                                         ("1d", [9], [3], {"pads": [1, 0], "strides": [2]}, 4),
                                         ("3d_default", [3, 4, 2], [2, 2, 2], {}, 6)):
        cb = int(np.prod(block))
        c.append(Case("col2im_" + name, [N("Col2Im", ["x", "i", "b"], ["y"], **attrs)], [("x", F, [2, 2 * cb, L])],
                      [("y", F, [2, 2] + image)], {"x": f32(2, 2 * cb, L)},
                      [init("i", np.array(image, np.int64)), init("b", np.array(block, np.int64))], opset=18))
    ccx = f32(4, 7, 3)
    c.append(Case("centercroppad_axes", [N("CenterCropPad", ["x", "s"], ["y"], axes=[0, 1])], [("x", F, [4, 7, 3])], [("y", F, [6, 4, 3])],
                  {"x": ccx}, [init("s", np.array([6, 4], np.int64))], opset=18))
    c.append(Case("centercroppad_all_int64", [N("CenterCropPad", ["x", "s"], ["y"])], [("x", I64, [5, 4])], [("y", I64, [2, 7])],
                  {"x": RNG.integers(0, 100, (5, 4)).astype(np.int64)}, [init("s", np.array([2, 7], np.int64))], opset=18))
    c.append(Case("centercroppad_negative_axis", [N("CenterCropPad", ["x", "s"], ["y"], axes=[-1])], [("x", F, [4, 7, 3])], [("y", F, [4, 7, 8])],
                  {"x": ccx}, [init("s", np.array([8], np.int64))], opset=18))
    upx = f32(1, 2, 2, 2)
    upi = np.array([[[[0, 3], [9, 14]], [[16, 19], [25, 30]]]], np.int64)
    c.append(Case("maxunpool_2d", [N("MaxUnpool", ["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
                  [("x", F, [1, 2, 2, 2]), ("i", I64, [1, 2, 2, 2])], [("y", F, [1, 2, 4, 4])], {"x": upx, "i": upi}, opset=11))
    c.append(Case("maxunpool_output_shape", [N("MaxUnpool", ["x", "i", "s"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
                  [("x", F, [1, 2, 2, 2]), ("i", I64, [1, 2, 2, 2])], [("y", F, [1, 2, 5, 5])], {"x": upx, "i": upi},
                  [init("s", np.array([1, 2, 5, 5], np.int64))], opset=11, oracle="ref"))
    for r, align in ((2, 0), (2, 1), (3, 0), (3, 1)):
        size = [2, 3, 4, 5] if r == 2 else [2, 3, 3, 4, 5]
        c.append(Case("affinegrid_%dd_align%d" % (r, align), [N("AffineGrid", ["t", "s"], ["y"], align_corners=align)],
                      [("t", F, [2, r, r + 1])], [("y", F, [2] + size[2:] + [r])], {"t": f32(2, r, r + 1)},
                      [init("s", np.array(size, np.int64))], opset=20))
    rx = f32(2, 3, 8, 9)
    rrois = np.array([[0, 1.2, 0.4, 6.6, 5.5], [1, -2, -1, 3, 2], [1, 4, 4, 4, 4], [0, 0, 0, 17, 15]], np.float32)
    c.append(Case("maxroipool", [N("MaxRoiPool", ["x", "r"], ["y"], pooled_shape=[3, 4], spatial_scale=0.5)],
                  [("x", F, [2, 3, 8, 9]), ("r", F, [4, 5])], [("y", F, [4, 3, 3, 4])], {"x": rx, "r": rrois}, opset=7, oracle="ort"))
    dx = f32(1, 4, 6, 7)
    dw = f32(6, 2, 3, 3)
    doff = f32(1, 36, 6, 4, scale=1.5)
    dmask = RNG.uniform(0, 1, (1, 18, 6, 4)).astype(np.float32)
    c.append(Case("deformconv_groups_mask_bias", [N("DeformConv", ["x", "w", "o", "b", "m"], ["y"], group=2, offset_group=2, pads=[1, 1, 1, 1],
                                                     strides=[1, 2])],
                  [("x", F, [1, 4, 6, 7]), ("w", F, [6, 2, 3, 3]), ("o", F, [1, 36, 6, 4]), ("b", F, [6]), ("m", F, [1, 18, 6, 4])],
                  [("y", F, [1, 6, 6, 4])], {"x": dx, "w": dw, "o": doff, "b": f32(6), "m": dmask}, opset=19))
    c.append(Case("deformconv_plain", [N("DeformConv", ["x", "w", "o"], ["y"])],
                  [("x", F, [1, 2, 5, 5]), ("w", F, [3, 2, 2, 2]), ("o", F, [1, 8, 4, 4])], [("y", F, [1, 3, 4, 4])],
                  {"x": f32(1, 2, 5, 5), "w": f32(3, 2, 2, 2), "o": f32(1, 8, 4, 4, scale=0.7)}, opset=19))
    c.append(Case("refuse_deformconv_3d", [N("DeformConv", ["x", "w", "o"], ["y"])],
                  [("x", F, [1, 1, 3, 3, 3]), ("w", F, [1, 1, 2, 2, 2]), ("o", F, [1, 24, 2, 2, 2])], [("y", F, [1, 1, 2, 2, 2])],
                  {"x": f32(1, 1, 3, 3, 3), "w": f32(1, 1, 2, 2, 2), "o": f32(1, 24, 2, 2, 2)}, opset=19, refuse="two spatial axes"))
    # Unique: its output extents depend on the values, so both builds take
    # the runtime-dimension path
    def uq(x, axis, srt):
        if axis is None:
            slices = [(v,) for v in x.reshape(-1)]
        else:
            m = np.moveaxis(x, axis, 0)
            slices = [tuple(m[i].reshape(-1)) for i in range(m.shape[0])]
        keys = [tuple((1, 0.0) if isinstance(v, np.floating) and np.isnan(v) else (0, v) for v in s) for s in slices]
        first, grp = [], []
        for i, k in enumerate(keys):
            g = next((g for g, f in enumerate(first) if keys[f] == k), None)
            if g is None:
                g = len(first)
                first.append(i)
            grp.append(g)
        order = sorted(range(len(first)), key=lambda g: keys[first[g]]) if srt else list(range(len(first)))
        idx = np.array([first[g] for g in order], np.int64)
        rank = {g: k for k, g in enumerate(order)}
        y = x.reshape(-1)[idx] if axis is None else np.take(x, idx, axis=axis)
        return [y, idx, np.array([rank[g] for g in grp], np.int64), np.array([grp.count(g) for g in order], np.int64)]

    def uq_oracle(axis, srt, keep=(0, 1, 2, 3)):
        return lambda feeds: [uq(feeds["x"], axis, srt)[k] for k in keep]

    def uq_case(name, x, elem, axis=None, srt=None, keep=(0, 1, 2, 3), oracle=None, then=False):
        attrs = {}
        if axis is not None:
            attrs["axis"] = axis
        if srt is not None:
            attrs["sorted"] = srt
        names = ["y", "i", "v", "c"]
        outs = [names[k] if k in keep else "" for k in range(4)]
        while outs[-1] == "":
            outs.pop()
        ax = None if axis is None else axis % x.ndim
        yshape = [None] if ax is None else [None if d == ax else e for d, e in enumerate(x.shape)]
        shapes = {"y": yshape, "i": [None], "v": [x.size if ax is None else x.shape[ax]], "c": [None]}
        nodes = [N("Unique", ["x"], outs, **attrs)]
        gouts = [(names[k], elem if k == 0 else I64, shapes[names[k]]) for k in keep]
        orc = oracle or uq_oracle(ax, 1 if srt is None else srt, keep)
        if then:
            nodes.append(N("Mul", ["y", "y"], ["z"]))
            gouts = [("z", elem, yshape)] + gouts[1:]
            base = orc
            orc = oracle or (lambda feeds: [o * o if k == 0 else o for k, o in enumerate(base(feeds))])
        c.append(Case("unique_" + name, nodes, [("x", elem, list(x.shape))], gouts, {"x": x}, opset=11, oracle=orc))
    uqf = np.array([2.5, -1.0, 3.0, 1.0, 0.0, 2.5, -3.0, 1.0, -1.0], np.float32)
    uqn = np.array([2.5, -0.0, np.nan, 1.0, 0.0, 2.5, -3.0, np.nan, 1.0, -0.0], np.float32)
    uqa = RNG.integers(-2, 3, (2, 5, 3)).astype(np.float32)
    uqa[:, 3] = uqa[:, 1]
    uqa[:, 4] = uqa[:, 0]
    uqa[1, :, 2] = uqa[1, :, 0]
    uq_case("flat_sorted", uqf, F)
    uq_case("flat_unsorted", uqf, F, srt=0)
    uq_case("nan_negative_zero_sorted", uqn, F, oracle="ref")
    uq_case("nan_negative_zero_unsorted", uqn, F, srt=0, oracle="ref")
    uq_case("axis1_unsorted", uqa, F, axis=1, srt=0)
    uq_case("axis1_sorted", uqa, F, axis=1)
    uq_case("axis_negative_sorted", uqa, F, axis=-1)
    uq_case("int64_axis0", np.array([[5, -7], [3, 3], [5, -7], [-9, 0], [3, 3]], np.int64), I64, axis=0)
    uq_case("int8_unsorted", np.array([-3, 100, -128, 5, -3, 127, 5], np.int8), S8, srt=0)
    uq_case("int32_sorted", RNG.integers(-4, 4, 12).astype(np.int32), I32)
    uq_case("y_and_counts", uqf, F, keep=(0, 3))
    uq_case("y_and_inverse", uqf, F, keep=(0, 2))
    uq_case("then_mul", uqa, F, axis=1, srt=0, then=True)
    # SequenceErase between tensors, so every target runs it
    c.append(Case("sequenceerase_split_concat", [N("SplitToSequence", ["x", "l"], ["s"], axis=0), N("SequenceErase", ["s", "p"], ["t"]),
                                                 N("SequenceErase", ["t"], ["u"]), N("ConcatFromSequence", ["u"], ["y"], axis=0)],
                  [("x", F, [6, 3])], [("y", F, [None, 3])], {"x": f32(6, 3)},
                  [init("l", np.array([1, 2, 2, 1], np.int64)), init("p", np.array(-3, np.int64))], opset=11))
    c.append(Case("refuse_unique_int16", [N("Unique", ["x"], ["y"])], [("x", TensorProto.INT16, [5])], [("y", TensorProto.INT16, [None])],
                  {"x": np.array([3, 1, 3, 2, 1], np.int16)}, opset=11, refuse="Graph input 'x' has element type INT16 (5); runtime-dimension emission binds"))
    # Scan and SequenceMap between tensors, so every target runs the Loop
    # they are rewritten into (the reference implements neither form here)
    sb = helper.make_graph([N("Add", ["acc", "a"], ["acc2"]), N("Mul", ["a", "b"], ["prod"]), N("Identity", ["acc2"], ["run"])], "scan_body",
                           [helper.make_tensor_value_info("acc", F, [3]), helper.make_tensor_value_info("a", F, [3]),
                            helper.make_tensor_value_info("b", F, [3])],
                           [helper.make_tensor_value_info("acc2", F, [3]), helper.make_tensor_value_info("run", F, [3]),
                            helper.make_tensor_value_info("prod", F, [3])])
    c.append(Case("scan_reverse_axes", [N("Scan", ["init", "x", "y"], ["final", "runs", "prods"], num_scan_inputs=2, body=sb,
                                          scan_input_axes=[0, -1], scan_input_directions=[1, 0], scan_output_directions=[1, 0],
                                          scan_output_axes=[-1, 1])],
                  [("init", F, [3]), ("x", F, [4, 3]), ("y", F, [3, 4])], [("final", F, [3]), ("runs", F, [3, 4]), ("prods", F, [3, 4])],
                  {"init": f32(3), "x": f32(4, 3), "y": f32(3, 4)}, opset=16, oracle="ort", symbolic_axes=()))
    mb = helper.make_graph([N("Mul", ["e", "w"], ["m"])], "map_body",
                           [helper.make_tensor_value_info("e", F, ["k"]), helper.make_tensor_value_info("w", F, [])],
                           [helper.make_tensor_value_info("m", F, ["k"])])
    c.append(Case("sequencemap_split_concat", [N("SplitToSequence", ["x", "l"], ["s"], axis=0), N("SequenceMap", ["s", "w"], ["ms"], body=mb),
                                              N("ConcatFromSequence", ["ms"], ["y"], axis=0)],
                  [("x", F, [9]), ("w", F, [])], [("y", F, [9])], {"x": f32(9), "w": np.array(-1.5, np.float32)},
                  [init("l", np.array([2, 4, 3], np.int64))], opset=17, oracle="ort", symbolic_axes=()))
    # opset 23 and later: Swish, RMSNormalization, CumProd
    for alpha in (1.0, 0.25, -1.5):
        c.append(Case("swish_alpha_%s" % str(alpha).replace("-", "m").replace(".", "_"), [N("Swish", ["x"], ["y"], alpha=alpha)],
                      [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])], {"x": f32(3, 4, 5, scale=4.0)}, opset=24))
    rx = f32(2, 3, 4, 5)
    for axis, sshape in ((-1, [5]), (2, [4, 5]), (2, [1, 5]), (1, [3, 4, 5]), (0, [2, 3, 4, 5])):
        c.append(Case("rmsnorm_axis%s_scale%s" % (str(axis).replace("-", "m"), "x".join(map(str, sshape))),
                      [N("RMSNormalization", ["x", "s"], ["y"], axis=axis, epsilon=1e-3)],
                      [("x", F, [2, 3, 4, 5]), ("s", F, sshape)], [("y", F, [2, 3, 4, 5])], {"x": rx, "s": f32(*sshape)}, opset=23))
    c.append(Case("rmsnorm_scale_initializer", [N("RMSNormalization", ["x", "s"], ["y"])], [("x", F, [4, 6])], [("y", F, [4, 6])],
                  {"x": f32(4, 6)}, [init("s", f32(6))], opset=23))
    c.append(Case("refuse_rmsnorm_stash_type", [N("RMSNormalization", ["x", "s"], ["y"], stash_type=0)], [("x", F, [4, 6]), ("s", F, [6])],
                  [("y", F, [4, 6])], {"x": f32(4, 6), "s": f32(6)}, opset=23, refuse="stash_type = 0"))
    c.append(Case("refuse_rmsnorm_scale_inner_broadcast", [N("RMSNormalization", ["x", "s"], ["y"], axis=1)], [("x", F, [2, 4, 6]), ("s", F, [4, 1])],
                  [("y", F, [2, 4, 6])], {"x": f32(2, 4, 6), "s": f32(4, 1)}, opset=23, dynamic=False, refuse="scale must have the trailing normalized extents"))
    cpf = (RNG.uniform(0.5, 1.5, (3, 4, 5)) * RNG.choice([-1, 1], (3, 4, 5))).astype(np.float32)
    for dt, elem, x in ((np.float32, F, cpf), (np.int32, I32, RNG.integers(-9, 9, (3, 4, 5)).astype(np.int32)),
                        (np.int64, I64, RNG.integers(-9, 9, (3, 4, 5)).astype(np.int64))):
        for axis, exclusive, reverse in ((1, 0, 0), (-1, 1, 0), (0, 0, 1), (2, 1, 1)):
            c.append(Case("cumprod_%s_axis%s_ex%d_rev%d" % (np.dtype(dt).name, str(axis).replace("-", "m"), exclusive, reverse),
                          [N("CumProd", ["x", "a"], ["y"], exclusive=exclusive, reverse=reverse)],
                          [("x", elem, [3, 4, 5])], [("y", elem, [3, 4, 5])], {"x": x}, [init("a", np.array(axis, np.int64))], opset=26))
    c.append(Case("cumprod_axis_input", [N("CumProd", ["x", "a"], ["y"])], [("x", F, [3, 4]), ("a", I32, [])], [("y", F, [3, 4])],
                  {"x": cpf[0, :3, :4].copy(), "a": np.array(1, np.int32)}, opset=26))
    # BitCast, RotaryEmbedding, TensorScatter
    bf = f32(3, 4)
    bf.flat[0], bf.flat[1] = -0.0, np.inf
    c.append(Case("bitcast_float_to_int32", [N("BitCast", ["x"], ["y"], to=I32)], [("x", F, [3, 4])], [("y", I32, [3, 4])], {"x": bf}, opset=26))
    c.append(Case("bitcast_int32_to_float", [N("BitCast", ["x"], ["y"], to=F)], [("x", I32, [5])], [("y", F, [5])],
                  {"x": np.array([0, 1065353216, -1082130432, 8388608, 2139095040], np.int32)}, opset=26))
    c.append(Case("bitcast_uint8_to_int8", [N("BitCast", ["x"], ["y"], to=TensorProto.INT8)], [("x", TensorProto.UINT8, [6])],
                  [("y", TensorProto.INT8, [6])], {"x": np.array([0, 1, 127, 128, 200, 255], np.uint8)}, opset=26))
    c.append(Case("bitcast_int8_to_bool", [N("BitCast", ["x"], ["y"], to=B)], [("x", TensorProto.INT8, [4])], [("y", B, [4])],
                  {"x": np.array([0, 1, 1, 0], np.int8)}, opset=26))
    c.append(Case("refuse_bitcast_width", [N("BitCast", ["x"], ["y"], to=I64)], [("x", F, [4])], [("y", I64, [4])], {"x": f32(4)}, opset=26,
                  refuse="width"))
    rx3 = f32(2, 3, 16)
    rpos = np.array([[0, 5, 2], [7, 1, 3]], np.int64)
    rc, rs = f32(8, 4), f32(8, 4)
    c.append(Case("rotary_rank3_positions", [N("RotaryEmbedding", ["x", "c", "s", "p"], ["y"], num_heads=2)],
                  [("x", F, [2, 3, 16]), ("c", F, [8, 4]), ("s", F, [8, 4]), ("p", I64, [2, 3])], [("y", F, [2, 3, 16])],
                  {"x": rx3, "c": rc, "s": rs, "p": rpos}, opset=23))
    c.append(Case("rotary_rank4_interleaved_partial", [N("RotaryEmbedding", ["x", "c", "s"], ["y"], interleaved=1, rotary_embedding_dim=4)],
                  [("x", F, [2, 2, 3, 6]), ("c", F, [2, 3, 2]), ("s", F, [2, 3, 2])], [("y", F, [2, 2, 3, 6])],
                  {"x": f32(2, 2, 3, 6), "c": f32(2, 3, 2), "s": f32(2, 3, 2)}, opset=23))
    c.append(Case("rotary_rank4_positions_halves", [N("RotaryEmbedding", ["x", "c", "s", "p"], ["y"])],
                  [("x", F, [1, 2, 3, 8]), ("p", I64, [1, 3])], [("y", F, [1, 2, 3, 8])],
                  {"x": f32(1, 2, 3, 8), "p": np.array([[4, 0, 9]], np.int64)}, [init("c", f32(10, 4)), init("s", f32(10, 4))], opset=23))
    c.append(Case("refuse_rotary_odd_dim", [N("RotaryEmbedding", ["x", "c", "s"], ["y"], rotary_embedding_dim=3)],
                  [("x", F, [1, 2, 3, 6]), ("c", F, [1, 3, 1]), ("s", F, [1, 3, 1])], [("y", F, [1, 2, 3, 6])],
                  {"x": f32(1, 2, 3, 6), "c": f32(1, 3, 1), "s": f32(1, 3, 1)}, opset=23, dynamic=False, refuse="must be even"))
    tpast, tupd = f32(2, 3, 6, 4), f32(2, 3, 2, 4)
    c.append(Case("tensorscatter_linear_indices", [N("TensorScatter", ["p", "u", "w"], ["y"])],
                  [("p", F, [2, 3, 6, 4]), ("u", F, [2, 3, 2, 4]), ("w", I64, [2])], [("y", F, [2, 3, 6, 4])],
                  {"p": tpast, "u": tupd, "w": np.array([4, 1], np.int64)}, opset=24))
    c.append(Case("tensorscatter_circular", [N("TensorScatter", ["p", "u", "w"], ["y"], mode="circular")],
                  [("p", F, [2, 3, 6, 4]), ("u", F, [2, 3, 2, 4]), ("w", I64, [2])], [("y", F, [2, 3, 6, 4])],
                  {"p": tpast, "u": tupd, "w": np.array([5, 11], np.int64)}, opset=24))
    c.append(Case("tensorscatter_int64_axis1_no_indices", [N("TensorScatter", ["p", "u"], ["y"], axis=1)],
                  [("p", I64, [2, 5, 3]), ("u", I64, [2, 2, 3])], [("y", I64, [2, 5, 3])],
                  {"p": RNG.integers(-9, 9, (2, 5, 3)).astype(np.int64), "u": RNG.integers(-9, 9, (2, 2, 3)).astype(np.int64)}, opset=24))
    c.append(Case("refuse_tensorscatter_mode", [N("TensorScatter", ["p", "u"], ["y"], mode="wrap")],
                  [("p", F, [2, 5, 3]), ("u", F, [2, 2, 3])], [("y", F, [2, 5, 3])], {"p": f32(2, 5, 3), "u": f32(2, 2, 3)}, opset=24,
                  refuse="linear and circular"))
    # Attention-23/24: 4-D and 3-D, grouped query heads, masks, causal, past
    # caches with present outputs, nonpad lengths, scale, softcap, qk output
    aq, ak, av = f32(2, 4, 3, 8), f32(2, 2, 5, 8), f32(2, 2, 5, 6)
    c.append(Case("attention_gqa_4d", [N("Attention", ["q", "k", "v"], ["y"])],
                  [("q", F, [2, 4, 3, 8]), ("k", F, [2, 2, 5, 8]), ("v", F, [2, 2, 5, 6])], [("y", F, [2, 4, 3, 6])],
                  {"q": aq, "k": ak, "v": av}, opset=23))
    c.append(Case("attention_3d_causal_float_mask", [N("Attention", ["q", "k", "v", "m"], ["y"], q_num_heads=2, kv_num_heads=2, is_causal=1)],
                  [("q", F, [1, 4, 8]), ("k", F, [1, 4, 8]), ("v", F, [1, 4, 6]), ("m", F, [4, 4])], [("y", F, [1, 4, 6])],
                  {"q": f32(1, 4, 8), "k": f32(1, 4, 8), "v": f32(1, 4, 6), "m": f32(4, 4)}, opset=23))
    bm = np.ones((1, 2, 3, 4), np.bool_)
    bm[0, 1, :, 2] = False
    c.append(Case("attention_bool_mask", [N("Attention", ["q", "k", "v", "m"], ["y"])],
                  [("q", F, [1, 2, 3, 4]), ("k", F, [1, 2, 4, 4]), ("v", F, [1, 2, 4, 4]), ("m", B, [1, 2, 3, 4])], [("y", F, [1, 2, 3, 4])],
                  {"q": f32(1, 2, 3, 4), "k": f32(1, 2, 4, 4), "v": f32(1, 2, 4, 4), "m": bm}, opset=23))
    c.append(Case("attention_past_present_float_mask_scale", [N("Attention", ["q", "k", "v", "m", "pk", "pv"], ["y", "prk", "prv"], scale=0.3)],
                  [("q", F, [2, 2, 2, 4]), ("k", F, [2, 2, 3, 4]), ("v", F, [2, 2, 3, 5]), ("m", F, [2, 1, 2, 5]), ("pk", F, [2, 2, 2, 4]), ("pv", F, [2, 2, 2, 5])],
                  [("y", F, [2, 2, 2, 5]), ("prk", F, [2, 2, 5, 4]), ("prv", F, [2, 2, 5, 5])],
                  {"q": f32(2, 2, 2, 4), "k": f32(2, 2, 3, 4), "v": f32(2, 2, 3, 5), "m": f32(2, 1, 2, 5), "pk": f32(2, 2, 2, 4), "pv": f32(2, 2, 2, 5)},
                  opset=23))
    for mode in (0, 1, 2, 3):
        # mode 0 is the product before the softcap; the onnx 1.22.0 reference hands out
        # the softcapped value there, so that form is tested without a softcap
        cap = {} if mode == 0 else {"softcap": 2.0}
        c.append(Case("attention_softcap_qk_mode%d" % mode, [N("Attention", ["q", "k", "v"], ["y", "", "", "qk"], qk_matmul_output_mode=mode, **cap)],
                      [("q", F, [1, 2, 3, 4]), ("k", F, [1, 2, 4, 4]), ("v", F, [1, 2, 4, 4])], [("y", F, [1, 2, 3, 4]), ("qk", F, [1, 2, 3, 4])],
                      {"q": f32(1, 2, 3, 4, scale=2.0), "k": f32(1, 2, 4, 4, scale=2.0), "v": f32(1, 2, 4, 4)}, opset=23))
    c.append(Case("attention_nonpad_short_mask", [N("Attention", ["q", "k", "v", "m", "", "", "n"], ["y"])],
                  [("q", F, [2, 2, 2, 4]), ("k", F, [2, 2, 6, 4]), ("v", F, [2, 2, 6, 4]), ("m", F, [2, 1, 2, 4]), ("n", I64, [2])], [("y", F, [2, 2, 2, 4])],
                  {"q": f32(2, 2, 2, 4), "k": f32(2, 2, 6, 4), "v": f32(2, 2, 6, 4), "m": f32(2, 1, 2, 4), "n": np.array([3, 4], np.int64)}, opset=24))
    c.append(Case("refuse_attention_softmax_precision", [N("Attention", ["q", "k", "v"], ["y"], softmax_precision=11)],
                  [("q", F, [1, 2, 3, 4]), ("k", F, [1, 2, 4, 4]), ("v", F, [1, 2, 4, 4])], [("y", F, [1, 2, 3, 4])],
                  {"q": f32(1, 2, 3, 4), "k": f32(1, 2, 4, 4), "v": f32(1, 2, 4, 4)}, opset=23, refuse="softmax is FLOAT"))
    # CausalConvWithState and LinearAttention (opset 27)
    c.append(Case("causalconv_bias_past_silu", [N("CausalConvWithState", ["x", "w", "b", "p"], ["y", "s"], activation="silu")],
                  [("x", F, [2, 3, 5]), ("w", F, [3, 1, 4]), ("b", F, [3]), ("p", F, [2, 3, 3])], [("y", F, [2, 3, 5]), ("s", F, [2, 3, 3])],
                  {"x": f32(2, 3, 5), "w": f32(3, 1, 4), "b": f32(3), "p": f32(2, 3, 3)}, opset=27))
    c.append(Case("causalconv_plain", [N("CausalConvWithState", ["x", "w"], ["y", "s"])], [("x", F, [1, 4, 6]), ("w", F, [4, 1, 3])],
                  [("y", F, [1, 4, 6]), ("s", F, [1, 4, 2])],
                  {"x": f32(1, 4, 6), "w": f32(4, 1, 3)}, opset=27))
    c.append(Case("refuse_causalconv_activation", [N("CausalConvWithState", ["x", "w"], ["y", "s"], activation="gelu")], [("x", F, [1, 4, 6]), ("w", F, [4, 1, 3])],
                  [("y", F, [1, 4, 6]), ("s", F, [1, 4, 2])], {"x": f32(1, 4, 6), "w": f32(4, 1, 3)}, opset=27, refuse="none, silu and swish"))
    lq, lk, lv = f32(2, 3, 12), f32(2, 3, 6, scale=0.5), f32(2, 3, 4)
    for rule, extra, ins in (("linear", {}, ["q", "k", "v"]), ("gated", {"g": RNG.uniform(-1, 0, (2, 3, 6)).astype(np.float32)}, ["q", "k", "v", "", "g"]),
                             ("delta", {"bt": RNG.uniform(0, 1, (2, 3, 2)).astype(np.float32)}, ["q", "k", "v", "", "", "bt"]),
                             ("gated_delta", {"g": RNG.uniform(-1, 0, (2, 3, 2)).astype(np.float32), "bt": RNG.uniform(0, 1, (2, 3, 1)).astype(np.float32)},
                              ["q", "k", "v", "", "g", "bt"])):
        decl = [("q", F, [2, 3, 12]), ("k", F, [2, 3, 6]), ("v", F, [2, 3, 4])] + [(n, F, list(a.shape)) for n, a in extra.items()]
        feeds = {"q": lq, "k": lk, "v": lv}
        feeds.update(extra)
        c.append(Case("linearattention_%s" % rule, [N("LinearAttention", ins, ["y", "s"], q_num_heads=4, kv_num_heads=2, update_rule=rule)],
                      decl, [("y", F, [2, 3, 8]), ("s", F, [2, 2, 3, 2])], feeds, opset=27))
    c.append(Case("linearattention_past_scale", [N("LinearAttention", ["q", "k", "v", "p", "g", "bt"], ["y", "s"], q_num_heads=2, kv_num_heads=2, scale=0.4)],
                  [("q", F, [1, 2, 6]), ("k", F, [1, 2, 6]), ("v", F, [1, 2, 4]), ("p", F, [1, 2, 3, 2]), ("g", F, [1, 2, 2]), ("bt", F, [1, 2, 2])],
                  [("y", F, [1, 2, 4]), ("s", F, [1, 2, 3, 2])], {"q": f32(1, 2, 6), "k": f32(1, 2, 6, scale=0.5), "v": f32(1, 2, 4), "p": f32(1, 2, 3, 2, scale=0.3),
                                          "g": RNG.uniform(-1, 0, (1, 2, 2)).astype(np.float32), "bt": RNG.uniform(0, 1, (1, 2, 2)).astype(np.float32)}, opset=27))
    c.append(Case("refuse_linearattention_rule", [N("LinearAttention", ["q", "k", "v"], ["y", "s"], q_num_heads=2, kv_num_heads=2, update_rule="rwkv")],
                  [("q", F, [1, 2, 6]), ("k", F, [1, 2, 6]), ("v", F, [1, 2, 4])], [("y", F, [1, 2, 4]), ("s", F, [1, 2, 3, 2])],
                  {"q": f32(1, 2, 6), "k": f32(1, 2, 6), "v": f32(1, 2, 4)}, opset=27, refuse="linear, gated, delta and gated_delta"))
    # Constant on both paths, every value form (Constant-13), sparse_value
    # checked against onnxruntime (the reference evaluator returns it sparse)
    cx = f32(2, 3)
    for name, attrs in (("value", {"value": numpy_helper.from_array(f32(2, 3), "v")}), ("value_float", {"value_float": 2.5}),
                        ("value_floats", {"value_floats": [1.0, -2.0, 3.5]})):
        c.append(Case("constant_%s" % name, [N("Constant", [], ["k"], **attrs), N("Add", ["x", "k"], ["y"])], [("x", F, [2, 3])],
                      [("y", F, [2, 3])], {"x": cx}, opset=13))
    for name, attrs in (("value_int", {"value_int": 3}), ("value_ints", {"value_ints": [1, -2, 7]})):
        c.append(Case("constant_%s" % name, [N("Constant", [], ["k"], **attrs), N("Add", ["x", "k"], ["y"])], [("x", I64, [2, 3])],
                      [("y", I64, [2, 3])], {"x": np.arange(6, dtype=np.int64).reshape(2, 3)}, opset=13))
    c.append(Case("constant_bool", [N("Constant", [], ["k"], value=helper.make_tensor("v", B, [2, 3], [1, 0, 1, 0, 0, 1])),
                                    N("And", ["x", "k"], ["y"])], [("x", B, [2, 3])], [("y", B, [2, 3])],
                  {"x": np.array([[True, True, False], [True, False, True]])}, opset=13))
    c.append(Case("constant_feeds_shape", [N("Constant", [], ["s"], value_ints=[3, 2]), N("Reshape", ["x", "s"], ["y"])], [("x", F, [2, 3])],
                  [("y", F, [3, 2])], {"x": cx}, opset=13))
    for name, idx in (("linear", np.array([1, 4], np.int64)), ("coordinates", np.array([[0, 1], [1, 2]], np.int64))):
        sp = helper.make_sparse_tensor(numpy_helper.from_array(np.array([5.0, -6.0], np.float32), "v"), numpy_helper.from_array(idx, "i"), [2, 3])
        c.append(Case("constant_sparse_%s" % name, [N("Constant", [], ["k"], sparse_value=sp), N("Add", ["x", "k"], ["y"])],
                      [("x", F, [2, 3])], [("y", F, [2, 3])], {"x": cx}, opset=13, oracle="ort"))
    sp = helper.make_sparse_tensor(numpy_helper.from_array(np.array([2, 9], np.int64), "v"), numpy_helper.from_array(np.array([0, 5], np.int64), "i"), [6])
    c.append(Case("constant_sparse_int64", [N("Constant", [], ["k"], sparse_value=sp), N("Add", ["x", "k"], ["y"])],
                  [("x", I64, [6])], [("y", I64, [6])], {"x": np.arange(6, dtype=np.int64)}, opset=13, oracle="ort"))
    c.append(Case("refuse_constant_string", [N("Constant", [], ["k"], value_string="text"), N("Identity", ["x"], ["y"])], [("x", F, [2])],
                  [("y", F, [2])], {"x": f32(2)}, opset=13, refuse="attribute value_string holds text"))
    # a graph whose intermediates carry no value_info (as a function-expanded
    # one): the runtime-dimension path computes it
    c.append(Case("undeclared_intermediates", [N("Shape", ["x"], ["s"]), N("Constant", [], ["i"], value_ints=[1]), N("Gather", ["s", "i"], ["g"]),
                                               N("Cast", ["g"], ["gf"], to=F), N("Mul", ["x", "gf"], ["m"]), N("Relu", ["m"], ["y"])],
                  [("x", F, [2, 3])], [("y", F, [2, 3])], {"x": cx}, opset=13, infer=False))
    # Where whose condition is larger than X and Y (forum 993)
    for name, cs, xs, ys in (("cond_higher_rank", [2, 3, 4], [4], [3, 4]), ("cond_wider", [2, 3, 4], [2, 1, 4], [1, 1, 4]),
                             ("each_widens", [1, 3, 1], [2, 1, 1], [1, 1, 4])):
        c.append(Case("where_%s" % name, [N("Where", ["c", "x", "y"], ["z"])], [("c", B, cs), ("x", F, xs), ("y", F, ys)],
                      [("z", F, [2, 3, 4])], {"c": RNG.uniform(size=cs) > 0.5, "x": f32(*xs), "y": f32(*ys)}, opset=16))
    # Relu, Flatten, LessOrEqual and BatchNormalization on both paths
    rx = f32(2, 3, 4)
    c.append(Case("relu", [N("Relu", ["x"], ["y"])], [("x", F, [2, 3, 4])], [("y", F, [2, 3, 4])], {"x": rx}, opset=14))
    for axis in (0, 1, 2, 3, -1):
        fa = axis if axis >= 0 else axis + 3
        c.append(Case("flatten_axis%s" % str(axis).replace("-", "m"), [N("Flatten", ["x"], ["y"], axis=axis)], [("x", F, [2, 3, 4])],
                      [("y", F, [int(np.prod([2, 3, 4][:fa])), int(np.prod([2, 3, 4][fa:]))])], {"x": rx}, opset=13))
    c.append(Case("flatten_int64_opset9", [N("Flatten", ["x"], ["y"])], [("x", I64, [2, 3, 2])], [("y", I64, [2, 6])],
                  {"x": np.arange(12, dtype=np.int64).reshape(2, 3, 2)}, opset=9))
    c.append(Case("lessorequal_broadcast", [N("LessOrEqual", ["a", "b"], ["y"])], [("a", F, [2, 3]), ("b", F, [3])], [("y", B, [2, 3])],
                  {"a": np.array([[0, 1, 2], [3, np.nan, -1]], np.float32), "b": np.array([1, 1, -1], np.float32)}, opset=16))
    c.append(Case("lessorequal_int64", [N("LessOrEqual", ["a", "b"], ["y"])], [("a", I64, [4]), ("b", I64, [4])], [("y", B, [4])],
                  {"a": np.array([1, 2, 3, -4], np.int64), "b": np.array([2, 2, 2, -4], np.int64)}, opset=12))
    bx = f32(2, 3, 2, 2)
    bn_in = [init("s", f32(3)), init("bb", f32(3)), init("mu", f32(3)), init("var", np.abs(f32(3)) + 0.5)]
    c.append(Case("batchnorm_inference", [N("BatchNormalization", ["x", "s", "bb", "mu", "var"], ["y"], epsilon=1e-3)],
                  [("x", F, [2, 3, 2, 2])], [("y", F, [2, 3, 2, 2])], {"x": bx}, bn_in, opset=15))
    c.append(Case("batchnorm_opset9", [N("BatchNormalization", ["x", "s", "bb", "mu", "var"], ["y"])],
                  [("x", F, [2, 3, 2, 2])], [("y", F, [2, 3, 2, 2])], {"x": bx}, bn_in, opset=9, oracle="ort"))
    c.append(Case("batchnorm_training", [N("BatchNormalization", ["x", "s", "bb", "mu", "var"], ["y", "rm", "rv"], training_mode=1, momentum=0.8)],
                  [("x", F, [2, 3, 2, 2])], [("y", F, [2, 3, 2, 2]), ("rm", F, [3]), ("rv", F, [3])], {"x": bx}, bn_in, opset=15))
    # Neg and Abs of integers (the runtime-dimension path runs the lane's
    # integer kernel for an input declared INT32 or INT64)
    for op, dt, et in (("Neg", np.int64, I64), ("Abs", np.int64, I64), ("Neg", np.int32, I32), ("Abs", np.int32, I32)):
        xv = np.array([[3, -4, 0], [np.iinfo(np.int32).min + 1, 7, -9]], dt)
        c.append(Case("%s_%s" % (op.lower(), np.dtype(dt).name), [N(op, ["x"], ["y"])], [("x", et, [2, 3])], [("y", et, [2, 3])], {"x": xv}, opset=13))
    c.append(Case("refuse_input_float16_runtime", [N("Abs", ["x"], ["y"])], [("x", TensorProto.FLOAT16, [3])], [("y", TensorProto.FLOAT16, [3])],
                  {"x": np.ones(3, np.float16)}, opset=20, fixed=False, refuse="Graph input 'x' has element type FLOAT16 (10); runtime-dimension emission binds"))
    # Bernoulli and Multinomial: this compiler's specified generator
    bp = RNG.uniform(0, 1, (3, 4, 5)).astype(np.float32)
    bp.flat[:4] = [0.0, 1.0, np.nan, 0.5]
    c.append(Case("bernoulli_seed", [N("Bernoulli", ["x"], ["y"], seed=3.0)], [("x", F, [3, 4, 5])], [("y", F, [3, 4, 5])],
                  {"x": bp}, opset=15, oracle=bernoulli_oracle(0, 3.0, np.float32)))
    c.append(Case("bernoulli_model_seed_bool", [N("Bernoulli", ["x"], ["y"], dtype=B)], [("x", F, [3, 4, 5])], [("y", B, [3, 4, 5])],
                  {"x": bp}, opset=15, oracle=bernoulli_oracle(0, None, np.bool_)))
    c.append(Case("bernoulli_int64_after_abs", [N("Abs", ["x"], ["a"]), N("Bernoulli", ["a"], ["y"], dtype=I64, seed=-1.5)],
                  [("x", F, [3, 4, 5])], [("y", I64, [3, 4, 5])], {"x": (bp - 0.5).astype(np.float32)}, opset=15,
                  oracle=bernoulli_oracle(1, -1.5, np.int64)))
    c.append(Case("bernoulli_uint8", [N("Bernoulli", ["x"], ["y"], dtype=TensorProto.UINT8, seed=11.0)], [("x", F, [7])],
                  [("y", TensorProto.UINT8, [7])], {"x": bp.ravel()[:7].copy()}, opset=15, oracle=bernoulli_oracle(0, 11.0, np.uint8)))
    ml = np.array([[0.0, 1.0, -1.0, 2.0, 0.5], [3.0, -20.0, 3.0, 0.0, 1.0], [-5.0, -5.0, -5.0, -5.0, -5.0]], np.float32)
    c.append(Case("multinomial_default", [N("Multinomial", ["x"], ["y"], seed=2.0)], [("x", F, [3, 5])], [("y", I32, [3, 1])],
                  {"x": ml}, opset=7, oracle=multinomial_oracle(0, 2.0, 1, np.int32)))
    c.append(Case("multinomial_samples_int64", [N("Multinomial", ["x"], ["y"], seed=5.0, sample_size=9, dtype=I64)], [("x", F, [3, 5])],
                  [("y", I64, [3, 9])], {"x": ml}, opset=7, oracle=multinomial_oracle(0, 5.0, 9, np.int64)))
    c.append(Case("multinomial_model_seed", [N("Multinomial", ["x"], ["y"], sample_size=4)], [("x", F, [3, 5])], [("y", I32, [3, 4])],
                  {"x": ml}, opset=15, oracle=multinomial_oracle(0, None, 4, np.int32)))
    c.append(Case("refuse_bernoulli_float16_dtype", [N("Bernoulli", ["x"], ["y"], dtype=TensorProto.FLOAT16)], [("x", F, [4])],
                  [("y", TensorProto.FLOAT16, [4])], {"x": bp.ravel()[:4].copy()}, opset=15, refuse="is not implemented: Bernoulli writes"))
    c.append(Case("refuse_multinomial_float_dtype", [N("Multinomial", ["x"], ["y"], dtype=F)], [("x", F, [2, 3])],
                  [("y", F, [2, 1])], {"x": ml[:2, :3].copy()}, opset=7, refuse="is not an output type of Multinomial"))
    return c


def roialign_legacy(feeds):
    # the reference evaluator applies the opset-16 default (half_pixel) to an
    # opset-10 node, whose only behaviour is output_half_pixel: run the same
    # node at opset 16 with that mode spelled out
    node = helper.make_node("RoiAlign", ["x", "r", "i"], ["y"], output_height=2, output_width=2, spatial_scale=0.5,
                            coordinate_transformation_mode="output_half_pixel")
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 6, 7]),
                                        helper.make_tensor_value_info("r", TensorProto.FLOAT, [3, 4]),
                                        helper.make_tensor_value_info("i", TensorProto.INT64, [3])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 16)])
    return ReferenceEvaluator(m).run(None, feeds)


def reexported(node_dir: Path, corpus: Path) -> list[tuple[str, str | None]]:
    out = []
    for case_dir in sorted(p for p in node_dir.iterdir() if p.is_dir()):
        name = case_dir.name[5:] if case_dir.name.startswith("test_") else case_dir.name
        if not any(name == op or name.startswith(op + "_") for op in OPS):
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
        dest = corpus / ("r20_" + case_dir.name)
        shutil.copytree(case_dir, dest)
        onnx.save(m, str(dest / "model.onnx"))
        out.append(("r20_" + case_dir.name, None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="cases for the operators that complete the ONNX operator set")
    ap.add_argument("--pbcompiler", required=True, type=Path)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--work", type=Path, default=HERE / "work" / "ops")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-r20", action="store_true")
    args = ap.parse_args()
    ns.quiet_crashes()
    corpus = args.work / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    todo = []
    if not args.no_r20:
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
    tally = {"pass": 0, "type": 0, "expanded": 0, "form": 0}
    for refuse, r in sorted(records, key=lambda t: t[1]["case"]):
        if refuse is None:
            refuse = R20_REFUSED.get(r["case"])
        if refuse is not None:
            ok = r["outcome"] == "REFUSED" and refuse in r["detail"]
            tally["form"] += 1 if ok and r["case"].startswith("r20_") else 0
        elif r["case"].startswith("r20_") and r["outcome"] == "REFUSED" and any(
                t in r["detail"] for t in ("unsupported ONNX type", "unsupported runtime type", "float8", "is not implemented by fixed-shape emission; it casts to",
                                           "element type", "ONNX tensor type")):
            # an official case over element types the compiler does not carry
            # (FLOAT16, BFLOAT16, float8, the 2- and 4-bit types, UINT8, STRING)
            ok = True
            tally["type"] += 1
        elif r["case"].startswith("r20_") and r["case"].endswith("_expanded") and r["outcome"] == "REFUSED" and "has no concrete type/shape" in r["detail"]:
            # a function-expanded case whose intermediate values carry no
            # shapes, which the fixed-shape path needs (docs/VALIDATION.md)
            ok = True
            tally["expanded"] += 1
        else:
            ok = r["outcome"] == "PASS"
            tally["pass"] += 1 if ok and r["case"].startswith("r20_") else 0
        bad += 0 if ok else 1
        print("%-4s %-58s %-12s %-18s %s" % ("ok" if ok else "BAD", r["case"], r["outcome"], r.get("path") or "", r["detail"][:260]))
    print("ops: %d of %d as expected; re-imported official cases: %d PASS, %d refused for their element type, %d function-expanded "
          "without intermediate shapes, %d refused for a form not implemented" % (len(records) - bad, len(records), tally["pass"],
                                                                                  tally["type"], tally["expanded"], tally["form"]))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
