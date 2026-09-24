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
       "dropout", "castlike", "size")


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), opset=20, dynamic=True, fixed=True,
                 refuse=None, symbolic_axes=(0,), oracle=None):
        self.name, self.nodes, self.inputs, self.outputs, self.feeds = name, nodes, inputs, outputs, feeds
        self.initializers, self.opset, self.dynamic, self.fixed, self.refuse = list(initializers), opset, dynamic, fixed, refuse
        self.symbolic_axes, self.oracle = symbolic_axes, oracle


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
            if a.shape != b.shape or not np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=1e-3, atol=1e-6, equal_nan=True):
                raise SystemExit("%s output %d: onnxruntime and the reference evaluator disagree\n  ref %r\n  ort %r"
                                 % (case.name, i, a.ravel()[:12], b.ravel()[:12]))
    return got


def write(folder: Path, case: Case, symbolic: bool):
    m = model_for(case, symbolic)
    onnx.checker.check_model(m)
    outs = [] if case.refuse else expected(case, m)
    ns.write_case(folder, m, [case.feeds[n] for n, _, _ in case.inputs], outs)


def f32(*shape, scale=1.0):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


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
    return c


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
    tally = {"pass": 0, "type": 0, "expanded": 0}
    for refuse, r in sorted(records, key=lambda t: t[1]["case"]):
        if refuse is not None:
            ok = r["outcome"] == "REFUSED" and refuse in r["detail"]
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
          "without intermediate shapes" % (len(records) - bad, len(records), tally["pass"], tally["type"], tally["expanded"]))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
