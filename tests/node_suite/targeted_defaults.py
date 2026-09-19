#!/usr/bin/env python3
"""Targeted cases for attribute forms the official node tests do not reach at opset <= 20.

DEVELOPER CHECK ONLY, in the node_suite family. The official ONNX node tests for
Cast, Shape, Conv, ConvTranspose and LSTM are all generated at opset 22 or 25 in
onnx 1.22.0, above the harness's opset-20 ceiling, and none of them spells out a
default attribute value the way an exporter does. These cases do, at opset 20.

Every case is built twice when the form allows it:
  <name>_fixed    all input extents declared, so the compiler takes the fixed-shape path;
  <name>_dynamic  the leading extent is symbolic, so it takes the runtime-dimension path.

Expected outputs come from onnx.reference.ReferenceEvaluator (the specification's
own Python implementation) and are cross-checked against onnxruntime when it
accepts the model; a disagreement between the two stops the generator.

Cases whose name starts with refuse_ must end REFUSED, with a sentence that names
the operator and the attribute; everything else must PASS.

Usage (repository root):
  py -3.12 tests/node_suite/targeted_defaults.py --pbcompiler PATH [--cli PATH] [--only GLOB ...]
"""
from __future__ import annotations

import argparse
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
NP = {F: np.float32, I64: np.int64, I32: np.int32, B: np.bool_}


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), extra_opsets=(), dynamic=True,
                 fixed=True, refuse=None, ref_nodes=None, oracle=None):
        self.ref_nodes, self.oracle = ref_nodes, oracle
        self.name, self.nodes, self.inputs, self.outputs = name, nodes, inputs, outputs
        self.feeds, self.initializers, self.extra_opsets = feeds, list(initializers), list(extra_opsets)
        self.dynamic, self.fixed, self.refuse = dynamic, fixed, refuse


def vi(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def model_for(case: Case, symbolic: bool) -> onnx.ModelProto:
    def shape(s):
        if s is None or not symbolic or len(s) == 0:
            return s
        return ["N"] + list(s[1:])
    ins = [vi(n, e, shape(s)) for n, e, s in case.inputs]
    outs = [vi(n, e, ["%s_%d" % (n, i) for i in range(len(s))] if symbolic else s) for n, e, s in case.outputs]
    g = helper.make_graph(case.nodes, case.name, ins, outs, case.initializers)
    opsets = [helper.make_opsetid("", 20)] + [helper.make_opsetid(d, v) for d, v in case.extra_opsets]
    m = helper.make_model(g, opset_imports=opsets)
    m.ir_version = 9
    return m


def expected(case: Case, m: onnx.ModelProto):
    ref_model = m
    if case.ref_nodes is not None:
        # The reference evaluator's Conv auto_pad is not the specification's (it pads
        # VALID like SAME and reads the batch and channel extents as spatial ones), so
        # these cases feed it the explicit pads the specification text defines and
        # hold onnxruntime, running the auto_pad model itself, to the same answer.
        ref_model = onnx.ModelProto()
        ref_model.CopyFrom(m)
        del ref_model.graph.node[:]
        ref_model.graph.node.extend(case.ref_nodes)
    try:
        got = case.oracle(case.feeds) if case.oracle else ReferenceEvaluator(ref_model).run(None, case.feeds)
    except (RuntimeError, NotImplementedError) as exc:
        # The reference LSTM returns Y and Y_h only and runs one direction only;
        # onnxruntime is then the sole oracle.
        if ort is None or not ("Unable to find output name" in str(exc) or "num_directions" in str(exc)):
            raise
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        return [np.asarray(x) for x in sess.run(None, case.feeds)]
    if ort is not None:
        try:
            sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        except Exception:
            sess = None
        if sess is not None:
            try:
                other = sess.run(None, case.feeds)
            except Exception as exc:  # e.g. onnxruntime refuses dilation with SAME padding
                print("note: %s: onnxruntime cannot run it (%s); the reference alone decides" % (case.name, str(exc)[-90:]))
                other = []
            for i, (a, b) in enumerate(zip(got, other)):
                a, b = np.asarray(a), np.asarray(b)
                if a.shape != b.shape or not np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=1e-4, atol=1e-5):
                    sys.exit("targeted: %s output %d: the reference evaluator and onnxruntime disagree\n%r\n%r"
                             % (case.name, i, a, b))
    return [np.asarray(x) for x in got]


def write(folder: Path, case: Case, symbolic: bool):
    m = model_for(case, symbolic)
    onnx.checker.check_model(m)
    outs = [] if case.refuse else expected(case, m)
    names = [n for n, _, _ in case.inputs]
    ns.write_case(folder, m, [case.feeds[n] for n in names], outs)


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def cases() -> list[Case]:
    out: list[Case] = []
    x = np.array([[2.7, -2.7, 0.5, -0.5], [1.49, -1.51, 3.0, 0.0]], np.float32)
    # ---------------- Cast: saturate is a float8-only attribute
    for to, label in ((F, "float"), (I64, "int64"), (B, "bool")):
        out.append(Case("cast_saturate1_to_" + label, [helper.make_node("Cast", ["x"], ["y"], to=to, saturate=1)],
                        [("x", F, [2, 4])], [("y", to, [2, 4])], {"x": x}))
    out.append(Case("cast_saturate0_int64_to_float", [helper.make_node("Cast", ["x"], ["y"], to=F, saturate=0)],
                    [("x", I64, [2, 3])], [("y", F, [2, 3])], {"x": np.array([[1, -2, 3], [40, 0, -7]], np.int64)}))
    out.append(Case("refuse_cast_to_float8", [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT8E4M3FN, saturate=1)],
                    [("x", F, [2, 4])], [("y", TensorProto.FLOAT8E4M3FN, [2, 4])], {"x": x}, refuse="Cast"))
    # ---------------- Shape: start and end
    s = f32(2, 3, 4, 5)
    for label, attrs in (("start0", dict(start=0)), ("start1", dict(start=1)), ("end_neg1", dict(end=-1)),
                         ("start1_end_neg1", dict(start=1, end=-1)), ("start_neg10_clip", dict(start=-10)),
                         ("end10_clip", dict(end=10)), ("start_gt_end", dict(start=3, end=1)),
                         ("start_neg1", dict(start=-1)), ("start1_end2", dict(start=1, end=2))):
        n = helper.make_node("Shape", ["x"], ["y"], **attrs)
        ref = ReferenceEvaluator(helper.make_model(helper.make_graph([n], "t", [vi("x", F, [2, 3, 4, 5])], [vi("y", I64, ["r"])]),
                                                   opset_imports=[helper.make_opsetid("", 20)])).run(None, {"x": s})[0]
        out.append(Case("shape_" + label, [n], [("x", F, [2, 3, 4, 5])], [("y", I64, [len(ref)])], {"x": s}))
    # ---------------- LayerNormalization: stash_type
    ln_scale = numpy_helper.from_array(f32(6), "scale")
    ln_bias = numpy_helper.from_array(f32(6), "bias")
    out.append(Case("layernorm_stash1_axis_neg1",
                    [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y"], axis=-1, epsilon=1e-5, stash_type=1)],
                    [("x", F, [2, 3, 6])], [("y", F, [2, 3, 6])], {"x": f32(2, 3, 6)}, [ln_scale, ln_bias]))
    out.append(Case("refuse_layernorm_stash_type_0",
                    [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y"], axis=-1, stash_type=0)],
                    [("x", F, [2, 3, 6])], [("y", F, [2, 3, 6])], {"x": f32(2, 3, 6)}, [ln_scale, ln_bias], refuse="LayerNormalization"))
    # ---------------- LSTM: layout, input_forget, activations at their defaults
    hid, inp, seq, bat = 3, 4, 5, 2
    for direction, dirs in (("forward", 1), ("bidirectional", 2)):
        W = numpy_helper.from_array(f32(dirs, 4 * hid, inp) * 0.5, "W")
        R = numpy_helper.from_array(f32(dirs, 4 * hid, hid) * 0.5, "R")
        Bb = numpy_helper.from_array(f32(dirs, 8 * hid) * 0.1, "B")
        acts = ["Sigmoid", "Tanh", "Tanh"] * dirs
        node = helper.make_node("LSTM", ["x", "W", "R", "B"], ["Y", "Y_h", "Y_c"], hidden_size=hid, direction=direction,
                                layout=0, input_forget=0, activations=acts)
        out.append(Case("lstm_defaults_" + direction, [node], [("x", F, [seq, bat, inp])],
                        [("Y", F, [seq, dirs, bat, hid]), ("Y_h", F, [dirs, bat, hid]), ("Y_c", F, [dirs, bat, hid])],
                        {"x": f32(seq, bat, inp)}, [W, R, Bb]))
    W = numpy_helper.from_array(f32(1, 4 * hid, inp) * 0.5, "W")
    R = numpy_helper.from_array(f32(1, 4 * hid, hid) * 0.5, "R")
    out.append(Case("refuse_lstm_layout_1", [helper.make_node("LSTM", ["x", "W", "R"], ["Y", "Y_h", "Y_c"], hidden_size=hid, layout=1)],
                    [("x", F, [bat, seq, inp])], [("Y", F, [bat, seq, 1, hid]), ("Y_h", F, [bat, 1, hid]), ("Y_c", F, [bat, 1, hid])],
                    {"x": f32(bat, seq, inp)}, [W, R], dynamic=False, refuse="LSTM"))
    out.append(Case("refuse_lstm_input_forget_1", [helper.make_node("LSTM", ["x", "W", "R"], ["Y", "Y_h", "Y_c"], hidden_size=hid, input_forget=1)],
                    [("x", F, [seq, bat, inp])], [("Y", F, [seq, 1, bat, hid]), ("Y_h", F, [1, bat, hid]), ("Y_c", F, [1, bat, hid])],
                    {"x": f32(seq, bat, inp)}, [W, R], dynamic=False, refuse="LSTM"))
    # ---------------- Conv and ConvTranspose: auto_pad
    def conv_pads(pad, width, k, stride, dil):
        if pad == "VALID":
            return [0, 0]
        out_w = -(-width // stride)
        total = max(0, (out_w - 1) * stride + (k - 1) * dil + 1 - width)
        begin = total // 2 if pad == "SAME_UPPER" else total - total // 2
        return [begin, total - begin]

    def ref_shape(nodes, width, inits):
        m = helper.make_model(helper.make_graph(nodes, "t", [vi("x", F, [1, 2, width])], [vi("y", F, ["a", "b", "c"])], inits),
                              opset_imports=[helper.make_opsetid("", 20)])
        return list(ReferenceEvaluator(m).run(None, {"x": f32(1, 2, width)})[0].shape)

    for width, k, stride, dil in ((9, 3, 1, 1), (10, 4, 2, 1), (11, 3, 3, 2), (7, 5, 2, 1), (8, 2, 1, 3)):
        Wc = numpy_helper.from_array(f32(4, 2, k), "W")
        Bc = numpy_helper.from_array(f32(4), "B")
        for pad in ("NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"):
            attrs = dict(auto_pad=pad, kernel_shape=[k], strides=[stride], dilations=[dil], group=1)
            explicit = dict(kernel_shape=[k], strides=[stride], dilations=[dil], group=1,
                            pads=[1, 1] if pad == "NOTSET" else conv_pads(pad, width, k, stride, dil))
            if pad == "NOTSET":
                attrs["pads"] = [1, 1]
            n = helper.make_node("Conv", ["x", "W", "B"], ["y"], **attrs)
            r = [helper.make_node("Conv", ["x", "W", "B"], ["y"], **explicit)]
            out.append(Case("conv_%s_w%d_k%d_s%d_d%d" % (pad.lower(), width, k, stride, dil), [n],
                            [("x", F, [1, 2, width])], [("y", F, ref_shape(r, width, [Wc, Bc]))], {"x": f32(1, 2, width)},
                            [Wc, Bc], ref_nodes=r))
    for width, k, stride, dil in ((5, 3, 2, 1), (6, 4, 3, 1), (4, 3, 1, 1), (5, 3, 2, 2)):
        Wt = numpy_helper.from_array(f32(2, 3, k), "W")
        Bt = numpy_helper.from_array(f32(3), "B")
        for pad in ("NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"):
            attrs = dict(auto_pad=pad, kernel_shape=[k], strides=[stride], dilations=[dil], group=1)
            explicit = dict(kernel_shape=[k], strides=[stride], dilations=[dil], group=1)
            if pad == "NOTSET":
                attrs["pads"] = explicit["pads"] = [1, 1]
                attrs["output_padding"] = explicit["output_padding"] = [1] if stride > 1 else [0]
            elif pad == "VALID":
                explicit["pads"] = [0, 0]
            else:
                total = stride * (width - 1) + (k - 1) * dil + 1 - width * stride
                begin = total // 2 if pad == "SAME_UPPER" else total - total // 2
                explicit["pads"] = [begin, total - begin]
            n = helper.make_node("ConvTranspose", ["x", "W", "B"], ["y"], **attrs)
            r = [helper.make_node("ConvTranspose", ["x", "W", "B"], ["y"], **explicit)]
            out.append(Case("convtranspose_%s_w%d_k%d_s%d_d%d" % (pad.lower(), width, k, stride, dil), [n],
                            [("x", F, [1, 2, width])], [("y", F, ref_shape(r, width, [Wt, Bt]))], {"x": f32(1, 2, width)},
                            [Wt, Bt], ref_nodes=r))
    out.append(Case("refuse_conv_auto_pad_with_pads", [helper.make_node("Conv", ["x", "W", "B"], ["y"], auto_pad="SAME_UPPER", pads=[1, 1])],
                    [("x", F, [1, 2, 9])], [("y", F, [1, 4, 9])], {"x": f32(1, 2, 9)},
                    [numpy_helper.from_array(f32(4, 2, 3), "W"), numpy_helper.from_array(f32(4), "B")], refuse="Conv"))
    # ---------------- Resize: Kitten's two forms with every default attribute written out
    defaults = dict(antialias=0, cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0,
                    keep_aspect_ratio_policy="stretch")
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0], np.float32), "scales")
    for label, attrs in (("nearest_floor_asymmetric", dict(mode="nearest", nearest_mode="floor", coordinate_transformation_mode="asymmetric")),
                         ("linear_half_pixel", dict(mode="linear", nearest_mode="floor", coordinate_transformation_mode="half_pixel")),
                         ("linear_asymmetric", dict(mode="linear", coordinate_transformation_mode="asymmetric"))):
        a = dict(defaults)
        a.update(attrs)
        n = helper.make_node("Resize", ["x", "", "scales"], ["y"], **a)
        out.append(Case("resize_defaults_" + label, [n], [("x", F, [2, 3, 5])], [("y", F, [2, 3, 10])], {"x": f32(2, 3, 5)}, [scales]))
    roi = numpy_helper.from_array(np.array([], np.float32), "roi")
    n = helper.make_node("Resize", ["x", "roi", "scales"], ["y"], mode="linear", coordinate_transformation_mode="half_pixel")
    out.append(Case("resize_empty_roi_linear_half_pixel", [n], [("x", F, [2, 3, 5])], [("y", F, [2, 3, 10])], {"x": f32(2, 3, 5)}, [scales, roi]))
    n = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="linear", antialias=1)
    down = numpy_helper.from_array(np.array([1.0, 1.0, 0.5], np.float32), "scales")
    out.append(Case("refuse_resize_antialias_1", [n], [("x", F, [2, 3, 8])], [("y", F, [2, 3, 4])], {"x": f32(2, 3, 8)}, [down], refuse="Resize"))
    n = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest", coordinate_transformation_mode="asymmetric")
    out.append(Case("refuse_resize_nearest_mode_default_round_prefer_floor", [n], [("x", F, [2, 3, 5])], [("y", F, [2, 3, 10])],
                    {"x": f32(2, 3, 5)}, [scales], refuse="Resize"))
    # ---------------- ScatterND: reduction
    # The specification's pseudo-code, written out: the reference evaluator's max and
    # min index with the tuple array instead of the tuple, so it cannot be the oracle.
    def scatter_oracle(reduction):
        def run(feeds):
            data, indices, updates = feeds["x"], feeds["i"], feeds["u"]
            output = np.copy(data)
            for idx in np.ndindex(indices.shape[:-1]):
                key = tuple(indices[idx])
                if reduction == "none":
                    output[key] = updates[idx]
                elif reduction == "add":
                    output[key] = output[key] + updates[idx]
                elif reduction == "mul":
                    output[key] = output[key] * updates[idx]
                elif reduction == "max":
                    output[key] = np.maximum(output[key], updates[idx])
                else:
                    output[key] = np.minimum(output[key], updates[idx])
            return [output]
        return run
    data = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    for red in ("none", "add", "mul", "max", "min"):
        indices = np.array([[1], [0], [1]], np.int64) if red != "none" else np.array([[1], [0]], np.int64)
        updates = np.array([[5, -1, 2, 8], [0.5, 3, -4, 1], [2, 2, 9, -3]], np.float32)[:len(indices)]
        n = helper.make_node("ScatterND", ["x", "i", "u"], ["y"], reduction=red)
        out.append(Case("scatternd_reduction_" + red, [n],
                        [("x", F, [3, 4]), ("i", I64, list(indices.shape)), ("u", F, list(updates.shape))],
                        [("y", F, [3, 4])], {"x": data, "i": indices, "u": updates}, oracle=scatter_oracle(red)))
        di = np.array([[0, 2], [1, 1], [0, 2]], np.int64)[:len(indices)]
        du = np.array([7, -2, 5], np.int64)[:len(indices)]
        dd = np.arange(8, dtype=np.int64).reshape(2, 4)
        out.append(Case("scatternd_int64_reduction_" + red, [n],
                        [("x", I64, [2, 4]), ("i", I64, list(di.shape)), ("u", I64, list(du.shape))],
                        [("y", I64, [2, 4])], {"x": dd, "i": di, "u": du}, oracle=scatter_oracle(red)))
    # ---------------- ReduceSum: noop_with_empty_axes at its default
    axes = numpy_helper.from_array(np.array([-1], np.int64), "axes")
    n = helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=0, noop_with_empty_axes=0)
    out.append(Case("reducesum_noop0_last_axis", [n], [("x", F, [2, 3, 4])], [("y", F, [2, 3])], {"x": f32(2, 3, 4)}, [axes]))
    # ---------------- opset imports no node uses
    n = helper.make_node("Add", ["x", "x"], ["y"])
    out.append(Case("opset_unused_domains_add", [n], [("x", F, [2, 3])], [("y", F, [2, 3])], {"x": f32(2, 3)},
                    extra_opsets=[("ai.onnx.ml", 5), ("com.microsoft", 1), ("ai.onnx.training", 1)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="targeted default-attribute cases")
    ap.add_argument("--pbcompiler", required=True, type=Path)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--work", type=Path, default=HERE / "work" / "targeted")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    ns.quiet_crashes()
    corpus = args.work / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    names = []
    for case in cases():
        for variant, symbolic, wanted in (("fixed", False, case.fixed), ("dynamic", True, case.dynamic)):
            if not wanted:
                continue
            name = "%s_%s" % (case.name, variant)
            if args.only and not any(fnmatch.fnmatch(name, p) for p in args.only):
                continue
            write(corpus / name, case, symbolic)
            names.append((name, case))
    cli = args.cli or ns.build_cli(args.pbcompiler, args.work, 600)
    tools = ns.Tools(cli, args.pbcompiler, 120, 300, 60)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda nc: (nc[1], ns.run_case(corpus / nc[0], nc[0], args.work / "cases", tools, set(), False)), names))
    bad = 0
    for case, r in sorted(records, key=lambda t: t[1]["case"]):
        if case.refuse:
            ok = r["outcome"] == "REFUSED" and case.refuse in r["detail"]
        else:
            ok = r["outcome"] == "PASS"
        bad += 0 if ok else 1
        print("%-4s %-62s %-12s %-18s %s" % ("ok" if ok else "BAD", r["case"], r["outcome"], r.get("path") or "", r["detail"][:300]))
    print("targeted: %d of %d as expected" % (len(records) - bad, len(records)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
