#!/usr/bin/env python3
"""Coverage for If, Loop and the sequence operators beyond what the corpus scores at
opset <= 20.

DEVELOPER CHECK ONLY, in the node_suite family.

Three groups of cases:

  r20_<case>   The official SplitToSequence node tests, which onnx 1.22.0 publishes
               only at opset 24, and test_identity_sequence (opset 25).
               SplitToSequence has one definition (opset 11) and Identity has
               carried sequences since opset 14 with only element types added
               later, so the same model is re-imported at opset 20 and scored
               against the official output_*.pb files unchanged.

  <name>       Targeted forms the node tests do not reach: If branches of different
               rank reading an outer tensor, Loop as a while loop, zero trips,
               permuted carried values, nested Loop and If with captures two
               levels out, a subgraph name that shadows an outer one, a sequence
               grown over many iterations, every SplitToSequence split form and
               both ConcatFromSequence forms. Runtime dimensions throughout.
               Expected outputs come from onnxruntime, cross-checked against
               onnx.reference.ReferenceEvaluator where it runs the model.

  kitten_<name> The exact Kitten TTS nano 0.8 nodes: the repeat-interleave region
               (SequenceEmpty, both SplitToSequence nodes, the Loop with its
               12-node body, ConcatFromSequence) and both If nodes, cut out with
               their attributes and bodies unchanged, fed Kitten-shaped values.
               Needs --kitten MODEL.

Cases whose name starts with refuse_ must end REFUSED with a sentence containing the
given text; run_refuse_ cases must end RUN_ERROR with it; everything else must PASS.

Usage (repository root):
  py -3.12 tests/node_suite/targeted_control.py --pbcompiler PATH [--cli PATH] [--kitten MODEL] [--only GLOB ...]
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
N = helper.make_node


def tv(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def sv(name, elem, shape=None):
    return helper.make_value_info(name, helper.make_sequence_type_proto(helper.make_tensor_type_proto(elem, shape)))


def scalar(name, value, elem):
    return numpy_helper.from_array(np.array(value, dtype=helper.tensor_dtype_to_np_dtype(elem)), name)


def const(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


class Case:
    def __init__(self, name, nodes, inputs, outputs, feeds, initializers=(), opset=20, refuse=None, run_refuse=None):
        self.name, self.nodes, self.inputs, self.outputs, self.feeds = name, nodes, inputs, outputs, feeds
        self.initializers, self.opset, self.refuse, self.run_refuse = list(initializers), opset, refuse, run_refuse


def model_for(case: Case) -> onnx.ModelProto:
    g = helper.make_graph(case.nodes, case.name, case.inputs, case.outputs, case.initializers)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", case.opset)])
    m.ir_version = 9
    return m


def as_value(x):
    if isinstance(x, list):
        return [np.asarray(e) for e in x]
    return np.asarray(x)


def same(a, b) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=1e-5, atol=1e-6, equal_nan=True)


def expected(case: Case, m: onnx.ModelProto):
    options = ort.SessionOptions()
    options.log_severity_level = 4
    sess = ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
    got = [as_value(x) for x in sess.run(None, case.feeds)]
    try:
        ref = [as_value(x) for x in ReferenceEvaluator(m).run(None, case.feeds)]
    except Exception as exc:
        print("note: %s: the reference evaluator cannot run it (%s); onnxruntime alone decides" % (case.name, str(exc)[-100:]))
        ref = got
    for i, (a, b) in enumerate(zip(got, ref)):
        if not same(a, b):
            if not isinstance(a, list) and not isinstance(b, list) and a.size == b.size and same(a.ravel(), b.ravel()):
                # onnx 1.22's reference Loop stacks scalar scan outputs as [T, 1];
                # the specification stacks them along one new axis, [T], as
                # onnxruntime does.
                print("note: %s output %d: shapes %s (onnxruntime) and %s (reference) hold the same values; onnxruntime decides"
                      % (case.name, i, a.shape, b.shape))
                continue
            raise SystemExit("targeted_control: %s output %d: onnxruntime and the reference evaluator disagree; the case is not a reliable oracle" % (case.name, i))
    return got


def write_value(path: Path, value, name: str):
    if isinstance(value, list):
        path.write_bytes(numpy_helper.from_list(value, name).SerializeToString())
    else:
        path.write_bytes(numpy_helper.from_array(value, name).SerializeToString())


def write(folder: Path, case: Case):
    m = model_for(case)
    if not case.refuse:
        try:
            onnx.checker.check_model(m, full_check=True)
        except onnx.checker.ValidationError as exc:
            # An If whose branches return different ranks cannot declare an output
            # shape, and the checker insists on one for graph outputs; Kitten's own
            # graph carries exactly that. onnxruntime still runs it and decides.
            if "shape" not in str(exc):
                raise
    ds = folder / "test_data_set_0"
    ds.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(folder / "model.onnx"))
    for i, vi in enumerate(case.inputs):
        write_value(ds / ("input_%d.pb" % i), case.feeds[vi.name], vi.name)
    if case.refuse or case.run_refuse:
        return
    for i, (vi, value) in enumerate(zip(case.outputs, expected(case, m))):
        write_value(ds / ("output_%d.pb" % i), value, vi.name)


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def graph(nodes, name, inputs, outputs, initializers=()):
    return helper.make_graph(nodes, name, inputs, outputs, list(initializers))


def cases() -> list[Case]:
    c: list[Case] = []

    # --- sequences --------------------------------------------------------
    x, y, z = f32(2, 3), f32(4, 3), f32(1, 3)
    c.append(Case("seq_construct_at_length",
                  [N("SequenceConstruct", ["x", "y", "z"], ["s"]),
                   N("SequenceAt", ["s", "last"], ["at_last"]),
                   N("SequenceAt", ["s", "first"], ["at_first"]),
                   N("SequenceLength", ["s"], ["n"])],
                  [tv("x", F, ["a", 3]), tv("y", F, ["b", 3]), tv("z", F, ["c", 3])],
                  [tv("at_last", F, ["p", 3]), tv("at_first", F, ["q", 3]), tv("n", I64, []), sv("s", F, None)],
                  {"x": x, "y": y, "z": z}, [scalar("last", -1, I64), scalar("first", 0, I32)]))
    a, b, d = np.array([1, 2, 3], np.int32), np.array([4], np.int32), np.array([5, 6], np.int32)
    c.append(Case("seq_insert_positions",
                  [N("SequenceEmpty", [], ["e"], dtype=I32),
                   N("SequenceInsert", ["e", "a"], ["s1"]),
                   N("SequenceInsert", ["s1", "b", "zero"], ["s2"]),
                   N("SequenceInsert", ["s2", "d", "minus1"], ["s3"]),
                   N("ConcatFromSequence", ["s3"], ["joined"], axis=0),
                   N("SequenceLength", ["s2"], ["n2"])],
                  [tv("a", I32, ["i"]), tv("b", I32, ["j"]), tv("d", I32, ["k"])],
                  [sv("s3", I32, None), tv("joined", I32, ["t"]), tv("n2", I64, []), sv("s1", I32, None)],
                  {"a": a, "b": b, "d": d}, [scalar("zero", 0, I64), scalar("minus1", -1, I64)]))
    x = f32(7, 3)
    c.append(Case("split_scalar_uneven",
                  [N("SplitToSequence", ["x", "three"], ["s"], axis=0),
                   N("ConcatFromSequence", ["s"], ["back"], axis=0)],
                  [tv("x", F, ["a", 3])], [sv("s", F, None), tv("back", F, ["b", 3])],
                  {"x": x}, [scalar("three", 3, I64)]))
    x = f32(2, 6)
    c.append(Case("split_list_last_axis",
                  [N("SplitToSequence", ["x", "lengths"], ["s"], axis=-1),
                   N("SequenceAt", ["s", "one"], ["middle"])],
                  [tv("x", F, [2, "w"])], [sv("s", F, None), tv("middle", F, [2, "m"])],
                  {"x": x}, [const("lengths", np.array([1, 2, 3], np.int32)), scalar("one", 1, I64)]))
    x = f32(3, 4)
    c.append(Case("split_nosplit_keepdims0_stack",
                  [N("SplitToSequence", ["x"], ["s"], axis=1, keepdims=0),
                   N("ConcatFromSequence", ["s"], ["stacked"], axis=-1, new_axis=1),
                   N("ConcatFromSequence", ["s"], ["front"], axis=0, new_axis=1)],
                  [tv("x", F, ["a", "b"])], [sv("s", F, None), tv("stacked", F, ["p", "q"]), tv("front", F, ["r", "t"])],
                  {"x": x}))
    x = np.array([[True, False, True], [False, False, True]])
    c.append(Case("split_bool_keepdims1",
                  [N("SplitToSequence", ["x"], ["s"], axis=0, keepdims=1),
                   N("ConcatFromSequence", ["s"], ["joined"], axis=-1)],
                  [tv("x", B, ["a", 3])], [sv("s", B, None), tv("joined", B, [1, "b"])],
                  {"x": x}))

    # --- If ----------------------------------------------------------------
    then_g = graph([N("Squeeze", ["conv", "axis1"], ["squeezed"])], "then", [], [tv("squeezed", F, None)],
                   [const("axis1", np.array([1], np.int64))])
    else_g = graph([N("Mul", ["conv", "onef"], ["same"])], "else", [], [tv("same", F, None)], [scalar("onef", 1.0, F)])
    for flag in (True, False):
        conv = f32(1, 1, 9) if flag else f32(1, 2, 9)
        c.append(Case("if_rank_differs_%s" % ("then" if flag else "else"),
                      [N("Shape", ["conv"], ["shape"]), N("Gather", ["shape", "one"], ["dim1"]),
                       N("Equal", ["dim1", "one"], ["cond"]),
                       N("If", ["cond"], ["picked"], then_branch=then_g, else_branch=else_g),
                       N("Unsqueeze", ["picked", "axis1b"], ["restored"])],
                      [tv("conv", F, [1, "c", "w"])], [tv("picked", F, None), tv("restored", F, None)],
                      {"conv": conv}, [const("one", np.array([1], np.int64)), const("axis1b", np.array([1], np.int64))]))

    # --- Loop --------------------------------------------------------------
    # A while loop: no trip count; the body keeps going while its counter is below 3.
    body = graph([N("Add", ["count", "one"], ["next"]), N("Less", ["next", "three"], ["more"]),
                  N("Mul", ["acc", "two"], ["doubled"]), N("Sub", ["next", "zero"], ["scan"])],
                 "while_body", [tv("i", I64, []), tv("c", B, []), tv("count", I64, []), tv("acc", F, None)],
                 [tv("more", B, []), tv("next", I64, []), tv("doubled", F, None), tv("scan", I64, [])],
                 [scalar("one", 1, I64), scalar("three", 3, I64), scalar("two", 2.0, F), scalar("zero", 0, I64)])
    c.append(Case("loop_while",
                  [N("Loop", ["", "go", "start", "acc0"], ["final_count", "final_acc", "counts"], body=body)],
                  [tv("go", B, []), tv("start", I64, []), tv("acc0", F, ["n"])],
                  [tv("final_count", I64, []), tv("final_acc", F, ["n"]), tv("counts", I64, ["t"])],
                  {"go": np.array(True), "start": np.array(0, np.int64), "acc0": f32(4)}))
    c.append(Case("loop_while_cond_false",
                  [N("Loop", ["", "go", "start", "acc0"], ["final_count", "final_acc", "counts"], body=body)],
                  [tv("go", B, []), tv("start", I64, []), tv("acc0", F, ["n"])],
                  [tv("final_count", I64, []), tv("final_acc", F, ["n"]), tv("counts", I64, [0])],
                  {"go": np.array(False), "start": np.array(0, np.int64), "acc0": f32(4)}))
    # Carried values trade places each trip; the iteration number is used as data.
    body = graph([N("Cast", ["i"], ["fi"], to=F), N("Add", ["p", "fi"], ["p_next"]), N("And", ["c", "c"], ["c_out"])],
                 "swap_body", [tv("i", I64, []), tv("c", B, []), tv("p", F, None), tv("q", F, None)],
                 [tv("c_out", B, []), tv("q", F, None), tv("p_next", F, None), tv("p_next", F, None)])
    c.append(Case("loop_swap_carries",
                  [N("Loop", ["trips", "", "p0", "q0"], ["p_final", "q_final", "history"], body=body)],
                  [tv("trips", I64, []), tv("p0", F, ["n"]), tv("q0", F, ["n"])],
                  [tv("p_final", F, ["n"]), tv("q_final", F, ["n"]), tv("history", F, ["t", "n"])],
                  {"trips": np.array(5, np.int64), "p0": f32(3), "q0": f32(3)}))
    c.append(Case("loop_zero_trips_declared_scan",
                  [N("Loop", ["trips", "", "p0", "q0"], ["p_final", "q_final", "history"],
                     body=graph([N("And", ["c", "c"], ["c_out"]), N("Add", ["p", "q"], ["s"])],
                                "zero_body", [tv("i", I64, []), tv("c", B, []), tv("p", F, [3]), tv("q", F, [3])],
                                [tv("c_out", B, []), tv("q", F, [3]), tv("p", F, [3]), tv("s", F, [3])]))],
                  [tv("trips", I64, []), tv("p0", F, [3]), tv("q0", F, [3])],
                  [tv("p_final", F, [3]), tv("q_final", F, [3]), tv("history", F, ["t", 3])],
                  {"trips": np.array(0, np.int64), "p0": f32(3), "q0": f32(3)}))
    c.append(Case("run_refuse_loop_zero_trips_undeclared_scan",
                  [N("Loop", ["trips", "", "p0"], ["p_final", "history"],
                     body=graph([N("And", ["c", "c"], ["c_out"]), N("Neg", ["p"], ["s"])],
                                "zero_body2", [tv("i", I64, []), tv("c", B, []), tv("p", F, None)],
                                [tv("c_out", B, []), tv("p", F, None), tv("s", F, None)]))],
                  [tv("trips", I64, []), tv("p0", F, ["n"])],
                  [tv("p_final", F, ["n"]), tv("history", F, ["t", "n"])],
                  {"trips": np.array(0, np.int64), "p0": f32(3)}, run_refuse="does not declare every extent"))
    # Nested Loop and If reading values from two scopes out. The inner body
    # defines 'x_after', a name the top-level graph defines again after the loop
    # (valid ONNX: a subgraph may not repeat an enclosing definition that already
    # exists, but a later one is not visible to it), and both branches of the If
    # define 'branch' (siblings may reuse a name).
    inner_then = graph([N("Add", ["x_after", "bias"], ["branch"])], "inner_then", [], [tv("branch", F, None)])
    inner_else = graph([N("Sub", ["x_after", "x"], ["branch"])], "inner_else", [], [tv("branch", F, None)])
    inner = graph([N("Less", ["j", "two_i"], ["even"]),
                   N("Mul", ["v", "scale"], ["x_after"]),
                   N("If", ["even"], ["v_next"], then_branch=inner_then, else_branch=inner_else),
                   N("And", ["c2", "c2"], ["c2_out"])],
                  "inner", [tv("j", I64, []), tv("c2", B, []), tv("v", F, None)],
                  [tv("c2_out", B, []), tv("v_next", F, None), tv("x_after", F, None)])
    outer = graph([N("Loop", ["inner_trips", "", "w"], ["w_next", "xs"], body=inner),
                   N("ReduceSum", ["xs", "axes1"], ["xs_sum"], keepdims=0),
                   N("And", ["c1", "c1"], ["c1_out"])],
                  "outer", [tv("i", I64, []), tv("c1", B, []), tv("w", F, None)],
                  [tv("c1_out", B, []), tv("w_next", F, None), tv("xs_sum", F, None)])
    c.append(Case("loop_nested_if_captures_renaming",
                  [N("Mul", ["x", "half"], ["scale"]),
                   N("Loop", ["outer_trips", "", "w0"], ["w_final", "sums"], body=outer),
                   N("Add", ["x", "w_final"], ["x_after"])],
                  [tv("x", F, [1]), tv("bias", F, ["n"]), tv("w0", F, ["n"]), tv("outer_trips", I64, []), tv("inner_trips", I64, [])],
                  [tv("w_final", F, ["n"]), tv("sums", F, ["t", "n"]), tv("x_after", F, ["n"])],
                  {"x": np.array([3.0], np.float32), "bias": f32(4), "w0": f32(4),
                   "outer_trips": np.array(3, np.int64), "inner_trips": np.array(4, np.int64)},
                  [scalar("half", 0.5, F), scalar("two_i", 2, I64), const("axes1", np.array([1], np.int64))]))
    # A sequence built over 300 trips from tensors of changing length, then joined.
    grow = graph([N("Add", ["i", "one"], ["len"]), N("Unsqueeze", ["len", "axis0"], ["len1"]),
                  N("ConstantOfShape", ["len1"], ["ones"], value=numpy_helper.from_array(np.array([1], np.int64))),
                  N("Mul", ["ones", "i"], ["chunk"]),
                  N("SequenceInsert", ["seq", "chunk"], ["seq_next"]),
                  N("And", ["c", "c"], ["c_out"])],
                 "grow_body", [tv("i", I64, []), tv("c", B, []), sv("seq", I64, None)],
                 [tv("c_out", B, []), sv("seq_next", I64, None)],
                 [scalar("one", 1, I64), const("axis0", np.array([0], np.int64))])
    c.append(Case("loop_sequence_growth",
                  [N("SequenceEmpty", [], ["empty"], dtype=I64),
                   N("Loop", ["trips", "", "empty"], ["built"], body=grow),
                   N("ConcatFromSequence", ["built"], ["joined"], axis=0),
                   N("SequenceLength", ["built"], ["count"])],
                  [tv("trips", I64, [])], [tv("joined", I64, ["t"]), tv("count", I64, [])],
                  {"trips": np.array(300, np.int64)}))
    # Opset 11 spelling with a Constant node inside the branch.
    c.append(Case("if_opset11_constant_branches",
                  [N("If", ["cond"], ["out"],
                     then_branch=graph([N("Constant", [], ["t"], value=numpy_helper.from_array(np.array([1, 2], np.int64))),
                                        N("Unsqueeze", ["t"], ["tu"], axes=[0])], "t11", [], [tv("tu", I64, [1, 2])]),
                     else_branch=graph([N("Constant", [], ["e"], value=numpy_helper.from_array(np.array([7, 8, 9], np.int64))),
                                        N("Unsqueeze", ["e"], ["eu"], axes=[0])], "e11", [], [tv("eu", I64, [1, 3])]))],
                  [tv("cond", B, [])], [tv("out", I64, None)],
                  {"cond": np.array(False)}, opset=11))

    # Identity carrying a sequence: in a loop body (a hand-over) and at top level (a copy
    # of a sequence that is read again afterwards).
    ident_body = graph([N("Identity", ["seq"], ["seq_same"]), N("SequenceInsert", ["seq_same", "i"], ["seq_next"]),
                        N("And", ["c", "c"], ["c_out"])],
                       "ident_body", [tv("i", I64, []), tv("c", B, []), sv("seq", I64, None)],
                       [tv("c_out", B, []), sv("seq_next", I64, None)])
    c.append(Case("identity_sequence_loop_and_copy",
                  [N("SequenceEmpty", [], ["empty"], dtype=I64),
                   N("Loop", ["trips", "", "empty"], ["built"], body=ident_body),
                   N("Identity", ["built"], ["copy"]),
                   N("SequenceLength", ["built"], ["count"]),
                   N("ConcatFromSequence", ["copy"], ["joined"], axis=0, new_axis=1)],
                  [tv("trips", I64, [])], [tv("count", I64, []), tv("joined", I64, ["t"]), sv("copy", I64, None)],
                  {"trips": np.array(4, np.int64)}))

    # --- refusals ---------------------------------------------------------
    c.append(Case("refuse_loop_without_trips_or_cond",
                  [N("Loop", ["", "", "v"], ["v_out"],
                     body=graph([N("And", ["c", "c"], ["c_out"]), N("Neg", ["p"], ["p_out"])], "forever",
                                [tv("i", I64, []), tv("c", B, []), tv("p", F, None)], [tv("c_out", B, []), tv("p_out", F, None)]))],
                  [tv("v", F, ["n"])], [tv("v_out", F, ["n"])], {"v": f32(2)}, refuse="loop that never ends"))
    c.append(Case("refuse_concat_from_sequence_without_axis",
                  [N("SplitToSequence", ["x"], ["s"]), N("ConcatFromSequence", ["s"], ["y"])],
                  [tv("x", F, ["n"])], [tv("y", F, ["m"])], {"x": f32(3)}, refuse="has no axis attribute"))
    c.append(Case("refuse_split_keepdims_2",
                  [N("SplitToSequence", ["x"], ["s"], keepdims=2), N("ConcatFromSequence", ["s"], ["y"], axis=0)],
                  [tv("x", F, ["n"])], [tv("y", F, ["m"])], {"x": f32(3)}, refuse="keepdims=2 is not 0 or 1"))
    c.append(Case("refuse_sequence_scan_output",
                  [N("SequenceEmpty", [], ["e"]),
                   N("Loop", ["trips", ""], ["scans"],
                     body=graph([N("And", ["c", "c"], ["c_out"])], "seqscan",
                                [tv("i", I64, []), tv("c", B, [])], [tv("c_out", B, []), sv("e", F, None)]))],
                  [tv("trips", I64, [])], [tv("scans", F, None)], {"trips": np.array(2, np.int64)},
                  refuse="scan output 0 is a sequence"))
    c.append(Case("refuse_sequence_empty_float16",
                  [N("SequenceEmpty", [], ["e"], dtype=TensorProto.FLOAT16), N("SequenceLength", ["e"], ["n"])],
                  [], [tv("n", I64, [])], {}, refuse="dtype=10"))
    c.append(Case("refuse_softmax_opset11_in_body",
                  [N("If", ["cond"], ["y"],
                     then_branch=graph([N("Softmax", ["x"], ["sm"])], "sm_then", [], [tv("sm", F, None)]),
                     else_branch=graph([N("Identity", ["x"], ["id"])], "sm_else", [], [tv("id", F, None)]))],
                  [tv("cond", B, []), tv("x", F, ["a", "b"])], [tv("y", F, None)],
                  {"cond": np.array(True), "x": f32(2, 3)}, opset=11, refuse="current from opset 13"))
    c.append(Case("refuse_identity_sequence_opset13",
                  [N("SplitToSequence", ["x"], ["s"]), N("Identity", ["s"], ["t"]), N("ConcatFromSequence", ["t"], ["y"], axis=0)],
                  [tv("x", F, ["n"])], [tv("y", F, ["m"])], {"x": f32(3)}, opset=13,
                  refuse="Identity takes a sequence only from opset 14"))
    c.append(Case("run_refuse_sequence_at_out_of_range",
                  [N("SplitToSequence", ["x"], ["s"], axis=0), N("SequenceAt", ["s", "pos"], ["y"])],
                  [tv("x", F, ["n", 2]), tv("pos", I64, [])], [tv("y", F, [1, 2])],
                  {"x": f32(3, 2), "pos": np.array(3, np.int64)}, run_refuse="SequenceAt position 3 is outside"))
    # SequenceErase: a position, the default last element, a negative
    # position, an INT32 position, erasing down to an empty sequence
    x, y, z = f32(2, 3), f32(4, 3), f32(1, 3)
    c.append(Case("seq_erase_positions",
                  [N("SequenceConstruct", ["x", "y", "z"], ["s"]),
                   N("SequenceErase", ["s", "one"], ["s1"]),
                   N("SequenceErase", ["s1"], ["s2"]),
                   N("SequenceErase", ["s", "minus3"], ["s3"]),
                   N("ConcatFromSequence", ["s2"], ["joined"], axis=0),
                   N("SequenceLength", ["s3"], ["n3"]),
                   N("SequenceAt", ["s3", "zero"], ["first3"])],
                  [tv("x", F, ["a", 3]), tv("y", F, ["b", 3]), tv("z", F, ["c", 3])],
                  [sv("s1", F, None), tv("joined", F, ["t", 3]), tv("n3", I64, []), tv("first3", F, ["p", 3])],
                  {"x": x, "y": y, "z": z}, [scalar("one", 1, I64), scalar("minus3", -3, I64), scalar("zero", 0, I32)]))
    x = f32(5, 2)
    c.append(Case("seq_erase_to_empty",
                  [N("SplitToSequence", ["x", "lengths"], ["s"], axis=0),
                   N("SequenceErase", ["s", "first"], ["s1"]),
                   N("SequenceErase", ["s1"], ["s2"]),
                   N("SequenceLength", ["s2"], ["n"]),
                   N("SequenceLength", ["s"], ["n0"])],
                  [tv("x", F, ["a", 2])], [tv("n", I64, []), tv("n0", I64, []), sv("s1", F, None)],
                  {"x": x}, [const("lengths", np.array([2, 3], np.int64)), scalar("first", 0, I32)]))
    c.append(Case("run_refuse_sequence_erase_out_of_range",
                  [N("SplitToSequence", ["x"], ["s"], axis=0), N("SequenceErase", ["s", "pos"], ["t"]), N("SequenceLength", ["t"], ["y"])],
                  [tv("x", F, ["n", 2]), tv("pos", I64, [])], [tv("y", I64, [])],
                  {"x": f32(3, 2), "pos": np.array(3, np.int64)}, run_refuse="SequenceErase position 3 is outside"))
    return c


def reexported(node_dir: Path, corpus: Path) -> list[str]:
    out = []
    for case_dir in sorted(p for p in node_dir.iterdir() if p.is_dir() and
                           (p.name.startswith("test_split_to_sequence") or p.name == "test_identity_sequence")):
        m = onnx.load(str(case_dir / "model.onnx"), load_external_data=False)
        if ns.ai_onnx_opset(m) is None or ns.ai_onnx_opset(m) <= 20:
            continue
        for o in m.opset_import:
            if o.domain in ("", "ai.onnx"):
                o.version = 20
        onnx.checker.check_model(m, full_check=True)
        name = "r20_" + case_dir.name
        shutil.copytree(case_dir, corpus / name)
        onnx.save(m, str(corpus / name / "model.onnx"))
        out.append(name)
    return out


def kitten_cases(path: Path) -> list[Case]:
    """The Kitten nodes, unchanged, with their outer inputs as graph inputs."""
    model = onnx.load(str(path))
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    by_output = {o: n for n in g.node for o in n.output}
    wanted = ["/SequenceEmpty", "/SplitToSequence_1", "/SplitToSequence", "/Loop", "/ConcatFromSequence"]
    nodes = [next(n for n in g.node if n.name == w) for w in wanted]
    produced = {o for n in nodes for o in n.output}
    needed, used_inits = [], []
    for n in nodes:
        for name in list(n.input) + [x for a in n.attribute if a.type == onnx.AttributeProto.GRAPH for sn in a.g.node for x in sn.input]:
            if not name or name in produced or name in [v for v, _, _ in needed]:
                continue
            if name in inits:
                if inits[name] not in used_inits:
                    used_inits.append(inits[name])
                continue
            if name in by_output or name in [i.name for i in g.input]:
                needed.append((name, None, None))
    durations = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4], np.int64)
    tokens = len(durations)
    feeds = {"/Reshape_1_output_0": np.arange(tokens, dtype=np.int64), "/Where_1_output_0": durations,
             "/ConstantOfShape_2_output_0": np.ones(tokens, np.int64), "/Unsqueeze_4_output_0": np.array([tokens], np.int64)}
    body_inputs = [name for name, _, _ in needed if name not in feeds]
    if body_inputs:
        raise SystemExit("targeted_control: the Kitten Loop region reads %s, which this case does not feed" % body_inputs)
    inputs = [tv("/Reshape_1_output_0", I64, ["tokens"]), tv("/Where_1_output_0", I64, ["tokens"]),
              tv("/ConstantOfShape_2_output_0", I64, ["tokens"]), tv("/Unsqueeze_4_output_0", I64, [1])]
    out = [Case("kitten_repeat_interleave", nodes, inputs, [tv("/ConcatFromSequence_output_0", I64, ["frames"])], feeds, used_inits)]
    for if_name, conv in (("/If_1", "/N_proj/Conv_output_0"), ("/If", "/F0_proj/Conv_output_0")):
        node = next(n for n in g.node if n.name == if_name)
        cond = node.input[0]
        for label, value in (("then", f32(1, 1, 12)), ("else", f32(2, 1, 12))):
            flag = value.shape[0] == 1
            out.append(Case("kitten_%s_%s" % (if_name.strip("/").lower(), label), [node],
                            [tv(cond, B, [1]), tv(conv, F, ["b", 1, "w"])], [tv(node.output[0], F, None)],
                            {cond: np.array([flag]), conv: value},
                            [inits[x] for a in node.attribute for x in [i for sn in a.g.node for i in sn.input] if x in inits]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="lane cases for If, Loop and the sequence operators")
    ap.add_argument("--pbcompiler", required=True, type=Path)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--kitten", type=Path)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--work", type=Path, default=HERE / "work" / "control")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    if ort is None:
        sys.exit("targeted_control: onnxruntime is required for the expected outputs; install onnxruntime.")
    ns.quiet_crashes()
    corpus = args.work / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    todo = []
    for name in reexported(ns.corpus_dir(), corpus):
        if not args.only or any(fnmatch.fnmatch(name, p) for p in args.only):
            todo.append((name, None, None))
    all_cases = cases() + (kitten_cases(args.kitten) if args.kitten else [])
    for case in all_cases:
        if args.only and not any(fnmatch.fnmatch(case.name, p) for p in args.only):
            continue
        write(corpus / case.name, case)
        todo.append((case.name, case.refuse, case.run_refuse))
    cli = args.cli or ns.build_cli(args.pbcompiler, args.work, 600)
    tools = ns.Tools(cli, args.pbcompiler, 120, 300, 60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda t: (t[1], t[2], ns.run_case(corpus / t[0], t[0], args.work / "cases", tools, set(), False)), todo))
    bad = 0
    for refuse, run_refuse, r in sorted(records, key=lambda t: t[2]["case"]):
        if refuse is not None:
            ok = r["outcome"] == "REFUSED" and refuse in r["detail"]
        elif run_refuse is not None:
            ok = r["outcome"] == "RUN_ERROR" and run_refuse in r["detail"]
        else:
            ok = r["outcome"] == "PASS"
        bad += 0 if ok else 1
        print("%-4s %-50s %-12s %-18s %s" % ("ok" if ok else "BAD", r["case"], r["outcome"], r.get("path") or "", r["detail"][:300]))
    print("control: %d of %d as expected" % (len(records) - bad, len(records)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
