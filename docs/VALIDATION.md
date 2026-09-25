# Validation and known boundaries

[Home](../README.md) · [Build](BUILD.md) · [Quick start](QUICKSTART.md)

## Publication checks — September 8, 2026

The standalone Desktop export was tested on Windows x64 with PureBasic 6.21's
native assembly backend. No Python was used, no inference-engine DLL was used
for the demo, and no physical board was accessed.

| Check | Result | What it establishes |
|---|---|---|
| Build helper and both application entry points | PASS | The exported source closure builds with PureBasic. |
| CLI bounded-wire self-test and target listing | PASS | Basic parser checks and all five registered profiles are callable. |
| Native demo-model creation | PASS | ONNX examples can be created without Python or downloaded assets. |
| Generated Windows Add model, three changing-length requests | PASS | Repeated resident execution, correct values, request cleanup, and final release. |
| Dense INT8 source generation for Windows, Pi 4, UNO Q, Pico, Pico 2 | PASS | Five-target emission and checked packed-weight output for one small eligible graph. |
| Internal Windows dynamic-runtime diagnostics | PASS | Changing shapes, exact INT64 behavior, broadcast/views, capacity and release checks, plus 42,210 clipped-window cases. |
| Internal Windows SIMD diagnostics | PASS | SSE2/AVX tails, threaded GEMM, convolution variants, transposed convolution, LSTM, and reduction checks. |

Internal diagnostics are not part of the public source export. The arithmetic
walkthrough is included as a small, reproducible integration example. These are
bounded checks, not proof of every ONNX graph or all target instructions.

## AArch64 native math checks — September 10, 2026

The Pi 4 and UNO Q runtime now uses native binary32 arithmetic for the common
`Exp` range, with the same reduction, polynomial coefficients and rounding
sequence as the retained software implementation. There is no fused
multiply-add contraction. Exceptional inputs, subnormal inputs and values
outside the conservative `-87..88` range retain the software path. Generated
applications must retain their floating-point startup contract.

A 791-input internal differential test covers both dispatch rails, special
bit patterns, reduction boundaries and ordinary values. All 723 admitted and
68 fallback cases agree bit-for-bit with the previous implementation in the
independent instruction interpreter. A separate high-precision oracle measured
a worst error of 1.167302 ULP on this corpus; that is a sample measurement, not
a universal accuracy bound. Five structural negative controls reject changes
to the fallback, rails, rounding, polynomial and no-fusion contract.

The complete 791-input probe subsequently passed on Pi 4 silicon too. All
three raw result arrays were read back and compared, including exceptional
bit patterns. The real startup path reported FPCR zero, both result and direct
stack guards were intact, and the payload returned normally. Over the same
723 admitted inputs, the software/native loops measured 38,344,086/404,017
counter ticks at 54 MHz. These are bounded loop measurements, not application
or model latency.

A complete small resident model (`Conv → Transpose → MatMul → Tanh/Sigmoid`)
also passed both interpreter and Pi 4 hardware checks. One bind served widths
16 and 24. All 640 output values per precision, request cleanup, final release
and stack guard were checked; the default host gate also rejects eight
deliberately damaged variants.

| Precision | Previous W16 / W24 | Native-math W16 / W24 | Maximum scalar-oracle error |
|---|---:|---:|---:|
| FP32 | 266.25 / 380.16 ms | 9.50 / 13.72 ms | 6.57e-8 |
| INT8 | 266.45 / 380.12 ms | 9.61 / 13.38 ms | 0.012081 |

These single-run Pi 4 intervals include input/shape setup, inference, numerical
validation and output copying, measured with the 54 MHz architected counter.
They are not inference-only timings, a Kokoro benchmark, or evidence that INT8
is faster than FP32. The returning RAM tests did not replace the board's boot
image. UNO Q hardware was not tested.

Generating the same small model with the previous and updated tools changed
only the AArch64 math include for Pi 4/UNO Q. Windows, Pico and Pico 2 output
closures were unchanged, after excluding only the manifest's absolute input
path from comparison. The private verification scripts are not required to
build or use this source-only product.

## Node-test coverage — September 16, 2026

The official ONNX backend node tests are the measuring stick for how much of
ONNX this compiler handles. Each case is one small model with its inputs and
expected outputs. A developer harness compiles every case for the `windows`
target, builds one generic driver around the emitted source, runs it with the
case's inputs and compares every output.

| | |
|---|---|
| Corpus | onnx 1.22.0 package, 1,765 node cases, sha256 `4955443be1453848b40f58dd92d8e709907cb73a0141cf17d0a9346cb863ce6b` |
| Scored | the 964 cases whose model imports ai.onnx at opset 20 or lower (786 above opset 20 and 15 with no ai.onnx import are excluded) |
| Compiler source | commit `6bd59b0` |
| Tolerance | rtol 1e-3, atol 1e-7 for floating outputs, as the onnx backend test loader assigns; exact for integer and bool; shape and element type must match |
| **Score** | **PASS 95 of 135 attempted (70.4%); PASS 95 of 964 scored (9.9%)** |

| Outcome | Cases | Meaning |
|---|---:|---|
| PASS | 95 | Every output matched. |
| FAIL_NUMERIC | 40 | The model compiled, built and ran, and an output was wrong. |
| REFUSED | 829 | The compiler declined the model with a sentence. A coverage gap, not a wrong answer. |
| FAIL_SHAPE, FAIL_DTYPE, BUILD_ERROR, RUN_ERROR, HARNESS_ERROR | 0 | |

"Attempted" means the compiler emitted source. The 40 wrong answers are eight
defects in operators the compiler accepts, each filed in the forum's ONNX bugs
area before any change: Clip with runtime bounds (topic 848), LayerNormalization
axis (850) and its Mean/InvStdDev outputs (851), Softmax axis (852), MatMul of
two 1-D tensors (853), Pow with an integer operand (854), ScatterND reduction
(855) and BatchNormalization training mode (856). Since then, all eight have
been fixed.
A ninth defect, Gemm without a bias producing source that did not
build (849), was fixed in `6bd59b0`; the same corpus scored 94 of 135 at
`583e912`, before that fix.

Of the 829 refusals, 471 name an operator that is not implemented, 157 are the
runtime-dimension path accepting only opset 20, 102 come from shape inference
inside function-expanded tests, 90 are element types (8/16-bit and unsigned
integers, INT32 arithmetic, strings, FP16), and the rest are opsets below 7,
attribute variants and MatMul rank forms. The per-operator table, every refusal
grouped by operator and every failure with its detail are in
[`tests/node_suite/results/2026-09-16-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-16-onnx-1.22.0.md),
with one record per case in the matching `.json`.

The results files named above now hold the latest measurement, at commit
`8b7d9df`, after those eight fixes and the operators added the same day:
**PASS 166 of 166 attempted (100.0%); PASS 166 of 964 scored (17.2%)**,
798 refused, no wrong answer and no build, run or harness error. No case that passed at
`6bd59b0` fails. The table above is the first measurement, kept as it was
measured.

### Running the harness

The harness is a developer check. It and the targeted cases beside it are the
only Python files in this repository, and building or using the compiler never
needs them. It needs the
onnx 1.22.0 package, which carries the corpus, and the x64 build compiler from
[Build](BUILD.md). From the repository root:

```powershell
py -3.12 -m pip install onnx==1.22.0
py -3.12 tests\node_suite\node_suite.py --pbcompiler "<install>\Compilers\pbcompiler.exe" --from-commit HEAD
py -3.12 tests\node_suite\node_suite.py --pbcompiler "<install>\Compilers\pbcompiler.exe" --from-commit HEAD --self-check
```

`--from-commit` builds the compiler from that commit's `src/` and `runtime/`,
so uncommitted edits are not measured. The first command writes
`tests/node_suite/results/<date>-onnx-<version>.json` and `.md`; `--cases`
with glob patterns runs a subset and writes nothing. A full run took about 35
seconds on 16 parallel jobs. The harness refuses to score if the installed
onnx version or the corpus hash differs from the pin, and refuses to write a
score if any case ends in HARNESS_ERROR.

`--self-check` proves the harness can report a failure. On `6bd59b0` all six
checks passed: an expected element nudged to ten times the tolerance became
FAIL_NUMERIC at that index; two same-shape outputs compared in swapped order
became FAIL_NUMERIC while the unswapped control passed; a model using `Erf`
was REFUSED with `Erf` named; and `test_add` built and run twice from nothing
produced identical records and identical output bytes.

This measures the Windows target only. The fixed-shape emission it exercises
is shared by all five targets, but no bare-metal build, board run or timing
is implied.

## Attribute forms — September 16, 2026

An attribute is refused only when its **value** selects behaviour the kernel
does not compute. Before this change the runtime-dimension path refused an
attribute because it was present, even when it carried the value the ONNX
specification defines as its default, and the fixed-shape path had no
attribute check at all: it read the attributes it implements and ignored the
rest. Both paths now share one check, and every refusal is a sentence that
names the operator, the node, the attribute and the value.

| Operator | Accepted | Refused, with a sentence |
|---|---|---|
| Cast | `to` FLOAT, INT64, BOOL (and INT32 on the runtime-dimension path); `saturate` with any value, since it only affects float8 casts | a float8 or other unimplemented `to` |
| Shape | `start` and `end`, negative values and clamping included, on both paths | — |
| LSTM | `layout` 0, `input_forget` 0, `activations` equal to the default Sigmoid/Tanh/Tanh per direction (with `activation_alpha`/`activation_beta`, which those take no part of), direction forward or bidirectional | `layout` 1, `input_forget` 1, `clip`, other activations, direction reverse, the peephole input |
| LayerNormalization | `stash_type` 1 | `stash_type` 0; runtime dimensions also still refuse an axis other than -1 and the Mean/InvStdDev outputs |
| Conv, ConvTranspose | `auto_pad` NOTSET, VALID, SAME_UPPER and SAME_LOWER, with the padding computed as the specification defines it | `auto_pad` combined with non-zero `pads`; ConvTranspose `output_shape`; on runtime dimensions, more than one spatial axis |
| Resize | `mode` nearest or linear; `coordinate_transformation_mode` half_pixel or asymmetric; `nearest_mode` floor for nearest (any value for linear, where it has no effect); `antialias` 0; `exclude_outside` 0; `keep_aspect_ratio_policy` stretch, or any value without a sizes input; `cubic_coeff_a`, `extrapolation_value` and a roi input, which have no effect in those modes | other modes and transformations, `nearest_mode` absent (its default round_prefer_floor) or other than floor for nearest, `antialias` 1, `exclude_outside` 1, `axes` |
| ScatterND | `reduction` none, add, mul, max and min | — |
| ReduceMean, ReduceSum | `noop_with_empty_axes` with any value when the axes are given | axes absent (a reduction over every axis, or with `noop_with_empty_axes` 1 the identity) |
| (model) | an opset import of another domain, such as `ai.onnx.ml`, that no node uses | a node in a domain other than `ai.onnx`, by operator and node name |

Three defects in accepted operators were found and fixed on the way, each filed
in the forum's ONNX bugs area first: a refusal decided inside the
runtime-dimension validator's operator block ended the compiler with an access
violation instead of a sentence (topic 858); a Cast from float to an integer
type rounded on Windows while the bare-metal targets truncate (860; it now
converts toward zero everywhere, as the reference does); and a zero-element
initializer or folded constant was refused as an allocation failure (861).
The fixed-shape path's missing attribute check is topic 859, and ScatterND's
ignored reduction is 855.

| Check | Result |
|---|---|
| Targeted default-attribute cases, `tests/node_suite/targeted_defaults.py`: 75 cases, 73 of them built both with declared extents (fixed-shape) and with a symbolic extent (runtime-dimension), 148 builds; expected outputs from the ONNX reference evaluator cross-checked against ONNX Runtime | 148 of 148 as expected, including every refusal case refused with its sentence. The same builds on commit `583e912`, before this change: 45 of 148 (72 refused, 25 wrong values, 5 accepted forms that should have been refused, 1 compiler crash) |
| Official node tests, the same corpus and harness as above, on the Clip fix plus this change | PASS 105 of 135 attempted (the Clip fix alone: 101; measured before the LayerNormalization axis fix, which moves no case to PASS on its own): `test_scatternd_add`, `_multiply`, `_max` and `_min` moved from FAIL_NUMERIC to PASS; no other case changed outcome |
| Every Kitten TTS nano 0.8 node of Cast, Shape, LSTM, LayerNormalization, Conv, ConvTranspose, Resize, ScatterND and ReduceSum (234 nodes), compiled alone with its attributes and runtime dimensions | emitted for Windows, Pi 4, UNO Q and Pico 2; Pico 233, one ConvTranspose refused because its 2.50 MiB weights exceed the Pico's flash |
| The new portable entries (Shape start/end, Conv and ConvTranspose SAME padding, ScatterND add/min on FLOAT and max/mul on INT64, Cast to INT64) built for the Pi 4 and executed as A64 code in an independent instruction interpreter | all nine outputs match the reference; the same model also builds for Pico and Pico 2 |
| Kokoro-82M generated for Windows and Pi 4 before and after | model source, weight pack and manifest byte-identical; the runtime closure gains the two new runtime files, one include line in each dynamic runtime, and the truncating float-to-integer store. The Windows waveform for the same request is byte-identical |

The default forms emit exactly the calls they emitted before, so a model that
was already accepted produces the same source. The new forms call their own
runtime entries (`DShapeRange`, `DConvSame`, `DConvTransposeSame`,
`DScatterReduce`), which compute what the attribute selects and hand the rest to
the kernels the default forms use, so no vector body changed.

## Random operators — September 16, 2026

`RandomNormal`, `RandomNormalLike`, `RandomUniform` and `RandomUniformLike`
are compiled on both paths for every target. The ONNX specification leaves the
generator unspecified and a reference runtime does not reproduce its own
values, so this compiler specifies one, and the contract is exact: **one seed,
one request number and one shape give the same FLOAT values on every
target.**

| Part | Contract |
|---|---|
| Generator | Threefry-2x32, 20 rounds (Random123 key schedule; JAX `threefry_2x32`). 32-bit add, rotate and exclusive-or only. |
| Node key | `Threefry(key=(S,0), counter=(node index, d))`; `S` = IEEE-754 bits of the node's `seed` attribute with `d=1`, else the 32-bit model seed (default 0) with `d=0`. The index is the node's 0-based position in the graph. |
| Element `i` | `Threefry(key=node key, counter=(i, request number))` gives `w0, w1`. The request number is 0 after bind or seed change and rises by one for every execution that starts the graph. |
| Uniform | `low + (high-low) * ((w0>>8) * 2^-24)`; in [low, high], below 1 for the defaults. |
| Normal | Box-Muller, cosine branch, in single correctly rounded binary32 operations with no library call: `ln` by exact power-of-two split and atanh series, `sqrt` by a fixed Newton sequence, cosine by octant fold and Taylor chains. `\|z\| <= 5.77`. |
| Seed API | `PmOnnxSetRandomSeed` / `PmModelSetRandomSeed`; 32 bits; a wider value is refused. Manifest entry `random`. |
| Comparison | `--random-inputs` (window: "Random nodes as model inputs (comparison)") makes every random node an extra model input, after the graph's inputs in node order; manifest `random.nodes[].input_index`. It changes the model's inputs. |
| Claimed forms | FLOAT output only (`dtype` 1, or a dtype-less Like node on FLOAT input); finite `mean`/`scale`/`low`/`high`; `shape` up to eight axes; up to 4,294,967,295 elements. Everything else is refused with a sentence naming the operator, attribute and value. |

The onnx 1.22.0 node-test corpus has no case for these operators at opset 20 or
lower (its three `test_bernoulli*_expanded` cases are opset 22), so the checks
are the lane's own, run from the private tooling tree without a board:

| Check | Result | What it establishes |
|---|---|---|
| Threefry-2x32-20 known answers (three published vectors) | PASS | The reference is Threefry. |
| Generator source on Windows: known answers, 832 crafted words, 6 streams, two 4,000,000-element streams | PASS, bit-identical | The kernel equals the reference in full on the host. |
| Same source, Pi 4 build in the A64 interpreter: known answers, 832 words, 6 streams | PASS, bit-identical | Same bits on AArch64 hardware binary32 (UNO Q shares the backend). |
| Same source, Pico and Pico 2 builds in the ARM emulator: known answers, 832 words, 6 streams | PASS, bit-identical | Same bits with software binary32 (Cortex-M0+) and on the Cortex-M33. |
| Statistics over 4,000,000 values | PASS | Mean and variance within five standard errors; Kolmogorov-Smirnov below twice the 5% critical value; ranges; seed, request, node and seed-source changes decorrelate the stream. |
| Six mutants of a copy of the generator | all rejected | The checks can fail. |
| Eight node-test-style cases (both paths) through the node-test harness, then bit for bit | PASS | Validation, emission, execution, shape and element type. |
| Nine unclaimed forms on windows, pi4 and pico | refused with sentences | No unclaimed form is compiled. |
| Request sequences with seed changes and a rebind, both APIs | PASS, bit-identical | Request numbering and the seed procedure. |
| `--random-inputs`, both paths | PASS | Supplied values pass through exactly; a wrong shape is refused. |
| Pi 4 fixed-shape and runtime-dimension builds in the A64 interpreter | PASS, bit-identical | The emitted Pi 4 source, not only the kernel. |
| Pico, Pico 2 and UNO Q sources, both paths and both modes | emitted and built | Generation and a downstream build; no hardware execution claimed. |
| Kitten TTS nano 0.8's two random nodes, extracted with their consumers | validate and emit for windows and pi4 in both modes; run on Windows | The model that needed them. |
| Kokoro source closure (windows `--speech-ui`, pi4 `--kokoro-text`), before and after | identical | Models without random nodes are unchanged. |

## Normalisation and small kernels — September 16, 2026

`InstanceNormalization`, `TopK`, `ScatterElements`, `ReduceMax`, `ReduceProd`
and `Not` are new on both paths, for every target. `Identity`, until now an
alias on the fixed-shape path only, also runs on the runtime-dimension path,
and `Pad` gains the `edge` and `wrap` modes, the `axes` input and negative
pads on both. Each kernel is written from the ONNX operator specification and
names the section it implements. These are the operators the Kitten TTS nano
0.8 voice needs that are kernels rather than control flow.

| Operator | Implemented | Refused, with a sentence naming operator, node and value |
|---|---|---|
| InstanceNormalization | FLOAT; input rank 3 or more (N x C x D1 ... Dn); `epsilon`; opset 6 on | other element types; rank below 3 |
| TopK | `K` as a one-element INT64 input (opset 10 on), or the `k` attribute (before opset 10, fixed-shape path); `axis`, negative included; `largest`; `sorted` (sorted order is written for `sorted` 0 too, which the specification leaves unordered); FLOAT and INT64 (and INT32 on the runtime-dimension path); equal values keep index order, lower index first; K from 0 to the axis extent | other element types (UINT64, FLOAT16 ...); a K outside 0 to the extent stops the request with an error |
| ScatterElements | `reduction` none, add and mul (opset 16 on), max and min (opset 18 on); negative indices; INT32 or INT64 indices; FLOAT, INT64 and BOOL data (and INT32 on the runtime-dimension path); FLOAT max and min propagate a NaN; BOOL max is or, min is and; updates in row-major order | an unknown reduction; add or mul on BOOL; a reduction before the opset that defines it; an index outside [-s, s-1] stops the request with an error |
| ReduceMax, ReduceProd | `axes` as an input (opset 18 on) or as an attribute (before); `keepdims`; `noop_with_empty_axes`; rank 0; reduction over an empty set gives minus infinity, the type's minimum or False (ReduceMax) and 1 (ReduceProd); FLOAT, INT64 and, for ReduceMax, BOOL (and INT32 on the runtime-dimension path); a FLOAT NaN wins ReduceMax | an axis named twice or out of range; the axes form of the other opset range |
| Not | BOOL, opset 1 on | other element types |
| Identity | FLOAT, INT8, INT32, INT64 and BOOL tensors on the runtime-dimension path; any tensor on the fixed-shape path | optional values; a sequence is carried from opset 14, with the [sequence operators](#control-flow-and-sequences--september-16-2026) |
| Pad | `mode` constant, reflect, edge and wrap (wrap from opset 19); the `axes` input (opset 18 on); negative pads, which crop first, after which reflect, edge and wrap work on the cropped data as numpy.pad does on the sliced array; reflect of a single element is that element; `constant_value` read when the program runs; Pad-2's attribute form on the fixed-shape path; FLOAT, INT64 and BOOL (and INT32 on the runtime-dimension path) | wrap before opset 19; axes before 18; a copying mode on an axis cropped to nothing |

**Opsets.** On the runtime-dimension path each of these operators is accepted
from the oldest opset whose definition of it is the one implemented:
InstanceNormalization 6, TopK 10, ScatterElements 11, Pad 11, and ReduceMax,
ReduceProd, Not and Identity 1. Every other operator has the floor listed under
[control flow and sequences](#control-flow-and-sequences--september-16-2026). The fixed-shape path's audited range of 7 to 23 is widened below 7
only for a graph made entirely of these operators at opsets whose definitions
they keep, which is what lets the three official `Not` tests, written at opset
1, run.

**InstanceNormalization's numerical contract.** The reference defines the
order of every operation: the mean is four running sums, element k of a plane
going to sum k mod 4 in index order, combined as (s0 + s1) + (s2 + s3) and
divided by the element count; the variance is the same over (x - mean)^2; the
denominator is the correctly rounded square root of variance + epsilon; and
each output is `t = x - mean`, `t = scale * t`, `t = t / den`, `y = t + B`, one
binary32 operation per step. The SSE body on Windows and the Advanced SIMD body
on the Pi 4 and UNO Q perform exactly those operations four elements at a time,
so they are **bit-identical to the reference**, NaN payloads included. The
four-core convolution split can carry this operator, since channels are
independent, but no split was added: no timing has been measured on the board.

Two defects in the existing `Pad` were found and fixed on the way, each filed
first: the fixed-shape path padded with zero when `constant_value` arrived while
the program ran (topic 865), and reflect on the runtime-dimension path mirrored
into elements a negative pad had removed and refused a one-element axis (866).

| Check | Result |
|---|---|
| Official node tests at opset 20 or lower, through the harness | ReduceMax 10 of 10, ReduceProd 9 of 9, ScatterElements 7 of 7, Not 3 of 3 PASS; `test_identity_opt` (an optional value) refused |
| Official node tests published only above opset 20, re-imported at opset 20 by `tests/node_suite/targeted_norm_small.py` (the later definitions add element types only) | InstanceNormalization 2 of 2, TopK 6 of 6, Pad 6 of 6 and Identity 1 of 1 PASS; `test_top_k_uint64` refused with a sentence; `test_identity_sequence` passes with the [sequence operators](#control-flow-and-sequences--september-16-2026) |
| Targeted cases in the same script, each built with declared extents and with a symbolic extent: the Kitten forms, every width 1 to 9 and 130 for InstanceNormalization, TopK ties and K 0, every ScatterElements reduction on FLOAT, INT64 and BOOL, multi-axis and no-op reductions, the attribute form at opset 13, NaN in ReduceMax, every Pad mode on every axis and with negative pads, wraps longer than the axis, and two refusals | 115 builds of 63 cases, all as expected: every accepted form matches, both refusals name the operator and the value |
| SSE InstanceNormalization against the scalar reference on planes of every width 1 to 41 holding signed zeros, infinities, subnormals, the largest finite values and NaNs with payloads, with NaN, -0.0 and infinite scales and biases | 0 of 14,738 elements differ; a one-unit change of epsilon moves 410; the reference equals a float32 transcription of the defined order on 9,572 ordinary elements |
| Pi 4 build of a model using every one of these operators, run as A64 code in an independent instruction interpreter | all 16 outputs match the reference evaluator (InstanceNormalization bit for bit against the defined order); the Advanced SIMD body against the scalar reference: 0 of 5,166 elements differ on the special-value planes; removing one accumulation instruction from either vector body is detected (1,710 and 1,821 elements) |
| The same model built for Pico and Pico 2 and run in the ARM emulator | all 16 outputs match on both parts, exercising the 32-bit word arrays and the checked INT64 range |
| Every Kitten TTS nano 0.8 node of these operators (57 InstanceNormalization, 2 Identity, 2 Pad, and one each of TopK, ScatterElements, ReduceMax, ReduceProd and Not), compiled alone with the voice's attributes and ranks | emitted for Windows, Pi 4, UNO Q, Pico and Pico 2, and every one of the five sources builds |
| Kokoro-82M generated for Windows and Pi 4 before and after | Windows model source, both weight packs and both manifests byte-identical; the Pi 4 model source gains two include lines; the runtime closure gains the four new runtime files, and `DPad` moves out of both dynamic runtimes into them unchanged in its results. The Windows waveform for the same request is byte-identical |


## Control flow and sequences — September 16, 2026

`If`, `Loop`, `SequenceEmpty`, `SequenceConstruct`, `SequenceInsert`,
`SequenceAt`, `SequenceLength`, `SplitToSequence` and `ConcatFromSequence`
compile on the runtime-dimension path for every target. A graph that contains
any of them, or that has a sequence input or output, takes that path even when
every extent is fixed: a trip count or a branch decides shapes while the
program runs, which the fixed-shape planner cannot express. On that path a
`Constant` node is compiled as an initializer.

| Part | Contract |
|---|---|
| Subgraphs | A branch or loop body reads values of the enclosing graphs by name. A name a subgraph defines that the model also defines elsewhere is renamed before emission, so the innermost definition is the one read. Subgraph initializers are packed with the weights. Nesting is limited to 16 levels. Each subgraph becomes one generated procedure. |
| `If` | `cond` is a BOOL tensor holding exactly one element. The two branches may return different shapes and different ranks. |
| `Loop` | `M` is an INT64 tensor holding one element, `cond` a BOOL tensor holding one element; either may be absent, not both. The iteration number is fed as an INT64 scalar. A carried value is handed to the next iteration without a copy when nothing else reads it. Scan outputs are stacked along a new first axis. After zero iterations a scan output is `[0]` followed by the body's declared extents, and only when every one of them is declared; otherwise the request is refused, because ONNX defines no shape for it. |
| Sequences | Elements are FLOAT, INT32, INT64 or BOOL tensors. A sequence is one block: a header, one 48-byte record per element and the element data. On Windows the block comes from the checked heap, on the bare-metal targets from the caller's bound arena through the same allocator the tensors use. Capacity is exact where the element count is known when the block is made (`SplitToSequence`, `SequenceConstruct`, a copy) and grows geometrically otherwise; growth past the working memory is refused and nothing is truncated. `SequenceInsert` takes its input sequence over, instead of copying it, when nothing reads that input afterwards. An `Identity` whose input is a sequence (opset 14 and later) hands it on in the same way, or copies it when it is read again. |
| Sequence inputs and outputs | Supplied with `DSeqReset(id, kind)` and `DSeqAppendShape(id, kind, rank, *dims)`, which returns the address to fill; read with `DSeqCount`, `DSeqItemRank`, `DSeqItemDim`, `DSeqItemCount`, `DSeqItemBytes` and `DSeqItemData`. |
| Opsets | Every node on the runtime-dimension path, subgraphs included, is accepted from its operator's floor through opset 20: the oldest opset whose definition is the one the kernel computes. Until now that path took opset 20 only, apart from the operators in [normalisation and small kernels](#normalisation-and-small-kernels--september-16-2026). Floors: `Shape`, `Transpose`, `MatMul` 1; `Reshape` 5; `Exp`, `Log`, `Sqrt`, `Abs`, `Neg`, `Floor`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Cast` 6; `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Equal`, `Greater`, `Less`, `And`, `Sin`, `Cos`, `Atan`, `LSTM` 7; `Expand` 8; `ConstantOfShape`, `Where`, `NonZero` 9; `Slice` 10; `Gather`, `Concat`, `Range`, `Round`, `CumSum`, `ScatterND`, `Squeeze`, `Unsqueeze`, `Clip`, `Resize`, `Gemm`, `ReduceMean`, `ReduceSum` and the If, Loop and sequence operators 11; `GreaterOrEqual` 12; `Softmax` 13 (Softmax-11 flattens around the axis instead); `LayerNormalization` and `STFT` 17; any operator not listed, 20. Older attribute spellings stay refused by the attribute checks, except `axes` as an attribute of `Squeeze` and `Unsqueeze` (opset 11), which is compiled as the opset-13 input. |
| Refused, with a sentence | `Loop` with neither `M` nor `cond` (the specification defines it as a loop that never ends); a sequence as a `Loop` scan output; sequences of sequences, maps and optional values, and the Optional operators; other sequence element types; `SequenceErase`, `SequenceMap` in its unexpanded form, and `Scan`; a `Constant` holding strings or a sparse tensor. |

| Check | Result |
|---|---|
| Official node tests for these operators at opset 20 or lower, with the harness now carrying sequence inputs and outputs | 12 of 12 attempted PASS: `test_if`, `test_if_seq`, `test_loop11`, `test_loop13_seq`, `test_sequence_insert_at_back`, `_at_front`, and the six `test_sequence_map_*_expanded` cases. Refused by name: `test_if_opt` and `test_loop16_seq_none` (Optional) and the six unexpanded `SequenceMap` cases. The `SplitToSequence` node tests are published at opset 24 and are scored below |
| The whole corpus on `e0b46a4`, without and with this change | PASS 158 of 166 attempted without, 194 of 251 with. No case changed outcome except refusals: the twelve cases above and 24 others the opset floors now admit (the expanded window functions, `Expand`, `NonZero`, `Slice`, expanded `ReduceL1`) pass; 47 now build and stop while running, each with a sentence, because the runtime-dimension validator admits forms its kernels refuse only then (`CumSum` on DOUBLE or INT32, reductions over an axis other than the last or over every axis, `Resize` over more than the last axis; forum topic 869, and the same at opset 20); `test_stft` and `test_stft_with_window` end FAIL_NUMERIC on the imaginary part of the DC and Nyquist bins, round-off near 6e-4 on magnitudes near 2e3 where exactly zero is expected (topic 870) |
| Targeted cases, `tests/node_suite/targeted_control.py`: the three `SplitToSequence` node tests and `test_identity_sequence` re-imported at opset 20 and scored against their official outputs; every split and concatenation form; `If` branches of different rank; `Loop` as a while loop, with the condition false at entry, with zero trips, with carried values that trade places, nested with an `If` reading values two scopes out and a body name the outer graph defines again, and growing a sequence over 300 trips; `Identity` of a sequence; eight refusals and two refusals while running; and the Kitten cases below. Expected outputs from ONNX Runtime, cross-checked against the ONNX reference evaluator where it runs the model | 34 of 34 as expected |
| Kitten TTS nano 0.8's own nodes: the repeat-interleave region (`SequenceEmpty`, both `SplitToSequence`, the `Loop` and its 12-node body, `ConcatFromSequence`) cut out unchanged, and both `If` nodes, each run both ways | The Loop region's output equals `numpy.repeat(arange(T), durations)` exactly; all four `If` runs pass |
| The same cases built for the Pi 4 and executed as A64 code in an independent instruction interpreter, sequence inputs and outputs included | 37 of 37 match the expected outputs: every node test above, the Kitten cases (the Loop region takes 1.93 million instructions) and the targeted cases (the 300-trip sequence peaks at 1.09 MiB of arena) |
| Pico, Pico 2 and UNO Q | sources emitted and built for a nested Loop/If model and a sequence model; no execution claimed |
| Kokoro-82M generated for Windows and Pi 4 with and without this change, on the same commit | source closure identical, file for file |
| Windows dynamic-runtime and SIMD self-tests | PASS |

## INT8 — September 23, 2026

INT8 now quantizes activations with one scale per row (per token, per input
position across the channels of a convolution, per recurrent step), and a
model may carry a measured precision plan of wide weights (two INT8 planes,
16-bit activations). The pinned Kokoro-82M graph has one; every other model
is narrow. The arithmetic is defined step by step, each step exact in
integers or one binary32 operation, and every target is held to the same
bits. [Chapter 4 of the book](guide/04_reading_a_model.txt) states the
definition; [chapter 8](guide/08_how_it_is_proved.txt) how it is checked.

| Check | Result |
|---|---|
| Full Kokoro-82M at INT8 on Windows (i9-14900HX), reference input, against ONNX Runtime FP32 on the pinned graph | 0 of 143 durations changed; log-mel 0.588 dB (this tool's FP32: 0.294; FP16 storage: 0.442; the earlier INT8 scheme: 9 of 143, about 5.2 dB) |
| The same, a 295-token and a 69-token sentence | 0 of 295, 0.647 dB (FP16 storage: 1 of 295, 1.835 dB); 0 of 69, 0.442 dB |
| Kokoro weight pack at INT8 | 119,358,848 bytes, 183 quantized weights of which 59 wide |
| Windows speed, reference request, the two builds alternated, five rounds | FP32 5.51 s, INT8 4.47 s (medians); INT8 1.23 times faster |
| 51 kernel cases (MatMul, Gemm in four transpose forms, Conv1D with pads, strides to 6, dilation and groups, Conv2D, LSTM gate rows; narrow and wide; rows of zeros, below 1e-30, subnormal, exactly halfway) | Windows with AVX2, without it, and a numpy statement of the definition: bit-identical. Pi 4 (Advanced SIMD, run as A64 code), Pico and Pico 2 (run as Thumb code) in an instruction emulator, NaN-filled working memory: bit-identical. NaN/infinity and one-byte-short working memory refused |
| LSTM end to end | gate rows bit-identical on every target; outputs within 1.5e-7 of Windows relative to the largest (its sigmoid and tanh are each target's FP32 functions; the same layer in FP32: 2.5e-7) |
| Runtime mutants (rounding, chunk length, epilogue order, padding scales, NaN refusal, vector-body faults, stride phases, Windows plain path) | 17 of 17 caught; the four-core split: 4 of 4 |
| Pi 4 four-core INT8 convolution | whole = split in one to four partitions, each partition writes only its own channels, the vector tile runs and no FP32 tile after an INT8 weight is bound |
| A model with every INT8 operator form, fixed-shape and runtime-dimension, generated and built for Windows, Pi 4, Pico, Pico 2 | outputs bit-identical to Windows on every target (LSTM output within 1.8e-7) |
| Nine nodes of a real Kokoro request (wide text-encoder convolution, ALBERT FFN, stride-6 wide convolution, a 7.1-billion-MAC generator convolution, LSTM, Gemm) re-run on the Pi 4 build whole and on the Pico / Pico 2 builds cut to their first channels, rows or steps | every output bit equal to the Windows run (LSTM within 8.3e-6) |
| Pi 4 instruction count, real Kokoro convolutions (emulator; no cycle model) | 0.38-0.41 A64 instructions per multiply-accumulate on narrow layers against 0.63-0.68 for FP32; 0.88 against 0.82 wide; 1.58 times fewer over four layers |
| Pico and Pico 2, the gate model, the parts' own timers in the ARM emulator's cycle model | INT8 2.6-2.9 times faster than FP32 |
| Models without INT8 (four models, fp32/fp16/bf16/int4, five targets), this change against the previous compiler | 80 of 80 emitted sources, packs and manifests byte-identical; 32 of 32 fixed-shape PureMetal images byte-identical; Kokoro FP32 on Windows: source, pack and both waveforms byte-identical |
| Node-test corpus, this change against the previous compiler | identical outcomes, case for case |
| Full Kokoro at INT8 for the Pi 4 | generated and built into a payload image (1,512,760 bytes) carrying the INT8 vector kernels |

**Not established:** any board run. The Pi 4 speed and audio of this scheme
are to be measured on the board; the Pico cannot hold the speech model.

## Absent optional outputs — September 23, 2026

Forum topic 988: an LSTM whose only output is Y was accepted by the
fixed-shape path and emitted as `PmOnnxLstm\YH = ` with nothing after it, so
the generated program did not build; the runtime-dimension path refused the
same node. LSTM's kernels keep the running hidden and cell state in Y_h and
Y_c and build Y_h from Y, so a left-out LSTM output now gets a name no model
can spell (`__pmo_absent_<node>_<position>`) and, on the fixed-shape path, its
declared shape, before either path plans memory; nothing reads it and the
model's own outputs are unchanged. Every other left-out position is the null
address `0` on the fixed-shape path, as an empty name already was.

| Check | Result |
|---|---|
| `tests/node_suite/targeted_optional_outputs.py`: LSTM with Y only (with and without B), Y_h only, Y and Y_c, Y and Y_h; LayerNormalization with Y only and with InvStdDev only; BatchNormalization-15 in training mode with Y only; each fixed-shape and, where that path has the form, runtime-dimension (14 builds). Expected outputs from the ONNX reference evaluator cross-checked against ONNX Runtime, ONNX Runtime alone for Y_c (the reference evaluator's LSTM has no Y_c) | 14 of 14 as expected; the BatchNormalization case is refused with its sentence (the operator's own shape inference and ONNX Runtime require all three outputs in training mode). The released compiler (`e9df233`): 4 of 14 (three fixed-shape programs that did not build, one that crashed with Y_h named empty, one runtime-dimension run that failed, five refusals) |
| The same gate with `--mutants`: the correction taken out of each path in turn and the compiler rebuilt | 2 of 2 caught (5 of 7 fixed-shape and 5 of 6 runtime-dimension cases fail) |
| Models with no left-out output (four models, fp32/fp16/bf16/int4/int8, five targets), this change against the previous compiler | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape PureMetal images byte-identical |

## Operator set, first group — September 24, 2026

Fifty-two operators are new on both paths and for every target: the
elementwise functions Erf, Reciprocal, Ceil, Sign, Softplus, Softsign, Elu,
Selu, Celu, HardSigmoid, HardSwish, Mish, Gelu, ThresholdedRelu, Shrink,
IsNaN and IsInf; Min, Max, Sum and Mean over any number of inputs, Mod,
PRelu, Or and Xor with numpy broadcasting; ReduceMin, ReduceL1, ReduceL2,
ReduceSumSquare, ReduceLogSum and ReduceLogSumExp; ArgMax, ArgMin and
LogSoftmax; MaxPool (with Indices), AveragePool, LpPool and the three
global pools over one to three spatial axes; Split, Tile, DepthToSpace,
SpaceToDepth, Trilu, GatherElements, GatherND and OneHot; Einsum; Dropout in
inference; CastLike; and Size. ReduceSum and ReduceMean on the
runtime-dimension path now reduce over any set of axes, the empty set with
`noop_with_empty_axes` included, where they took one final axis.

**One kernel source, the same bits everywhere.** The kernels are one file,
`runtime/tensor_ops.pmi`, included unchanged by the Windows program and by
the programs for the Pi 4, the UNO Q, the Pico and the Pico 2, on both
paths. Every floating-point step is one binary32 operation, and the
functions the kernels need - exp, expm1, log, log1p, tanh, erf and a
correctly rounded sqrt - are written into the file as fixed sequences of
such operations rather than taken from each target's library. Two
properties of the host compiler had to be designed around: its float
comparisons do not order a NaN the way IEEE 754 does (a NaN compares below
and equal to everything), and it adds -0 and -0 to +0. So every kernel
decides NaN on the bits before it compares, adds through a helper where a
negative zero can arise, and writes every NaN it produces as the one quiet
NaN `$7FC00000` (the x87 and the Arm cores make different NaNs for an
invalid operation). The definition is `tools/onnx/tests/Diagnostics/ops_scheme.py`
in the compiler repository, the kernels written out in numpy one binary32
operation per line.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 189 kernel cases - every math function over tables with signed zeros, subnormals, infinities, NaN and every branch boundary; every elementwise operator; Min/Max/Sum/Mean with one to three broadcast inputs and NaNs; Mod both ways and by zero; every reduction, including empty axes; ArgMax/ArgMin ties and NaN; LogSoftmax on every axis; 17 pooling forms in 1, 2 and 3 dimensions; permutation, Tile, Split, Trilu, GatherElements, GatherND, OneHot, nine Einsum equations - against the numpy definition | Windows, Pi 4 (A64 in unicorn), Pico and Pico 2 (Thumb in unicorn): 189 of 189 bit-identical on every target |
| The same with `--mutants`: 18 planted defects (a shorter Taylor series, the lost low part of ln 2, a missing rounding step, a NaN that stops winning, broadcasting, pooling, index and Einsum faults) | 18 of 18 caught |
| Accuracy of the math functions against the true function, 80,000 arguments each | exp 1.1, expm1 1.9, log 2.6, log1p 3.8, tanh 1.9, erf 0.8 units in the last place at most; sqrt correctly rounded (50,000 of 50,000) |
| `tests/node_suite/targeted_ops.py`: the official node tests of these operators published only above opset 20, re-imported at opset 20; and 119 targeted forms (every attribute, both opset forms of the reductions and of Split, broadcasting, ties, NaN, the pooling shape rules, all four Split forms, Einsum with a diagonal, implicit output and ellipsis), each built with declared extents and with a symbolic extent; expected outputs from the ONNX reference evaluator, cross-checked against ONNX Runtime | 375 of 375 as expected. Re-imported official cases: 75 PASS; 65 refused for their element type (FLOAT16, BFLOAT16, DOUBLE, the float8, 4-bit and 2-bit types, UINT8); 2 function-expanded cases whose intermediate values carry no shape. Every refusal case is refused with its sentence |
| Official node tests at opset 20 or lower (the corpus and harness of the node-test coverage section), this change against the previous compiler | PASS 202 of 964 before, 442 after (442 of 455 attempted); no case that passed fails; 36 fewer cases stop while running (DReduce's any-axes form), none newly fails. Every case: [`tests/node_suite/results/2026-09-24-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `49145a7` |
| `ops_targets_gate.py` (compiler repository, beside the kernel check): the 224 targeted builds that must pass, compiled for the Pi 4, Pico and Pico 2 by the path the model selects, the generated programs run in unicorn (A64; Thumb) | 672 of 672 as expected: every output within the node-suite tolerance of the reference and bit-identical to the Windows program's; the INT64 forms the 32-bit targets' contract cannot prove in range (a CastLike to INT64, INT64 ReduceL1 and ReduceSumSquare) refused on the Pico and Pico 2 with the contract's sentence |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4/int8, five targets) | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape PureMetal images byte-identical; the kernels are included only where a node uses them |
| Kokoro-82M FP32 for Windows, reference, 295-token and 69-token requests | source and pack byte-identical; all three waveforms byte-identical |

**Refused, with a sentence naming operator, node and value:** element types
outside FLOAT, INT32 (runtime-dimension path), INT64 and BOOL; Mod with
fmod = 0 on FLOAT; Gelu approximations other than none and tanh; LogSoftmax
before opset 13 on an axis other than the last (its older definition
flattens); Dropout in training mode with a nonzero ratio (it draws random
masks); an Einsum ellipsis on the runtime-dimension path (the operand ranks
are not known when the source is written); pooling with explicit pads and
an auto_pad; and on the Pico and Pico 2, an INT64 Sum, PRelu, ReduceL1 or
ReduceSumSquare whose result the 32-bit INT64 contract cannot prove in range.

## Operator set, second group — September 24, 2026

Twenty-six more operators, on both paths and for every target unless noted:
Tan, Asin, Acos, Sinh, Cosh, Asinh, Acosh and Atanh; BitwiseNot,
BitwiseAnd, BitwiseOr and BitwiseXor on INT32 and INT64; Hardmax,
LpNormalization, MeanVarianceNormalization, LRN and GroupNormalization
(opset 18 to 20); EyeLike and Det; Compress and ReverseSequence; RNN and GRU
(forward, reverse and bidirectional, sequence lengths, initial state, clip,
the Sigmoid, Tanh and Relu activations, GRU's linear_before_reset);
RoiAlign (avg and max, half_pixel and output_half_pixel);
NonMaxSuppression, always on the runtime-dimension path because its output
size depends on the scores; and Upsample, opset 7 to 9, fixed shapes only
(nearest on any axes, linear on the last two). Their kernels join the first group's
in `runtime/tensor_ops.pmi` under the same three rules. tan reduces its
argument by pi/2 in integer arithmetic against 288 bits of 2/pi, so a large
argument keeps its quadrant on every target; asin and acos use a fitted
polynomial; the hyperbolic functions are built on the file's exp,
expm1, log and log1p.

A node whose first output is left out (a GRU or RNN listing only `Y_h`)
now builds on the fixed-shape path: the generated procedure receives the
null address for it, as forum 988 made it for the other positions.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 268 kernel cases - the first group's 189, and tan, asin, acos and the hyperbolic functions over special values, branch boundaries and large arguments; the bitwise operators; Hardmax with a tie; every normalization on several axes; LRN with even and odd sizes; Det with pivoting and singular matrices; RNN and GRU in every direction with lengths; NonMaxSuppression with both box formats; RoiAlign in both modes; Upsample - against the numpy definition | Windows, Pi 4 (A64 in unicorn), Pico and Pico 2 (Thumb in unicorn): 268 of 268 bit-identical on every target |
| The same with `--mutants`: 28 planted defects, the first group's 18 and ten more (tan's reduction started at the wrong limb, the quadrant rule, a shorter asin polynomial, LRN's scale, Det's sign on a swap, EyeLike's diagonal, GRU's update gate, NonMaxSuppression's union, RoiAlign's interpolation, Upsample's scale) | 28 of 28 caught |
| Accuracy of the new math functions against the true function, 80,000 arguments each | tan 3.9, asin 1.8, acos 1.2, sinh 1.8, cosh 1.5, asinh 4.2, acosh 1.1, atanh 4.1 units in the last place at most |
| `tests/node_suite/targeted_ops.py`: 512 cases - the first group's; the official cases of the second group's operators published only above opset 20, re-imported at opset 20; and 99 new builds of the second group's forms with declared extents and with a symbolic extent; expected outputs from the ONNX reference evaluator cross-checked against ONNX Runtime; ONNX Runtime alone where the reference cannot run a form (RNN and GRU with lengths or both directions, NonMaxSuppression without thresholds); a numpy definition, cross-checked against ONNX Runtime where it runs, where the reference lacks a form or answers otherwise than the definition (GroupNormalization-18, Upsample-7, LpNormalization, LRN, bilinear Upsample) | 512 of 512 as expected. Re-imported official cases: 109 PASS (75 before); 67 refused for their element type; 2 function-expanded cases whose intermediate values carry no shape; RNN and GRU in batch-first layout refused with their sentence. Every refusal case is refused with its sentence |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 442 of 964 before, 468 after (468 of 481 attempted); no case that passed fails. The 26 new passes are the Compress, Hardmax, LRN, MeanVarianceNormalization, NonMaxSuppression and ReverseSequence cases; the official cases of the other new operators are published above opset 20 and are scored by `targeted_ops.py` above. The official bitwise cases are over INT8, INT16, INT32 and the unsigned types with declared extents and stay refused for their element type. Every case: [`tests/node_suite/results/2026-09-24b-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24b-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `19d5028` |
| `ops_targets_gate.py`: the 320 targeted builds of both groups that must pass, compiled for the Pi 4, Pico and Pico 2 and run in unicorn | 960 of 960 as expected: every output within the node-suite tolerance of the reference and bit-identical to the Windows program's; on the Pico and Pico 2, besides the first group's compile-time refusals, the bitwise forms whose INT64 inputs hold values beyond 32 bits refused at the request by the contract's check (their twins with values inside the range run and match) |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `a2aecab` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape PureMetal images byte-identical |
| Kokoro-82M FP32 for Windows | source and pack byte-identical to the previous compiler's; the reference request's waveform byte-identical |

**Refused, with a sentence naming operator, node and value:** Upsample on
the runtime-dimension path and from opset 10 (deprecated for Resize), with
a scale below 1, or linear on an axis before the last two;
GroupNormalization from opset 21 (its scale becomes per channel); Hardmax
before opset 13 on an axis other than the last; LpNormalization with p
other than 1 and 2; RNN and GRU with activation_alpha or activation_beta,
activations other than Sigmoid, Tanh and Relu, or layout 1; RoiAlign
coordinate modes and pooling modes outside those named above; EyeLike to
an element type other than FLOAT, INT64 and BOOL on the fixed-shape path;
Compress on the fixed-shape path with a condition that is not a constant
(the runtime-dimension path takes it); and, on the Pico and Pico 2, a
bitwise operator's INT64 input outside the 32-bit range, refused at the
request by the contract's check (on the fixed-shape path and for the
two-input operators; BitwiseNot on the runtime-dimension path carries all
64 bits).

## GridSample and Scatter — September 24, 2026

GridSample-16 and GridSample-20 on both paths and every target: the modes
nearest, linear and cubic (bilinear and bicubic at opset 16), the paddings
zeros, border and reflection, align_corners, over one to three spatial axes
(opset 20; two at opset 16), cubic over two. Its kernel joins
`runtime/tensor_ops.pmi` under the same rules: the coordinate arithmetic is
single binary32 operations, nearest rounds half to even as the reference
does, and the taps are summed innermost axis first. Where the reference
evaluator and ONNX Runtime disagree - cubic with border padding, where the
reference evaluator clamps the sampling coordinate as well as each tap -
the kernel follows ONNX Runtime and PyTorch, which clamp only the taps. A
NaN grid coordinate is out of range under zeros padding and the first
element's position on its axis under the others, and an infinite one is
treated as 2^22 (reflection: the low border); the reference evaluator
raises on both, so these are the kernel's own rules.

Scatter, the deprecated operator of opsets 9 and 10, is read as
ScatterElements without a reduction on both paths.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 310 cases - the 268 before, and 42 GridSample forms (one, two and three spatial axes; every mode, padding and align_corners; NaN and infinite grid coordinates, coordinates beyond 2^22, a NaN in X) | Windows, Pi 4, Pico and Pico 2: 310 of 310 bit-identical to the definition; `--mutants` 32 of 32 caught (four new: the cubic coefficient, reflection's parity, the align_corners extent, round half to even) |
| `tests/node_suite/targeted_ops.py`: 598 cases - the 512 before; the 18 official GridSample cases, published at opset 22, re-imported at opset 20; 64 new GridSample builds and 4 Scatter builds with declared extents and a symbolic extent. Expected outputs from the ONNX reference evaluator cross-checked against ONNX Runtime; ONNX Runtime alone for cubic with border padding; the opset-16 modes through the reference evaluator's opset-20 node | 598 of 598 as expected; re-imported official cases 127 PASS (109 before), every GridSample case among them |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 468 of 964 before, 470 after (test_scatter_with_axis and test_scatter_without_axis); no case that passed fails. Every case: [`tests/node_suite/results/2026-09-24c-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24c-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `4de6b7e` |
| `ops_targets_gate.py` on the 62 new builds that must pass, compiled for the Pi 4, Pico and Pico 2 and run in unicorn | 186 of 186: within the node-suite tolerance and bit-identical to the Windows program's |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `d6aa9d7` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical |
| Kokoro-82M FP32 for Windows | source and pack byte-identical to the previous compiler's; the reference request's waveform byte-identical |

**Refused, with a sentence:** GridSample-16's modes under their GridSample-20
names and the reverse; a GridSample-16 input that is not 4-D; four or more
spatial axes; cubic over other than two; element types other than FLOAT;
Scatter from opset 11, where it is deprecated for ScatterElements.

## The quantized operators — September 24, 2026

QuantizeLinear and DequantizeLinear (opset 10 to 20, per tensor and, from
13, per axis), DynamicQuantizeLinear, MatMulInteger, QLinearMatMul,
ConvInteger and QLinearConv, on both paths and every target. They are the
operators of a model quantized before export, and they bring UINT8 and INT8
tensors - graph inputs, outputs, initializers and values - to both paths,
and INT32 values to the fixed-shape path. Cast converts between UINT8, INT8,
INT32, INT64, FLOAT and BOOL on both paths (integer conversions keep the low
bits, as numpy does; a float converts toward zero first). On the fixed-shape
path a UINT8, INT8 or INT32 value is produced only by these operators, Cast
and the operators that rename (Identity, Reshape, Flatten, Squeeze,
Unsqueeze); an operator that would compute one otherwise is refused with a
sentence naming the type.

The kernels join `runtime/tensor_ops.pmi`. QuantizeLinear divides and rounds
half to even in binary32, as the reference does. QLinearMatMul and
QLinearConv requantize as the reference does - the INT32 accumulator times
the binary32 multiplier `scale_a * scale_b / scale_y` in binary64, the zero
point added in binary64, rounded half to even - and binary64 is not an
operation the file may use, so the kernel reproduces it exactly: the
product is formed in 12-bit integer digits, rounded to 53 bits as binary64
rounds it, and the sum's own rounding is applied where it can move the
result, within a hair of a half. An INT32 is converted to binary32 on the
bits, ties to even, so every target converts one above 2^24 the same way.
Where the reference's answer is not defined, the kernel defines it: a NaN
quantizes to the low end of the range (what the reference's numpy gives on
x86); infinities saturate; DynamicQuantizeLinear refuses a NaN or an
infinite input at run time.

A defect found on the way, and fixed: on the Pico and Pico 2 the
runtime-dimension runtime converted an integer to binary32 by returning it
from a binary32 procedure, which returns its bits, not its value; a Cast
from an integer type to FLOAT there gave denormals. An assignment converts.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 386 cases - the 310 before; QuantizeLinear per tensor and per axis over special values and halves; DequantizeLinear of UINT8, INT8 and INT32 (above 2^24 too); DynamicQuantizeLinear including all-zero, all-negative and NaN inputs; the requantization over 756 accumulator, multiplier and zero-point triples, to UINT8 and to INT8, where 39 and 29 of the results differ from the exactly rounded ones because binary64 rounds twice - the kernel must reproduce each; the INT32 conversion; MatMulInteger and QLinearMatMul with batch broadcasting, 1-D operands and per-column zero points and scales; ConvInteger and QLinearConv over one to three spatial axes with groups, strides, dilations, pads, bias and per-channel zero points and scales | Windows, Pi 4, Pico and Pico 2: 386 of 386 bit-identical to the definition; `--mutants` 40 of 40 caught (eight new, among them the product's 53-bit rounding and the sum's rounding window) |
| `tests/node_suite/targeted_ops.py`: 665 cases - the 598 before; the official QuantizeLinear, DequantizeLinear and QLinearMatMul cases published above opset 20, re-imported at 20; 53 new builds of the quantized operators and of Cast | 665 of 665 as expected; re-imported official cases 135 PASS (127 before), every QuantizeLinear, DequantizeLinear and QLinearMatMul case over UINT8 and INT8 among them; the float8 forms refused for their element type |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 470 of 964 before, 490 after: ConvInteger (2), DynamicQuantizeLinear (3), MatMulInteger, QLinearConv, and 13 cases the new element types admit: Equal, Greater, GreaterOrEqual, Less and LessOrEqual over INT8, UINT8 and INT32 inputs, and a function-expanded Clip over INT8; no case that passed fails. Every case: [`tests/node_suite/results/2026-09-24d-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24d-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `08fae2c` |
| `ops_targets_gate.py`: the 432 targeted builds that must pass, all groups, compiled for the Pi 4, Pico and Pico 2 and run in unicorn | 1,296 of 1,296 as expected, bit-identical to the Windows program; it found the defect below first |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `5148bac` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical. The runtime-dimension support files differ: they read and write UINT8 and INT8 |
| Kokoro-82M FP32 for Windows (runtime-dimension path) | source and pack byte-identical; the reference request's waveform byte-identical with the new support files |

**Refused, with a sentence:** a per-row zero point for MatMulInteger's A (the
reference evaluator broadcasts it along the wrong axis, and ONNX Runtime
does not implement it); QuantizeLinear of other than FLOAT; DequantizeLinear
of other than UINT8, INT8 and INT32; the float8, 4-bit, 2-bit and 16-bit
quantized types; blocked quantization (opset 21); a scale of more than one
value before opset 13; FLOAT16 scales.

## Group C, first batch: windows, DFT, MelWeightMatrix and the losses — September 24, 2026

HannWindow, HammingWindow and BlackmanWindow (opset 17), DFT (opset 17 and
20: forward, inverse, one-sided, and the real inverse of a one-sided
spectrum, over any axis before the last), NegativeLogLikelihoodLoss and
SoftmaxCrossEntropyLoss (opset 12 on; none, sum and mean, weights,
ignore_index, the optional log_prob) on both paths and every target, and
MelWeightMatrix (opset 17) from initializers.

The kernels join `runtime/tensor_ops.pmi`. sin and cos over the binary32
range are Tan's integer reduction by pi/2 followed by its kernels. DFT is
computed by its definition, each output a sum in index order of binary32
products; its twiddle factors take their quarter turn from the integers
(4r = q n + rem, the angle within it pi/2 * rem / n), so a multiple of a
quarter turn gives exact 0 and 1. MelWeightMatrix is not a kernel: a
binary32 computation of its band edges disagreed with the reference on one
of 980 parameter sets tried, because a band edge that falls within 1e-8 of
a bin boundary decides a whole column. When its five inputs are
initializers the compiler computes the matrix before either path runs, the
edges as the reference computes them (the mel endpoints in binary32, the
bins in binary64), and the node becomes an initializer; otherwise it is
refused with a sentence.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 476 cases - the 386 before; the three windows, periodic and symmetric, sizes 1 to 400; DFT real and complex, lengths 1 to 31 against a signal of 6, both directions, one-sided, a NaN; NegativeLogLikelihoodLoss with INT32 and INT64 targets, each reduction, weights and ignore_index, and a target outside the classes | Windows, Pi 4, Pico and Pico 2: 476 of 476 bit-identical to the definition; `--mutants` 45 of 45 caught (five new: a Hamming coefficient, a twiddle quadrant, the inverse's sign, the one-sided inverse's doubling, the weighted mean's divisor) |
| `tests/node_suite/targeted_ops.py`: 729 cases - the 665 before and 64 new builds of these operators (MelWeightMatrix before a Mul and before an Add, both paths) | 729 of 729 as expected. Multi-node cases now carry their intermediate shapes, as an exporter writes them |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 490 of 964 before, 538 after; no case that passed fails. The 48 new passes are the window, DFT and loss cases. Two new FAIL_NUMERIC cases, test_dft_inverse and its opset-19 twin, are the reference's round-off: an imaginary part the reference gives as 1.9e-7 is exactly 0 here, and the corpus's absolute tolerance is 1e-7. test_melweightmatrix is refused: its parameters are graph inputs. The function-expanded SoftmaxCrossEntropyLoss cases stay refused (their intermediate values carry no shape). Every case: [`tests/node_suite/results/2026-09-24e-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24e-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `e54abcc` |
| `ops_targets_gate.py`: every targeted build that must pass, compiled for the Pi 4, Pico and Pico 2 and run in unicorn | 492 builds, 1,476 runs: 1,473 bit-identical to the Windows program; the three others were a MelWeightMatrix case followed by a MatMul, whose Windows kernel sums in another order than the bare-metal one - the case now multiplies instead, and the 60 builds of this batch then pass 180 of 180 |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `c995378` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

**Refused, with a sentence:** a window's output_datatype other than FLOAT;
MelWeightMatrix whose inputs are not all initializers, whose parameters do
not describe a filter bank, or whose bands pass the Nyquist bin; DFT with a
last axis other than 1 or 2, a signal on the last axis, or (fixed shapes) a
length or axis that is not a constant; a loss reduction other than none,
sum and mean; a target outside the classes at run time.

## Group C, second batch: Col2Im, CenterCropPad, MaxUnpool, AffineGrid, MaxRoiPool, DeformConv — September 24, 2026

Col2Im (opset 18, one to three spatial axes), CenterCropPad (18, any
element type the compiler carries), MaxUnpool (9 and 11, with and without
output_shape), AffineGrid (20, two and three spatial axes), MaxRoiPool (1)
and DeformConv (19, two spatial axes, groups, offset groups, mask and bias)
on both paths and every target. Their kernels join
`runtime/tensor_ops.pmi` under the same rules. Col2Im adds the block
elements in the reference's order; DeformConv samples bilinearly with
zeros outside, as the reference's GridSample does, and sums each input
channel's kernel positions in order.

Two operators follow one implementation where the others differ.
MaxUnpool with output_shape: the indices address the inferred shape, whose
block sits at the start of the larger output, as the reference and the
official test data have it; ONNX Runtime reads them as indices into the
output. MaxRoiPool has no reference implementation; it follows ONNX
Runtime (the box corners scaled and rounded half away from zero, the bins
the float extent divided by the pooled extent, an empty bin 0, a maximum
that a NaN never wins).

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 499 cases - the 476 before; Col2Im over one to three axes with pads, strides and dilations and a NaN; CenterCropPad cropping, padding and both on FLOAT, INT64 and BOOL; MaxUnpool with output_shape, INT32 indices and an index outside; AffineGrid in two and three dimensions, both alignments, an extent of 1; MaxRoiPool with boxes outside the input, empty bins and a NaN; DeformConv with groups, offset groups, mask, bias, strides, dilations and an offset far outside | Windows, Pi 4, Pico and Pico 2: 499 of 499 bit-identical to the definition; `--mutants` 51 of 51 caught (six new) |
| `tests/node_suite/targeted_ops.py`: 765 cases - the 729 before; the official Col2Im, CenterCropPad, MaxUnpool, AffineGrid and DeformConv cases published above opset 20, re-imported at 20; 32 new builds (MaxRoiPool against ONNX Runtime alone, MaxUnpool with output_shape against the reference alone) | 765 of 765 as expected; re-imported official cases 139 PASS (135 before) |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 538 of 964 before, 553 after: the Col2Im, CenterCropPad and AffineGrid cases; no case that passed fails. Every case: [`tests/node_suite/results/2026-09-24f-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24f-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `29d2b42` |
| `ops_targets_gate.py`: every targeted build that must pass, on the Pi 4, Pico and Pico 2 in unicorn | 522 builds, 1,566 runs: 1,546 bit-identical to the Windows program; the other 20 are the 10 builds whose INT64 input lies outside the signed 32-bit range, refused on the Pico and Pico 2 as the INT64 contract requires |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `41ff4a6` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

**Refused, with a sentence:** DeformConv over other than two spatial axes
(the reference implements two); image_shape, block_shape, shape, size and
output_shape that are not constants on the fixed-shape path; shapes that
disagree with the attributes; an index outside the unpooled shape or a
batch index outside the input at run time.


## Group C, third batch: Unique and SequenceErase — September 24, 2026

Unique (opset 11: with and without an axis, sorted or in order of first
occurrence, any of its four outputs) on FLOAT, UINT8, INT8, INT32, INT64
and BOOL, and SequenceErase (opset 11, a position or the default last
element) for every target. Both take the runtime-dimension path even when
every extent is declared: Unique's output extents depend on the input's
values, and a sequence decides its length while the program runs. Unique's
kernel joins `runtime/tensor_ops.pmi` under the same rules;
SequenceErase is one procedure appended to each sequence runtime.

Unique follows the reference, which follows numpy: two FLOAT values are
equal when both are NaN or when they compare equal, so -0 and +0 are one
value (the first occurrence's bits are kept), and a NaN sorts after every
number. ONNX Runtime keeps every NaN apart, so the targeted cases with a NaN
are held to the reference alone. For an axis other than 0 with `sorted = 0`
the onnx 1.22.0 reference fails (it reorders along axis 0); those cases are
held to a definition written into the harness, which ONNX Runtime agrees
with.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 525 cases - the 499 before; Unique flat and along an axis, sorted and not, a NaN and -0, INT64 values outside the signed 32-bit range, INT32, INT8 and UINT8 order, BOOL, and outputs left out | Windows, Pi 4, Pico and Pico 2: 525 of 525 bit-identical to the definition (the INT64 values outside the signed 32-bit range refused through the INT64 range check on the Pico and Pico 2, as the contract says); `--mutants` 54 of 54 caught (three new: the NaN order, INT8's sign, the stride of an axis slice) |
| `tests/node_suite/targeted_ops.py`: 795 cases - the 765 before; 26 Unique builds (flat, axes 1 and -1, INT64, INT8, INT32, NaN and -0, outputs left out, a Mul after it), an INT16 input refused, and SequenceErase between two tensors | 795 of 795 as expected |
| `tests/node_suite/targeted_control.py`: 32 cases - the 29 before; SequenceErase at a position, at the default, at a negative position, down to an empty sequence, and a position outside the sequence refused at run time | 32 of 32 as expected |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 553 of 964 before, 559 after: the six Unique cases; no case that passed fails |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 28 builds, 84 runs: 84 of 84 bit-identical to the Windows program |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `29d2b42` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

A model that uses a sequence operator exports the sequence runtime, which
now carries one more procedure (`DSeqErase`); nothing else in it changed.

**Refused, with a sentence:** Unique on an element type other than those
six; Unique on the fixed-shape path when that path is forced; a
SequenceErase position outside the sequence, or an empty sequence, at run
time. SequenceErase, refused in the section on control flow and sequences,
now compiles.

## Group C, third batch (second half): Bernoulli and Multinomial — September 24, 2026

Bernoulli (opset 15) and Multinomial (opset 7) join the random operators
on both paths and every target, under the contract of the
[random operators](#random-operators--september-16-2026): the same node
key, the same element counter, the same uniform `u = (w0 >> 8) * 2^-24`.

| Part | Contract |
|---|---|
| Bernoulli | FLOAT probabilities in; `1` where `u < p`, else `0`, as FLOAT, UINT8, INT8, INT32, INT64 or BOOL (`dtype`; FLOAT without one). So `P(1) = p` for `p` in [0, 1], `p = 1` always gives 1, `p = 0` and a NaN give 0. This is the operator's text and the reference's; the ONNX function body of Bernoulli (`Greater(RandomUniformLike(x), x)`) gives 1 with probability `1 - p`, and is not followed. |
| Multinomial | FLOAT `[batch_size, class_size]` of unnormalized log-probabilities in; `[batch_size, sample_size]` INT32 or INT64 class indices out. Draw `i` of row `b` is element `b * sample_size + s` of the stream. Row weights `exp(x - max)` (the binary32 `exp` of `runtime/tensor_ops.pmi`) are summed in class order; the draw picks the first class whose running sum exceeds `u * total`, else the last class with a positive weight, else class 0 (a row whose weights are not finite). The draws are written into the output buffer first and replaced by indices last to first, so no scratch is needed. |
| Comparison | `--random-inputs` turns these nodes into model inputs too; the manifest gives each node's `dtype`, and a supplied tensor of another element type or shape fails the request with a sentence. |
| Claimed forms | Bernoulli on FLOAT input; Multinomial on FLOAT input of rank 2 with at least one class, `sample_size` of at least 1. Every other `dtype`, a non-FLOAT input and an unknown attribute are refused with a sentence naming the node, the attribute and the value. |

The onnx 1.22.0 corpus has no case for either operator at opset 20 or lower
(its Bernoulli cases are opset 22), so the checks are these:

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 537 cases - the 525 before; Multinomial over random rows, logits 100 apart (an overflowing `exp` without the shift), equal logits, `-inf` and `+inf` and NaN in a row, 40 classes, a draw of `1 - 2^-24`, INT32 and INT64 in place | Windows, Pi 4, Pico and Pico 2: 537 of 537 bit-identical to the definition; `--mutants` 57 of 57 caught (three new: the max shift, the running sum, the draw's scale) |
| `tests/node_suite/targeted_ops.py`: 813 cases - the 795 before; Bernoulli with a seed, with the model seed, after another node (index 1), as FLOAT, BOOL, INT64 and UINT8; Multinomial with one and nine samples, INT32 and INT64, seeded and not; FLOAT16 Bernoulli and FLOAT Multinomial refused. Expected values from the contract written out in the harness (Threefry checked against two published answers there) | 813 of 813 as expected |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 14 builds, 42 runs: 42 of 42 bit-identical to the Windows program |
| `random_operator_gate.py` (private tree), all parts, with a new part for these two in comparison mode | 235 checks PASS: node-test forms, refusals, seed and request sequences, comparison mode for all six operators, Kitten, the Pi 4 fixed-shape and runtime-dimension sources in the A64 interpreter, and the Pico, Pico 2 and UNO Q sources built |
| Statistics from the contract over 400,000 draws (seed 3) and 20,000 draws of a five-class row | Bernoulli frequency within 2.4 standard errors of p for p = 0.01, 0.1, 0.5, 0.9; Multinomial chi-square 2.15 on 4 degrees of freedom |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 559 of 964 before and after; no case changes (the corpus has none for these operators) |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `43a461e` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

A model that uses a random operator exports the generator and its
wrappers, which now carry the appended procedures (`PmRandomBernoulli`,
`PmRandomUniformDraws`; `DRandomBernoulli`, `DRandomMultinomial`,
`DRandomFedKind`); a model with Multinomial also carries `tensor_ops.pmi`.

## Group C, fourth batch: Scan and SequenceMap — September 24, 2026

Scan (opset 9 to 20) and SequenceMap (opset 17 to 20) compile on the
runtime-dimension path for every target, with no new runtime: straight
after loading, `onnx_control.pbi` rewrites each into the `Loop` it
describes, with the nodes the specification implies, and the Loop emitter
of the [control-flow section](#control-flow-and-sequences--september-16-2026)
does the rest.

| Operator | Rewritten as |
|---|---|
| Scan | Trip count `Gather(Shape(x0), axis0)`. In the body, scan input `j`'s slice is `Gather(xj, i, axis_j)`, or `Gather(xj, M-1-i, axis_j)` for direction 1; the state values are the Loop's carried values. Scan output `k` is the Loop's stacked output, then `Slice` with step -1 for direction 1 and `Transpose` to move the new axis to `scan_output_axes[k]`. |
| SequenceMap | Trip count `SequenceLength(s0)`. In the body, a sequence input's element is `SequenceAt(s, i)` and a tensor input is read unchanged; output `k` starts as `SequenceEmpty` of the body output's declared element type and grows by `SequenceInsert`. That is the operator's ONNX function body. |

The added nodes carry out an operator whose own opset floor has already
been checked, so their floors are not checked again. Without that, a Scan
in an opset-9 model would be refused for the Gather it becomes.

| Check | Result |
|---|---|
| `tests/node_suite/targeted_control.py`: 41 cases - the 32 before; Scan with two scan inputs on different axes, reversed inputs and outputs, negative axes, output axes other than 0, no state, INT64 state with a scalar scan output, a Scan inside a Loop body that reads the Loop's carried value; SequenceMap with two outputs of two element types, a sequence and a tensor input, two INT32 sequences; Scan-8 and SequenceMap at opset 16 refused | 41 of 41 as expected; the expected values come from ONNX Runtime where the reference does not implement the form (scan axes other than 0, reversed directions) |
| `tests/node_suite/targeted_ops.py`: 817 cases - the 813 before; a Scan with reversed directions and moved axes, and a SequenceMap between two tensors, so that every target runs them | 817 of 817 as expected |
| `ops_targets_gate.py`: those builds on the Pi 4, Pico and Pico 2 in unicorn | 4 builds, 12 runs: 12 of 12 bit-identical to the Windows program |
| `tests/node_suite/pi4_control_gate.py` with the A64 interpreter: the new control cases on the Pi 4 | 7 of 7 bit-identical to the expected outputs (the Scan and SequenceMap cases that must pass) |
| Official node tests at opset 20 or lower, this change against the previous compiler | PASS 559 of 964 before, 568 after: the three Scan-9 cases and the six SequenceMap cases; no case that passed fails. test_scan_sum (Scan-8) is refused by name |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `4e6d5cd` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

**Refused, with a sentence:** Scan-8 (a batch axis and `sequence_lens`;
the definition from opset 9 is implemented); SequenceMap before opset 17;
a negative scan axis on an input with no declared rank; a scan output
axis other than 0 on a body output with no declared rank; a SequenceMap
body output with no carried element type declared. SequenceMap in its
unexpanded form and Scan, both refused in the section on control flow and
sequences, now compile.

## Opsets 21 to 27 — September 24, 2026

Both paths now accept models that import ai.onnx opsets up to 27, the
newest onnx 1.22.0 defines. Until now the runtime-dimension path stopped at
20 and the fixed-shape path at 23. Each operator is still accepted only
from its floor, and now also only up to its **ceiling**: the last opset
whose definition of it is the one the kernels compute.

The ceilings come from an audit of onnx 1.22.0's schema history. Every
version above 20 of every operator the compiler lists was compared with the
version before it: attributes (names, types, defaults, whether required),
inputs, outputs, arity and the operator's text. `src/compiler/onnx_opsets.pbi`
holds the result.

| Finding | Operators | Treatment |
|---|---|---|
| Only more element types (bfloat16, float16, int4, uint4, the float8 and float4 formats, int2, uint2) | 110 of the 121 version steps compared | Accepted through 27; the element-type checks refuse those types |
| A new attribute whose absence keeps the older form | DequantizeLinear-21 (block_size), -23 (output_dtype); QuantizeLinear-21 (block_size, output_dtype), -23 (precision) | Accepted; the attribute checks refuse the attribute when it is present |
| A new attribute that acts only on types not carried | Cast-24 and CastLike-24 (round_mode, float8 only); Range-27 (stash_type, float16 and bfloat16 only) | Accepted |
| Text for types not carried | QuantizeLinear-25 (the int2 and uint2 ranges) | Accepted |
| Windows that would start in the right padding are dropped | MaxPool-22, AveragePool-22 | Accepted: the kernels already drop them, and the official `-22` ceil_mode cases pass |
| A different operation | GroupNormalization-21 (scale and bias per channel, not per group) | Ceiling 20: refused by name at opset 21 and later |

`opset_ceiling_check.py` in the compiler repository's private tooling repeats
the audit against the installed onnx package. It fails when a changed
version is accepted, when a ceiling is lower than the history needs, or when
the accepted maximum passes onnx's newest opset. It has three mutants: the
GroupNormalization entry dropped, a needless Conv-22 entry added, and the
maximum set to 28. All three are caught.

The node-test harness now scores every case at opset 27 or lower, 1,750 of
the corpus's 1,765 (15 import no ai.onnx opset). Random operators have no
comparable values: the reference draws with numpy and this compiler with
its own specified generator. A case that uses one is therefore scored
PASS_SHAPE: shape and element type checked, and 0/1 outputs staying 0/1,
but not the values. PASS_SHAPE is its own outcome, counted apart from PASS,
from the commit that adds Swish, RMSNormalization and CumProd (the
2026-09-24g files predate it).

| Check | Result |
|---|---|
| `opset_ceiling_check.py --mutants` | PASS: 182 operators listed, 121 version steps above 20 compared; 3 of 3 mutants caught |
| Official node tests, this change against the previous compiler under the same harness (opset 27 or lower) | PASS 737 of 1,750 before, 827 after, and in both one case PASS_SHAPE (test_bernoulli_seed, a random operator: shape and element type checked, not values; the published 2026-09-24g files count it as PASS, the harness has since given it its own outcome): 90 cases at opsets 22 to 27 now compile and pass (63 at opset 25, 15 at 27, 9 at 24, 3 at 22); no case that passed fails, and the 964 cases at opset 20 or lower score exactly as before (568). Seven new RUN_ERROR cases are refusals when the program runs: Dropout in training mode with a nonzero ratio (four, opset 22), and float16 or bfloat16 inputs the driver cannot supply (three). Every case: [`tests/node_suite/results/2026-09-24g-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24g-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `39a0748` |
| Models at opset 20 or lower: the four identity models (fp32/fp16/bf16/int4, five targets) and every targeted suite, this change against `bd07ecf` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs. Targeted suites unchanged: ops 817, defaults 148, norm_small 132, control 41, optional outputs 14 |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

## Above opset 20, first batch: Swish, RMSNormalization, CumProd — September 24, 2026

Swish (opset 24), RMSNormalization (23) and CumProd (26), three of the nine
operators onnx defines only above opset 20, compile on both paths for every
target. Their kernels join `runtime/tensor_ops.pmi` under the same rules.

| Operator | Computed as |
|---|---|
| Swish | `x / (1 + exp(-alpha x))`, the binary32 `exp` of the kernel set; FLOAT |
| RMSNormalization | `stash_type` 1: per row of the axes from `axis` on, the sum of `x*x` in order, divided by the count, plus epsilon; the correctly rounded square root; `(x / r) * scale`. FLOAT X and scale; a scale whose extents, leading 1s aside, are the trailing normalized extents |
| CumProd | running products along the axis (a constant on the fixed-shape path, a one-element INT32 or INT64 tensor on the runtime-dimension path), exclusive and reverse as CumSum has them; FLOAT one binary32 multiply per step, INT32 and INT64 wrapping at their width |

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 566 cases - the 537 before; Swish with alpha 1.702; RMSNormalization over one and two axes, three scale shapes, a zero row and a NaN; CumProd on FLOAT with a NaN, INT32 with wrapping products and INT64, every axis, exclusive and reverse | Windows, Pi 4, Pico and Pico 2: 566 of 566 bit-identical to the definition; `--mutants` 60 of 60 caught (three new: RMSNormalization's mean, CumProd's exclusive order, Swish's sign) |
| `tests/node_suite/targeted_ops.py`: 864 cases - the 817 before; 47 new builds (Swish at three alphas; RMSNormalization over five axis and scale forms and with an initializer scale; CumProd on FLOAT, INT32 and INT64 in four axis, exclusive and reverse forms, and with its axis as an input; stash_type 0 and a scale broadcast inside the normalized block refused) | 864 of 864 as expected |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 44 builds, 132 runs: 132 of 132 bit-identical to the Windows program |
| Official node tests at opset 27 or lower, this change against the previous compiler | PASS 827 of 1,750 before, 849 after (and one PASS_SHAPE in both): the 19 RMSNormalization cases, test_swish and the two INT32 CumProd cases; no case that passed fails. The function-expanded RMSNormalization cases stay refused (their folded Shape has no graph value), and the other CumProd cases are DOUBLE, refused by type. Every case: [`tests/node_suite/results/2026-09-24h-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24h-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `b8e35e8`, the first published run that counts PASS_SHAPE apart |
| `random_operator_gate.py` after the harness change below | 235 checks PASS |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `16ab909` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

**The harness's PASS_SHAPE.** The node-test harness now ends a case that
uses a random operator in its own outcome, PASS_SHAPE, counted apart from
PASS: shape and element type checked, 0/1 outputs staying 0/1, values not
compared. Harnesses whose expected values are this compiler's generator
contract (`targeted_ops.py`, the private random gate) still compare every
value and end those cases in PASS.

## Windows multicore — September 24, 2026

A Windows model now runs on every logical processor its process may use,
counted when the model is bound: the process's affinity mask, every active
processor of every group when the process is not narrowed to part of one,
lowered by a default CPU set. `--threads N` and `PmModelSetThreads` /
`PmOnnxSetThreads` can only lower it. Bind takes the workers (creating any
that do not exist yet) and unbind parks them - blocked, running nothing -
for the next bind; no operator creates a thread. Operators are split by output
elements, rows, channels, positions or recurrent units, each output computed
by one task in the one-thread order, so the thread count never changes a bit
([chapter 5](guide/05_what_comes_out.txt)). The operators whose kernels are in
`runtime/tensor_ops.pmi` run on the calling thread: that file keeps its
parameters and scratch in shared globals and is one source for every target.

| Check | Result |
|---|---|
| `mt_split_gate.py` (compiler repository, `tools/onnx/tests/Diagnostics`): every operator the runtime splits - element-wise, row-wise, FP32 and INT8 products and Gemm, every convolution form, transposed convolution, the LSTM (FLOAT, mixed and INT8, hidden sizes 35 and 36), STFT, data movement, reductions, InstanceNormalization, TopK, Pad - at 1, 2, 3 and 8 threads, natural and forced smallest split, odd shapes | 144 cases x 8 configurations, every output CRC equal to the one-thread run's; every case split when forced, the large ones naturally too, a 101-element add did not |
| The same gate's affinity program: the process narrows its own mask to 1, F, FF, 5555, F0F0, every other processor, all but one, all | detected count, default pool size and the processor of every task follow the mask in every case; a request above the mask gives the mask's count; the allowed-processor rule on two- and three-group machines and with a CPU set |
| The same gate's exit program, launched as the node-suite harness launches (fault dialogs off): 180 ends straight after bind, after one parallel operator, and after an unbind | 180 of 180 clean |
| `--mutants`: a reduction summed in two halves, every STFT worker in one working buffer, the pool returning before its workers finish, the count ignoring the mask, bind returning before the workers have started, workers spinning on slots in the language's memory, unbind ending the workers instead of parking them | 7 of 7 red (the last three: 58, 177 and 16 of 180 exits faulted) |
| Kokoro-82M, reference request, FP32 and INT8, threads 1, 2, 4, 8, 16, 32 and main's CLI (and again at 1 and 32 on the merged tree against `d455afb`'s CLI) | every waveform byte-identical to main's: FP32 `5993F501...`, INT8 `086B8558...` |
| Kokoro under `start /affinity` 1, F, FF, 5555 and all | threads used = the mask's processor count: 1, 4, 8, 8 and 32; FP32 13.7, 5.4, 3.0, 2.4 and 1.7 s, INT8 10.9, 4.7, 2.6, 2.8 and 1.6 s (one run each); every waveform byte-identical to the one-thread run |
| `targeted_ops.py`, `node_suite.py`, `int8_model_gate.py`, each run with every processor and again with `--threads 1` (`mt_threads_rerun.py`), on this change merged onto `d455afb` | targeted_ops 864 of 864 as expected; node suite PASS 849 and PASS_SHAPE 1 of 1,750 at both counts, every outcome equal between them and to the published `2026-09-24h` record; int8_model_gate PASS; the 1,607 Windows result files of the two runs byte-identical. Before the merge (on `c995378`) the same comparison over `targeted_defaults.py` 148, `targeted_norm_small.py` 132, `targeted_control.py` 29 and `targeted_optional_outputs.py` 14 of 14 was identical too |
| `ops_kernel_check.py` on the merged tree; before the merge, `ops_kernel_check.py --mutants`, `int8_kernel_check.py --mutants`, `ops_targets_gate.py`, `int8_conv_split_gate.py --mutants` and `kokoro_int8_nodes.py` | 566 of 566 on Windows, Pi 4, Pico and Pico 2. Before the merge: 386 of 386 and 40 of 40 mutants; 51 INT8 cases bit-identical everywhere and 17 of 17 mutants; 1,296 of 1,296 target builds bit-identical to the 32-thread Windows program; 4 of 4; the nine real Kokoro INT8 nodes dumped from the 32-thread build bit-identical on Pi 4, Pico and Pico 2 |
| Six models (fp32, fp16, bf16, int4, int8; two of them with operator-set kernels), this change against main's CLI | Pi 4, UNO Q, Pico and Pico 2: 108 of 108 output sets (source, pack, manifest, runtime folder) byte-identical, 15 refused alike. Windows: packs and manifests identical; the source differs by the dispatch lines only - `PmPoolStart()` at bind, `PmPoolStop()` at unbind, the four-line `PmModelSetThreads` / one-line `PmOnnxSetThreads` procedure with its two comment lines, and `#PMO_THREADS = N` when `--threads` is given |

Speed, Kokoro-82M reference request on the i9-14900HX (8 performance cores
with two threads each, 16 efficiency cores; 32 logical processors), the
second request of each run, configurations interleaved, five rounds; this
laptop throttles, so only numbers from the same round compare:

| threads | FP32 median (min) | INT8 median (min) |
|---|---:|---:|
| main's CLI (per-operator threads, at most 8, in convolution, the product and the INT8 tiles only) | 6.55 s (5.58) | 5.09 s (4.51) |
| 1 | 11.22 s (9.34) | 10.14 s (7.19) |
| 2 | 5.27 s (4.51) | 4.64 s (3.15) |
| 4 | 3.38 s (2.42) | 3.08 s (1.84) |
| 8 | 1.90 s (1.61) | 2.33 s (1.30) |
| 16 | 1.58 s (1.41) | 1.66 s (1.11) |
| 32 (the default here) | **1.39 s (1.33)** | **1.24 s (0.99)** |

32 threads against one: 8.1 times (FP32) and 8.2 (INT8); against main's CLI
in the same rounds 4.7 and 4.1 times; against the published 5.51 s and
4.47 s, 4.0 and 3.6. The curve flattens after 8 threads, and a per-node
profile at 16 and 32 threads says where: convolution (0.74 s of 1.59 at 32)
is compute-bound and the efficiency cores add about half again; the
element-wise operators (0.30 s) stream 20 MB tensors and stay at the
memory's bandwidth from 16 threads on; the six recurrent layers (0.04 s) are about 2,100
strictly ordered steps; and on this laptop more active cores lower the
clock of all of them. Workers block between operators rather than spin:
blocking measured as fast as spinning 10 or 50 us, and spinning 2 ms made
32 threads slower than 16 (3.6 s), because spinning workers spend the
package power the working ones need.

A defect found while merging, and fixed: with every processor in use, 2 of
817 targeted builds and 1 node-suite case ended with an access violation
after writing correct results (0 at `--threads 1`). The cause is the exit,
not the arithmetic: ending the workers at unbind leaves work for Windows'
own thread-pool workers, and a program that ends straight after races them
- reproduced in a twelve-line program with no model and no pool (89 of 600
exits, fault dialogs off). Two earlier forms of the same race were found on
the way: ending the program while a worker was still starting (bind now
waits for every worker to block), and workers spinning on slots in the
language's memory, which End releases (the slots are system memory, and
workers no longer spin). Workers are now parked at unbind and reused at the
next bind; the gate's exit program holds all three.

`--threads 1` is slower than main's CLI: main was not single-threaded - its
convolution, product and INT8 tiles created up to eight threads per call -
and `--threads 1` now means the calling thread alone.

## Above opset 20, second batch: BitCast, RotaryEmbedding, TensorScatter — September 24, 2026

BitCast (opset 26), RotaryEmbedding (23) and TensorScatter (24) compile on
both paths for every target.

| Operator | Computed as |
|---|---|
| BitCast | the input's bytes, little-endian, read as `to`: FLOAT and INT32; INT64; UINT8, INT8 and BOOL. A `to` of another width is refused (the specification requires the same width) |
| RotaryEmbedding | X as `[B, S, hidden]` (with `num_heads`) or `[B, H, S, D]`; the caches `[max_position, rd/2]` read at `position_ids`, or `[B, S, rd/2]` without them; the first `rd` elements of each head rotate in halves or interleaved pairs, `c*x1 - s*x2` and `s*x1 + c*x2`, each product and sum one binary32 step; the rest is copied. A position id outside the caches fails the request |
| TensorScatter | the past cache with the update written along `axis` at each sample's write index (0 without `write_indices`), linear or circular; any carried element type. A linear write past the cache fails the request |

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 575 cases - the 566 before; RotaryEmbedding in rank 3 and 4, halves and interleaved, partial rotation, with and without position ids; TensorScatter linear and circular, with and without write indices, FLOAT, INT64 and UINT8, and a linear write past the cache | Windows, Pi 4, Pico and Pico 2: 575 of 575 bit-identical to the definition; `--mutants` 62 of 62 caught (two new: the rotation's sign, the circular modulo) |
| `tests/node_suite/targeted_ops.py`: 889 cases - the 864 before; BitCast FLOAT to INT32 (with -0 and infinity), INT32 to FLOAT, UINT8 to INT8, INT8 to BOOL; RotaryEmbedding rank 3 with position ids, rank 4 interleaved with a partial rotation, rank 4 with initializer caches; TensorScatter linear with indices, circular, INT64 along axis 1 without indices; a width change, an odd rotary dimension and an unknown mode refused | 889 of 889 as expected |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 20 builds, 60 runs: 60 of 60 bit-identical to the Windows program |
| Official node tests at opset 27 or lower, this change against the previous compiler | PASS 849 of 1,750 before, 866 after (and one PASS_SHAPE in both): six BitCast cases, the eight RotaryEmbedding cases and the three TensorScatter cases; no case that passed fails. The function-expanded RotaryEmbedding cases stay refused (their folded Shape has no graph value); BitCast of DOUBLE, UINT16 and UINT32 is refused by type. Every case: [`tests/node_suite/results/2026-09-24i-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24i-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `0e4e197` |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `628ee1c` (the multicore Windows runtime) | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

## Above opset 20, third batch: Attention — September 24, 2026

Attention (opsets 23 and 24) compiles on both paths for every target: Q, K
and V all 3-D (`[B, S, heads x size]` with `q_num_heads` and
`kv_num_heads`) or all 4-D; grouped-query and multi-query heads; a FLOAT
mask (added) or BOOL mask (false masks), broadcast from rank 1 to 4 and
shorter than the total sequence (the rest masked); `is_causal`, aligned
upper-left and after the past; past and present key and value caches;
`nonpad_kv_seqlen`; `scale` (default `1/sqrt(head size)`); `softcap`; the
four `qk_matmul_output_mode`s. `softmax_precision` other than FLOAT is
refused. Its kernel joins `runtime/tensor_ops.pmi`: `sqrt(scale)` on Q and
K, the dot product summed in order, the tanh softcap, the bias, the softmax
as max, `exp(s - max)`, a sum in order and a division, and the weighted
sum of V in order, each one binary32 step. A row whose every position is
masked gives NaN, as the reference's softmax does.

**One disagreement with the onnx 1.22.0 reference.** `qk_matmul_output_mode`
0 is, in the operator's text, the output of the QK product, before the
softcap. The reference's implementation hands out the softcapped value
there when `softcap` is set. This compiler follows the text. No official
case combines mode 0 with a softcap, and the targeted case for mode 0 runs
without one.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 580 cases - the 575 before; Attention in five forms: grouped heads, 3-D with a causal float mask and a scale, past caches with a BOOL mask and a softcap, nonpad lengths with a short mask, multi-head causal; every output (Y, present key and value, the qk output in all four modes) | Windows, Pi 4, Pico and Pico 2: 580 of 580 bit-identical to the definition; `--mutants` 65 of 65 caught (three new: the grouped head's KV head, causal alignment after the past, the softcap's place before the mask) |
| `tests/node_suite/targeted_ops.py`: 909 cases - the 889 before; nine Attention forms on both paths and softmax_precision DOUBLE refused | 909 of 909 as expected |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 18 builds, 54 runs: 54 of 54 bit-identical to the Windows program |
| Official node tests at opset 27 or lower, this change against the previous compiler | PASS 866 of 1,750 before, 928 after (and one PASS_SHAPE in both): the 62 Attention cases whose inputs are FLOAT, every one of them attempted; no case that passed fails. The float16 cases are refused by type and the function-expanded ones stay refused (their folded batch size has no graph value). Every case: [`tests/node_suite/results/2026-09-24j-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24j-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `03e2bd3` |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `14b8423` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

## Above opset 20, fourth batch: LinearAttention and CausalConvWithState — September 24, 2026

LinearAttention and CausalConvWithState (opset 27, the newest onnx 1.22.0
defines) compile on both paths for every target. That completes the nine
operators onnx defines only above opset 20.

| Operator | Computed as |
|---|---|
| CausalConvWithState | depthwise over `[B, C, L]` with weight `[C, 1, K]`: the past state (zeros without one) followed by the input, each output the products with its window summed in order, the bias added; `silu` and its alias `swish` as `y / (1 + exp(-y))`; the present state the window's last `K-1` positions |
| LinearAttention | the four update rules (`linear`, `gated`, `delta`, `gated_delta`), token by token as the operator defines the recurrence: the state times `exp(decay)` (per head or per key dimension), the value less the state's read of the key times `beta` (per head or one per token), the key's outer product added, the query's read of the state times the scale (default `1/sqrt(d_k)`); grouped-query heads share their KV head's state; the past state or zeros on entry, the present state out. `chunk_size` is a tuning hint and changes nothing |

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 589 cases - the 580 before; CausalConvWithState with and without bias, past and silu, K of 1, 3 and 4; LinearAttention under all four rules, grouped heads, decay per head and per key, beta per head and per token, with and without a past, with a scale | Windows, Pi 4, Pico and Pico 2: 589 of 589 bit-identical to the definition; `--mutants` 68 of 68 caught (three new: the present state's offset, the delta rule's sign, the grouped head's state) |
| `tests/node_suite/targeted_ops.py`: 927 cases - the 909 before; CausalConvWithState with bias, past and silu, and plain; LinearAttention under each rule with grouped heads and both outputs, and with a past and a scale; an unknown activation and an unknown rule refused | 927 of 927 as expected |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 14 builds, 42 runs: 42 of 42 bit-identical to the Windows program |
| Official node tests at opset 27 or lower, this change against the previous compiler | 952 PASS against 928: the 24 that change are the 11 CausalConvWithState and 13 LinearAttention cases in FP32, refused before and passing now; their FP16 cases stay refused for their element type and their function-expanded forms for their folded tensors; nothing else changes. Every case: [`tests/node_suite/results/2026-09-24k-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24k-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `327f6ef` |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `b26ea37` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images byte-identical; the 16 runtime-dimension sources for the Pi 4 and the Pico build; no support file differs |
| Kokoro-82M FP32 for Windows | source, pack and support files byte-identical |

## Constant on both paths, function-expanded graphs, and the runtime-dimension gaps — September 24, 2026

The node-test results showed four gaps in forms the compiler accepts
elsewhere, and three wrong answers. Each wrong answer was filed in the
forum's ONNX bugs area before it was fixed.

| Change | What it does |
|---|---|
| Constant on the fixed-shape path | Every top-level `Constant` node becomes the initializer it names before anything is planned, as the runtime-dimension path already did: `value`, `value_float`, `value_floats`, `value_int`, `value_ints`, and now `sparse_value` on both paths. The reader keeps the SparseTensorProto, and the lowering writes it out densely: zero everywhere except at its indices, which may be `[NNZ]` linear positions or `[NNZ, rank]` coordinates. An index outside the shape is refused. `value_string` and `value_strings` are refused by name. |
| Function-expanded graphs | A graph whose node outputs carry no declared type and shape, such as the intermediates of an expanded function, is computed by the runtime-dimension path, which works shapes out as the program runs. Before, the fixed-shape path refused it ("folded tensor ... has no graph value" or "has no concrete type/shape"), so no model that compiled before takes a different path now. A value declared with a named extent still gets the fixed-shape refusal ("still has a dynamic extent"), as `--shape` documents, and a model compiled with `--trace-input` keeps its traced shapes. |
| Relu, Flatten, BatchNormalization and LessOrEqual on the runtime-dimension path | These four were the only operators the fixed-shape path accepts and the runtime-dimension emitter lacked. Relu and BatchNormalization run the fixed-shape path's own kernels (`PmTensorRelu`, `PmTensorBatchNorm`, `PmTensorBatchNormTraining`), and Flatten is a sized copy. They are wrappers in `runtime/tensor_dynamic_ops.pmi`, included only when a node uses them. BatchNormalization covers opsets 9 to 15, with training mode and its running statistics from 14. LessOrEqual is a new variadic-kernel code (12): its output is BOOL, and it is false whenever either FLOAT side is NaN. |
| Runtime-dimension input types | A graph input whose element type the runtime-dimension runtime does not bind (FLOAT16, DOUBLE, INT16, the unsigned types above 8 bits and others) is refused when the model is compiled, naming the input and its type. Before, the program was generated and then failed when it received its first request. |
| Where, forum 993 | On the runtime-dimension path the output is now sized by broadcasting the condition, X and Y together, as Where-16 requires. It had been sized from X and Y alone, so a condition with more dimensions, or larger ones, failed at run time. The fix changes only `DWhere` in `runtime/tensor_dynamic_portable.pmi` and `runtime/tensor_dynamic_windows.pbi`. |
| Neg and Abs of integers, forum 994 | On the fixed-shape path, Neg and Abs of INT64 ran the FLOAT kernel on integer bits and gave wrong values, and INT32 was refused. On the runtime-dimension path both failed at run time. Both paths now run the operator-set lane's integer kernel (`PmOpUnary` codes 29 and 30), which wraps as two's complement, like numpy's int32 and int64. On the Pico targets an INT64 result outside INT32 trips the INT64 range check. |

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 597 cases - the 589 before; LessOrEqual on FLOAT (NaN on either side, signed zeros, infinities, ties), INT32, INT64 and INT64 beyond 32 bits; Neg and Abs of INT32 and INT64 including INT32's most negative value | Windows, Pi 4, Pico and Pico 2: 597 of 597 bit-identical to the definition; `--mutants` 70 of 70 caught (two new: LessOrEqual made strict, and Abs that negates) |
| `runtime_mutants.py` (new, beside the kernel check): the runtime-dimension support files with a planted defect, rebuilt into the CLI and run through the cases that cover it | 2 of 2 caught: `DWhere` sized from X and Y alone, planted in the Windows runtime and caught by the Windows cases, and planted in the portable runtime and caught by the Pi 4, Pico and Pico 2 cases |
| `tests/node_suite/targeted_ops.py`: 990 cases - the 927 before; every Constant value form on both paths, including sparse with linear and coordinate indices and an INT64 one (sparse checked against ONNX Runtime; the reference evaluator returns it still sparse); `value_string` refused; a graph with no value_info on its intermediates; Where with a condition of higher rank, a wider condition, and each input widening one axis; Relu; Flatten at every axis and on INT64 at opset 9; LessOrEqual with NaN and broadcasting, and on INT64; BatchNormalization at opset 15, at opset 9 (checked against ONNX Runtime; the reference evaluator's BatchNormalization-9 disagrees with it) and in training mode; Neg and Abs of INT32 and INT64; a FLOAT16 input refused on the runtime-dimension path | 990 of 990 as expected. Of the re-imported official cases, 141 pass, against 139 before: the two function-expanded ones without intermediate shapes now take the runtime-dimension path. `refuse_unique_int16` is now refused one step earlier, for its INT16 graph input |
| `ops_targets_gate.py`: the new builds on the Pi 4, Pico and Pico 2 in unicorn | 60 builds, 180 runs. 156 are bit-identical to the Windows program. The 18 BatchNormalization runs pass the node-suite tolerance but do not match the Windows bits: both paths run the fixed-shape runtime's own kernel, and the host compiler keeps its intermediates wider than binary32. The gate now lists such cases by name (`TOLERANCE_ONLY`) and still requires the tolerance. 6 INT64 Constant forms are refused on the Pico and Pico 2, as the INT64 contract requires |
| `tests/node_suite/pi4_control_gate.py` (Constant lowering and opset floors are shared with control flow) | 29 of 41, the same with this change as with the previous compiler. The other 12 are refusal cases, refused by both compilers with the same sentences |
| Official node tests at opset 27 or lower, this change against the previous compiler | PASS 952 of 1,750 before, 1,219 after, with PASS_SHAPE going from 1 to 2 (test_bernoulli_seed_expanded). Of the 267 new passes, 266 are function-expanded cases, among them every expanded Attention, CausalConvWithState, LayerNormalization, RMSNormalization and NLLLoss case whose inputs are FLOAT. The other is a model with a Constant node. Ten cases that failed at run time are now refused when compiled, for their FLOAT16, BFLOAT16 or DOUBLE inputs on the runtime-dimension path; RUN_ERROR goes from 18 to 8. No case that passed fails. Every case: [`tests/node_suite/results/2026-09-24l-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-24l-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `34a7d86` |
| Models that use none of these operators (four models, fp32/fp16/bf16/int4, five targets), this change against `2c8bf02` | 80 of 80 emitted sources, packs and manifests and 32 of 32 fixed-shape images are byte-identical, and the 16 runtime-dimension sources for the Pi 4 and the Pico build. The only support files that differ are `tensor_dynamic_portable.pmi` and `tensor_dynamic_windows.pbi`, each in one hunk inside `DWhere` (`@@ -1706,21 +1706,40 @@` and `@@ -711,13 +711,22 @@`). Both CLIs were built from CRLF trees |
| Kokoro-82M, FP32 and INT8, for Windows | FP32 and INT8: the source, the pack and every support file are byte-identical, except `tensor_dynamic_windows.pbi`, which differs only inside `DWhere` (`@@ -711,13 +711,22 @@`) |

## NaN on every path: the host compiler's float comparisons — September 24, 2026

A NaN reached eleven kernels and came out wrong: Equal, Greater and
GreaterOrEqual answered true (forum 995), Clip replaced it with the upper bound
(forum 997), and Cast to BOOL, Floor, Round, Relu, Tanh, ScatterND's max and min
reductions, runtime-dimension NonZero and Pow by a one-element NaN exponent
each had their own wrong answer (forum 998). Floor and Round also failed on
infinities and on values of magnitude 2^23 and above, which Int() overflowed.
All were filed before any change and reproduce on the released CLI.

**Two causes.** Most were the host compiler. PureBasic 6.21 (Windows x64) does
not answer a float comparison with a NaN operand the way IEEE 754 does, and
its answer depends on the operand order and on whether a side is a literal;
PureBasic 6.41 answers every one as IEEE 754 does. A probe with v = NaN:

| Comparison | 6.21 | 6.41 and IEEE 754 |
|---|---|---|
| `a > b`, `a >= b`, `a = b` (variables) | true | false |
| `a <> b` (variables) | false | true |
| `a < b`, `a <= b` (variables) | false | false |
| `v < 0.0`, `v <= 0.0`, `v < -10.0` (literal on the right) | true | false |
| `v > 10.0`, `0.0 <= v` | false | false |

Built with 6.41, the compiler before this change already passes the
comparisons, Clip, Relu, Tanh, Cast to BOOL, NonZero and the Pow shortcut.
The rest were logic that no compiler fixes: Floor and Round overflowed Int()
on NaN, infinities and large values, and ScatterND's max and min dropped a
NaN update and overwrote a NaN already in the data, where numpy.maximum and
numpy.minimum let a NaN win.

**The fix: guards only where a compiler needs them.** A program built from our
source may be built with an older PureBasic, so the Windows runtime keeps a
test on the float's bits in front of every comparison whose NaN answer
matters, but compiles it only when the compiler needs it:
`tensor_fp32_windows.pbi` sets `#PMO_HOST_NAN_BUG` to 1 when
`#PB_Compiler_Version` is below 641, and each guarded comparison is a
`CompilerIf #PMO_HOST_NAN_BUG = 1` with the guarded form, else the plain one.
With 6.41 the plain comparison is compiled and the guard costs nothing. The
portable runtime (Pi 4, UNO Q, Pico, Pico 2) sets the constant to 0 and
carries no guard: PureMetal's float comparisons answer a NaN as IEEE 754 does.
The source the fixed-shape emitter writes selects the same way, and so do the
operator-set lane's `PmOpGt`, `PmOpLt` and `PmOpLe` (IEEE 754's ordered
comparisons, used by NonMaxSuppression, RoiAlign and Multinomial). What ONNX
requires of a NaN or an infinity is kept on every path, as exact tests on the
bits: Floor and Round, ScatterND's max and min, Range, NonZero, Cast to BOOL,
the Pow shortcut and RoiAlign's sample coordinate.

What the guards cost where they are compiled, measured with 6.41 on 16M
elements (the Windows serial kernels, best of seven): Relu 2%, Clip 2 to 4%,
Tanh 3%, Greater 35%, Equal 114%. That is why they are compiled only for an
old compiler. Cast to BOOL, written on the bits, is twice as fast as before
(20 ms against 39 ms); Floor's test for 2^23 and above costs 3%, the price of
a right answer. Kokoro-82M on Windows, this change against the compiler before it, both built with 6.41 (15 requests each, interleaved): FP32 1,260.6 ms median against 1,254.2, INT8 1,003.7 against 1,001.8, within the run-to-run spread, the same output bits.

| Kernel | Now |
|---|---|
| Equal, Greater, GreaterOrEqual, Less, LessOrEqual (both paths) | false whenever a NaN takes part (guarded below 6.41) |
| Clip, Relu, Tanh (both paths) | a NaN passes through (guarded below 6.41) |
| Cast FLOAT to BOOL (both paths) | a NaN is true, as numpy's: the bits, on every compiler |
| Floor, Round (both paths) | a NaN, an infinity and any value of magnitude 2^23 or more pass through unchanged (they are already integers) |
| ScatterND reduction max and min (both paths) | numpy.maximum and numpy.minimum: a NaN on either side wins |
| NonZero (runtime-dimension; fixed-shape emitted) | a NaN is not zero: the bits |
| Pow with a one-element exponent (runtime-dimension, Windows) | the square shortcut is taken only for an exponent whose bits are exactly 2.0 |
| Pow with an integer base (fixed-shape emitted) | a NaN exponent is refused at run time (guarded below 6.41) |
| Range (both paths) | a NaN start, limit or delta is refused at run time with a sentence |
| NonMaxSuppression, RoiAlign, Multinomial (operator-set lane) | IEEE 754's ordered comparisons (guarded below 6.41); a NaN RoiAlign sample coordinate samples nothing, as for DeformConv |
| ReduceMean before opset 18, ReduceSum before 13 (runtime-dimension) | the axes attribute is now implemented, lowered to the axes input the later definitions take |

**Reviewed and left alone.** Loop bounds and every integer comparison; the
comparisons already behind a NaN test (the unary math, Fmod, the reductions
and ArgMax/ArgMin, LogSoftmax, the pools, Hardmax, Det, LessOrEqual, TopK's
comparator, ScatterElements, GridSample's coordinates, QuantizeLinear,
DynamicQuantizeLinear, Unique, BilinearZero, the RNN and GRU activations);
LeakyRelu and Sigmoid, whose every branch carries a NaN through; Softmax's
maximum in both runtimes and in the fixed-shape emitter, and Attention's,
where a row holding a NaN comes out all NaN whichever element is taken as the
maximum; the random generator's internal comparisons, which never see model
data; the LSTM's NEON activations, whose inputs the LSTM refuses when they are
not finite. The compile-time checks of attribute values inside the compiler
itself were not part of this audit.

**Both branches are gated.** The gates run on 6.41, which compiles the plain
branch; they also build the guarded branch by forcing `#PMO_HOST_NAN_BUG` to
1 (the kernel check's `windows-hostbug` target, and `runtime_mutants.py`'s
`force` entries), so the code an older compiler builds is checked and
mutated too.

| Check | Result |
|---|---|
| `ops_kernel_check.py`: 604 cases - the 597 before; NonMaxSuppression with NaN and infinite box coordinates, RoiAlign with a NaN feature and a NaN region corner in both modes, Multinomial with a NaN after the largest logit | Windows (both branches), Pi 4, Pico and Pico 2: 604 of 604 bit-identical to the definition on each; `--mutants` 72 of 72 caught (new: the RoiAlign NaN coordinate, on the Pi 4, and `PmOpGt`'s guard inverted, on the forced branch) |
| `runtime_mutants.py`: the guarded branch forced and every NaN case run on it; each guard inverted on the forced branch; the plain branch and every semantic rewrite mutated | 23 of 23 as required: the forced branch passes every NaN case; each guard inverted on it is caught (Relu, Clip, Tanh, DBinary, the emitted comparison); the plain comparisons, Floor and Round above 2^23 in both runtimes, the exact-bits tests (Cast to BOOL in both runtimes and the emitter, NonZero, the Pow shortcut), ScatterND in both runtimes and the emitter, DWhere and ReduceMean's axes attribute are each caught on the path that compiles them |
| `tests/node_suite/targeted_ops.py`: 1,059 cases - the 990 before; every comparison with NaN, signed zeros and infinities; 33 kernels given NaN and infinities on both paths (the ONNX reference decides; numpy for ScatterND's max and min, whose reference indexes wrongly); ReduceMean-13 and ReduceSum-11 with their axes attribute | 1,059 of 1,059 as expected |
| `ops_targets_gate.py`: those cases, and the Cast, Clip, Floor, Round and Tanh cases, on the Pi 4, Pico and Pico 2 | 101 builds, 303 runs: 243 bit-identical to the Windows program; 60 within the node-suite tolerance, each named with its reason in `TOLERANCE_ONLY` (the 18 BatchNormalization runs, and 42 of the base runtime's Neg, Cos, Log, Sqrt, Sin, Atan and Sigmoid on NaN and signed zeros: see the limitations) |
| Official node tests at opset 27 or lower, this change against the previous compiler, both built with 6.41 | PASS 1,219 of 1,750 before, 1,239 after, the same as with 6.21: the 20 function-expanded LayerNormalization and MeanVarianceNormalization cases, which reduce with ReduceMean's axes attribute; no case that passed fails. Every case: [`tests/node_suite/results/2026-09-25a-onnx-1.22.0.md`](../tests/node_suite/results/2026-09-25a-onnx-1.22.0.md) and `.json`, measured with `--from-commit` at `986eac9`, built with PureBasic 6.41 |
| Models that use none of these forms (four models, fp32/fp16/bf16/int4, five targets), this change against `c0d7c5b`, both built with 6.41 | 80 of 80 emitted sources, packs and manifests byte-identical; the 16 runtime-dimension sources build; 16 of the 32 fixed-shape Pico and Pico 2 images differ, because they link the changed `tensor_fp32.pmi` procedures. Output bits without NaN: the four models' Windows outputs byte-identical (`output_identity_check.py`), and all 30 outputs of their Pi 4, Pico and Pico 2 programs byte-identical |
| Kokoro-82M, FP32 and INT8, for Windows, against `c0d7c5b`, both built with 6.41 | Source and pack byte-identical; `tensor_dynamic_windows.pbi`, `tensor_dynamic_forms_windows.pbi` and `tensor_fp32_windows.pbi` differ (the table below); the output for the reference request byte-identical, FP32 and INT8 |

Every support-file procedure this change touches (the emitted text changes only for FLOAT comparisons, Cast to BOOL, NonZero, ScatterND max and min, Range and Pow with an integer base):

| Support file | Procedure | |
|---|---|---|
| `tensor_dynamic_forms_portable.pmi` | `DScatterReduce` | changed |
| `tensor_dynamic_forms_windows.pbi` | `DScatterReduce` | changed |
| `tensor_dynamic_portable.pmi` | `DNonZero` | changed |
| `tensor_dynamic_portable.pmi` | `DNonZeroAt` | added |
| `tensor_dynamic_portable.pmi` | `DPut` | changed |
| `tensor_dynamic_portable.pmi` | `DRange` | changed |
| `tensor_dynamic_windows.pbi` | `DBinary` | changed |
| `tensor_dynamic_windows.pbi` | `DIndexTask` | changed |
| `tensor_dynamic_windows.pbi` | `DNonZero` | changed |
| `tensor_dynamic_windows.pbi` | `DNonZeroAt` | added |
| `tensor_dynamic_windows.pbi` | `DPut` | changed |
| `tensor_dynamic_windows.pbi` | `DRange` | changed |
| `tensor_fp32.pmi` | `PmTensorFloor` | changed |
| `tensor_fp32.pmi` | `PmTensorIsNan` | added |
| `tensor_fp32.pmi` | `PmTensorNanAt` | added |
| `tensor_fp32.pmi` | `PmTensorRoundEven` | changed |
| `tensor_fp32_windows.pbi` | `PmTensorClipSerial` | changed |
| `tensor_fp32_windows.pbi` | `PmTensorFloorSerial` | changed |
| `tensor_fp32_windows.pbi` | `PmTensorIsNan` | added |
| `tensor_fp32_windows.pbi` | `PmTensorNanAt` | added |
| `tensor_fp32_windows.pbi` | `PmTensorReluSerial` | changed |
| `tensor_fp32_windows.pbi` | `PmTensorRoundEvenSerial` | changed |
| `tensor_fp32_windows.pbi` | `PmTensorTanhValue` | changed |
| `tensor_ops.pmi` | `PmOpGt` | added |
| `tensor_ops.pmi` | `PmOpMnMax` | changed |
| `tensor_ops.pmi` | `PmOpMultinomial` | changed |
| `tensor_ops.pmi` | `PmOpNmsSuppress` | changed |
| `tensor_ops.pmi` | `PmOpRoiAlign` | changed |
| `tensor_ops.pmi` | `PmOpRoiSample` | changed |


## Explicit limitations

- This compiler implements a **validated subset**, not the entire ONNX specification.
  Unsupported operators, attributes, element types, and shape forms must be
  rejected. Both paths accept each operator from the oldest opset whose
  definition it implements through its ceiling, opset 27 at most
  ([opsets 21 to 27](#opsets-21-to-27--september-24-2026)); see
  [control flow and sequences](#control-flow-and-sequences--september-16-2026).
  No accepted form is currently known to compute a wrong answer; see
  [node-test coverage](#node-test-coverage--september-16-2026) and, for NaN,
  [NaN on every path](#nan-on-every-path-the-host-compilers-float-comparisons--september-24-2026).
- PureBasic before 6.41 answers a float comparison with a NaN operand by the
  operand order and by literals, not as IEEE 754 does; 6.41 answers as IEEE
  754 does. The Windows runtime and the source the compiler writes carry a
  test on the float's bits in front of every comparison whose NaN answer
  matters, compiled only when the compiler is older than 6.41
  (`#PMO_HOST_NAN_BUG`), so a program built with either is right and one
  built with 6.41 pays nothing for it. A kernel written with a plain
  comparison is wrong for NaN on Windows when built with an older compiler.
- Some kernels of the base runtime (not of the operator-set lane) match the
  Windows program on the Pi 4, Pico and Pico 2 within the node-suite
  tolerance but not bit for bit. `ops_targets_gate.py` names each in
  `TOLERANCE_ONLY` with its reason and still holds it to the tolerance:
  BatchNormalization on both paths (the host compiler keeps its
  intermediates wider than binary32); Neg, Cos, Log, Sqrt and Sin of a NaN
  or of an invalid argument (Windows writes x86's default NaN, `FFC00000`,
  the targets `7FC00000`); Log, Sqrt, Sin, Atan and Sigmoid, one unit in the
  last place on some arguments, where Windows is the correctly rounded side;
  and Sin(-0) on the Pico and Pico 2 and Atan(-0) on all three targets,
  which give +0. Binary32 base kernels used by both paths, with NaN written
  as `7FC00000` as the operator-set lane writes it, are the next change, and
  will empty the list.
- Windows x64 is the verified host. Linux/macOS hosting, other PureBasic
  versions, and alternate PureBasic backends are not certified by this export.
- Five-target **generation** does not establish downstream bare-metal builds,
  bootability, hardware numerical accuracy, or timing. The later Pi 4 checks
  above cover only their named fixtures, not other boards or arbitrary models.
- Dynamic requests still require sufficient working memory. A successful
  generation is not an unlimited-dimensions or arbitrary-input promise.
- INT8 is selective; FP16/BF16/INT4 storage savings do not remove the resident
  decoded FP32 or activation-memory costs. Compare output quality yourself.
- SIMD/NEON may change floating-point summation order. Do not demand bitwise
  equality where the execution contract calls for numerical tolerances.
- Model execution uses generated globals. Serialize calls to one model instance.
- On Windows the kernels of `runtime/tensor_ops.pmi` (pooling, the quantized
  operators, GRU and RNN and the rest of the 2026 operator set) run on one
  thread; every other operator uses the model's worker pool.
- Random operators follow this compiler's specified generator; their values
  match another runtime only through `--random-inputs`. FLOAT output only.
- Optional reference tracing needs a separately supplied compatible ONNX Runtime
  DLL. Generated inference does not use it.
- Full speech assets are excluded, and no native pack-construction tool is
  included. See [Kokoro requirements](KOKORO.md) before expecting the reader to run.

## Before trusting your own model

Begin with known inputs and reference outputs in FP32. Check shape and element
types, the manifest's memory accounting, actual target memory use, error paths,
and repeated requests. Then compare reduced precision on representative data.
For bare metal, validate on the intended hardware before making correctness or
performance claims. Keep model inputs and weights from trusted sources; format
checks are not a substitute for an independent security audit.
