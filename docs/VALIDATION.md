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
| Official node tests at opset 20 or lower (the corpus and harness of the node-test coverage section), this change against the previous compiler | PASS 202 of 964 before, 442 after; no case that passed fails; 36 fewer cases stop while running (DReduce's any-axes form), none newly fails |
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


## Explicit limitations

- This compiler implements a **validated subset**, not the entire ONNX specification.
  Unsupported operators, attributes, element types, and shape forms must be
  rejected. Runtime-dimension generation accepts each operator from the oldest opset whose
  definition it implements, through opset 20; see
  [control flow and sequences](#control-flow-and-sequences--september-16-2026).
  No accepted form is currently known to compute a wrong answer; see
  [node-test coverage](#node-test-coverage--september-16-2026).
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
