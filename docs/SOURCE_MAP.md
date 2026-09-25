# Source file guide

[Home](../README.md) · [Build](BUILD.md) · [Architecture](ARCHITECTURE.md)

The application and all generation logic are PureBasic. The `.pmi` files are
BASIC runtime source templates: some are shared with PureBasic, while the
target-specific templates are emitted for a separate PureMetal application build.
No proprietary language compiler implementation is included. The Python files
are the node-test harness and its cases under `tests/node_suite/`, and the host
tools of the Pi 4 example under `examples/pi4/` - developer checks and board
tools that building and using the compiler never need.

## Start here

| File | Role |
|---|---|
| [Build.pb](../Build.pb) | Separate developer helper; builds the native CLI and GUI with the installed PureBasic compiler. |
| [PureMetalOnnxCompiler.pb](../src/PureMetalOnnxCompiler.pb) | CLI entry point, command dispatch, inspection, checks, generation, and optional tracing. |
| [PureMetalOnnxCompilerUI.pb](../src/PureMetalOnnxCompilerUI.pb) | Graphical entry point and its native include closure. |
| [CreateDemoModels.pb](../examples/CreateDemoModels.pb) | Creates the arithmetic demo models from embedded protobuf bytes. |
| [RunTwiceWindows.pb](../examples/RunTwiceWindows.pb) | Complete consumer of a resident generated Windows model, with changing input lengths. |
| [ResidentProgress.md](ResidentProgress.md) | Developer contract for resident progress callbacks, cancellation, non-reentry, and platform service boundaries. |

## Raspberry Pi 4 examples

Everything under `examples/pi4/`. The `.pi4` and `.pbi` files are PureMetal source
for a board payload, not PureBasic; the `.py` files are host-side tools.

| File | Role |
|---|---|
| [README.md](../examples/pi4/README.md) | Index of the Pi 4 examples. |
| [kokoro-sentence/README.md](../examples/pi4/kokoro-sentence/README.md) | The recipe: pinned inputs, the two compiler builds, the model and asset packs with their hashes, the payload build, the board steps and the measured result. |
| [KokoroSentence.pi4](../examples/pi4/kokoro-sentence/KokoroSentence.pi4) | The Anvil payload that spoke the sentence: service table, four-core entry, MMU and caches, model and text binds, one render, KTRC trace, per-node tick rows, result block and sample hash. |
| [node_output_map.pbi](../examples/pi4/kokoro-sentence/node_output_map.pbi) | Per-node first output tensor and output count for the pinned graph, read only by the runner's optional NaN scan. |
| [kokoro_readback.py](../examples/pi4/kokoro-sentence/kokoro_readback.py) | Host tool: the four-core dispatch counter, the run's trace and result block, and the waveform checked against the board's own SHA-256; uses the Anvil repository's console client and readback reader by path. |
| [f32_to_wav.py](../examples/pi4/kokoro-sentence/f32_to_wav.py) | Host tool: raw FLOAT32 samples to a 16-bit PCM WAV and a lossless float WAV, counting clamped samples. |

## Compiler modules

All of these live under `src/compiler/`.

| File | Responsibility |
|---|---|
| [onnx_wire.pbi](../src/compiler/onnx_wire.pbi) | Bounded protobuf wire reader: varints, field keys, length-delimited spans, and input limits. |
| [onnx_model.pbi](../src/compiler/onnx_model.pbi) | Model, graph, node, attribute, shape, and tensor parsing. Initializer slices reference immutable model bytes. |
| [npy_reader.pbi](../src/compiler/npy_reader.pbi) | Checked C-order numeric NPY input reader for reference/trace workflows. Rejects unsupported layouts. |
| [onnx_rank_infer.pbi](../src/compiler/onnx_rank_infer.pbi) | Infers justified types, ranks, and dimensions for temporary trace outputs. |
| [onnx_proto_writer.pbi](../src/compiler/onnx_proto_writer.pbi) | Writes temporary protobuf graph output descriptions for tracing without editing the original model. |
| [onnx_runtime.pbi](../src/compiler/onnx_runtime.pbi) | Optional ONNX Runtime C API bridge for host reference execution, pinned to API version 29. Not generated inference. |
| [onnx_trace.pbi](../src/compiler/onnx_trace.pbi) | Loads supplied feeds, runs optional host tracing, and records concrete output shapes. |
| [onnx_ir.pbi](../src/compiler/onnx_ir.pbi) | Target-neutral graph representation, constant folding, aliases, lifetimes, and arena planning. |
| [onnx_quant.pbi](../src/compiler/onnx_quant.pbi) | Eligible constant-weight INT8 conversion, scales, layouts, the measured wide-weight precision plan of the pinned speech model, and quantization scratch planning. |
| [onnx_storage.pbi](../src/compiler/onnx_storage.pbi) | FP16/BF16/INT4 storage conversion, rounding, decoded-memory budgets, and decoder template embedding. |
| [onnx_pack.pbi](../src/compiler/onnx_pack.pbi) | Deterministic checked PMONNXW weight-pack construction, identity, CRC, and alignment. |
| [onnx_targets.pbi](../src/compiler/onnx_targets.pbi) | Five target profiles: dialect, suffix, launch contract, kernels, and capacity defaults. |
| [onnx_compile.pbi](../src/compiler/onnx_compile.pbi) | Generation coordinator: options, validation, routing (a graph with undeclared values goes to the runtime-dimension path), capacity checks, packing, and manifests. |
| [onnx_opsets.pbi](../src/compiler/onnx_opsets.pbi) | The highest ai.onnx opset accepted (27) and the operators whose definition changed above 20 (their ceilings), used by both paths; the private opset_ceiling_check.py holds it to onnx's schema history. |
| [onnx_emit_random.pbi](../src/compiler/onnx_emit_random.pbi) | The random operators (RandomNormal, RandomUniform and their Like forms, Bernoulli, Multinomial) in both paths: validation sentences, emitted calls, the seed procedure, the manifest entry and the `--random-inputs` comparison mode. |
| [onnx_emit.pbi](../src/compiler/onnx_emit.pbi) | Fixed-shape source generation and the `PmOnnx...` caller-owned-memory API. |
| [onnx_emit_norm_small.pbi](../src/compiler/onnx_emit_norm_small.pbi) | Fixed-shape forms of InstanceNormalization, TopK, ScatterElements (and Scatter-9, read as it), ReduceMax, ReduceProd, Not and Pad: refusal sentences, the per-node call into the kernels, and the opset floors of these operators. |
| [onnx_forms.pbi](../src/compiler/onnx_forms.pbi) | Attribute forms shared by both emitters: which attribute values each operator implements, and the refusal sentence naming operator, node, attribute and value for the rest. |
| [onnx_dynamic_emit.pbi](../src/compiler/onnx_dynamic_emit.pbi) | Runtime-dimension source generation, resident constants, request lifetimes, `PmModel...` API, and embedded runtime export. |
| [onnx_control.pbi](../src/compiler/onnx_control.pbi) | If, Loop and the sequence operators, first half: Constant lowering (dense and sparse, for both paths), Scan and SequenceMap rewritten into Loop, subgraph scopes by renaming, value kinds (tensor or sequence), per-operator opset floors, and validation sentences. |
| [onnx_control_emit.pbi](../src/compiler/onnx_control_emit.pbi) | If, Loop and the sequence operators, second half: one generated procedure per subgraph, the inline control code, and the move-or-copy decisions from last uses. |
| [onnx_dynamic_norm_small.pbi](../src/compiler/onnx_dynamic_norm_small.pbi) | Runtime-dimension forms of the same operators and of Identity: validation by form, element type and opset, and their call text. |
| [onnx_emit_ops.pbi](../src/compiler/onnx_emit_ops.pbi) | The operators that complete the operator set (Erf ... Einsum, Dropout, CastLike, Size; Tan ... Atanh, the bitwise operators, Hardmax, the normalizations, EyeLike, Det, Compress, ReverseSequence, Upsample, RNN, GRU, NonMaxSuppression, RoiAlign, GridSample; the quantized operators QuantizeLinear ... QLinearConv; the windows, DFT and the losses; Col2Im, CenterCropPad, MaxUnpool, AffineGrid, MaxRoiPool, DeformConv; Unique; Swish, RMSNormalization, CumProd, BitCast, RotaryEmbedding, TensorScatter, Attention, LinearAttention, CausalConvWithState): which they are and their opset floors, Einsum's equation parser, the arena scratch Det and the recurrent operators need, and their fixed-shape forms - refusal sentences and the per-node procedure that fills the kernel parameters. |
| [onnx_dynamic_ops.pbi](../src/compiler/onnx_dynamic_ops.pbi) | Runtime-dimension forms of the same operators: validation by attribute, declared element type and opset, and their call text. |
| [onnx_kokoro.pbi](../src/compiler/onnx_kokoro.pbi) | Native preparation and checking of Kokoro reference inputs; shared text/asset validation. Not a dictionary or voice-pack builder. |
| [kokoro_asset_pack.pbi](../src/compiler/kokoro_asset_pack.pbi) | Native checked PMVOICE construction from caller-supplied raw voices; payload hashes, CRC, finite-value checks and no-overwrite publication. No download or dictionary construction. |
| [onnx_ui.pbi](../src/compiler/onnx_ui.pbi) | Model/target/precision controls, generation worker, CLI logging/cancellation, and Kokoro input preparation. |

## Tensor execution sources

These sources are embedded into the generator and exported beside generated models.
An emitted application uses its exported runtime, not files from this checkout.

| File | Responsibility |
|---|---|
| [tensor_fp32.pmi](../runtime/tensor_fp32.pmi) | Portable tensor foundation and scalar kernels, and the INT8 scheme: its portable definition and the AArch64 Advanced SIMD bodies that compute the same bits. Its square root is correctly rounded on every target (fsqrt, vsqrt.f32, or the integer root on the Pico): the same bits as the Windows program. |
| [tensor_pool_windows.pbi](../runtime/tensor_pool_windows.pbi) | The Windows worker pool: taken when a model is bound and parked (blocked, running nothing) when it is unbound, never ended; how many processors the process may use (affinity mask, processor groups, default CPU set), lowered by `--threads` or the run-time setting; tasks over disjoint output ranges, the caller's floating-point environment in every worker, no nesting. |
| [tensor_fp32_windows.pbi](../runtime/tensor_fp32_windows.pbi) | Windows-native tensor foundation and host execution support; its operators split across the pool by output elements, rows or channels, each output computed exactly as on one thread. |
| [tensor_simd_windows.pbi](../runtime/tensor_simd_windows.pbi) | Native x64 SSE2/AVX kernels: dense products split by row and column blocks, convolution by position blocks and output-row groups, recurrent steps by units, reductions by rows, all on the pool. No inference DLL. |
| [tensor_fp32_neon.pmi](../runtime/tensor_fp32_neon.pmi) | AArch64 NEON acceleration selected by the Pi 4 and UNO Q profiles, including the four-core split of FP32 and INT8 convolution. |
| [tensor_fp32_a64.pmi](../runtime/tensor_fp32_a64.pmi) | Retained ordered scalar-register AArch64 acceleration; distinct from the current NEON profile. |
| [tensor_dynamic_windows.pbi](../runtime/tensor_dynamic_windows.pbi) | Runtime tensor descriptors, dimensions, typed operators, checked heap allocation, errors, and live/peak storage tracking; its index loops split across the pool, and the block cache that spares a multi-threaded model the page faults of fresh large blocks. |
| [tensor_dynamic_portable.pmi](../runtime/tensor_dynamic_portable.pmi) | Portable dynamic tensor implementation using a caller-owned arena with splitting and coalescing. |
| [tensor_control_windows.pbi](../runtime/tensor_control_windows.pbi) | Sequences as one block each, value moves and copies for If and Loop, and the sequence operators, on Windows. Exported only for a model that uses them. |
| [tensor_control_portable.pmi](../runtime/tensor_control_portable.pmi) | The same for the bare-metal targets, allocating from the caller's bound arena. |
| [tensor_dynamic_forms_windows.pbi](../runtime/tensor_dynamic_forms_windows.pbi) | Windows runtime entries for attribute forms that need a run-time extent: Shape start/end, Conv and ConvTranspose auto_pad SAME, ScatterND reductions. |
| [tensor_dynamic_forms_portable.pmi](../runtime/tensor_dynamic_forms_portable.pmi) | The same entries for the bare-metal targets. |
| [tensor_norm_small_windows.pbi](../runtime/tensor_norm_small_windows.pbi) | Windows kernels for InstanceNormalization (scalar reference and bit-identical SSE body), TopK, ScatterElements, ReduceMax/ReduceProd, Not and Pad. |
| [tensor_norm_small.pmi](../runtime/tensor_norm_small.pmi) | The same kernels for the bare-metal targets, with the bit-identical Advanced SIMD InstanceNormalization body on AArch64. |
| [tensor_dynamic_norm_small_windows.pbi](../runtime/tensor_dynamic_norm_small_windows.pbi) | Runtime-dimension wrappers for those kernels and for Identity on Windows: shape and type checks, outputs, scratch. |
| [tensor_dynamic_norm_small_portable.pmi](../runtime/tensor_dynamic_norm_small_portable.pmi) | The portable twin of the Windows wrappers, in caller-owned arena memory. |
| [tensor_features_all.pmi](../runtime/tensor_features_all.pmi) | Feature definitions enabling the complete runtime surface when needed. Generated programs select their required features. |
| [tensor_ops.pmi](../runtime/tensor_ops.pmi) | The kernels of the operators in onnx_emit_ops.pbi: elementwise functions, variadic and broadcasting operators, reductions, pooling, data movement, Einsum, the normalizations, Det, the recurrent operators, NonMaxSuppression, RoiAlign, Upsample, GridSample, the quantized operators (with the exact binary64 requantization their reference performs), the windows, DFT and the losses, Col2Im, CenterCropPad, MaxUnpool, AffineGrid, MaxRoiPool, DeformConv, Unique, Multinomial's class choice, RMSNormalization, CumProd, RotaryEmbedding, TensorScatter, Attention, LinearAttention and CausalConvWithState, and the math they need (exp, expm1, log, log1p, tanh, erf, a correctly rounded sqrt, tan with an integer reduction by pi/2, asin, acos and the hyperbolic functions) as fixed sequences of single binary32 operations. One source for every target, included unchanged, so every target computes the same bits. |
| [tensor_dynamic_ops.pmi](../runtime/tensor_dynamic_ops.pmi) | Runtime-dimension wrappers for those kernels: shape and type checks at run time, outputs, parameters (a > b that can meet a NaN is PmOpGt, IEEE 754's ordered comparison, which tests the bits first only for a host compiler older than 6.41); and the runtime-dimension Relu, Flatten and BatchNormalization, over the fixed-shape kernels. One source for both dialects. |
| [tensor_dynamic_ops_windows.pbi](../runtime/tensor_dynamic_ops_windows.pbi), [tensor_dynamic_ops_portable.pmi](../runtime/tensor_dynamic_ops_portable.pmi) | The one line each dialect differs in for those wrappers: how a tensor's extent is read. |
| [tensor_random.pmi](../runtime/tensor_random.pmi) | The specified generator behind the random operators: Threefry-2x32-20, the node and element keys, the uniform and Box-Muller transforms in single binary32 operations, and Bernoulli's and Multinomial's draws. One source for every target, included unchanged. |
| [tensor_random_dynamic_windows.pbi](../runtime/tensor_random_dynamic_windows.pbi) | Runtime-dimension wrappers for the random operators on Windows, including the `--random-inputs` type and shape checks. |
| [tensor_random_dynamic_portable.pmi](../runtime/tensor_random_dynamic_portable.pmi) | The portable twin of the Windows wrappers. |
| [weight_storage.pmi](../runtime/weight_storage.pmi) | Reduced-weight decoder template; its type placeholder is substituted during emission, so it is not a standalone include. |
| [math_a64.pmi](../runtime/math/math_a64.pmi) | Bare-metal AArch64 math source template. |
| [math_m0.pmi](../runtime/math/math_m0.pmi) | Bare-metal Cortex-M0+ math source template. |
| [math_m33.pmi](../runtime/math/math_m33.pmi) | Bare-metal Cortex-M33 math source template. |

## Optional speech and text adapters

These implement application-level speech support, not general ONNX graph semantics.
The checked external assets they consume are described in [Kokoro](KOKORO.md).

| File | Responsibility |
|---|---|
| [pcm_fp32.pmi](../runtime/pcm_fp32.pmi) | Checked FP32-to-PCM conversion and WAV support with finite-value checks and saturation. |
| [kokoro_assets.pmi](../runtime/kokoro_assets.pmi) | Token vocabulary, strict UTF-8, voice/job binary formats, identity and integrity validation. |
| [kokoro_g2p.pmi](../runtime/kokoro_g2p.pmi) | Checked English pronunciation lookup, supported inflections, numbers, phoneme overrides, and unknown-word spelling. |
| [kokoro_g2p_extra.pmi](../runtime/kokoro_g2p_extra.pmi) | Checked supplementary pronunciation-pack attachment and lookup. |
| [kokoro_speech_windows.pbi](../runtime/kokoro_speech_windows.pbi) | Generated Windows speech window; binds assets, prepares inputs, invokes the resident model, and reports errors. |
| [kokoro_reader_windows.pbi](../runtime/kokoro_reader_windows.pbi) | Book-style chunking and normalization, bounded producer/consumer queue, Windows audio, Pause/Resume/Stop. |
| [kokoro_voices_windows.pbi](../runtime/kokoro_voices_windows.pbi) | Local prepared-voice discovery, labeling, deduplication, and selection. Does not download voices. |
| [kokoro_text_portable.pmi](../runtime/kokoro_text_portable.pmi) | Resident AArch64 text-to-model adapter with caller-owned immutable assets and borrowed output buffers; no OS or board I/O. |

## Developer checks

| File | Responsibility |
|---|---|
| [node_suite.py](../tests/node_suite/node_suite.py) | Node-test coverage harness: locates the pinned ONNX backend node-test corpus, compiles each case for Windows, builds and runs it, compares outputs, writes the score. `--self-check` proves it reports failures. |
| [node_driver_windows.pbi](../tests/node_suite/node_driver_windows.pbi) | The one generic driver the harness builds around each emitted model; serves both emitted APIs through a small binary request and result format. |
| [targeted_control.py](../tests/node_suite/targeted_control.py) | Targeted If, Loop and sequence cases, the SplitToSequence node tests re-imported at opset 20, and Kitten TTS nano's own control-flow nodes. |
| [pi4_control_gate.py](../tests/node_suite/pi4_control_gate.py) | Runs node-suite case folders on the Pi 4 target in an independent A64 instruction interpreter. |
| [targeted_defaults.py](../tests/node_suite/targeted_defaults.py) | Opset-20 cases for attribute forms the node tests do not reach: defaults written out, auto_pad, Shape slices, ScatterND reductions, and the forms that must be refused; each built for both emitters and checked against the ONNX reference and ONNX Runtime. |
| [targeted_ops.py](../tests/node_suite/targeted_ops.py) | The operators of onnx_emit_ops.pbi: their official node tests published above opset 20, re-imported at opset 20, and targeted forms built for both emitters, checked against the ONNX reference and ONNX Runtime; the refusals that must name the form. |
| [targeted_optional_outputs.py](../tests/node_suite/targeted_optional_outputs.py) | Nodes with optional outputs left out (LSTM, LayerNormalization, BatchNormalization), for both emitters; `--mutants` rebuilds the compiler without the forum 988 correction and requires the cases to fail. |
| [results/](../tests/node_suite/results/) | Published scores: one `.md` summary and one `.json` record per case for each measured run. |

The rest of `tests/` is not part of this export; see [validation](VALIDATION.md).

## Which file should I change?

| Change | Start with |
|---|---|
| Add a target | `onnx_targets.pbi`, then runtime export and launch handling in the emitters. |
| Add an ONNX operator | Parser/IR validation, both applicable emitters, and both relevant execution runtimes. |
| Change supported precision | `onnx_quant.pbi` or `onnx_storage.pbi`, packing, runtime kernels/decoder, and capacity reporting. |
| Improve Windows tensor speed | `tensor_simd_windows.pbi`, `tensor_pool_windows.pbi`; preserve shape, tail, and numerical contracts, and split only by outputs: every value computed by one task in the one-thread order (`mt_split_gate.py` holds it at 1, 2, 3 and 8 threads). |
| Improve AArch64 tensor speed | `tensor_fp32_neon.pmi`; verify tolerances and memory access bounds. |
| Change the generator window | `onnx_ui.pbi`; keep generation separate from downstream compilation. |
| Change speech controls or reading | `kokoro_speech_windows.pbi` and `kokoro_reader_windows.pbi`. |
| Change pronunciation behavior | `kokoro_g2p.pmi`, `kokoro_g2p_extra.pmi`, and the adapters' normalization boundaries. |

Rebuild the generator after changing any embedded runtime. Regenerate and
separately rebuild model applications to pick up those changes.
