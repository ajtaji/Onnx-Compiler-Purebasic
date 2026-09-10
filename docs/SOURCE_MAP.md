# Source file guide

[Home](../README.md) · [Build](BUILD.md) · [Architecture](ARCHITECTURE.md)

The application and all generation logic are PureBasic. The `.pmi` files are
BASIC runtime source templates: some are shared with PureBasic, while the
target-specific templates are emitted for a separate PureMetal application build.
No Python files or proprietary language compiler implementation are included.

## Start here

| File | Role |
|---|---|
| [Build.pb](../Build.pb) | Separate developer helper; builds the native CLI and GUI with the installed PureBasic compiler. |
| [PureMetalOnnxCompiler.pb](../src/PureMetalOnnxCompiler.pb) | CLI entry point, command dispatch, inspection, checks, generation, and optional tracing. |
| [PureMetalOnnxCompilerUI.pb](../src/PureMetalOnnxCompilerUI.pb) | Graphical entry point and its native include closure. |
| [CreateDemoModels.pb](../examples/CreateDemoModels.pb) | Creates the arithmetic demo models from embedded protobuf bytes. |
| [RunTwiceWindows.pb](../examples/RunTwiceWindows.pb) | Complete consumer of a resident generated Windows model, with changing input lengths. |
| [ResidentProgress.md](ResidentProgress.md) | Developer contract for resident progress callbacks, cancellation, non-reentry, and platform service boundaries. |

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
| [onnx_quant.pbi](../src/compiler/onnx_quant.pbi) | Eligible constant-weight INT8 conversion, scales, layouts, and quantization scratch planning. |
| [onnx_storage.pbi](../src/compiler/onnx_storage.pbi) | FP16/BF16/INT4 storage conversion, rounding, decoded-memory budgets, and decoder template embedding. |
| [onnx_pack.pbi](../src/compiler/onnx_pack.pbi) | Deterministic checked PMONNXW weight-pack construction, identity, CRC, and alignment. |
| [onnx_targets.pbi](../src/compiler/onnx_targets.pbi) | Five target profiles: dialect, suffix, launch contract, kernels, and capacity defaults. |
| [onnx_compile.pbi](../src/compiler/onnx_compile.pbi) | Generation coordinator: options, validation, routing, capacity checks, packing, and manifests. |
| [onnx_emit.pbi](../src/compiler/onnx_emit.pbi) | Fixed-shape source generation and the `PmOnnx...` caller-owned-memory API. |
| [onnx_dynamic_emit.pbi](../src/compiler/onnx_dynamic_emit.pbi) | Runtime-dimension source generation, resident constants, request lifetimes, `PmModel...` API, and embedded runtime export. |
| [onnx_kokoro.pbi](../src/compiler/onnx_kokoro.pbi) | Native preparation and checking of Kokoro reference inputs; shared text/asset validation. Not a dictionary or voice-pack builder. |
| [kokoro_asset_pack.pbi](../src/compiler/kokoro_asset_pack.pbi) | Native checked PMVOICE construction from caller-supplied raw voices; payload hashes, CRC, finite-value checks and no-overwrite publication. No download or dictionary construction. |
| [onnx_ui.pbi](../src/compiler/onnx_ui.pbi) | Model/target/precision controls, generation worker, CLI logging/cancellation, and Kokoro input preparation. |

## Tensor execution sources

These sources are embedded into the generator and exported beside generated models.
An emitted application uses its exported runtime, not files from this checkout.

| File | Responsibility |
|---|---|
| [tensor_fp32.pmi](../runtime/tensor_fp32.pmi) | Portable tensor foundation and scalar kernels, plus selected target INT8 inner loops. |
| [tensor_fp32_windows.pbi](../runtime/tensor_fp32_windows.pbi) | Windows-native tensor foundation and host execution support. |
| [tensor_simd_windows.pbi](../runtime/tensor_simd_windows.pbi) | Native x64 SSE2/AVX kernels, threaded dense operations, convolution, and reductions. No inference DLL. |
| [tensor_fp32_neon.pmi](../runtime/tensor_fp32_neon.pmi) | AArch64 NEON acceleration selected by the Pi 4 and UNO Q profiles. |
| [tensor_fp32_a64.pmi](../runtime/tensor_fp32_a64.pmi) | Retained ordered scalar-register AArch64 acceleration; distinct from the current NEON profile. |
| [tensor_dynamic_windows.pbi](../runtime/tensor_dynamic_windows.pbi) | Runtime tensor descriptors, dimensions, typed operators, checked heap allocation, errors, and live/peak storage tracking. |
| [tensor_dynamic_portable.pmi](../runtime/tensor_dynamic_portable.pmi) | Portable dynamic tensor implementation using a caller-owned arena with splitting and coalescing. |
| [tensor_features_all.pmi](../runtime/tensor_features_all.pmi) | Feature definitions enabling the complete runtime surface when needed. Generated programs select their required features. |
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

## Which file should I change?

| Change | Start with |
|---|---|
| Add a target | `onnx_targets.pbi`, then runtime export and launch handling in the emitters. |
| Add an ONNX operator | Parser/IR validation, both applicable emitters, and both relevant execution runtimes. |
| Change supported precision | `onnx_quant.pbi` or `onnx_storage.pbi`, packing, runtime kernels/decoder, and capacity reporting. |
| Improve Windows tensor speed | `tensor_simd_windows.pbi`; preserve shape, tail, and numerical contracts. |
| Improve AArch64 tensor speed | `tensor_fp32_neon.pmi`; verify tolerances and memory access bounds. |
| Change the generator window | `onnx_ui.pbi`; keep generation separate from downstream compilation. |
| Change speech controls or reading | `kokoro_speech_windows.pbi` and `kokoro_reader_windows.pbi`. |
| Change pronunciation behavior | `kokoro_g2p.pmi`, `kokoro_g2p_extra.pmi`, and the adapters' normalization boundaries. |

Rebuild the generator after changing any embedded runtime. Regenerate and
separately rebuild model applications to pick up those changes.
