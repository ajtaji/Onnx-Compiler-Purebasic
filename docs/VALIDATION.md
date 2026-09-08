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

## Explicit limitations

- This compiler implements a **validated subset**, not the entire ONNX specification.
  Unsupported operators, attributes, element types, and shape forms must be
  rejected. Runtime-dimension generation checks its implemented opset-20 surface.
- Windows x64 is the verified host. Linux/macOS hosting, other PureBasic
  versions, and alternate PureBasic backends are not certified by this export.
- Five-target **generation** does not establish downstream bare-metal builds,
  bootability, hardware numerical accuracy, or timing. Those were not tested here.
- Dynamic requests still require sufficient working memory. A successful
  generation is not an unlimited-dimensions or arbitrary-input promise.
- INT8 is selective; FP16/BF16/INT4 storage savings do not remove the resident
  decoded FP32 or activation-memory costs. Compare output quality yourself.
- SIMD/NEON may change floating-point summation order. Do not demand bitwise
  equality where the execution contract calls for numerical tolerances.
- Model execution uses generated globals. Serialize calls to one model instance.
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
