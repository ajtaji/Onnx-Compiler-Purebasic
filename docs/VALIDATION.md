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

## Explicit limitations

- This compiler implements a **validated subset**, not the entire ONNX specification.
  Unsupported operators, attributes, element types, and shape forms must be
  rejected. Runtime-dimension generation checks its implemented opset-20 surface.
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
