# Integrating generated models

[Home](../README.md) · [Quick start](QUICKSTART.md) · [Architecture](ARCHITECTURE.md)

## Choose the destination, not the machine running this tool

The generator runs on Windows. The selected target controls the emitted BASIC
dialect, runtime sources, capacity defaults, and launch contract.

| CLI target | Build generated application with | Application supplies |
|---|---|---|
| `windows` | PureBasic x64, native backend, thread-safe and optimizer | Input data, output handling, and the weight-pack file. |
| `pi4` | Separate PureMetal toolchain for AArch64 | Startup, memory map, weights, working arena, and device I/O. |
| `unoq` | Separate PureMetal AArch64 UEFI integration | UEFI application/loader, weight loading, arena, and I/O. This profile is not a sketch for a companion microcontroller. |
| `pico` | Separate PureMetal RP2040 toolchain | Firmware startup, working arena, and I/O; checked weights are embedded. |
| `pico2` | Separate PureMetal RP2350 Arm toolchain | Firmware startup, working arena, and I/O; this is the Cortex-M33 profile, not RISC-V. |

**PureMetal is not distributed here.** If you only have PureBasic, Windows is
the directly buildable execution target. Generating another dialect does not
install its compiler or produce flashable firmware.

The PureMetal Forge download from [puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/)
is that toolchain, and it also includes this generator ready to run in
`PureMetalForge\tools\onnx\`. With the download, use
`PureMetalForge\tools\onnx\PureMetalOnnxCompilerCLI.exe` wherever a command
below says `.\bin\PureMetalOnnxCompilerCLI.exe`; the `pico2` command below was
run that way. The download's generator is the one built for that release, so
the [Pi 4 Kokoro sentence](../examples/pi4/kokoro-sentence/README.md), which
calls `PmOnnxAnvilEnter`, needs a newer one built from this repository.

## Generation commands

After creating the demo models, choose any one of the five target IDs:

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --compile output\dense.onnx --output output\dense-pico2 --target pico2 --precision int8
if ($LASTEXITCODE -ne 0) { throw 'Do not use stale output: generation failed' }
```

The output prefix must name a file prefix **inside an existing directory**.
The generated source, `.pmw`, `.json` manifest, and `.runtime/` directory form a
set. Keep them together, and use a new prefix when comparing different targets
or precision choices. Do not treat an older file as new output after a failure.

## Two model interfaces

Read the API in the emitted source: fixed-shape and runtime-dimension models
have different entry points. Neither is a one-shot executable.

### Fixed shapes — `PmOnnx...`

The generated header and manifest describe exact input/output layouts and the
weight and arena sizes.

1. Supply aligned, non-overlapping weight and working-memory buffers for the
   model's complete lifetime. Use `#PMO_WEIGHT_FILE_BYTES` and `#PMO_ARENA_BYTES`.
2. Call `PmOnnxBindMemory(weights, weightBytes, arena, arenaBytes)` and check success.
3. Fill each buffer returned by `PmOnnxInputAddress(index)` using the declared type and shape.
4. Call `PmOnnxExecuteBound()` and check success before consuming outputs.
5. Read buffers returned by `PmOnnxOutputAddress(index)` before the next execution.
6. Repeat input/execution/output as needed. Call `PmOnnxUnbindMemory()` before
   freeing your buffers. Use `PmOnnxLastErrorNode()` for failing-node context.

Input/output indices are zero-based. The Pico profiles emit an embedded weight
array; use the generated source's address and size, not a guessed flash address.
Binding does not transfer ownership of caller memory.

### Runtime dimensions — `PmModel...`

On Windows, call `PmModelInitialize(weightPath)` once. For each request:

1. Reset previous request tensors with `PmModelResetRequest()`.
2. Clear the request error as demonstrated in the example.
3. Get tensor IDs from `PmModelInput(index)`, allocate with the declared type,
   rank, and this request's dimensions, and fill the tensor data.
4. Call `PmModelExecute()`; check its result and `DError`.
5. Consume output tensor IDs returned by `PmModelOutput(index)` while their storage is live.

Finally call `PmModelClose()`. The [complete Windows example](../examples/RunTwiceWindows.pb)
shows this lifecycle, including cleanup and validation. These generated globals
represent one model instance: serialize requests rather than calling it concurrently.

Portable output replaces file-loading initialization with
`PmModelBindMemory(weights, weightBytes, arena, arenaBytes)`. Assets remain
caller-owned. Its dynamic allocator uses your arena; allocation errors must be
handled. Its `DError` is an ASCII string pointer, unlike the Windows string.
The runtime currently caps its working arena at 1 GiB. This is not a promise
that every model/request fits that limit.

## Precision is a tradeoff, not a fit guarantee

| Option | What changes | What does not automatically shrink |
|---|---|---|
| `fp32` | Baseline floating-point weights and execution. | Nothing reduced. |
| `int8` | Eligible constant linear weights; dynamic per-row activation quantization, INT32 accumulation, FP32 boundaries; the same bits on every target. | Untransformed operators and general activation storage. |
| `fp16` | Stored weights, decoded once for resident use. | Decoded FP32 weights and activation requirements. |
| `bf16` | Stored weights with a different precision/range tradeoff, decoded once. | Decoded FP32 weights and activations. |
| `int4` | Aggressively reduced stored weights with scales, decoded once. | Resident FP32 decode buffers and activations. |

Check the actual manifest and compare numerical accuracy on representative
inputs. Choosing INT8 does not convert every operation into integer arithmetic;
choosing INT4 does not make an arbitrarily large model fit a microcontroller.

## Pico capacity

The profiles reserve space for application/runtime overhead and reject model
requirements beyond their configured budgets. Those checks do not know the
final size of your complete firmware or all of your peripheral buffers. Inspect
the downstream compiler/linker map and leave room for stacks and application state.
Runtime-dependent requests can still fail allocation and require error handling.

Full Kokoro-82M and its text assets are **not Pico-sized models**, even with reduced
weight storage. Use the Pico targets for suitably small models.

## AArch64 compiler compatibility

The native CRC32, LSTM and trigonometry routines use PureMetal's invocation-local
frame calling convention. Their scalar arguments remain live in `x0` through
`x3` at the first inline-assembly instruction, after generated parameter homing.
They do not reload parameters from process-wide absolute symbols. Use a matching
PureMetal compiler; compatibility with older compiler builds is not established.

After updating these runtime sources, rebuild the ONNX generator, regenerate
the application's complete source/runtime set, and separately rebuild the
application. Do not mix old generated runtime files with a new compiler ABI.
Invocation-local frames alone do not make the model's shared runtime state
safe for concurrent requests; the serialization requirement above still applies.

## A worked Pi 4 example

[`examples/pi4/kokoro-sentence`](../examples/pi4/kokoro-sentence/README.md) is a
complete `pi4` application: an Anvil payload that binds the full Kokoro-82M
model and its text assets at fixed addresses, calls `PmOnnxAnvilEnter` so the
convolutions run on four cores, renders one sentence and leaves the samples for
the host. Its README builds the payload that ran on 2026-09-16 byte for byte
from a clean checkout and lists the board steps. Further Pi 4 examples go
beside it under [`examples/pi4/`](../examples/pi4/README.md).

## Hardware proof

This export was verified by building and executing Windows examples, and by
generating source for all five profiles. No board was accessed for this export.
AArch64/Pico source generation is not a downstream firmware build, a boot test,
or hardware numerical/timing proof. Those belong to the target application's
integration and test process.

Subsequent bounded Pi 4 resident-model and native-math measurements are recorded
in [Validation](VALIDATION.md#aarch64-native-math-checks--september-10-2026).
They do not certify UNO Q or Pico hardware, arbitrary model sizes, or speech
latency. Regenerate the source closure and separately rebuild your application
to incorporate runtime changes; an already-built model does not update itself.
