# Kokoro speech integration

[Home](../README.md) · [Targets](TARGETS.md) · [Source map](SOURCE_MAP.md)

Kokoro is an optional application of the general model compiler. It is not a
requirement for building the tools or running the arithmetic examples.

## What is included

- Native generation for the supported full Kokoro-82M graph.
- A Windows speech window with local voice selection and book-style queued
  reading, Pause, Resume, and Stop.
- A resident text adapter for the Pi 4 and UNO Q AArch64 profiles.
- Checked text, pronunciation, token, and voice format readers written in BASIC.

The model is initialized once. New text becomes new tensor inputs; it does
not generate or compile another model per word or sentence.

## What you must supply separately

**This repository does not contain the model or prepared speech data, and does
not currently include a PureBasic asset-pack builder.** The compiler is fully
buildable without them, but the speech demo is not ready to speak from a fresh
checkout until compatible prepared assets are supplied.

| Asset | Contract expected by the current adapter |
|---|---|
| Full Kokoro ONNX model | Pinned full-precision v1.0 export, 325,532,232 bytes; SHA-256 below. |
| `kokoro_us_english.pmg2p` | Checked base English pronunciation pack, 12,613,954 bytes. |
| `kokoro_us_extra.pmg2p` | Checked supplementary pronunciation pack, 5,397,230 bytes. Required by the Windows reader; optional but recommended in the portable adapter. |
| One or more `.pmvoice` files | Checked PMVOICE format, 522,368 bytes each, with the expected model identity and dimensions. |

Pinned model SHA-256:

```text
8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb
```

File size alone is not enough: loaders also validate the format and identity.
Raw upstream voice `.bin` files, dictionary JSON, and arbitrary quantized model
exports cannot simply be renamed into these formats. Obtain compatible prepared
packs and keep their accompanying licenses. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Windows reader

With the compatible full model available, create an output directory and generate:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
.\bin\PureMetalOnnxCompilerCLI.exe --compile 'C:\Models\kokoro.onnx' --output output\kokoro-reader --target windows --precision fp32 --speech-ui
```

Build the emitted `.pb` application separately in PureBasic x64 with thread-safe
and optimizer enabled. Keep its `.pmw` and `.runtime/` files with the source/build.
In the speech window, select your `.pmvoice` and base `.pmg2p` files. Place
`kokoro_us_extra.pmg2p` beside the base pronunciation file. Voice discovery is
local; it does not fetch a voice catalog from the Internet.

Use FP32 as your correctness baseline. INT8 and other reductions can be compared
by generating distinct output prefixes. Performance depends on the graph,
processor, request length, and precision; no real-time reading guarantee is made.

## Pi 4 and UNO Q

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --compile 'C:\Models\kokoro.onnx' --output output\kokoro-pi4 --target pi4 --precision fp32 --kokoro-text
```

Use `--target unoq` for the UNO Q AArch64 profile. The generated runtime exports
the portable text adapter. The host application binds the model's weights/arena,
then calls `PmKokoroTextBind()` with immutable pronunciation and voice buffers.
Use the emitted adapter's prepare/render/voice-selection API and check each
return value and error field.

The adapter supports one serialized request at a time, limits a text request
to 4,096 UTF-8 bytes and the supported phoneme sequence length, and returns
borrowed model output. Consume or copy it before resetting the request. Audio
is 24 kHz mono floating-point PCM; playback, device drivers, asset loading, and
book-sized chunk scheduling belong to the application. Closing the text adapter
does not free the caller's immutable assets.

The normal text route is US English. It normalizes common book typography and
uses the checked dictionaries and supported word rules. Unknown words may be
spelled out; explicit `/phoneme/` overrides are vocabulary-checked. This is not
a claim of unrestricted multilingual linguistic coverage.

## Pico is not a full-Kokoro destination

The Pico and Pico 2 profiles exist for small models. Full Kokoro and its text
assets cannot fit their memory budgets. The speech/text switches explicitly
reject unsupported target combinations rather than offering a misleading demo.
