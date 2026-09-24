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

**This repository does not contain the model or prepared speech data.** Native
PureBasic voice-pack and pronunciation-pack builders are included. The compiler
is fully buildable without them, but the speech demo is not ready to speak from
a fresh checkout until compatible prepared assets are supplied.

| Asset | Contract expected by the current adapter |
|---|---|
| Full Kokoro ONNX model | Pinned full-precision v1.0 export, 325,532,232 bytes; SHA-256 below. |
| `kokoro_us_english.pmg2p` | Checked base English pronunciation pack, 12,613,954 bytes. |
| `kokoro_us_extra.pmg2p` | Checked supplementary pronunciation pack, 5,397,245 bytes. Required by the Windows reader; optional but recommended in the portable adapter. |
| One or more `.pmvoice` files | Checked PMVOICE format, 522,368 bytes each, with the expected model identity and dimensions. |

Pinned model SHA-256:

```text
8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb
```

File size alone is not enough: loaders also validate the format and identity.
Raw upstream voice `.bin` files, dictionary JSON, and arbitrary quantized model
exports cannot simply be renamed into these formats. Obtain compatible prepared
packs and keep their accompanying licenses. See [third-party notices](../THIRD_PARTY_NOTICES.md).

### Getting the generator

The commands on this page use `.\bin\PureMetalOnnxCompilerCLI.exe`, built from
this repository ([Build](BUILD.md)). The PureMetal Forge download from
[puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/) also includes the
generator, ready to run, as `PureMetalForge\tools\onnx\PureMetalOnnxCompilerCLI.exe`,
with the window `PureMetalOnnxCompiler.exe` beside it. That copy is the one built
for the release, and for speech the difference shows. Run with the pinned sources
below, the 2026-09-14 download's generator gave:

| Output | Result |
|---|---|
| `af_heart.pmvoice` | 522,368 bytes, identical to `main`'s |
| `kokoro_us_english.pmg2p` | 12,613,954 bytes, identical to `main`'s |
| `kokoro_us_extra.pmg2p` | **5,397,230 bytes, 66,304 entries: the older supplementary pack**, without the model's own name. `main`'s `--kokoro-verify-extra-dictionary` refuses it with `PMG2P-3020`. |
| `--compile ... --target pi4 --precision fp32 --kokoro-text` | the same `.pmw` weight pack; source without `PmOnnxAnvilEnter`, so a payload that calls it for four cores does not build |

For the supplementary pack and the four-core Pi 4 entry, build the generator
from this repository.

### Prepare a voice without Python

Supply a compatible upstream raw voice file: exactly 510 rows of 256
little-endian FP32 values (522,240 bytes). It must belong to the pinned Kokoro
model above, not an arbitrary model with the same dimensions. The tool validates
the data layout and records the pinned model identity; dimensions alone cannot
prove that an unknown voice is semantically compatible.

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-voice C:\Models\af_heart.bin --output C:\Models\af_heart.pmvoice
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-verify-voice C:\Models\af_heart.pmvoice
```

Optionally supply `--source-sha256` with the expected raw-file digest from a
trusted source. Packing rejects NaN/infinity, validates the completed header,
checksums and payload, and refuses to overwrite an existing destination.
This only prepares a voice asset: it neither downloads data nor invokes a
language compiler. Retain the voice's license and attribution.

For a standalone PureBasic entry point, build
[`examples/KokoroVoicePack.pb`](../examples/KokoroVoicePack.pb) as a console
application and use `pack RAW OUTPUT [EXPECTED_SHA256]` or `verify PMVOICE`.
The resulting pack is shared by the Windows and supported bare-metal adapters.

### Prepare the pronunciation packs without Python

Supply the pinned upstream dictionary sources. The base pack is built from the
Misaki US-English `us_gold.json` and `us_silver.json`; the supplementary pack is
built from `cmudict.dict` and the finished base pack, which it consults so that
it only adds words the base pack cannot already answer. Both sources are checked
by SHA-256 against the pinned revisions before anything is parsed, so a newer or
re-saved copy is refused rather than quietly producing a different pack.

**The pinned digests include the line endings.** The Misaki files are pinned
with CRLF endings, as a Windows clone with `core.autocrlf=true` checks them out;
the raw files are LF and are refused. `cmudict.dict` is pinned as the LF file.
These commands produce exactly the pinned bytes:

```powershell
git clone -c core.autocrlf=true https://github.com/hexgrad/misaki.git
git -C misaki checkout fba1236595f2d2bf21d414ba6e57d25256afada3
git clone -c core.autocrlf=false https://github.com/cmusphinx/cmudict.git
git -C cmudict checkout 74790861f652b15e4ac49015a90074ad62a27690
```

`us_gold.json` is then `misaki\misaki\data\us_gold.json` (sha256 `a1391113...`),
`us_silver.json` beside it (`a7d8629f...`), and `cmudict\cmudict.dict`
(`81917843...`).

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-base-dictionary C:\Misaki\us_gold.json --silver C:\Misaki\us_silver.json --output C:\Assets\kokoro_us_english.pmg2p
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-extra-dictionary C:\CMU\cmudict.dict --base C:\Assets\kokoro_us_english.pmg2p --output C:\Assets\kokoro_us_extra.pmg2p
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-verify-base-dictionary C:\Assets\kokoro_us_english.pmg2p
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-verify-extra-dictionary C:\Assets\kokoro_us_extra.pmg2p
```

Each pack is deterministic: the same pinned sources always produce the same
bytes, down to the hash-slot probe order. Packing refuses a malformed lexicon
line, a repeated word, a pronunciation outside the pinned Kokoro vocabulary, a
key longer than 63 bytes, and an existing destination; every refusal names its
numeric code and what to check. The completed header, extents, payload CRC-32,
every hash slot and every pronunciation codepoint are validated before the
output is published, and the verify commands re-run those checks on a file at
rest. `--self-test` also runs the packing core's own checks. This only prepares
pronunciation assets: it neither downloads data nor invokes a language compiler.
Retain each dictionary's license and attribution.

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

At `--precision int8` this model is compiled with a measured precision plan:
the text, duration and pitch front end and three nodes next to the waveform
take wide (two-plane, 16-bit-activation) weights and the rest narrow INT8.
On the reference input it predicts the same durations as FP32 for all 143
tokens and scores 0.588 dB of log-mel distance against ONNX Runtime FP32
(FP16 weight storage: 0.442). See [the book, chapter 4](guide/04_reading_a_model.txt).

## Pi 4 and UNO Q

**A complete, reproducible Pi 4 example:** [`examples/pi4/kokoro-sentence`](../examples/pi4/kokoro-sentence/README.md)
is the runner that spoke a 295-token sentence on a bare-metal Pi 4 on four
cores, its host tools, and the recipe from a clean checkout to the WAV, with
every expected hash. The book's chapter 11 walks through it.

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

### Measured Pi 4 status (2026-09-10)

The generated FP32 runtime at revision `9c6c96e` produced the complete sentence
“The sky is blue, and the sun is shining.” on a Pi 4 running Anvil, with the
processor measured at approximately 1.5 GHz. This was resident-model inference,
not a model recompile or host inference-engine output.

- Output: 68,400 finite FP32 samples, 24 kHz mono, 2.85 seconds of audio.
- Inference: 433.587 seconds, approximately **152 times slower than real time**.
- Full-buffer comparison with the pinned model's reference: maximum absolute
  error 0.092810, RMSE 0.004057, correlation 0.998205; no alignment or trimming.
- Raw little-endian FP32 output SHA-256:
  `dd596a19999e0e8017b879a935a98bd18fafb4aaa8a55e60bbc7aabc2dfc8cc4`.

This is a correctness checkpoint, **not usable interactive speech performance**.
A second, longer request hit the test's combined 900-second deadline; that run
does not establish two-request completion. Per-operation profiling and further
kernel work are in progress. These measurements do not establish performance
or physical execution on UNO Q, Windows, or other targets.

The normal text route is US English. It normalizes common book typography and
uses the checked dictionaries and supported word rules. Unknown words may be
spelled out; explicit `/phoneme/` overrides are vocabulary-checked. This is not
a claim of unrestricted multilingual linguistic coverage.

## Pico is not a full-Kokoro destination

The Pico and Pico 2 profiles exist for small models. Full Kokoro and its text
assets cannot fit their memory budgets. The speech/text switches explicitly
reject unsupported target combinations rather than offering a misleading demo.
