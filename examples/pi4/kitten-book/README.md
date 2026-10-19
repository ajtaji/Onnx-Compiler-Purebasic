# Kitten nano book reader on Anvil

This example generates Rosie speech on a Raspberry Pi 4 and sends it to an
A2DP speaker through Anvil's existing Bluetooth stack. All four cores compute
speech; only the primary core performs Bluetooth I/O. There is no operating
system and no host inference engine in the playback path.

Text preparation runs on the host. It supplies phonemes, voice style and speed,
not rendered audio. Plain-text input is supported; convert an EPUB or PDF to
text first. The preparation tool uses the released Kitten 0.8.1 front end,
`phonemizer`, `espeakng-loader` and NumPy. It never imports ONNX Runtime.

## Prepare the model and text

Use the files from `KittenML/kitten-tts-nano-0.8-fp32`, revision
`7a1db645b1f3ab9420761d87428e042b9cec3f26`: the ONNX model, `voices.npz` and
`config.json`. Extract `onnx_model.py` and `preprocess.py` from the released
Kitten 0.8.1 distribution into a separate front-end folder.

```powershell
PureMetalOnnxCompilerCLI.exe --compile assets/kitten_tts_nano_v0_8.onnx --output model/kitten --target pi4 --precision int8
py prepare_book.py --assets assets --frontend frontend-0.8.1 --text chapter.txt --out chapter.ktbk --limit 24
```

The default is Rosie (`expr-voice-4-f`) with its upstream speed prior, 0.8.
The default 100-character chunk budget reduces attention work. The upstream
chunker can exceed that budget for an indivisible segment; the tool refuses
more than 512 tokens. Each generated waveform loses its final 5,000 samples,
as in the released front end. The reader also shortens boundary quiet above
100 ms leading / 150 ms trailing, retaining interior pauses and fading shortened
edges for 5 ms. Provenance includes source and pack hashes.
Use `--limit 0` for the complete input. Packs are limited to the 16 MiB input
slot; divide larger books into chapters or parts.

## Prepare and build the reader

Use an 80-byte private Bluetooth configuration. At offset 48 it contains
`ADDR` followed by the six-byte controller address in display order; at offset
64, `PEER` followed by the target speaker address in display order. An optional
saved bond at offset zero contains `BOND` followed by the 22-byte record returned
by `HciLinkKeyGet`. The reader checks that the bond belongs to the selected peer.
Keep this file and run readbacks private. A successful connection can produce a
new bond in RAM at `0x53010000`; save it privately for subsequent connections.
The reader uses a separate controller identity so another application's saved
pairing can remain available. The shared speaker still needs to be free.

```powershell
py prepare_reader.py --anvil ANVIL_ROOT --model model/kitten.pi4 --book chapter.ktbk --firmware BRCMBT.HCD --private-config speaker.bin --out reader
$env:PMF_ROOT='ANVIL_ROOT'
PureMetalForge.exe --compile reader/KittenBookReader.pi4 -t pi4 --load-addr 0x3000000 --bss-addr 0x1000000 --stack-addr 0x7F00000 --entry-returns --wants-services -o reader/KittenBookReader.img
py ANVIL_ROOT/tools/board_model_run.py reader/book.run.json --console-ip PI_ADDRESS --lease-file BOARD_SHARE_FILE --lane kitten-book --lease-wait 1
```

The manifest bounds a verification run to 240 seconds. Choose a short part or
set a chunk limit for that run. For a complete long chapter, use the USB launch
below, or explicitly choose a suitable host wait. The reader's own timeout is
per inference request, so a long book is not stopped after four minutes.

## Resident memory and USB launch

| Item | Address |
|---|---:|
| Linked payload / BSS / stack | `0x03000000` / `0x01000000` / `0x07F00000` |
| INT8 weights | `0x44000000` |
| Book inputs, at most 16 MiB | `0x52000000` |
| Bluetooth firmware / private configuration | `0x53000000` / `0x53010000` |
| Circular FP32 audio queue, 12 MiB | `0x57000000` |
| Model arena, 1 GiB | `0x60000000` |

These slots are for the tested Pi 4 Anvil map; check another monitor's map
before reusing them. The speaker firmware tested here is 63,806 bytes. Other
firmware lengths require changing `HciLoadPatch` in the reader template.

The tested stick holds the verified weight packs, reader, complete chapter and
private speaker configuration under these names:

```text
load 2:/models/KITTEN8.PMW 44000000
load 2:/models/ALICE1.KTB 52000000
load 1:/BRCMBT.HCD 53000000
load 2:/models/KITBT.CFG 53010000
load 2:/models/KITREAD.PMF 3000000
deadman 15
boot mem 3000000
```

After return, `deadman off` leaves the monitor ready for another application.
No automatic boot setting is changed. FP32 weights are separately retained as
`2:/models/KITTENFP.PMW`; this reader uses the INT8 pack.

## Timing and acceptance

Measure inference, complete chunk production, and Bluetooth starvation
separately. Two prefetched chunks can conceal a sustained speed deficit.
The playback metadata at `0x58001000` reports setup/fault status, SBC packet
counts, starvation events, encoding time, maximum pump gap and generation
counters. Offsets 136/144/152 hold total production ticks, concurrent
production ticks and concurrent generated samples. Divide ticks by the
architectural timer frequency and samples by 24,000. The first 128 chunk
records are retained at `0x58002000`, 32 bytes each.

The idle callback runs only on the primary core, is throttled to one
millisecond, and must never call the model runtime. The ring writer publishes
only complete, finite chunks. The player linearly upsamples 24 kHz speech to
48 kHz, encodes stereo SBC and sends three frames per packet with a bounded
lead. End-of-stream drain time is excluded from starvation accounting.

Numerical optimization checks preserve the previous native waveform bit for
bit. INT8 quantization remains an explicit precision choice. A separate
captured-noise FP32 comparison against ONNX Runtime still fails its existing
max-absolute and RMS thresholds; no tolerances were loosened for these speed
changes. UNO Q has not been hardware-tested for this update.

## Measured Pi 4 result

| Excerpt (24 chunks) | Concurrent production | New speech | Work/audio | Starvation |
|---|---:|---:|---:|---:|
| The Pool of Tears | 106.552 s | 110.527 s | 0.96403 | 0 |
| A Mad Tea-Party | 100.790 s | 106.023 s | 0.95064 | 0 |
| Alice's Evidence | 123.166 s | 130.894 s | 0.94096 | 0 |

All three measured excerpts generated faster than playback, with no recorded starvation.

The full-waveform control measured 1.313 seconds for Hello, excluding speaker setup. The book reader still prefetches two chunks before connecting. The separate warm-reply mode below starts from generated waveform windows while the rest is computed; its latency and quality checks are reported separately.

The reader refuses SBC frames with an invalid sync header (playback fault 4).


## Warm replies and route instructions

Use `prepare_reply.py` for interactive replies with the model already resident
and Bluetooth already connected. The same host-prepared input format is used.
The generated `ReplyStart(index)`, `ReplyDrain()` and `ReplyPoll()` procedures
can be called from an enclosing application loop. `ReplyStart` is synchronous,
but begins playback from early windows before synthesis finishes. Poll while
waiting for input; the primary-core idle callback services Bluetooth during
inference. Drain the preceding queue before submitting an independent reply.

```powershell
py prepare_reply.py --anvil ANVIL_ROOT --model model/kitten.pi4 --inputs directions.ktbk --firmware BRCMBT.HCD --private-config speaker.bin --out replies
PureMetalForge.exe --compile replies/KittenReplySession.pi4 -t pi4 --load-addr 0x3000000 --bss-addr 0x1000000 --stack-addr 0x7F00000 --entry-returns --wants-services -o replies/KittenReplySession.img
py ANVIL_ROOT/tools/board_model_run.py replies/reply.run.json --console-ip PI_ADDRESS --lease-file BOARD_SHARE_FILE --lane kitten-replies --lease-wait 1
py read_reply_results.py replies/runs/RUN_ID
```

The supplied Main is a bounded demonstration: it warms the model, opens A2DP
once, reads each prepared input as a fresh request, then returns and releases
Bluetooth. In an application, retain the initialized model and connection
around the application's own request loop. No TruckGPS source is changed.
The input table supports 1..128 short replies. Long requests can exceed the
bounded temporal scratch map; split them into shorter phrases. The adapter
requires the validated Kitten 0.8 INT8 emitted graph and refuses other graphs
instead of assuming that their tensor identifiers have the same meaning.

Additional temporary slots are `0x48000000` (latent features, 2 MiB),
`0x48200000` (F0), `0x48400000` (spectrogram, 4 MiB), and `0x4A000000`
(assembled output, 16 MiB). These returning-payload slots do not alter boot
media. They must be reserved before embedding the example in another app.

For route lookahead or fixed prompts, add `--prepared` and choose a new output
folder. This generates complete instructions on the Pi before opening A2DP,
stores up to 64 MiB of audio at `0x4C000000`, and uses the same reply API for
later cached requests. The model remains loaded. Preparation and request
timings are reported separately; this mode does not claim fresh synthesis
in a few milliseconds. `ReplyCachePrepare` runs during initial warmup.

Fresh replies retain a common excitation phase across all windows and blend
joins for 20 ms. Their per-window normalization differs from the full-waveform
model. The control using one full window preserves the reference samples;
the streamed path is separately checked for clear, continuous playback.

Measured four-core GPS tests, with upstream Rosie speed 0.8:

| Path | First voice packet | Active stream position | Queue starvation |
|---|---:|---:|---:|
| Fresh short directions | about 0.80 s | about 0.86-0.89 s | 0 |
| Fresh longer directions | about 1.02-1.29 s | about 1.10-1.37 s | 0 |
| Prepared directions | 3-8 ms | about 80-85 ms before retained leading quiet | 0 |

Prepared active-audio positions are 0.08-0.32 seconds including leading quiet.
The listener confirmed the fresh directions clear and continuous. The speaker's
additional acoustic delay was not measured. Fresh model work/audio ratios are
still above one on these short clips; the initial queued speech permits
continuous playback. Use the full-waveform reader for sustained books, and
route lookahead to leave the inference cores available during directions.

The USB reply recipes use the existing weights, firmware and speaker bond:

```text
load 2:/models/KITTEN8.PMW 44000000
load 2:/models/GPSREPLY.KTB 52000000
load 1:/BRCMBT.HCD 53000000
load 2:/models/KITBT.CFG 53010000
load 2:/models/KITREPLY.PMF 3000000
deadman 15
boot mem 3000000
```

Use `KITNAV.PMF` instead of `KITREPLY.PMF` for the prepared-instruction version.
After either demonstration returns, use `deadman off`. Neither changes auto boot.
