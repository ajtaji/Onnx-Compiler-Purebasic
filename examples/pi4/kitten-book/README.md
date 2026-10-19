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

The tested stick already holds both weight packs. After storing the built
reader, complete chapter and private configuration under these example names,
launch them as follows; those three additional files were not installed during
the recorded run because its console readback stopped responding:

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

The 24-chunk production Bluetooth run sent 140.688 seconds of audio. Concurrent production took 128.014 seconds for 126.281 seconds of newly generated audio: 1.01372 seconds of work per second of speech. It recorded 0 starvation events, 0 receive errors and 0 secondary-core audio calls. Maximum audio-service gap was 51.571 ms. Sustained faster-than-real-time playback is still unachieved; initial prefetch conceals the remaining deficit.

With weights resident, five short phrases took 1.369, 1.753, 2.202, 2.417 and 3.758 seconds of model execution. The shortest was 'Hello'; its paced audio lasts 1.397 seconds. These exclude Bluetooth connection and initial playback buffering. A 0.5-1 second request-to-speech start has not been demonstrated. The current reader prefetches two chunks before connecting; persistent connection and first-phrase scheduling remain needed for interactive startup. The complete-chunk ONNX interface supplies no partial waveform during inference.
