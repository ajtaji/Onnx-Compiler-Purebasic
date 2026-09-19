# A sentence from Kokoro-82M on a bare-metal Raspberry Pi 4

[Pi 4 examples](../README.md) · [Kokoro speech](../../../docs/KOKORO.md) · [Targets](../../../docs/TARGETS.md) · [The book](../../../docs/GUIDE.md)

On 2026-09-16 a Raspberry Pi 4 with no operating system on it ran the full
Kokoro-82M speech model, in FP32, on all four cores, and spoke this:

> I am the kokoro speech AI model on the raspberry Pi 4. How do I sound, good?
> I am running on bare metal, with no operating system, under Anvil, a mini
> kernel and boot loader, on all four cores of the Raspberry Pi 4. Both Anvil
> and I were compiled by Pure Metal Forge.

This folder is everything that made it, and this page is the recipe, from a
clean checkout to the WAV.

| File | What it is |
|---|---|
| [`KokoroSentence.pi4`](KokoroSentence.pi4) | The runner: the Anvil payload that binds the model and the assets, turns on four cores, renders the sentence and leaves the samples in memory. Every step is commented. |
| [`node_output_map.pbi`](node_output_map.pbi) | A per-node table the runner's optional NaN scan reads. Off in this run, and kept so the source builds the exact payload that ran. |
| [`kokoro_readback.py`](kokoro_readback.py) | Host tool: reads the run's trace and result block, the four-core dispatch counter, and the waveform, with the board's own SHA-256 as the witness. |
| [`f32_to_wav.py`](f32_to_wav.py) | Host tool: turns the raw samples into a 16-bit WAV and a lossless 32-bit float WAV. |

> **The two `.py` files are developer tools for talking to the board.** Building
> this compiler, and generating or building a model with it, needs no Python.
> They use the Anvil repository's own host code by path (`board_run.py`'s console
> client and `anvil_readback.py`'s reader); nothing is copied from it.

## What came out, and what it measured

Measured on the board on 2026-09-16, Anvil build 142:

| | |
|---|---|
| tokens | 295 |
| samples | 453,000 FP32, 24 kHz mono: **18.875 s of audio** |
| time on the board | **158.2 s** from the start of the model bind to the end of the request (8,540,801,559 ticks of the 54 MHz generic timer), about 8.4 times slower than real time |
| four-core dispatches | the monitor's counter read **84** after the run, on a monitor reset just before it: one per convolution split across the cores |
| payload status | `KrResultStatus = 0`, trace stage 9 at node 2462 |
| raw samples | 1,812,000 bytes, sha256 `05c30419a882ad6a094a05fe0b7bbe4e96d549c20896a5522991c652d5296f6f`, equal to the board's own `sha256sum` of the range |
| WAV, 16-bit | 906,044 bytes, sha256 `a764205813ee082c365ccd510cb24402f6357e452c3cfb584556295513207080`, no sample clamped |

**Against ONNX Runtime.** The same text, packs and voice through ONNX Runtime
1.29.0 on a PC CPU give 453,000 samples as well. Compared sample for sample, with
no alignment and no trimming:

| max absolute difference | RMSE | correlation |
|---:|---:|---:|
| 0.259917 | 0.006588 | 0.996006 |

Every kernel the board runs is held bit-identical to this compiler's scalar
reference (no fused multiply-add, no reassociation), and the four-core split was
proven to return the same bytes as one core on the two shorter test sentences.
So the difference is between that reference and ONNX Runtime. The book's chapter
8 traces its mechanism on a shorter sentence: a divergence of about 6e-5 in the
text encoder, inside ONNX Runtime's own fusion and threading noise, amplified by
a cumulative-sum, multiply and resize chain into the phase of a sine whose
arguments are so large that one float32 step moves it by as much as 0.0156. This
sentence is 4.4 times longer than that one and diverges further, which fits the
same mechanism, but it has not been re-traced node by node. 0.26 is above the
0.1 limit set for the short test sentences, and it is recorded as a failure of
that limit, not explained away.

## What you need

- **A Raspberry Pi 4 Model B with at least 4 GB.** The run used a 4 GB board. The
  runner's working memory ends at `$A0000000`, which a 2 GB board does not have.
- **A boot volume for Anvil**: a FAT32 USB stick or SD card. The run booted from a
  USB stick, which also held the weight pack.
- **An Ethernet cable** from the Pi straight to a **Windows** PC. The monitor's
  network console and uploads run over it. The first-boot serial adapter is
  described in the Anvil skeleton's README.
- On the PC: **Git**, **PureBasic 6.21 x64** (to build this compiler),
  **PureMetal Forge** from [puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/)
  (to build the payload), and **Python 3** for the host tools. The Forge download
  also includes a ready-built copy of this compiler. The 2026-09-14 copy cannot
  build this example; the 2026-09-17 copy can, but not the exact payload that
  ran. Step 3 shows both. Anvil's
  `pi4_upload.py` also needs **pyserial** (`python -m pip install pyserial`),
  because it imports the serial console module even when it talks over the network.
- About 1 GB of disk.

## The recipe

Everything below runs in **Windows PowerShell** from one working folder. The
desk half, steps 1 to 6, was followed on 2026-09-16 from fresh clones and
produced the payload that ran, byte for byte. Where a command's output matters,
the expected value is under it.

### 1. Check out the two repositories, side by side

```powershell
$root = "$HOME\pi4-kokoro"
New-Item -ItemType Directory -Force $root | Out-Null
Set-Location $root
git clone https://github.com/ajtaji/Onnx-Compiler-Purebasic.git
git clone https://github.com/ajtaji/Anvil.git
```

The runner includes `../../../../Anvil/RaspberryPi4/Lib/mmu.pi4`, so the Anvil
checkout must be a folder called `Anvil` beside this repository's.

> **Keep `$root` short.** The model's runtime files land several folders deep
> inside it, and a path past the classic 260-character Windows limit makes step 4
> fail with `Cannot create generated runtime source`, which does not mention the
> length (forum topic 868, open). `$HOME\pi4-kokoro` is well inside it.

### 2. Get the model, the voice and the two dictionaries

```powershell
New-Item -ItemType Directory -Force downloads | Out-Null
$hf = 'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231'
curl.exe -L -o downloads\model.onnx "$hf/onnx/model.onnx"
curl.exe -L -o downloads\af_heart.bin "$hf/voices/af_heart.bin"
git clone -c core.autocrlf=true https://github.com/hexgrad/misaki.git
git -C misaki checkout fba1236595f2d2bf21d414ba6e57d25256afada3
git clone -c core.autocrlf=false https://github.com/cmusphinx/cmudict.git
git -C cmudict checkout 74790861f652b15e4ac49015a90074ad62a27690
Get-FileHash downloads\model.onnx, downloads\af_heart.bin, misaki\misaki\data\us_gold.json, misaki\misaki\data\us_silver.json, cmudict\cmudict.dict
```

| file | bytes | sha256 |
|---|---:|---|
| `model.onnx` (Kokoro-82M v1.0, FP32) | 325,532,232 | `8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb` |
| `af_heart.bin` (the voice) | 522,240 | `d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b` |
| `us_gold.json` | 3,093,057 | `a13911134e702c8fd1d79ea47cdc64f146aec1deabf4638a9a8cdd71da5b87a3` |
| `us_silver.json` | 3,192,879 | `a7d8629ff02614cd37837f44c41f0085f55bed4f0eb2c169964f0077bb102d6b` |
| `cmudict.dict` | 3,618,488 | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |

> **The line endings are part of the pin.** The Misaki digests are of the files
> with CRLF line endings, which is what `core.autocrlf=true` checks out; the same
> files downloaded raw (LF, 3,000,469 bytes for gold) are refused by the pack
> builder. The CMUdict digest is of the LF file, so that clone turns
> `core.autocrlf` off. Both clone commands above were run and give the hashes in
> the table.

> The two model files were not downloaded again for the desk run of this recipe:
> copies already on disk with the same SHA-256 were used, and the two URLs were
> checked against the server's own published size and SHA-256 for each file.

### 3. Build the compiler, twice

**The compiler in the PureMetal Forge download.** The download includes this
ONNX compiler ready to run, in `PureMetalForge\tools\onnx\`, and for most work
that is the easiest way to get it. Which release you have decides what it can do
here.

**The 2026-09-14 release cannot build this example.** Its copy was built on
2026-09-13, before the model source learned `PmOnnxAnvilEnter`, the call this
runner makes to put the convolutions on four cores. On 2026-09-16, from fresh
clones, steps 4 to 6 were run with it in place of `bin\`, with the download
extracted into `$root` as step 6 describes:

```powershell
$onnx = "$root\PureMetalForge\tools\onnx\PureMetalOnnxCompilerCLI.exe"
$ex = "$root\Onnx-Compiler-Purebasic\examples\pi4\kokoro-sentence"
New-Item -ItemType Directory -Force "$ex\model" | Out-Null
& $onnx --compile "$root\downloads\model.onnx" --output "$ex\model\kokoro-pi4" --target pi4 --precision fp32 --kokoro-text
```

and the three pack commands of step 5 with `& $onnx` in place of
`.\bin\PureMetalOnnxCompilerCLI.exe`. It gave:

| file | from the download's compiler | same as below? |
|---|---|---|
| `kokoro-pi4.pmw` | 324,621,440 bytes, `a41177210505c80c...` | yes |
| `af_heart.pmvoice` | 522,368 bytes, `79ef729cc9aa7ec4...` | yes |
| `kokoro_us_english.pmg2p` | 12,613,954 bytes, `0d791ecc12e11d77...` | yes |
| `kokoro_us_extra.pmg2p` | 5,397,230 bytes, `e9eee914562c0a218dd10ed42364d38124d29478a3edbf89234d8e315bea7f63` | **no**: the older 66,304-entry supplement, without the model's own name (forum topic 871); a compiler built from `main` refuses it with `PMG2P-3020` |
| `kokoro-pi4.pi4` | 788,990 bytes, `accb8c9240fc5536747942b2717f8a3517fbc8219ff155edd6df2d880d173bef` | **no**: older runtime kernels, and no `PmOnnxAnvilEnter` |

Step 6 then stops, with exit code `1`:

```text
SEMANTIC ERROR: 'PmOnnxAnvilEnter' at KokoroSentence.pi4 line 397, in procedure krbindservices() is not defined - there is no procedure and no intrinsic with that name.
                Check the spelling, and check that the library declaring it is included.
pmfc: semantic analysis failed
pmfc: BUILD FAILED - no output produced.
```

**The 2026-09-17 release builds it, but not the payload that ran.** Its copy is
ONNX commit `658566d`. Run the same way on 2026-09-17, it gave all four assets
byte-identical to the tables in steps 4 and 5 (`kokoro-pi4.pmw` and all three
packs, including the 5,397,245-byte supplement), and the runner built with it.
The payload was 1,590,676 bytes, sha256
`61668df99f5c7d7d34f42f3409b97872bd424662cd4964ac776ed9a250a7285f`, because
its model source comes from later runtime kernels (789,716 bytes, not 789,585).
That payload has not been run on a board.

So to reproduce the exact payload that ran, build the compiler from source as
below, which needs PureBasic.

The model source that ran was emitted by this compiler at commit **`6b6b770`**.
Later commits changed runtime kernels, so they emit a different payload; one
built from `main` has not been run on a board. The pronunciation supplement, on
the other hand, needs `main`: the native packer could not build it before
`b48cd05` (forum topic 871). So build both:

```powershell
$pb = 'C:\Program Files\PureBasic\Compilers\pbcompiler.exe'   # your PureBasic 6.21 x64
Set-Location "$root\Onnx-Compiler-Purebasic"
New-Item -ItemType Directory -Force bin | Out-Null
& $pb src\PureMetalOnnxCompiler.pb /CONSOLE /THREAD /OPTIMIZER /OUTPUT bin\PureMetalOnnxCompilerCLI.exe
git checkout 6b6b770
& $pb src\PureMetalOnnxCompiler.pb /CONSOLE /THREAD /OPTIMIZER /OUTPUT bin\OnnxCli-6b6b770.exe
git checkout main
```

`bin\` is ignored by Git, so both executables survive the checkouts.
`Build.pb` does the first build for you if you prefer the IDE.

### 4. Emit the model for the Pi 4

```powershell
$ex = "$root\Onnx-Compiler-Purebasic\examples\pi4\kokoro-sentence"
New-Item -ItemType Directory -Force "$ex\model" | Out-Null
.\bin\OnnxCli-6b6b770.exe --compile "$root\downloads\model.onnx" --output "$ex\model\kokoro-pi4" --target pi4 --precision fp32 --kokoro-text
```

| file | bytes | sha256 |
|---|---:|---|
| `kokoro-pi4.pmw`, the weight pack | 324,621,440 | `a41177210505c80c08836acbb919283d140cb94ddfce89ed2dc732ed8c0d5413` |
| `kokoro-pi4.pi4`, the model source | 789,585 | `1dc51126a2ef266a84a9694d47358c1f51509513c3efd6b32a9a9b58e5b5eafe` |
| `kokoro-pi4.kokoro.pi4`, its entry file | 243 | `b443f672ea10401e4bc35d87192f8ee6f4a1744b688d803e8e6025ebd2dbb397` |

The weight pack is the file that ran. The two source files are the files that
ran with one difference: their output name appears in five include lines, and
the run used a different name. The nine files in `kokoro-pi4.runtime\` match the
run's after line endings are normalised; a checkout gives them CRLF. Neither
difference reaches the payload (step 6).

### 5. Build the voice and the pronunciation packs

```powershell
New-Item -ItemType Directory -Force "$root\assets" | Out-Null
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-voice "$root\downloads\af_heart.bin" --output "$root\assets\af_heart.pmvoice" --source-sha256 d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-base-dictionary "$root\misaki\misaki\data\us_gold.json" --silver "$root\misaki\misaki\data\us_silver.json" --output "$root\assets\kokoro_us_english.pmg2p"
.\bin\PureMetalOnnxCompilerCLI.exe --kokoro-pack-extra-dictionary "$root\cmudict\cmudict.dict" --base "$root\assets\kokoro_us_english.pmg2p" --output "$root\assets\kokoro_us_extra.pmg2p"
Get-FileHash "$root\assets\*"
```

| file | bytes | sha256 | goes to |
|---|---:|---|---|
| `kokoro_us_english.pmg2p` | 12,613,954 | `0d791ecc12e11d77b5d961718128cf945a67ab53274816a94b9d8eecddbc0b88` | `$54000000` |
| `kokoro_us_extra.pmg2p` | 5,397,245 | `73be678aa7d1e8cd1b8f87681f4ee1a14f26ac7298fad1063136813064ea8525` | `$55000000` |
| `af_heart.pmvoice` | 522,368 | `79ef729cc9aa7ec47025ece5c3d6fa230404e0f4d061465d47a71afc4b3d06ce` | `$56000000` |
| `kokoro-pi4.pmw` (step 4) | 324,621,440 | `a41177210505c80c...` | `$40000000` |

All three are byte-identical to the assets the board bound.

### 6. Build the payload

Download PureMetal Forge from [puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/)
and extract it; the zip holds a `PureMetalForge` folder. The run's payload was
reproduced with the 2026-09-14 release: zip sha256
`e2048ddb400581a5d7d1e6c3bb7d8c0c13ce4cd2225ca1fd42b87de510cf9624`,
`PureMetalForge.exe` sha256
`8f2e43d4b260c5fa049762e6212c4355ea9cf49d30c0c10092783d3a664a5367`.

```powershell
$pmf = "$root\PureMetalForge\PureMetalForge.exe"
Set-Location $ex
New-Item -ItemType Directory -Force build | Out-Null
$build = Start-Process $pmf -ArgumentList '--compile KokoroSentence.pi4 -t pi4 --load-addr 0x500000 --bss-addr 0x1000000 --stack-addr 0x7F00000 --entry-returns --wants-services -o build\kokoro-sentence.img' -Wait -PassThru -NoNewWindow
$build.ExitCode
Get-FileHash build\kokoro-sentence.img.pmf
```

Expect exit code `0` and

```text
build\kokoro-sentence.img.pmf   1,578,488 bytes   sha256 58b402ea88f92c414a752df39f5ef4b006021a905018f3145df3fb4b02da63b3
```

**That is the payload that ran on 2026-09-16, byte for byte.** The `.img` beside
it is the raw image (1,578,360 bytes); the `.pmf` is the container the board
takes. `build\kokoro-sentence.img.sym` is the symbol file the host tool reads.

The 2026-09-17 release's `PureMetalForge.exe` (sha256
`5898e744e6e688528abb0e3f704404e52afe3a1967ac4c7ad1113b276b0c2194`) builds
the same `58b402ea...` from the same `6b6b770` model source.

### 7. Get Anvil running

Follow the Anvil repository's boot-volume skeleton,
[`Boards/RaspberryPi4/sdcard`](https://github.com/ajtaji/Anvil/tree/main/Boards/RaspberryPi4/sdcard):
format the stick or card FAT32, copy the three Raspberry Pi firmware files,
`config.txt` as `CONFIG.TXT` and `kernel8.img` as `KERNEL8.IMG`. That
`kernel8.img` is **Anvil build 151**
(sha256 `932e5437e11b45420f268066dfc2d9acedf5068f28f6cb045f2f03cd8e9ad75b`), the
first build with the `readback` command step 10 uses. Connect the cable, power
the board, and find its console:

```powershell
python "$root\Anvil\tools\anvil_wifi.py" --find
$board = '192.168.137.1'   # the address that answered
```

On the run's bench the board answered on `192.168.137.1` over the cable.

### 8. Upload, run, read back and make the WAV: one command

Steps 8 to 11 below are one invocation of Anvil's `board_model_run.py`, driven by
[`../kokoro-sentence.run.json`](../kokoro-sentence.run.json), the manifest that
names the four assets with their addresses and SHA-256, the payload, the deadman,
the waveform region and the WAV:

```powershell
$tools = "$root\Anvil\tools"
python "$tools\board_model_run.py" "$ex\..\kokoro-sentence.run.json" --console-ip $board --no-lease
```

On a board other people share, replace
`--no-lease` with `--lease-file <the shared-board file> --lane "<your name>"` and
the tool takes and releases the lease itself.

What it does, with no step of yours in between:

- **An asset already in the board's memory is not sent again.** For each asset
  the board computes `sha256sum` of the range it would go to; equal to the
  manifest's digest, it prints `resident` and moves on. Otherwise it uploads the
  file and checks the board's SHA-256 of the memory it landed in. A reset empties
  memory, so the first run after one uploads everything.
- It uploads the payload to `$03000000`, arms the 15-second deadman and enters it.
- The moment the return line arrives it reads the 1,812,000 samples back with
  `readback`, checks them against the board's own `sha256sum` of the range, writes
  `run\<time>\kokoro-sentence.wav` (16-bit, 24 kHz) and prints one line:
  `WAV ready: <path>`.
- It then reads the trace, turns the deadman off and writes
  `run\<time>\kokoro-sentence.json` with every step's time.

It exits `0` only if the payload returned `0` and every region verified. The
manual steps below are what it does, kept as a reference.

> **This manifest has not been run on a board yet.** The tool was proven on
> 2026-09-16 with the Kitten TTS manifest beside it, `kitten-rosie.run.json`. The
> Kokoro manifest's digests are the ones in this recipe's tables; the waveform's
> expected digest is the one under step 10.

## Steps 8 to 11 by hand, for reference

### 8. Put the four assets in the board's memory

Each upload streams the file over TCP and then compares the board's own SHA-256
of the memory it landed in with the file on disk; a mismatch exits non-zero.
`--cache` turns the board's caches on for the digest, which is otherwise the slow
half.

```powershell
$tools = "$root\Anvil\tools"
python "$tools\pi4_upload.py" "$root\assets\kokoro_us_english.pmg2p" --console net --console-ip $board --board-ip $board --addr 0x54000000 --cache
python "$tools\pi4_upload.py" "$root\assets\kokoro_us_extra.pmg2p" --console net --console-ip $board --board-ip $board --addr 0x55000000 --cache
python "$tools\pi4_upload.py" "$root\assets\af_heart.pmvoice" --console net --console-ip $board --board-ip $board --addr 0x56000000 --cache
python "$tools\pi4_upload.py" "$ex\model\kokoro-pi4.pmw" --console net --console-ip $board --board-ip $board --addr 0x40000000 --cache
```

On the run, the three small files went up and verified like this:

```text
  VERIFIED: 12613954 bytes, sha256 equal, 1389 ms on the board (8868 KB/s).
  VERIFIED: 5397245 bytes, sha256 equal, 595 ms on the board (8858 KB/s).
  VERIFIED: 522368 bytes, sha256 equal, 56 ms on the board (9109 KB/s).
```

**The run did not upload the weights.** They had been uploaded and verified on
an earlier day and saved to the USB stick as `MODEL.PMW` with
`save MODEL.PMW 40000000 13595480`; after the reset before the run, the board
read them back into memory at its console:

```text
pmf> load MODEL.PMW 40000000
Loaded 324621440 bytes into memory, filling 40000000 through 5359547F.
```

Either way, the payload itself checks the pack's header, identity and CRC-32
when it binds, and refuses with status 3 rather than computing from wrong
weights. Everything in memory is lost at a reset, so the assets go up again
after every reset.

### 9. Run it

Read the four-core dispatch counter, run the payload, read the counter again.

```powershell
python "$ex\kokoro_readback.py" cores --console-ip $board
python "$tools\board_run.py" "$ex\build\kokoro-sentence.img.pmf" --console-ip $board --board-ip $board --addr 0x03000000 --deadman 15 --expect-x0 0 --timeout 700 --out "$ex\run" --name kokoro-sentence
python "$ex\kokoro_readback.py" cores --console-ip $board
```

Do not wrap `board_run.py` in an outer timeout and do not pipe it through `tail`
(Anvil's [`docs/BOARD_RUN.md`](https://github.com/ajtaji/Anvil/blob/main/docs/BOARD_RUN.md)
says why). The run staged the container at `$03000000`, clear of the payload's
own load address, armed the 15-second deadman, which the runner services at
every node, and entered it:

```text
  stage   03000000
  sent 1578488 bytes in 0.17 s
  verified on the board: 1578488 bytes, digest matches
  deadman 15 s
  boot mem 3000000 shot
```

The counter's difference should be **84**. `board_run.py` also expects a
screenshot of the display when the payload returns; the run's board had a
display attached.

> **On 2026-09-16 `board_run.py` reported that the payload never returned, and
> the payload had finished.** The console lost the return line during a long
> silent run. That is why the runner writes its own trace: go to step 10 and
> read it rather than running again. Build 142 also kept the four cores after a
> four-core payload until a reset; build 145 and later give them back at the
> payload boundary.

### 10. Read the result and the waveform back

```powershell
python "$ex\kokoro_readback.py" status --console-ip $board
python "$ex\kokoro_readback.py" wave --console-ip $board
```

`status` reads the KTRC trace at `$57E00000` and the result block (its addresses
come from `build\kokoro-sentence.img.sym`). A finished run reads trace stage 9 at
node 2462, status 0, 295 tokens and 453,000 samples. Any other status is named
in words; the full list is in the runner's `KrFinish` comment.

`wave` asks the board for `sha256sum 57400000 1BA620`, reads the 1,812,000 bytes
with `readback`, asks for the digest again, and saves `run\kokoro-sentence.f32`
only if all three digests agree and the XOR of the samples equals the one the
payload computed on the board. On the run the board's own digest was:

```text
pmf> sha256sum 57400000 1BA620
sha256sum over 1812000 bytes, 57400000 .. 575BA61F ...
The sha256sum is 05c30419a882ad6a094a05fe0b7bbe4e96d549c20896a5522991c652d5296f6f
```

> **`kokoro_readback.py` has not yet been run against a board.** Its decoding and
> its refusals pass a self-test against a scripted console
> (`python kokoro_readback.py --self-test`). On 2026-09-16 the waveform came back
> through build 142's older `memory` dump path, which took 1,048 s at 1.7 KB/s;
> the Anvil repository measured `readback` on build 151 reading the same 1,812,000
> bytes in 1.28 s.

### 11. Make the WAV

```powershell
python "$ex\f32_to_wav.py"
```

It reads `run\kokoro-sentence.f32` and writes `run\kokoro-sentence.wav`,
`run\kokoro-sentence.f32.wav` and `run\kokoro-sentence.wav.json`. From the run's
samples it prints:

```text
...\run\kokoro-sentence.wav: 453000 samples, 18.875 s at 24000 Hz, 0 clamped
  pcm16   906044 bytes  sha256 a764205813ee082c365ccd510cb24402f6357e452c3cfb584556295513207080
  float32 1812044 bytes  sha256 3e217288ec5da58ec0c62fc72449adec566c8138349da12e694b81b246379af6
```

## Speaking something else

Change the bytes under `KrText3` in `KokoroSentence.pi4`, one `Data.a` value per
UTF-8 byte, and `#KR_TEXT3_BYTES` to match. PowerShell gives you both:

```powershell
$text = 'Hello from the Pi.'
[Text.Encoding]::UTF8.GetBytes($text) -join ','
[Text.Encoding]::UTF8.GetByteCount($text)
```

Set `#KR_EXPECT_TOKENS3` to the sentence's token count if you know it. If it is
wrong the model still runs and the samples are still copied, the payload returns
320 instead of 0, and `status` prints the real count. Keep the audio under
21.8 seconds: the copy slot holds 524,288 samples, and a longer sentence is
refused with 313 rather than cut short. A changed runner is a different payload,
so its hash is no longer the one above.

## What is not in this repository

The model, the voice, the dictionaries and every file the recipe makes. The
weights, packs and WAV come from running it; `model\`, `build\` and `run\` are
ignored by Git. Keep the licences of what you download: Kokoro-82M and Misaki
are Apache-2.0, and CMUdict carries its own BSD-style terms (see
[third-party notices](../../../THIRD_PARTY_NOTICES.md)).
