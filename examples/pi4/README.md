# Raspberry Pi 4 examples

[Home](../../README.md) · [Targets](../../docs/TARGETS.md) · [Kokoro speech](../../docs/KOKORO.md)

Programs that run a model this compiler emitted on a Raspberry Pi 4, with no
operating system, as payloads of the [Anvil](https://github.com/ajtaji/Anvil)
monitor. Each folder is one example with its own recipe, from a clean checkout
to the result on the host.

| Example | What it does | Proven on a board |
|---|---|---|
| [kokoro-sentence](kokoro-sentence/README.md) | Speaks one sentence with the full Kokoro-82M speech model, FP32, on all four cores, and brings the audio back as a WAV. | 2026-09-16: 295 tokens, 453,000 samples, 18.875 s of audio, 158.2 s on the board. |

Each example also has a **run manifest** beside its folder for Anvil's
`tools/board_model_run.py`, which does a whole board run in one command:
uploads only the assets that are not already in the board's memory, runs the
payload, reads the result back and writes the WAV.

| Manifest | Run |
|---|---|
| [kokoro-sentence.run.json](kokoro-sentence.run.json) | the Kokoro sentence above; its recipe's step 8 |
| [kitten-rosie.run.json](kitten-rosie.run.json) | a Kitten TTS nano sentence in the Rosie voice, for files prepared outside this repository (give `--base` the folder that holds them). Proven on 2026-09-16, Anvil build 160: 595,600 samples, 69.3 s on the board, WAV line 5.1 s after the return; a second run found both assets already in memory and took 126.5 s of board time instead of 180.0 s. |

The payloads are PureMetal BASIC (`.pi4`), built with
[PureMetal Forge](https://puremetalforge.ajtaji.com/). The Python files in these
folders are host-side developer tools for talking to the board. Building this
compiler, and generating or building a model with it, never needs them.
