# The ONNX Compiler Guide

[Home](../README.md) · **[Read the book (PDF)](guide/OnnxCompilerGuide.pdf)**

The book for this repository: what this compiler does, in the order you need
it, with every command shown as it was actually run and every board number
printed with the conditions it was measured under.

The reference pages under `docs/` answer one question each. This book is the
path through them.

## Chapters

| # | Chapter | What it covers |
|---:|---|---|
| 1 | [What this compiler is](guide/01_what_this_compiler_is.txt) | Ahead-of-time compilation; source out, not binaries; the four things it deliberately will not do; the five target profiles; bind once and submit many. |
| 2 | [Building the tool from source](guide/02_building_the_tool.txt) | The ready-built compiler in the PureMetal Forge download and when it is too old; `Build.pb`, the two executables and why they are a pair, what `bin/` holds, `--self-test`, and why `runtime/` is needed at build time. |
| 3 | [Your first model](guide/03_your_first_model.txt) | The two demo models, `--inspect`, one `--compile`, building `RunTwiceWindows.pb`, and three requests of different lengths through one resident model. |
| 4 | [Reading a model](guide/04_reading_a_model.txt) | The complete operator subset, the audited opset range, fixed shapes versus runtime dimensions, `--shape`, the exact text of every refusal, control flow and sequences, the random operators and `--random-inputs`, and what each `--precision` really changes. |
| 5 | [What comes out](guide/05_what_comes_out.txt) | The emitted source, the checked `.pmw` pack, the manifest field by field, the runtime folder, both emitted APIs, the random seed procedure, and the progress/cancellation callback. |
| 6 | [The five targets](guide/06_the_five_targets.txt) | Windows; Pi 4 and UNO Q with the Anvil payload contract and its single-core fallback; Pico and Pico 2 capacity, with the refusal that shows its arithmetic. |
| 7 | [Speech, end to end](guide/07_speech_end_to_end.txt) | The pinned model, the four checked assets and the pack/verify commands, `--speech-ui` and `--kokoro-text`, a resident runner in outline, and the measured speed and accuracy today. |
| 8 | [How it is proved](guide/08_how_it_is_proved.txt) | The independent oracle, bit identity where it can be claimed, what the published checks establish, the random operators' cross-target contract, the node-test score and how to run its harness, the control-flow cases and their Pi 4 interpreter runs, the failure that is still recorded as a failure, and why the rest of `tests/` is not tracked. |
| 9 | [Contributing](guide/09_contributing.txt) | The source map seam, the numerical contract (no FMA, no reassociation), how a kernel is proved in six steps, and licensing. |
| 10 | [Where everything is](guide/10_where_everything_is.txt) | Every reference page and what it owns, how this book is rendered, and the rule that keeps it current. |
| 11 | [Running a model on the Raspberry Pi 4: the Kokoro sentence](guide/11_the_kokoro_sentence_on_the_pi4.txt) | The published runner and host tools behind the sentence a bare-metal Pi 4 spoke on four cores; the pinned inputs, the two compiler builds and why the download's compiler cannot build it, the payload reproduced byte for byte, the board steps, and what it measured against ONNX Runtime. |

## The source of each chapter

Chapters are plain text with the same `@CHAPTER` / `@SECTION` /
`@FRAG` / `@NOTE` / `@TRAP` markup the PureMetal Forge books use.
`guide/00_OUTLINE.txt` documents the markup and the rules the chapters are
written to; files beginning `00_` are never rendered.

## Rendering

The PDF is produced by the PureMetal Forge help renderer, pointed at this
folder, so this book prints identically to the other books rather than
drifting into its own dialect:

```text
python tools/build_help_pdf.py ^
    --book <this checkout>\docs\guide ^
    --out  <this checkout>\docs\guide\OnnxCompilerGuide.pdf ^
    --title "The ONNX Compiler Guide" ^
    --subtitle "Ahead-of-time ONNX compilation to readable BASIC source." ^
    --section-nav
```

A commit that changes what a developer does — a command, a flag, a file
format, an emitted API, an install step, a capacity rule — updates the
affected chapter and re-renders the PDF **in the same commit**.
