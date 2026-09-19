# Your first resident model

[Home](../README.md) · [Build](BUILD.md) · [Targets](TARGETS.md)

This walkthrough uses only PureBasic and the sources in this repository.
It does not download a model or load another inference engine.

## 1. Get the generator

**The easiest way:** download PureMetal Forge from
[puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/) and extract it. The
download includes this generator ready to run in `PureMetalForge\tools\onnx\`:
`PureMetalOnnxCompilerCLI.exe` and the window, `PureMetalOnnxCompiler.exe`.
Wherever a command below says `.\bin\PureMetalOnnxCompilerCLI.exe`, use that file
instead, for example:

```powershell
$onnx = 'C:\PureMetalForge\tools\onnx\PureMetalOnnxCompilerCLI.exe'   # where you extracted the download
& $onnx --compile output\twice.onnx --output output\twice --target windows
```

Steps 2 and 4 still use PureBasic, because they build programs from this
repository. Steps 2 to 5 were run with the command-line generator from the
2026-09-14 download and gave the output shown here.

**To build it from source** instead, for example to change it, open
[Build.pb](../Build.pb) in PureBasic x64 and run it. It creates the CLI
and graphical generator in `bin/`. See [compiler settings](BUILD.md).

## 2. Create the demonstration models

Open [CreateDemoModels.pb](../examples/CreateDemoModels.pb) and run it.
The program writes two small, valid ONNX protobuf models into `output/`:

- `twice.onnx`: an Add graph with runtime input length, computing `y = x + x`.
- `dense.onnx`: a fixed-shape MatMul graph with a 2-by-2 diagonal weight matrix.

These bytes are embedded in the PureBasic example; no protobuf package is needed.

## 3. Generate Windows source

In PowerShell, from the checkout root:

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --compile output\twice.onnx --output output\twice --target windows
if ($LASTEXITCODE -ne 0) { throw 'Model generation failed' }
```

Keep the generated `.pb`, `.pmw`, manifest, and runtime directory together.
The tool has now finished its job: **source generation**, not an executable build.

Alternatively, launch `bin/PureMetalOnnxCompiler.exe`, choose the model, output
prefix, Windows target, and FP32 precision, then start generation. The window
uses the sibling CLI and displays its result.

## 4. Run three requests without reloading

Open [RunTwiceWindows.pb](../examples/RunTwiceWindows.pb). Enable **thread-safe
executable** and **optimizer**, select the x64 native backend, and run it.
This is a separate application build performed by you in PureBasic.

Expected output includes:

```text
PASS request 1: 3 values
PASS request 2: 6 values
PASS request 3: 9 values
```

The example initializes the weights once, fills a new input tensor on each
request, checks every output value, resets request storage, and finally closes
the model. Open the source to see the complete ownership and error handling.

## 5. Try packed INT8 weights

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --compile output\dense.onnx --output output\dense-windows --target windows --precision int8
```

The dense example has an eligible constant matrix. `twice.onnx` has no weights
to reduce, so selecting INT8 for that model cannot demonstrate compression.
Inspect the manifest for actual transformed weights and memory requirements.

## Next steps

Use [target integration](TARGETS.md) for bare-metal output, [architecture](ARCHITECTURE.md)
to understand the two generated APIs, and [Kokoro](KOKORO.md) for speech-specific
assets. For your own model, begin with FP32 and compare outputs against known
reference inputs before measuring speed or trying reduced precision.
