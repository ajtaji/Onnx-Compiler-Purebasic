# Build in PureBasic

[Home](../README.md) · [Quick start](QUICKSTART.md) · [Source map](SOURCE_MAP.md)

## You may not need to build it

The PureMetal Forge download from [puremetalforge.ajtaji.com](https://puremetalforge.ajtaji.com/)
includes both executables, built and ready to run, in `PureMetalForge\tools\onnx\`:
`PureMetalOnnxCompilerCLI.exe`, `PureMetalOnnxCompiler.exe` (the window) and
`NATIVE_COMPILER_GUIDE.txt`. The checks under [Check the result](#check-the-result)
pass on the 2026-09-14 download's copy.

Build from source when you are changing the compiler, or when you need something
committed here after your download was built. A download carries the compiler as
it was when that release was built: the 2026-09-14 release's executables were
built on 2026-09-13. The [Pi 4 Kokoro sentence](../examples/pi4/kokoro-sentence/README.md)
needs a newer compiler than that, and its recipe says what differs.

## Requirements

- PureBasic **Windows x64**. Tested with **6.21** and its native assembly backend.
- A writable checkout directory.
- No Python, ONNX package, ONNX Runtime DLL, model download, or PureMetal compiler.

Other host operating systems/backends are not validated. Windows-specific APIs
are used by the UI and emitted Windows execution libraries.

## Option A — build helper

Open [`Build.pb`](../Build.pb) in the 64-bit PureBasic IDE and press F5.
The helper locates the compiler belonging to that PureBasic installation and
builds the following files sequentially:

| Source | Output | Compiler options |
|---|---|---|
| `src/PureMetalOnnxCompiler.pb` | `bin/PureMetalOnnxCompilerCLI.exe` | Console, thread-safe, optimizer |
| `src/PureMetalOnnxCompilerUI.pb` | `bin/PureMetalOnnxCompiler.exe` | Thread-safe, optimizer, DPI-aware |

This helper is developer infrastructure, not part of the ONNX generator.
Its compiler invocation must not be copied into model-generation or inference paths.

## Option B — build the entry points directly

In the PureBasic IDE, open each source listed above, enable its listed options,
and use **Create Executable** with the corresponding output path. Create `bin/`
if it does not exist. Keep the two executables together: the window launches
the sibling CLI to perform generation.

Equivalent PowerShell, with the path adjusted to your installation:

```powershell
$pb = 'C:\Path\To\PureBasic\Compilers\pbcompiler.exe'
New-Item -ItemType Directory -Force bin | Out-Null
& $pb src/PureMetalOnnxCompiler.pb /CONSOLE /THREAD /OPTIMIZER /OUTPUT bin/PureMetalOnnxCompilerCLI.exe
if ($LASTEXITCODE -ne 0) { throw 'CLI build failed' }
& $pb src/PureMetalOnnxCompilerUI.pb /THREAD /OPTIMIZER /DPIAWARE /OUTPUT bin/PureMetalOnnxCompiler.exe
if ($LASTEXITCODE -ne 0) { throw 'UI build failed' }
```

## Check the result

```powershell
.\bin\PureMetalOnnxCompilerCLI.exe --self-test
.\bin\PureMetalOnnxCompilerCLI.exe --list-targets
```

Then run the [arithmetic demo](QUICKSTART.md). See the
[validation record](VALIDATION.md) for the additional internal runtime checks
performed before publication; diagnostic sources are not part of this export.

## Why the runtime folder is required

`src/compiler/onnx_dynamic_emit.pbi` embeds runtime files using `IncludeBinary`.
`onnx_storage.pbi` embeds the reduced-weight decoder, and `onnx_kokoro.pbi`
includes shared asset/text code directly. These are source files, despite the
name of the embedding instruction. The build needs the complete `runtime/`
folder, including its BASIC target templates and `math/` sources.

Changes to embedded runtime source require rebuilding the CLI. Existing generated
applications do not update themselves: regenerate their source closure and
perform a separate application build.

## Output and redistribution

### Optional private release destination

A source checkout can be the authority for a separate product's executables.
For that developer workflow only, an ignored `build.local.ini` may select a
private PowerShell release-build script:

```ini
[Build]
ReleaseScript = C:\YourProduct\tools\build_onnx.ps1
```

When this file exists, `Build.pb` delegates to that script with `-SourceRoot`
pointing to this checkout and `-Compiler` pointing to the running PureBasic
installation. The script owns building both tools, provenance checks, signing,
and installing them in the product's release directory. An optional `ReleaseHost`
entry selects the absolute path of the PowerShell executable used by that private
workflow; otherwise Windows PowerShell is used. A nonzero script exit
is a failed build. A missing/invalid configured script fails rather than quietly
building elsewhere. Review and trust any script you configure here.

Without the local file, the ordinary Python-free `bin/` build above is unchanged.
The local file, private integration tools, signing material, and machine paths
do not belong in the public repository. This is a developer build hook only;
the model generator still never invokes a downstream language compiler.

The generator recognizes both this checkout's runtime layout and an installed
PureMetal runtime layout when locating optional speech assets. Its embedded
execution sources always come from the repository used to build it.

### Published files

The repository tracks source and documentation, not executable builds, signing
credentials, model weights, or local settings. `bin/` and generated `output/`
files are ignored by Git. This checkout does not need files from another project
directory to build its two tools.
