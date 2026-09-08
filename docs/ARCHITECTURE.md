# How the compiler is organized

[Home](../README.md) · [Source map](SOURCE_MAP.md) · [Validation](VALIDATION.md)

## The boundary that matters

```text
ONNX file
    |
    v
Bounded native protobuf reader
    |
    v
Graph validation + constants + shapes + memory/precision planning
    |
    +---- fixed shapes --------> PmOnnx... source API
    |
    +---- runtime dimensions --> PmModel... source API
                                  |
                                  v
                 Source + checked weights + runtime source + manifest
                                  |
                          END OF THIS TOOL
                                  |
                 Separate PureBasic / PureMetal application build
                                  |
                     Initialize once; submit many requests
```

The GUI is a native client of the sibling CLI. It does not hide a Python process
or compile an executable for each request. `--build` is explicitly refused.

## Graph and memory planning

The parser checks protobuf boundaries and keeps immutable model spans where
possible. The IR owns graph semantics, supported forms, constants, aliases,
and lifetimes. Targets select dialect and kernels rather than changing the
meaning of a graph operation.

Fixed-shape emission assigns storage using known lifetimes. Runtime-dimension
emission separates persistent model initialization from request computation;
temporary request tensors are released without discarding the model's weights.
Windows uses checked heap allocation, while portable targets use caller-provided
working memory. Tensor IDs and generated globals currently form a single,
serialized model instance rather than a reentrant multi-instance object API.

## Precision and packing

Eligible INT8 paths use packed constant weights and integer accumulation, while
unsupported transformations stay on the FP32 path. FP16/BF16/INT4 are storage
reductions with explicit resident decode costs. Packs carry validation data,
and generated code checks that the supplied weights match the expected model.
The source, pack, and manifest must therefore be deployed as a matching set.

## Source closure

The generator embeds its required runtime **source** at build time. Generation
exports selected sources into the model's runtime directory. That is why
`runtime/` belongs in this repository and why changing it requires a generator
rebuild before regenerating an application.

The math templates are the minimum target runtime sources, not a redistribution
of the PureMetal compiler. Generated bare-metal code still needs application
startup, a memory map, and the separately supplied language toolchain.

## Optional reference engine

The native ONNX Runtime C API bridge is for optional trace/reference commands
using a separately supplied compatible DLL (C API version 29). It is not linked
into generated model execution and is not required for the arithmetic quick
start, generator builds, or the resident Kokoro emission path. An NPY reader
does not imply a Python dependency: its format is parsed directly in PureBasic.

## Speech is an adapter

Kokoro tensor execution, text-to-phoneme preparation, voice selection, and audio
playback are separate layers. The Windows reader chunks text and queues audio;
the portable text adapter deliberately contains no filesystem, audio driver,
compiler invocation, or board-access code. The application supplies those services.
