# Third-party software and data

[Home](README.md) · [Project license](LICENSE)

The MIT license in this repository covers the project source and documentation
published here. It does not relicense a compiler, inference engine, model,
dictionary, voice, or other external asset that you obtain separately.

## Included material

The checkout contains the native ONNX tool sources, required BASIC runtime/math
templates, small original arithmetic-model examples expressed as PureBasic data,
and documentation/artwork. It does not contain a vendor compiler implementation,
ONNX Runtime binaries, a Kokoro model, pronunciation dictionary payloads, or voice data.

## Separately supplied dependencies and assets

| Item | Relationship to this project | Where to check terms |
|---|---|---|
| PureBasic | Required to build the Windows tools and emitted Windows applications. Not redistributed here. | [PureBasic](https://www.purebasic.com/) and the license accompanying your installation. |
| PureMetal | Separate language toolchain for generated bare-metal source. Its implementation and licensing protections are not included or changed. | Terms accompanying your separate PureMetal distribution. |
| ONNX | Model interchange format parsed by this implementation. An ONNX package is not required to build or run the examples. | [ONNX project](https://github.com/onnx/onnx). |
| ONNX Runtime | Optional host reference/trace engine only. Not required by generated inference and not bundled. | [ONNX Runtime project and license](https://github.com/microsoft/onnxruntime). |
| Kokoro | Optional speech model and voices supplied separately. Compatibility requires the pinned model/checked asset formats. | [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M). |
| Misaki dictionaries | Source data for compatible external English pronunciation packs; no dictionary payload is included. | [Misaki project](https://github.com/hexgrad/misaki). |
| CMU Pronouncing Dictionary | Source data for compatible external supplementary pronunciation packs; no dictionary payload is included. | [CMUdict project](https://github.com/cmusphinx/cmudict). |

Retain the notices and satisfy the terms accompanying the **specific versions**
you download, transform, or redistribute. A prepared binary pack is still derived
data; changing its format does not remove its original licensing requirements.
The source compiler does not grant redistribution rights to model assets.

Project/product names identify compatibility or provenance, not affiliation or endorsement.
