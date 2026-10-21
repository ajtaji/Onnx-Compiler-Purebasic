# Preparing immutable INT8 weights once

The Pi 4 AArch64 runtime has optional caller-owned caches for constant matrix
and convolution weights. Preparing them once after binding avoids repeating
layout conversion during requests. Source weights stay unchanged. These are
weight layouts, not stored inference results or recorded speech.

The optional developer tool uses only Python's standard library:

```text
python tools/prepare_immutable_weight_caches.py generated/model.pi4 --output generated/weight-caches.pmi
```

Run it on a fresh dynamic Pi 4 INT8 model emitted by the current compiler.
The destination must not exist. It writes a native include and a JSON plan with
the source digest, selected constant tensor metadata and exact byte requirements
for three buffers: matrix layout, convolution layout and convolution proof masks.
The compiler and generated application do not need Python at runtime.

Include the result **after** the generated model. After a successful bind,
call `PmModelPrepareWeightCaches(matrix,matrixBytes,conv,convBytes,proof,proofBytes)`
once, with three distinct caller-owned buffers. A return of one means ready;
zero means preparation failed and all these caches are disabled. The generated
constants `#PMO_PREPARED_MATRIX_BYTES`, `#PMO_PREPARED_CONV_BYTES` and
`#PMO_PREPARED_PROOF_BYTES` state minimum sizes. Each nonempty buffer must be
16-byte aligned and entirely within the Pi 4's 32-bit physical address range.
Zero-sized cache classes need no buffer.

Preparation checks capacities, buffer overlap with each other, weights and the
bound tensor arena, and the selected live descriptors against generated metadata.
The caller must also keep these buffers clear of firmware, stacks, peripherals
and other application state. Preparation is synchronous: perform it while
inference is idle. Do not prepare or clear caches from a runtime callback.

Keep the model bound, its source weights unchanged, and the three buffers live
between serialized requests. `PmModelResetRequest()` releases request storage
without clearing these layouts. Before closing, rebinding, replacing weights
or reclaiming cache storage, call `PmModelClearWeightCaches()` while idle.
Never reuse the generated helper with a different emitted model; regenerate it.
It is one helper for one model, with the same global-state serialization
requirement as the model runtime.

Selection is based on operator, storage kind and dimensions. Eligible static
MatMul right operands have rank two, INT8 or wide INT8 storage, at least 65,536
elements and 16 columns. Eligible Conv weights have rank three, narrow INT8
storage, 64/128/256/512 input channels per group, output channels divisible by
four, and multiple taps. The runtime still checks the actual execution form;
grouped, strided and wide convolutions retain their existing fallback.
No model name or tensor name is used. Temporary operands are never selected.
Each cache class supports at most 64 entries; oversized plans fail explicitly.
The tool accepts the current emitted metadata layout, not arbitrary BASIC source.

For 256/512-channel convolutions, the proof mask is zero and the packed kernel
uses its safe two-product integer path. Larger proof groups cannot reuse the
64/128-channel mask layout without overlapping bit fields. Tap order, scaling
and floating-point reduction order are unchanged.

## Measured scope

On 21 October 2026, a silent four-core Pi 4 Kokoro-82M mixed INT8 comparison
rendered eight complete TruckGPS directions, totaling 27.6 seconds of audio.
Preparing eight matrices and 24 existing eligible convolutions reduced paired
synthesis from 99.904 to 83.657 seconds (16.26%). Preparation took 0.092 seconds.
Enabling the larger-channel packed path in a second paired test, with 49
convolutions prepared, reduced 83.623 to 80.896 seconds (3.26%). Every output
sample remained bit-identical in both comparisons; all memory guards passed.
The larger plan uses 11,993,088 / 14,942,208 / 138,240 bytes across its buffers.

These are silent whole-waveform synthesis results. The final ratio is 2.93
seconds of compute per second of speech: **Kokoro real-time generation and
subsecond warm speech remain unmet.** They do not measure map concurrency,
Bluetooth onset or exact agreement with an FP32 reference. The mixed INT8
baseline itself has known reference differences. Models with other shapes may
benefit differently or have no eligible weights.

The rebuilt standard compiler then emitted a fresh model/runtime closure.
Using the generated helper, all eight directions completed in 80.978 seconds
and reproduced the previous optimized candidate byte for byte. Six native
refusal checks (arena overlap, weight overlap, overlapping caches, misalignment,
short capacity and changed tensor metadata) left caches disabled and guards
intact. Eight host metadata safety tests passed. The standalone
`tests/pi4_large_packed_conv_gate.pi4` also passed all 12 cases with no model
assets on the physical four-core Pi. Both standard compiler applications were
rebuilt and the CLI self-test passed. These checks do not certify other targets.
