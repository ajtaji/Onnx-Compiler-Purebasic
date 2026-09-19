# Resident progress and cancellation

Reusable generated runtimes expose `DSetProgressCallback(callback)`. The
callback has the form `callback(phase, node)` and returns nonzero to continue
or zero to cancel. Passing `0` disables it. No callback is installed by
default, so applications that do not opt in retain their existing behavior.

| Phase | Value | `node` |
|---|---:|---:|
| `PMD_PROGRESS_BIND_BEFORE` | 1 | -1 |
| `PMD_PROGRESS_BIND_AFTER` | 2 | -1 |
| `PMD_PROGRESS_NODE_BEFORE` | 3 | generated node index |
| `PMD_PROGRESS_NODE_AFTER` | 4 | generated node index |

Callbacks execute synchronously on the inference thread and must not re-enter
the resident model or Kokoro adapter. Generated stage/chunk, bind/init, reset,
close, input/output, and execute entry points reject callback re-entry before
mutating state. Kokoro adapter close, bind, voice, prepare, and render entry
points follow the same rule. A rejected re-entry records an error and sets
`DCancel`.

Returning zero or setting `DCancel` requests cancellation. The callback first
unwinds normally; the operation then follows its ordinary failure and cleanup
path. Inspect `DError`, `DNode`, and `DCancel`, perform the normal close or reset
for the request, and clear cancellation only when deliberately starting a new
request.

Progress points are node boundaries. They make multi-node inference observable
and cancellable, but do not bound latency inside an operator, CRC pass,
allocation, or platform service. Platform-specific code may use the callback
to service a watchdog or perform required cache maintenance, but those policies
do not belong in the target-neutral tensor core. Keep callbacks short and
non-reentrant.

The portable callback and emitted regression gates establish the
target-neutral contract. Watchdog and cache integration still require evidence
on each applicable platform. A complete end-to-end Kokoro Pi run has not yet
proved this progress mechanism.

