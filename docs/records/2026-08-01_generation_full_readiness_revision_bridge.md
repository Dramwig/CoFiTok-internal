# Full Readiness Revision Bridge

Date: 2026-08-01

## Scope

The stability-full CUDA readiness artifact remains source-bound to revision
`5dd3488ac9b30274f4960195e252cc9fdb161002`. Later control-plane hardening did
not change the 250M training implementation, but a source-bound readiness
artifact must not silently authorize a different checkout.

This change adds an immutable readiness revision bridge between the source
readiness checkout and the eventual full-training checkout. It does not run
CUDA readiness, launch full 300K training, or weaken the separate human
authorization requirement for that launch.

## Bridge Contract

The bridge replays and rehashes:

- the source readiness report and all seven of its physical source files;
- the source and target isolated deployment receipts;
- the source and target Git identity and ancestor relationship;
- the current target CUDA/runtime environment fingerprint;
- a 66-blob training-critical Git manifest covering package/runtime files,
  both 250M configs, training/monitor/audit scripts, and milestone evaluation.

Every training-critical blob must be identical across the two revisions.
The full-training runbook is split at `monitor_report_passes() {`. The training
execution suffix must be byte-identical. The authorization preamble is also
fail-closed: after exactly removing the six bridge-only bindings, replacing
the complete bridge-validator command with the source readiness-validator
command, and restoring the sample reserve from `116640` to `16384`, the target
preamble must be byte-identical to the source preamble. Extra commands,
missing bindings, command-line drift, and moved execution markers are rejected.

The bridge report stores both normalized preamble hashes and the training
execution suffix hash so terminal completion auditing can independently replay
the proof.

## Completion Runway

Readiness remains backward-compatible with the historical `16384` training
sample allowance. The full launch receipt requires `116640` samples:

- `16384` training-time allowance;
- `100256` formal post-evaluation and prefix-diagnostic allowance.

With the latest observed checkpoint size, the aggregate reserve requires
`180880415360` free bytes. The last read-only estimate observed
`199258644480` free bytes, leaving `18378229120` bytes of headroom. Launch and
resume must recompute this capacity; the estimate is not durable authorization.

## Downstream Binding

The original full launch receipt was schema v2 and contained ten rehashed
sources, including the bridge. The 2026-08-03 quality-prerequisite follow-up
advances this to schema v3 and eleven sources by also binding the passing frozen
stability supplemental. The post-training supervisor now requires schema v3 and
physically rehashes all eleven sources. The terminal completion auditor adds an
independent `stability_full_readiness_revision_bridge` check and replays the
bridge before accepting launch, training, post-evaluation, export, or final
completion evidence.

The first Linux isolated deployment attempt for `35a9b99` failed before receipt
creation because the server Git version does not support `git ls-tree --format`.
The deployment helper removed its temporary checkout and left the formal
repository unchanged. The manifest builder now parses the stable default
`ls-tree` record format, preserving the same blob/path contract across old and
new Git versions.

The final target revision and deployment receipt are intentionally not written
here until the compatibility fix is committed, bundled, validated on Linux,
and deployed to a new immutable checkout. The active readiness waiter is not
replaced, and the active matched 50K queue is not modified.
