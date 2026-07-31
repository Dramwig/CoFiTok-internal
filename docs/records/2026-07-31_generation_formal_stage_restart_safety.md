# Formal post-evaluation restart safety (2026-07-31)

## Problem

The stability post-training supervisor retried whole shell runbooks. Formal
sampling itself supported resume, but several expensive downstream stages did
not have a safe replay contract:

- checkpoint mechanism evaluation and 50K torch-fidelity metrics overwrote
  reports on every retry;
- inference preflight overwrote its report;
- smoke inference used `--overwrite` and regenerated PNGs;
- artifact export was internally reusable, but its report was still rewritten;
- a final gate could exist before the comparison table, causing the supervisor
  to skip an incomplete post-evaluation tail;
- an existing terminal completion report could be accepted without re-reading
  all physical evidence.

A process interruption after a costly command succeeded but before the parent
recorded success could also trigger a full rerun.

## Source-bound stage receipts

`scripts/run_generation_stage_once.py` provides a shared fail-closed stage state
machine. Every protected stage declares:

- the exact clean Git checkout;
- the exact command and working directory;
- SHA256/byte identities for input files;
- content-addressed identities for input directory trees;
- the complete expected output file/tree set.

The first invocation rejects any unreceipted output. A completed invocation is
reused only after all inputs, the command, Git provenance, the worker sidecar,
and every output identity are recomputed and match exactly. Source, request,
receipt, worker-result, output, or symlink drift exits with code `86`. The
post-training supervisor treats that code as nonretryable.

Failed or interrupted attempts move partial outputs to attempt-numbered archive
paths before retrying. Existing files are never adopted by name.

## Parent-crash recovery

The stage runner launches an independent worker. The worker records the real
child PID and atomically writes a source-bound result sidecar after the command
exits. If the parent wrapper is interrupted after the costly command succeeds,
a later invocation waits for any still-live worker/child process, validates the
sidecar and outputs, and completes the receipt without rerunning the command.

Recovery also covers the narrow launch-to-receipt window where the parent has
spawned the worker but has not yet persisted the worker PID. If the result
sidecar still says `running`, the replay path waits for the recorded worker (or
its command child when the worker has already disappeared), reloads the atomic
sidecar, and only then decides whether to adopt or archive outputs. A live
worker can therefore finish successfully without its stale in-memory
`running` record being misclassified as a failed attempt.

This closes the success-to-receipt crash window for long checkpoint evaluation,
50K FID/IS/precision/recall, export preflight, and smoke inference.

## Protected formal stages

The stability full post-evaluation runbook now receipts:

1. CoFiTok and dense checkpoint evaluation;
2. CoFiTok and dense 50K generation metrics;
3. deterministic visual audit panels;
4. the final scientific gate;
5. the strong-baseline comparison report.

The metrics stages bind both generated image trees and the actual ImageNet-256
validation image tree. Formal sampling continues to use its existing immutable
manifest, progress report, sample-set digest, and `--resume` protocol.

The inference export runbook receipts:

1. CoFiTok and dense EMA-only inference artifacts plus integrity manifests;
2. CoFiTok and dense release-authorized real-forward preflights;
3. CoFiTok and dense smoke inference reports and PNG trees.

The prior unconditional smoke `--overwrite` flags were removed.

## Terminal re-audit

The post-training supervisor always replays the full post-evaluation runbook;
receipts skip valid completed work and continue any missing tail stage. It no
longer returns success solely because a completion report exists.

The completion runbook removes the previous completion report before every
audit. The audit then rehashes checkpoints, formal samples, the real set,
inference artifacts, smoke PNGs, source reports, and comparison sources. It now
also decodes and rehashes all three visual-audit PNGs, requires canonical paths
inside the formal visual-audit directory, and rejects extra unreported PNGs.

## Verification

Local verification on `scale/generation-large-capacity`:

- focused restart/replay tests passed, with only the Windows symlink-creation
  case skipped because it requires elevated privileges;
- the complete pytest suite passed;
- all formal runbook CLI entrypoints, including the nested stage runner
  contract, passed the live `--help` audit;
- `git diff --check` passed.

The Linux exact-commit rehearsal and all tracked `bash -n` checks are required
before deployment.

## Runtime boundary

This work is CPU-only code hardening. It does not modify the formal repository,
stop or restart the active stability 50K queue, run CUDA readiness, authorize
full 300K training, or create a full-training launch receipt.
