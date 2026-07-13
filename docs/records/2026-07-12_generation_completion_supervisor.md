# Bounded generation completion supervisor (2026-07-12)

## Problem

The completion pipeline spans 10K post-evaluation, two alternating 300K full
trainings, milestone evaluation, and two 50K formal sample sets. Exact checkpoint
resume protects model state, but an exited shell previously required manual
relaunch. A naive automatic restart would also be unsafe because it could retry a
failed scientific gate or repeat expensive completed evaluation.

## Supervisor policy

`generation_completion_supervisor.sh` launches the existing locked completion
pipeline as a child. It uses a separate `flock`, atomic status JSON, and at most
four attempts with exponential delays of 60, 120, and 240 seconds (capped at 900
seconds when configured for more attempts).

The structured classifier retries only:

- `posteval_10pct` before a promotion gate exists;
- `full_training`, which resumes exact model/optimizer/scheduler/RNG/sampler and
  reconciles canonical metrics;
- `full_posteval`, whose numbered samples and atomic progress are resumable.
- `inference_export`, whose verified EMA artifacts and atomic smoke reports are
  idempotent for the same source checkpoint SHA.

It never retries `preconditions`, `promotion_gate`, `final_gate`,
`completion_audit`, or unknown stages. These may represent scientific failure,
provenance conflict, or an unsafe state and therefore require inspection.
Storage-capacity preflight failure uses exit code `78` and is likewise
non-retryable until an operator has inspected or expanded the project filesystem.

## Idempotency

The completion pipeline now validates an existing scaling gate instead of
recomputing completed 10K evaluation. It skips final post-evaluation only when
the final gate, comparison, and visual audit all exist.

The full-training runbook checks each paired milestone report before work. A
completed milestone is skipped only when both protected checkpoint payloads and
integrity sidecars still exist. If one method has advanced beyond a milestone,
the protected checkpoint is reused; advancing without that checkpoint fails
immediately. This permits recovery after interruption between CoFiTok and dense
segments without changing revision or repeating already paired milestones.

## Deployment

The remote deploy helper now launches the supervisor under `nohup`, records a
supervisor PID/log, and syntax-checks both supervisor and child runbooks after
the full remote test suite. Re-running deployment while that PID is alive is
idempotent.

The supervisor does not survive a server power cycle by itself. After a power
cycle, rerunning the same pinned deploy command is safe; it finds existing
artifacts and resumes through the same bounded policy.
