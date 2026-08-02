# Generation checkpoint evaluation completed replay (2026-08-03)

## Problem

Formal generation post-evaluation runs the checkpoint mechanism diagnostic on
1,024 validation images before long-form sampling and metrics. Sampling already
has an immutable manifest and exact resume, but
`evaluate_generation_checkpoint.py` previously rewrote its only JSON report on
every invocation. If a later post-evaluation stage failed, restarting the
runbook recomputed an already completed CoFiTok or dense diagnostic. A changed
timestep, seed, order count, checkpoint, or evaluator checkout could also reuse
the same output path without an immutable request boundary.

## Completed-replay contract

Checkpoint evaluation now writes
`checkpoint_evaluation_manifest.json` before the expensive dataloader/model
loop. The manifest binds:

- the physical checkpoint path, bytes, SHA256, step, format, and artifact type;
- the adjacent integrity sidecar path, bytes, and SHA256;
- evaluator Git revision, branch, and tracked-dirty state;
- image count, timestep, random-order count, seed, weights, and precision;
- the canonical output directory and report path.

`--resume` supports three fail-closed states:

1. An empty output directory starts a fresh evaluation and publishes the
   immutable manifest.
2. A matching manifest without a report reruns that evaluation from the
   beginning. The diagnostic does not claim batch-level exact resume.
3. A matching completed report is reused without loading the model or touching
   the GPU after independently verifying the checkpoint and sidecar, manifest
   identity, request, Git identity, config/timestep bounds, deterministic order
   set, evaluated count, and positive runtime provenance.

An existing report without its manifest, a request or checkpoint drift,
unexpected files, a symlinked evidence path, or a malformed/tampered report is
rejected. Invoking the evaluator again without `--resume` also refuses to
overwrite evidence.

The stability 50K post-eval, future full 50K post-eval, and shared 300K
milestone runner now always pass `--resume`. The locked legacy 10% evaluation
is not migrated or rerun.

## Scope and deployment boundary

This is post-evaluation reliability hardening; it does not change training,
EMA weights, sampling, metrics, promotion thresholds, or full-300K
authorization. The active stability dense trainer and its existing post-eval
waiter remain on their immutable remote revisions and were not modified. This
contract applies only after the change is deployed through a separately bound
checkout/receipt.

## Verification

CPU subprocess tests cover fresh execution, fresh `--resume`, completed report
reuse without rewriting bytes or mtime, rerunning an incomplete manifest,
request-drift rejection, inconsistent-report rejection, and refusal to
overwrite existing evidence. Runbook tests bind the three formal callers to
the new resume contract. Full repository pytest and shell syntax results are
recorded in the implementing commit.
