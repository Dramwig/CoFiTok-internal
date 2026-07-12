# Generation resume metrics reconciliation (2026-07-12)

## Problem

Production checkpoints already restore model, EMA, optimizer, scheduler,
scaler, RNG, and consumed sampler position. A hard interruption can still leave
`train_metrics.jsonl` ahead of the latest durable checkpoint. Resuming from
that checkpoint and appending new rows would duplicate or decrease step IDs,
making the training history non-canonical even when model-state recovery is
exact.

Starting without `--resume` in an output directory that retained a checkpoint
but lost its metrics file was a related overwrite risk.

## Implementation

`cofitok.training.metrics.reconcile_metrics_for_resume` now runs immediately
after checkpoint state is restored and before the resumed run manifest is
written. It:

- validates every existing JSONL row before modifying the file;
- keeps the latest occurrence of each step at or before the checkpoint;
- orders the retained canonical history strictly by step;
- preserves checkpoint-ahead and superseded duplicate rows in a
  content-addressed `train_metrics_orphaned_*` archive;
- atomically rewrites the active JSONL and writes an atomic reconciliation
  report containing row counts and the orphan SHA256;
- records the reconciliation result in `run_manifest.json` and therefore in
  the final training report.

`ensure_fresh_training_output` runs before dataloader/model construction when
no resume was requested. It rejects an output directory containing a run
manifest, training report, latest pointer, metrics, checkpoint (including a
temporary checkpoint), orphan archive, or reconciliation report.

Train and validation iterators are created only after checkpoint RNG and
sampler state restoration. The production ImageNet transform is deterministic
(resize, tensor conversion, normalization), so worker-local augmentation RNG
does not create an uncheckpointed data path; consumed sampler position remains
the authoritative data-order state.

The validation iterator is advanced by the number of scheduled validation
events implied by the restored step. Intermediate `--stop-after-steps`
boundaries no longer create an unscheduled validation event; validation runs
only at the configured interval or the true configured training endpoint.

## Verification

Unit tests cover checkpoint-ahead rows, duplicate trajectories, unchanged
histories, malformed JSON with zero mutation, missing metrics, and fresh-run
state refusal. A local 16x16 random CPU integration run stopped at step 1,
resumed from the full checkpoint, and completed step 2 with:

```text
resume_status: unchanged
resume_step: 1
completed_steps: 2
training_complete: true
metric_steps: 1,2
```

Formal training remains server-only. This local run verifies control flow and
state bookkeeping without producing paper evidence.

An automated trajectory test also compares an uninterrupted two-step CPU run
against a one-step checkpoint plus exact resume. Model, EMA, optimizer,
scheduler, Python/NumPy/Torch RNG, sampler state, losses, gradient norm, LR,
and validation MSE are bitwise/equality identical at step 2.
