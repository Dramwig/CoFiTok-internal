# Generation training-exposure guard

Date: 2026-08-19

## Problem

Generation reports already recorded optimizer steps, effective batch and
images seen, and the quality-bridge preparation computed equivalent epochs.
However, the preparation validator did not verify the exposure fields and the
terminal result did not surface the validated exposure contract. A downstream
reader could therefore treat equal `50K` labels as equal training exposure
across `imagenet_256_10pct` and full `imagenet_256`.

That interpretation is materially wrong:

- 10% ImageNet-256, 50K x batch 64: `3.2M` images and `24.9685942` epochs;
- full ImageNet-256, 50K x batch 64: `3.2M` images and `2.4977228` epochs;
- full ImageNet-256, 100K x batch 64: `6.4M` images and `4.9954456` epochs.

The full-data 100K bridge uses twice the source pair's raw training images but
only `20.0069%` of its dataset-normalized epoch exposure.

## Implementation

Added `cofitok.training_exposure` with strict validation of:

- formal dataset provenance and train split size;
- completed/target steps;
- micro-batch, accumulation and effective batch;
- exact `samples_seen = completed_steps * effective_batch`;
- exact partial/complete training status;
- completed and target equivalent epochs.

Added `scripts/build_generation_training_exposure_audit.py`, which binds each
input report by absolute path, bytes and SHA256, compares step/image/epoch
budgets separately, and permanently disallows sample-quality or method-ranking
claims from exposure evidence alone.

The existing quality-bridge preparation validator now checks its source/full
train counts, images seen, equivalent epochs and milestone exposure fields.
`build_quality_bridge_result` surfaces the validated exposure contract for
future execution revisions.

## Frozen evidence

```text
artifacts/reports/generation/training_exposure_audit_2026-08-19
```

`training_exposure_report.json` is `6,298` bytes with SHA256
`e1a557a4c54a0813af438a3237bfb4e540944198937062c4b26ea25bea7edaef`.

Sources:

- frozen 10% CoFiTok training report: `10,628` bytes, SHA256
  `8eeeb5e1cac3a3fe38d20fbc26be7af88788910f4dc86c5d34548be50a1b77fd`
- frozen 10% dense training report: `9,832` bytes, SHA256
  `68374ce7a33f2fe3979f53d255785250ac6aeb60db80934f1bf25e319216bb6c`
- full-data quality-bridge preparation: `23,398` bytes, SHA256
  `7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea`
- full-data CoFiTok 50K training report: `11,245` bytes, SHA256
  `6ac71cc89e5f5769fe5d4b4e9e6789f1ea377ccd9179f435e23ce4e27ea81cd7`,
  reused from the separately frozen 50K milestone archive.

## Verification and boundary

The focused exposure and quality-bridge test group passes (`32` tests). Tests
cover partial and complete training reports, 10%/full dataset normalization, inconsistent
sample arithmetic, inconsistent completion state, corrupted formal split
identity, comparison boundaries, source binding, and mutated preparation
exposure fields.

No active runbook, remote checkout, checkpoint, sample set, GPU process or
authorization artifact was modified. The active `cf0e5faa...` quality bridge
continues under its original pinned code. This hardening is for post-run audit
and subsequent execution revisions; it cannot alter or retroactively rewrite
the immutable running experiment.
