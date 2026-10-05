# Stability matched 35K EMA-teacher schedule trajectory

Date: 2026-08-03

## Purpose

The active dense-identity recovery crossed an exact checkpoint-aligned
35,000-step boundary. This record freezes the matched CoFiTok/dense trajectory
through that boundary and, unlike the earlier 20K snapshot, includes the first
six fixed-validation events in the predeclared EMA-teacher warmup window from
30K through 35K.

This is a read-only training-space diagnostic. No trainer, controller, monitor,
watchdog, waiter, checkpoint, queue, lock, or GPU process was modified. Formal
sample quality remains assigned to the existing exact-50K EMA post-evaluation,
frozen supplemental, and class-fidelity evidence chain.

## Bound sources and integrity

Both methods use the clean immutable training identity
`2c2c1f5166b73d4f28df93b276901671ac1a7836` on
`scale/generation-stability-50k-preflight`, the same formal
`imagenet_256_10pct` dataset identity, runtime-environment identity, effective
batch 64, log interval 50, and validation interval 1,000.

The builder retained exactly 701 rows and 2,240,000 images per method through
step 35,000. Every numeric metric is finite; optimizer steps are strictly
increasing; and `samples_seen = step * 64`. All 35 scheduled validation events
match in step, event index, validation batch, fixed noise seed, and image count.
The resolved pair contract has no issues.

The live dense recovery checkpoint at the frozen boundary is bound by the
required integrity policy:

- checkpoint: `checkpoint_step_00035000.pt`;
- bytes: `1,006,120,214`;
- SHA256: `470740ea8b52172070ac7198c480dade4414421369a6e33cb408ba8532cf371d`;
- the retained adjacent sidecar binds the same step, bytes, SHA, dataset
  identity, runtime identity, branch, and revision.

The live `latest.json`, watchdog, and pair-monitor snapshots were collected only
after the run had advanced beyond 40K, so they are deliberately excluded from
this 35K pack rather than being misrepresented as boundary-time evidence. The
separate 40K pack binds those later observations. Lock/process health remains a
live operational claim, not part of this frozen trajectory report.

## Result

Across the full 1K--35K fixed-validation window, CoFiTok and dense have mean
epsilon MSE `0.0303214049` and `0.0303540541`. CoFiTok is lower by `0.107561%`
in the ratio of means; lower-event counts are `14 / 21`. At the exact 35K
endpoint CoFiTok is higher by `0.107953%`.

The predeclared schedule regimes show:

- rollout-consistency warmup, 1K--9K: CoFiTok mean is lower by `0.067741%`,
  with lower-event counts `4 / 5`;
- rollout-consistency full scale, 10K--35K: CoFiTok mean is lower by
  `0.122864%`, with lower-event counts `10 / 16`;
- EMA-teacher inactive, 1K--29K: CoFiTok mean is lower by `0.156157%`, with
  lower-event counts `12 / 17`;
- EMA-teacher warmup, 30K--35K: CoFiTok mean is higher by `0.172577%`, with
  lower-event counts `2 / 4`; the maximum absolute event-relative delta in
  this window is `0.550214%`.

The teacher warmup therefore remains finite and closely matched, but six
scheduled events are not evidence of a causal teacher benefit or durable
quality advantage. This report does not establish free-generation quality,
FID/IS/precision/recall, rollout stability, promotion, or full-300K readiness.
Every authorization field remains false. Required next evidence is exact
healthy matched 50K completion followed by formal EMA post-evaluation.

## Evidence

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_35k_schedule_trajectory_2026-08-03/
```

Key SHA256 values:

- `trajectory_report.json`:
  `e02d63c966dd6450aa40df4a8a94364ed02c4db4da106c3fbbb277bbad4c7329`;
- CoFiTok 35K metrics prefix:
  `364f08b9646913394c936c0308dcbc5b4c35331bcac3c55c0833c4d68c1cba12`;
- dense 35K metrics prefix:
  `fbcc22912b3d3378bbf26535be63bdd99a9a805de0a8ce6aad12a4a8f1a7ded0`;
- CoFiTok run manifest:
  `3b554762a5ee75d77fbf66af616c6eaab31ae24dd7a6f40f1f9d4f747bc06519`;
- dense run manifest:
  `a049a97d8a4e9f4dc4761a9269e50ae55f53ec28b9971d14a7d968bfd7e9f9fe`;
- dense 35K checkpoint integrity sidecar:
  `d38f6596b33874fd7a11d60b697e9f227b50de593c9fd1d4b972390bb8b7df5c`.

## Verification

- `tests/test_build_generation_matched_training_trajectory.py`: `13 passed`;
- rebuilding from the four frozen builder inputs produced byte-identical
  `trajectory_report.json` with the same
  `e02d63c966dd6450aa40df4a8a94364ed02c4db4da106c3fbbb277bbad4c7329`
  SHA256;
- the `ml-training-recipes` exact runtime, checkpoint, and EMA inference
  boundaries were retained: this snapshot diagnoses training continuity only
  and leaves formal EMA sample quality to the frozen post-training chain.
