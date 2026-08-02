# Stability matched 20K schedule trajectory

Date: 2026-08-03

## Purpose

The active dense-identity recovery has crossed an exact checkpoint-aligned
20,000-step boundary. This record freezes a source-bound matched trajectory at
that boundary, extending the earlier exact-12K schedule analysis without using
the live, growing tail of either metrics file.

This is a read-only training-space diagnostic. No trainer, controller, monitor,
watchdog, waiter, checkpoint, queue, lock, or GPU process was modified. The
formal sample-quality decision remains assigned to the existing 50K EMA
post-evaluation and frozen supplemental queue.

## Bound sources and integrity

Both methods use the clean immutable training identity
`2c2c1f5166b73d4f28df93b276901671ac1a7836` on
`scale/generation-stability-50k-preflight`, the same formal
`imagenet_256_10pct` dataset identity, the same runtime-environment identity,
effective batch 64, log interval 50, and validation interval 1,000.

The builder retained exactly 401 rows and 1,280,000 images per method through
step 20,000. Every numeric metric is finite; optimizer steps are strict; and
`samples_seen = step * 64`. All 20 scheduled validation events match in step,
event index, validation batch, fixed noise seed, and image count.

The live dense checkpoint at this boundary is also bound by the required
integrity policy:

- checkpoint: `checkpoint_step_00020000.pt`;
- bytes: `1,006,120,214`;
- SHA256: `86b4f9bd3a76168d94c9c3420346ca1c66dbab9a024dd2d45221edad20162f1f`;
- adjacent sidecar and `latest.json` both bind the same step, bytes, SHA,
  dataset identity, runtime identity, branch, and revision.

The recovery controller still holds the advisory write lock on fd 6 at
`dense_recovery.lock`.

## Result

Across the full 1K--20K fixed-validation window, CoFiTok and dense have mean
epsilon MSE `0.0319879465` and `0.0320814812`. CoFiTok is lower by `0.291554%`
in the ratio of means, while lower-event counts are tied `10 / 10`.

The predeclared rollout-consistency regimes remain well behaved:

- warmup, 1K--9K: CoFiTok mean is lower by `0.067741%`, with lower-event
  counts `4 / 5`;
- full scale, 10K--20K: CoFiTok mean is lower by `0.481945%`, with lower-event
  counts `6 / 5`;
- exact 20K endpoint: CoFiTok is higher by `0.185878%`.

The event-to-event result therefore remains closely matched rather than
showing a durable one-sided advantage. EMA-teacher consistency is still
inactive before its predeclared 30K start, so this evidence says nothing about
the later teacher warmup or full-scale regimes.

This result supports continued execution of the existing dense run. It does
not establish free-generation quality, FID/IS/precision/recall, rollout
stability, a causal EMA-teacher benefit, promotion, or full-300K readiness.
Every authorization field in the report remains false. Required next evidence
is exact healthy matched 50K completion followed by formal EMA post-evaluation.

## Evidence

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_20k_schedule_trajectory_2026-08-03/
```

Key SHA256 values:

- `trajectory_report.json`:
  `7c682e4ca5dba5634b003f6ff1ee06801963454341247b74e31bc35f58ab483a`;
- CoFiTok 20K metrics prefix:
  `199c725ace63ca7cbe6cac4f055e015354039df66956c049a566bd2ddba6eb62`;
- dense 20K metrics prefix:
  `a8d9c37cb386cd2cea6eb801d70cf8e323081c2b6642983b131865d3fef3b1b4`;
- CoFiTok run manifest:
  `3b554762a5ee75d77fbf66af616c6eaab31ae24dd7a6f40f1f9d4f747bc06519`;
- dense run manifest:
  `a049a97d8a4e9f4dc4761a9269e50ae55f53ec28b9971d14a7d968bfd7e9f9fe`.

## Verification

- `tests/test_build_generation_matched_training_trajectory.py`: `13 passed`;
- rebuilding from the five frozen evidence files produced byte-identical
  `trajectory_report.json` with the same
  `7c682e4ca5dba5634b003f6ff1ee06801963454341247b74e31bc35f58ab483a`
  SHA256.
