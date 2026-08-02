# Stability matched 12K schedule-transition trajectory

Date: 2026-08-03

## Purpose

The earlier exact-10K report established a closely aligned fixed-validation
epsilon trajectory through the rollout-consistency warmup boundary. The active
dense run subsequently completed the scheduled 11K and 12K validation events.
This update adds schedule-aware evidence so later 15K, 40K, and 50K audits do
not mix warmup, full rollout, and EMA-teacher phases into one undifferentiated
mean.

No remote process, checkpoint, queue, control file, or GPU state was modified.
The authoritative JSONL files and run manifests were copied read-only, then
reduced locally to byte-exact prefixes ending at the last complete scheduled
validation event, step 12,000. The live step 12,650 observation was not used as
a scientific cutoff.

## Implementation

`scripts/build_generation_matched_training_trajectory.py` now emits schema 2
and derives the two shared schedule contracts from both resolved manifests:

- rollout consistency: start 0, warmup 10,000, full scale at 10,000;
- EMA-teacher consistency: start 30,000, warmup 10,000, full scale at 40,000.

Every paired validation event is assigned a predeclared phase. The builder
independently verifies that both methods recorded the same finite schedule
scale, that inactive or disabled scales are zero, warmup scales remain below
one, and full-scale values equal one. Each nonempty phase receives the same
paired epsilon summary as the global report. Regime-level significance and
quality claims remain explicitly disabled.

The existing trust checks remain unchanged: clean expected training identity,
formal dataset/runtime provenance, matched generation-pair contract, parameter
gap below 2%, exact log/evaluation schedules, all numeric metrics finite, exact
sample arithmetic, and event-by-event validation provenance equality.

## Result

Both methods pass through step 12,000 and 768,000 images with 241 logged rows
and 12 paired fixed-validation events. Across the full window, CoFiTok/dense
mean validation epsilon MSE is `0.0325498291 / 0.0327016859`, a relative
difference of `-0.464370%`; lower-event counts are tied `6 / 6`.

The predeclared rollout regimes are:

- warmup, steps 1K--9K: relative delta of means `-0.067741%`;
- full scale, steps 10K--12K: relative delta of means `-1.664242%`, with only
  three events;
- step-12K endpoint: `-4.507906%` relative for CoFiTok.

The full-scale subset is too short for a durable-trend conclusion and contains
different deterministic validation batches across events. The result only
shows that no immediate shared-epsilon collapse is visible after rollout
consistency reaches full weight. It does not establish sample quality, formal
gate success, promotion, or full-300K readiness. EMA-teacher consistency is
still inactive throughout this evidence window.

Evidence:

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_12k_schedule_trajectory_2026-08-03/
```

`trajectory_report.json` has SHA256
`64d291ae5730ce80d7b1656728e97dbf930aac68149332375fab8c2dbe8e7d62`.
Exact matched 50K completion and formal EMA sample evaluation remain the next
decision-grade evidence.
