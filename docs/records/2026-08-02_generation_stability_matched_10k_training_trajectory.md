# Stability matched 10K training-trajectory evidence

Date: 2026-08-02

## Purpose

The active stability pair had reached a shared 10K horizon, but the operational
monitor intentionally reports health rather than scientific paired trends. A
source-bound diagnostic was therefore added to test whether CoFiTok and the
matched dense baseline were already diverging on the shared fixed-validation
stream before waiting for terminal 50K EMA sampling.

No remote process, checkpoint, control file, or GPU state was modified. The
authoritative metrics and run manifests were copied read-only, and the growing
logs were reduced to byte-exact prefixes ending at step 10,000.

## Implementation

`scripts/build_generation_matched_training_trajectory.py` validates:

- the clean expected Git revision and branch;
- identical formal dataset and runtime-environment identities;
- the existing `generation_pair_contract`, including matched data, diffusion,
  runtime, optimizer, shared backbone fields, and shared consistency losses;
- a parameter gap no greater than 2%;
- the exact log schedule, finite numeric rows, and
  `samples_seen = step * effective_batch`;
- the exact scheduled validation steps and complete provenance metadata;
- event-by-event equality of validation step, event index, batch index, image
  count, and fixed noise seed.

It binds both the complete observed input identity and the raw byte prefix
through the cutoff. Optional `--snapshot-dir` materializes those immutable
prefixes, so later dense log appends do not change the evidence.

The report does not compare total loss or wall-clock efficiency. Total loss is
not a shared objective because CoFiTok includes factorization-only auxiliaries,
and the dense run had a known period of unrelated FieldScope GPU contention.

## Result

Both methods pass the contract through 10,000 steps and 640,000 images. The ten
fixed-validation events have mean epsilon MSE `0.0326398859` for CoFiTok and
`0.0326587601` for dense, or `-0.057792%` relative for CoFiTok. The endpoint
difference is only `+0.034889%`; the largest absolute event difference is
`1.444119%`. CoFiTok is lower on four events and dense on six.

The correct interpretation is that the shared validation trajectory remains
closely aligned through the rollout-consistency warmup boundary. It neither
establishes a quality win nor reveals an early shared-epsilon collapse. Formal
generation quality still requires exact matched 50K completion and the existing
EMA post-evaluation waiter.

Evidence:

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_10k_training_trajectory_2026-08-02/
```

The machine-readable report has SHA256
`2576f53574667ca0e0a8172a3501e9e5baf8f0255a6b6cd5d961797e2a267476`.
Its claim boundary explicitly keeps sample-quality, formal-gate, promotion, and
full-300K authorization false.
