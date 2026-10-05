# Stability matched 40K EMA-teacher full-scale trajectory

Date: 2026-08-03

## Purpose

The active dense-identity recovery crossed the exact 40,000-step checkpoint and
the predeclared point at which EMA-teacher consistency reaches full scale. This
record freezes the matched CoFiTok/dense training trajectory through that
boundary, including all ten warmup validation events from 30K--39K and the
first full-scale event at 40K.

This remains a read-only training-space diagnostic. No trainer, controller,
monitor, watchdog, waiter, checkpoint, queue, lock, or GPU process was changed.
The existing frozen post-training chain remains the only source of formal
sample-quality evidence.

## Resume-aware source contract

The earlier trajectory builder required every row after step 1 to lie exactly
on the 50-step logging grid. At 40K that correctly surfaced one additional
canonical CoFiTok row at step 36,546: training had exactly resumed from
`checkpoint_step_00036545.pt`, and the trainer intentionally logs the first
post-resume optimizer step.

Schema 3 of `scripts/build_generation_matched_training_trajectory.py` now
accepts such a row only when all of the following are source-bound in the run
manifest:

- the resume path has the exact `checkpoint_step_XXXXXXXX.pt` form;
- `metrics_resume_reconciliation.schema_version == 1`;
- reconciliation status is `unchanged` or `reconciled`;
- the checkpoint step equals the reconciliation `resume_step`;
- unchanged/reconciled orphan counts and content-addressed orphan evidence are
  internally consistent;
- the irregular row is exactly `resume_step + 1`.

Arbitrary off-grid rows continue to fail closed. In the frozen CoFiTok source,
the resume step is 36,545, reconciliation is `unchanged`, 731 rows were
retained, zero rows were orphaned, and the only admitted boundary row is
36,546. Dense did not resume and admits no off-grid row.

## Bound sources and checkpoint integrity

Both methods remain bound to clean immutable training Git
`2c2c1f5166b73d4f28df93b276901671ac1a7836` on
`scale/generation-stability-50k-preflight`, the same formal
`imagenet_256_10pct` dataset identity, runtime-environment identity, effective
batch 64, shared loss schedules, and valid pair contract.

The frozen prefixes contain 802 CoFiTok rows and 801 dense rows, 2,560,000
images per method, and 40 matched fixed-validation events. All numeric values
are finite; steps are strictly increasing; and every row satisfies
`samples_seen = step * 64`. Validation step, event index, batch index, image
count, and fixed noise seed match exactly.

The physical dense checkpoint was independently rehashed:

- checkpoint: `checkpoint_step_00040000.pt`;
- bytes: `1,006,120,214`;
- SHA256: `cb040fb971550bcab42bda32e44d9d962b8044af33e60caac414d56d6062a68f`;
- sidecar and `latest.json` bind the same step, bytes, SHA, Git, dataset, and
  runtime identities.

## Result

Across 1K--40K, mean fixed-validation epsilon MSE is `0.0299178669` for
CoFiTok and `0.0299400262` for dense. CoFiTok is lower by `0.074012%` in the
ratio of means; lower-event counts are `15 / 25`. At the exact 40K endpoint,
CoFiTok is lower by `0.005674%`.

Predeclared schedule regimes are:

- rollout warmup, 1K--9K: CoFiTok mean is lower by `0.067741%`, lower-event
  counts `4 / 5`;
- rollout full scale, 10K--40K: CoFiTok mean is lower by `0.076061%`,
  lower-event counts `11 / 20`;
- EMA teacher inactive, 1K--29K: CoFiTok mean is lower by `0.156157%`,
  lower-event counts `12 / 17`;
- EMA teacher warmup, 30K--39K: CoFiTok mean is higher by `0.202082%`,
  lower-event counts `2 / 8`;
- EMA teacher full scale at 40K: CoFiTok is lower by `0.005674%` in the first
  and only event available in this phase.

The exact transition is finite and matched, but one full-scale event cannot
establish a teacher benefit. This report contains no generated samples and does
not support FID/IS/precision/recall, rollout-quality, promotion, or full-300K
authorization. All authorization fields remain false; exact matched 50K plus
formal frozen EMA post-evaluation remains required.

## Evidence

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_40k_schedule_trajectory_2026-08-03/
```

Key SHA256 values:

- `trajectory_report.json`:
  `3339269636570be75bf53ba811eda018cad8ab01405aaf196838d8c691ee217d`;
- schema-3 builder:
  `8e2bcfb8b82b6c551472ed826df43c9eb1b4ab61f71873a289d00c2851df7131`;
- CoFiTok 40K metrics prefix:
  `7a2ba2b87d9aacc8afb4bd3d90c0375d245a4fcc353333a6f958eef0fa91f03e`;
- dense 40K metrics prefix:
  `6d5279bef872bc3389bcc9ac80b22ad4da32be03a5b378454254bd6b2f7bb7de`;
- CoFiTok/dense run manifests:
  `3b554762a5ee75d77fbf66af616c6eaab31ae24dd7a6f40f1f9d4f747bc06519` /
  `a049a97d8a4e9f4dc4761a9269e50ae55f53ec28b9971d14a7d968bfd7e9f9fe`;
- dense 40K checkpoint sidecar:
  `6c1257964de2548deb8d9d378ee2577d6c7985a2c859cc8c3a54a5a3848fa301`.

## Verification

- trajectory-builder tests: `17 passed`;
- builder, metrics-resume, and generation-pair-contract focused suites:
  `37 passed`;
- rebuilding from the four permanent builder inputs produced byte-identical
  report SHA256
  `3339269636570be75bf53ba811eda018cad8ab01405aaf196838d8c691ee217d`;
- the three 35K live-state attachments that were actually captured after 40K
  were removed rather than being represented as 35K boundary evidence;
- following `ml-training-recipes`, physical checkpoint identity and EMA
  inference quality remain separate: the trajectory proves continuity, while
  frozen EMA sampling must prove generation quality.

An attempted full local suite produced no failure output but was terminated by
the outer command timeout before pytest emitted a terminal summary; it is not
reported as a pass. No local pytest/Python child remained afterward.
