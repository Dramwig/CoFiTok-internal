# Full-data matched 34K training trajectory

Date: 2026-08-19

## Outcome

The full ImageNet-256 quality bridge now has an immutable matched trajectory
through step 34,000, or 2,176,000 training images per method. The result remains
an optimization tie and does not establish a CoFiTok generation advantage.

The report and exact metrics-prefix snapshots are:

```text
artifacts/reports/generation/matched_training_trajectory_step_00034000_2026-08-19/
```

Identities:

| artifact | bytes | SHA256 |
|---|---:|---|
| `trajectory_report.json` | 40,152 | `2dc484a523602e182568345b63d50771d88b7df2b44055d12a7a710262d3d513` |
| CoFiTok metrics through 34K | 640,712 | `0df343335476205f23e23330da94aebdb19a81d8b0325c8f887305ee085595ac` |
| dense metrics through 34K | 595,327 | `7aceea7592493642520b9da76a76ec8ec3f1ebdeb9f947913955232ea22d8361` |

The source-bound builder is revision
`40ee55202545ea01e48fb6642d1b5a51b5264663` on
`analysis/generation-resume-aware-matched-trajectory-v1`, with builder SHA256
`2477345f81a99895f73b5c9a93844c8005ca959a2ed74ab3fa5a077c4753c21a`.
Rebuilding from the frozen snapshots produced the same report SHA256 byte for
byte.

## Results

Across all 34 paired fixed-validation events:

| measure | CoFiTok | dense identity | relative CoFiTok delta |
|---|---:|---:|---:|
| mean epsilon MSE | 0.0304513134 | 0.0304278563 | +0.07709% |
| lower-MSE events | 17 | 17 | tie |
| step-34K epsilon MSE | 0.0286979601 | 0.0287152678 | -0.06027% |

During the predeclared EMA-teacher warmup regime at steps 30K through 34K:

| measure | CoFiTok | dense identity | relative CoFiTok delta |
|---|---:|---:|---:|
| mean epsilon MSE | 0.0257247746 | 0.0256777916 | +0.18297% |
| lower-MSE events | 1/5 | 4/5 | dense favored descriptively |

During the full-scale rollout-consistency regime at steps 10K through 34K,
CoFiTok mean epsilon MSE was `0.0295543431` versus `0.0295331015` for dense,
a relative delta of `+0.07192%`; event wins were 13 versus 12.

These differences are small, change sign at the latest event, and provide no
stable optimization advantage signal. They also provide no evidence of a
CoFiTok-specific optimization collapse.

## Boundary

This trajectory contains no generated-sample metrics and is not a formal 50K
gate substitute. It does not support a generation-quality claim, promotion,
full training, or release. Required next evidence remains exact healthy matched
50K completion followed by the paired EMA sampling evaluation. The independent
40K schedule waiter must first verify that the EMA-teacher warmup reached full
scale with finite, source-bound rows.
