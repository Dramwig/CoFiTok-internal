# CIFAR10 P0 Completion

Date: 2026-07-10

Scope: close CIFAR10 P0 matrix gaps under the existing 32x32, 3k-step smoke
protocol.

Runbook:

```text
artifacts/runbooks/p0_cifar10_completion_2026-07-10.sh
```

Primary outputs:

```text
artifacts/reports/summary_2026-07-10_cifar_p0/
artifacts/reports/baselines/summary_2026-07-10_cifar_p0/
artifacts/reports/paper_comparison_matrix_2026-07-10_cifar_p0/
```

Data adapter:

```text
docs/experiment_conditions/cifar10_imagefolder_2026-07-10.md
docs/experiment_conditions/cifar10_imagefolder_manifest_2026-07-10.json
```

The image-folder derivative contains 50,000 train and 10,000 test RGB PNG
images converted from the canonical CIFAR10 Python batches. It is only for
external baseline adapters.

Completed rows:

- Internal ablations: no-prefix-loss, clean-monotonic, simultaneous-components,
  and deep-`S_k` all have train + quality reports.
- Same-backbone dense epsilon now has generated-quality evidence.
- Improved DDPM and EDM now have train + generated-quality evidence.

Key CIFAR10 generated metrics:

| Method | Steps | NFE | Lowres Frechet | Inception Frechet |
|---|---:|---:|---:|---:|
| CoFiTok | 3000 | 50 | 7.2663 | 313.8881 |
| Dense epsilon | 3000 | 50 | 6.4290 | 320.1376 |
| Improved DDPM | 3000 | 50 | 7.8610 | 498.6459 |
| EDM | 3000 | 79 | 2.8681 | 351.1658 |

Matrix status counts after this step:

```text
completed=52
missing=20
needs_adapter=32
protocol_blocked=8
```

Interpretation: CIFAR10 is a smoke-scale protocol. It now verifies that the P0
pipeline and adapters close end-to-end on a small dataset, but it should not
override the formal64 conclusion that EDM is strongest on the main generated
Frechet-style metrics.
