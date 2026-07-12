# Tiny External P0 Baselines

Date: 2026-07-09

Scope: fill the `tiny_imagenet_200` external P0 baseline rows for the current
formal64 paper table.

Runbook:

```text
artifacts/runbooks/p0_tiny_external_baseline_queue_2026-07-09.sh
```

New/updated outputs:

```text
artifacts/reports/baselines/summary_2026-07-09_plus_edm_tiny/
artifacts/reports/formal64_p0_horizontal_table_2026-07-09_tiny_external/
artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_tiny_external/
```

Tiny ImageNet external baseline metrics:

| Method | Steps | NFE | Lowres Frechet | Inception Frechet |
|---|---:|---:|---:|---:|
| Improved DDPM | 5000 | 50 | 9.0747 | 381.2383 |
| EDM | 5000 | 79 | 3.0996 | 268.7190 |

EDM-specific data adapter:

```text
docs/experiment_conditions/tiny_imagenet_200_edm_rgb_train_2026-07-09.md
docs/experiment_conditions/tiny_imagenet_200_edm_rgb_train_manifest_2026-07-09.json
```

Reason: Tiny ImageNet train has 1,821 grayscale JPEG files. EDM's
`ImageFolderDataset` asserts uniform RGB shape, so an EDM-only derived train
directory was created with RGB hardlinks plus grayscale-to-RGB PNG conversions.
No symlinks were used.

Matrix status counts after this step:

```text
completed=45
missing=26
needs_adapter=32
partial=1
protocol_blocked=8
```

Remaining relevant gaps:

- `Dense epsilon` on `tiny_imagenet_200` was later completed by
  `docs/records/2026-07-09_tiny_dense_generated_quality.md`.
- `imagenet_1k_64x64_hf` still lacks external EDM/improved-DDPM rows, but this
  dataset is not the strict exact-source ImageNet-64 row.
- `cifar10`, ImageNet-256, and P1 tokenizer/AR baselines should remain separate
  protocol decisions, not silently mixed into the formal64 MVP table.
