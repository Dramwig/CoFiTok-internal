# D-AR Official ImageNet-256 50K Completion - 2026-07-10

## Result

Status: `completed_eval_only`

D-AR official pretrained ImageNet-256 eval-only finished on `pro6000`.
This is a secondary related-method result only. It must not be merged into the
P0 same-dataset, same-budget all-dataset generation table.

## Metrics

| method | dataset | protocol | samples | FID | sFID | IS | precision | recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-AR | imagenet_256 | official pretrained eval-only | 50000 | 2.6281 | 6.7195 | 285.8914 | 0.8029 | 0.5884 |

## Evidence

Standard report:

```text
CoFiTok-internal/artifacts/reports/baselines/d_ar/official_imagenet256_50k_2026-07-10/baseline_eval_report.json
```

Secondary related-method table:

```text
CoFiTok-internal/artifacts/reports/baselines/official_related_methods_2026-07-10/official_related_methods_table.md
```

Remote sample NPZ:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.npz
```

Run status:

```text
status=finished rc=0 finished_at=2026-07-10T08:30:54+08:00
sample_shape=(50000,256,256,3)
png_count=50048
```

## Interpretation

This strengthens related-method coverage against the D-AR nearest-neighbor
threat, but it does not change the P0 fairness conclusion:

- P0 completed table remains the same-dataset, same-budget CoFiTok / dense /
  EDM / Improved-DDPM comparison.
- D-AR remains protocol-blocked for all-dataset retraining because the official
  public assets target ImageNet-256 class-conditional pretrained evaluation.
- The D-AR result can be cited only in a separate official-checkpoint
  ImageNet-256 related-method table.
