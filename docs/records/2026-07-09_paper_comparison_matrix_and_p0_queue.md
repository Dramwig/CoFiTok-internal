# Paper Comparison Matrix and P0 Formal-64 Queue

Date: 2026-07-09

Goal: move from prepared datasets and cloned baseline repos toward the full
method x dataset training/testing table needed for paper-level comparison.

## Current Audit

Current evidence was audited with:

```bash
python scripts/build_paper_comparison_matrix.py \
  --output-dir artifacts/reports/paper_comparison_matrix_2026-07-09
```

Result before the new queue:

| Status | Count |
|---|---:|
| completed | 14 |
| partial | 1 |
| missing | 41 |
| needs_adapter | 56 |

Interpretation:

- Existing completed cells are concentrated in CIFAR-10, Tiny ImageNet-200, and
  `imagenet_1k_64x64_hf`.
- `downsampled_imagenet_64`, `ffhq_64`, `afhqv2_64`, `imagenet_256_10pct`, and
  `imagenet_256` had no paper-matrix train/quality/generated rows yet.
- External baselines are cloned and pinned, but still need adapters and fair
  train/eval records.

## Data Loader Fix

Updated `src/cofitok/data/registry.py` so prepared-image datasets can be loaded
from metadata manifests:

- `ffhq_64`: uses `metadata/image_manifest.jsonl` to avoid treating shard
  directories as classes.
- `afhqv2_64`: uses `label_name` when present.
- `imagenet_256` / `imagenet_256_10pct`: supports manifest split filtering and
  `train` / `val` class-folder layouts.

Remote smoke-forward reports were written:

```text
/root/autodl-tmp/CoFiTok/checkpoints/smoke_ffhq64_2026-07-09/report.json
/root/autodl-tmp/CoFiTok/checkpoints/smoke_afhqv2_64_2026-07-09/report.json
/root/autodl-tmp/CoFiTok/checkpoints/smoke_imagenet_256_10pct_2026-07-09/report.json
/root/autodl-tmp/CoFiTok/checkpoints/smoke_imagenet_256_2026-07-09/report.json
```

Remote dataset probe:

| Dataset | Split used | Count | Tensor shape | Classes |
|---|---|---:|---|---|
| `ffhq_64` | val fallback to unsplit images | 70,000 | `(3, 64, 64)` | `unlabeled` |
| `afhqv2_64` | val fallback to train | 15,803 | `(3, 64, 64)` | `cat`, `dog`, `wild` |
| `imagenet_256_10pct` | val | 50,000 | `(3, 256, 256)` | ImageNet wnids |
| `imagenet_256` | val | 50,000 | `(3, 256, 256)` | ImageNet wnids |

## Queue Launched

Runbook:

```text
artifacts/runbooks/p0_formal64_queue_2026-07-09.sh
```

Remote launch:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
nohup bash artifacts/runbooks/p0_formal64_queue_2026-07-09.sh \
  > artifacts/logs/p0_formal64_queue_2026-07-09.log 2>&1 < /dev/null &
```

PID:

```text
619525
```

Log:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/logs/p0_formal64_queue_2026-07-09.log
```

Queue contents:

| Dataset | Methods | Evidence |
|---|---|---|
| `downsampled_imagenet_64` | CoFiTok full, dense epsilon, channel-mask | train + quality; generated-quality for CoFiTok/dense; order eval for CoFiTok |
| `ffhq_64` | CoFiTok full, dense epsilon, channel-mask | train + quality; generated-quality for CoFiTok/dense; order eval for CoFiTok |
| `afhqv2_64` | CoFiTok full, dense epsilon, channel-mask | train + quality; generated-quality for CoFiTok/dense; order eval for CoFiTok |

Current observed start state:

- First job: `train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda`.
- GPU became active after startup, confirming the queue entered training.

## Next Required Work

1. Monitor queue completion and regenerate the paper comparison matrix from
   `artifacts/reports/summary_2026-07-09_p0_formal64`.
2. If P0 formal-64 rows look stable, add 20k confirmation configs for the best
   CoFiTok/dense settings on strict `downsampled_imagenet_64`, `ffhq_64`, and
   `afhqv2_64`.
3. Start external baseline adapters in order: same evaluation protocol first,
   then `improved_diffusion` / EDM, then D-AR.
