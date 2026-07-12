# ImageNet-256 10pct P0 Completion

Date: 2026-07-10

Scope: close `imagenet_256_10pct` P0 cells under the 256-light protocol. This is
not the full `imagenet_256` result.

Runbook:

```text
artifacts/runbooks/p0_imagenet256_10pct_queue_2026-07-10.sh
```

Config manifest:

```text
docs/experiment_conditions/imagenet_256_p0_config_manifest_2026-07-10.json
```

Primary outputs:

```text
artifacts/reports/summary_2026-07-10_imagenet256_10pct_p0/
artifacts/reports/baselines/summary_2026-07-10_imagenet256_10pct_p0/
artifacts/reports/paper_comparison_matrix_2026-07-10_imagenet256_10pct_p0/
artifacts/reports/baselines/improved_diffusion/improved_diffusion_imagenet_256_10pct_5000steps_ddim50_2026-07-10_imagenet256_10pct_p0/
artifacts/reports/baselines/edm/edm_imagenet_256_10pct_5000steps_edm40_2026-07-10_imagenet256_10pct_p0/
```

Protocol summary:

```text
dataset: imagenet_256_10pct
image_size: 256
internal train steps: 5000
internal batch size: 4
internal K: 4
internal token channels: 16
internal base channels: 24
quality eval: 256 val images, fixed timestep 500, prefix budgets [1, 2, 4]
generated eval: 512 samples, 2048 real images
external baselines: Improved-DDPM DDIM50, EDM 40 steps / NFE 79
```

Completed `imagenet_256_10pct` P0 rows:

```text
cofitok_light: train, quality, generated_quality
same_backbone_dense: train, quality, generated_quality
channel_mask: train, quality
cofitok_no_prefix_loss: train, quality
cofitok_no_monotonic_loss: train, quality
cofitok_simultaneous: train, quality
cofitok_deep_synthesis: train, quality
improved_diffusion: baseline_train, baseline_eval
edm: baseline_train, baseline_eval
```

Generated-quality smoke metrics:

| Method | Samples | Real | NFE / steps | Lowres Frechet | Inception Frechet |
|---|---:|---:|---:|---:|---:|
| CoFiTok K4 light | 512 | 2048 | DDIM50 | 10.4423 | 414.1863 |
| Same-backbone dense | 512 | 2048 | DDIM50 | 10.5062 | 337.8743 |
| Improved-DDPM | 512 | 2048 | 50 | 10.3989 | 419.5981 |
| EDM | 512 | 2048 | 79 | 10.3918 | 368.6788 |

Fixed-noise denoising quality at budget 4:

| Variant | PSNR | Lowres Frechet | Inception Frechet | LPIPS Alex | Component mean abs cosine |
|---|---:|---:|---:|---:|---:|
| channel_mask | 13.8070 | 2.2793 | 338.2386 | 0.8782 | 0.0632 |
| clean_monotonic_ablation | 13.8528 | 0.9032 | 308.7236 | 0.8142 | 0.8198 |
| deep_synthesis_ablation | 14.4828 | 0.5812 | 325.1934 | 0.8572 | 0.9519 |
| epsilon_only | 13.8669 | 0.9934 | 330.7862 | 0.8920 | 0.1133 |
| light_denoise_path | 13.6268 | 1.0566 | 330.2744 | 0.8962 | 0.8101 |
| no_prefix_loss_ablation | 13.8654 | 1.4068 | 323.1635 | 0.8814 | 0.7236 |
| simultaneous_predictor | 13.8503 | 0.8423 | 312.5193 | 0.8626 | 0.7986 |

Overall matrix status after this step:

```text
completed=63
missing=9
needs_adapter=32
protocol_blocked=8
```

Remaining P0 missing cells are only full `imagenet_256`: 7 internal rows plus
EDM and Improved-DDPM. D-AR remains `protocol_blocked` and must not be counted
as a completed baseline. P1 FlexTok/MAR/TiTok/ReTok still need adapters.

Interpretation:

- This 10pct scaling run supports feasibility of the 256-light protocol and
  closes the `imagenet_256_10pct` P0 matrix cells.
- It does not support a broad unconditional generation-quality win. EDM remains
  strongest or tied on generated Frechet-style metrics here; same-backbone dense
  is stronger on generated Inception Frechet.
- The defensible claim remains ordered restricted denoising-token
  factorization and prefix-controllable denoising, not SOTA image generation.
