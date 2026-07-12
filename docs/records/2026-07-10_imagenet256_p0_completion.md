# ImageNet-256 Full P0 Completion

Date: 2026-07-10

Scope: close the remaining full `imagenet_256` P0 cells and rebuild the current
P0 method x dataset table.

Runbook:

```text
artifacts/runbooks/p0_imagenet256_queue_2026-07-10.sh
```

Primary outputs:

```text
artifacts/reports/summary_2026-07-10_imagenet256_p0/
artifacts/reports/baselines/summary_2026-07-10_imagenet256_p0/
artifacts/reports/paper_comparison_matrix_2026-07-10_imagenet256_p0/
artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/
artifacts/reports/baselines/improved_diffusion/improved_diffusion_imagenet_256_5000steps_ddim50_2026-07-10_imagenet256_p0/
artifacts/reports/baselines/edm/edm_imagenet_256_5000steps_edm40_2026-07-10_imagenet256_p0/
```

Protocol summary:

```text
dataset: imagenet_256
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

Completed full `imagenet_256` P0 rows:

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
| CoFiTok K4 light | 512 | 2048 | DDIM50 | 10.5431 | 417.2551 |
| Same-backbone dense | 512 | 2048 | DDIM50 | 10.2803 | 354.4296 |
| Improved-DDPM | 512 | 2048 | 50 | 10.3998 | 414.2991 |
| EDM | 512 | 2048 | 79 | 10.3975 | 368.4282 |

Fixed-noise denoising quality at budget 4:

| Variant | PSNR | Lowres Frechet | Inception Frechet | LPIPS Alex | Path AUC | Component mean abs cosine |
|---|---:|---:|---:|---:|---:|---:|
| channel_mask | 13.7612 | 2.4336 | 340.7570 | 0.8794 | 3.5299 | 0.0685 |
| clean_monotonic_ablation | 13.6543 | 1.2163 | 312.4116 | 0.8209 | 0.1210 | 0.8143 |
| deep_synthesis_ablation | 14.6533 | 0.5301 | 322.0835 | 0.8278 | 0.0521 | 0.9553 |
| epsilon_only | 13.8765 | 1.0281 | 323.9321 | 0.8360 | 1.4917 | 0.1082 |
| light_denoise_path | 13.8089 | 1.0088 | 329.2057 | 0.8438 | 0.1255 | 0.8134 |
| no_prefix_loss_ablation | 13.3884 | 1.6402 | 324.3095 | 0.8812 | 0.1731 | 0.7219 |
| simultaneous_predictor | 13.9744 | 0.9143 | 313.5437 | 0.8475 | 0.1292 | 0.8035 |

Current P0 table status:

```text
rows=80
completed=72
protocol_blocked=8
missing=0
```

Overall matrix status after this step:

```text
completed=72
needs_adapter=32
protocol_blocked=8
missing=0
```

D-AR remains `protocol_blocked` for all eight datasets and must not be counted
as a completed baseline. P1 FlexTok/MAR/TiTok/ReTok remain `needs_adapter`.

Interpretation:

- The configured P0 matrix now has no missing rows.
- Full ImageNet-256 does not support a broad unconditional generation-quality
  win: same-backbone dense is stronger on generated Inception Frechet, and EDM
  remains a strong monolithic pixel baseline.
- The useful CoFiTok evidence remains prefix-controllable denoising and ordered
  restricted factorization. On full ImageNet-256, CoFiTok light has much lower
  path AUC than dense epsilon (`0.1255` vs `1.4917`), but generated quality must
  be described conservatively.
