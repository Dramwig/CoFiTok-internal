# ImageNet HF 64 P0 5k Completion

Date: 2026-07-10

Scope: close the remaining 5k-aligned `imagenet_1k_64x64_hf` P0 cells and
rebuild the current P0 64/CIFAR horizontal table.

Runbooks:

```text
artifacts/runbooks/p0_imagenet_hf_external_baselines_2026-07-10.sh
artifacts/runbooks/p0_imagenet_hf_5k_internal_gaps_2026-07-10.sh
```

New configs:

```text
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_5k_cuda.json
configs/baselines/improved_diffusion/imagenet_1k_64x64_hf.json
configs/baselines/edm/imagenet_1k_64x64_hf.json
```

Primary outputs:

```text
artifacts/reports/summary_2026-07-10_imagenet_hf_5k_internal_gaps/
artifacts/reports/baselines/summary_2026-07-10_imagenet_hf_5k_internal_gaps/
artifacts/reports/paper_comparison_matrix_2026-07-10_imagenet_hf_5k_internal_gaps/
artifacts/reports/p0_64_horizontal_table_2026-07-10_imagenet_hf_5k_complete/
```

`imagenet_1k_64x64_hf` key 5k metrics:

| Method | Status | NFE | Lowres Frechet | Inception Frechet | Denoise PSNR | Path AUC |
|---|---|---:|---:|---:|---:|---:|
| CoFiTok | completed | 50 | 8.5337 | 374.3523 | 15.062 | 0.0641 |
| Dense epsilon | completed | 50 | 7.6747 | 388.1463 | 15.131 | 0.8837 |
| Improved DDPM | completed | 50 | 8.8830 | 378.3325 |  |  |
| EDM | completed | 79 | 3.2285 | 258.4005 |  |  |
| D-AR | protocol_blocked |  |  |  |  |  |

P0 64/CIFAR table status:

```text
rows=60
completed=54
protocol_blocked=6
partial=0
missing=0
```

Overall matrix status after this step:

```text
completed=54
missing=18
needs_adapter=32
protocol_blocked=8
```

Remaining P0 missing cells are `imagenet_256` and `imagenet_256_10pct` only.
D-AR remains `protocol_blocked` by adapter/protocol feasibility and must not be
counted as a completed baseline. P1 repositories are cloned and pinned, but
FlexTok/MAR/TiTok/ReTok still need adapters before they enter result tables.
