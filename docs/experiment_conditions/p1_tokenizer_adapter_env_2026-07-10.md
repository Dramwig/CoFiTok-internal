# P1 Tokenizer Adapter Environment - 2026-07-10

## Scope

This record covers the eval-only P1 tokenizer reconstruction adapters for:

- `ml_flextok`
- `titok_1d_tokenizer`

This is not a retraining protocol. It evaluates official pretrained tokenizer reconstruction after resizing inputs to 256x256.

## Remote Environment

- Host: `pro6000`
- Project root: `/root/autodl-tmp/CoFiTok`
- Code root: `/root/autodl-tmp/CoFiTok/CoFiTok-internal`
- Conda env: `pf-vlm`
- PyTorch in smoke reports: `2.7.1+cu128`
- GPU smoke device: `cuda`
- Hugging Face cache: `/root/autodl-tmp/CoFiTok/checkpoints/hf_cache`
- Network setup:

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

## Dependency Changes

Installed into `pf-vlm` during P1 adapter setup:

```bash
pip install hydra-core omegaconf einops diffusers mup
```

Observed import status after install:

- `ml_flextok`: import passed.
- `titok_1d_tokenizer`: import passed.
- `mar`: import passed.
- `retok`: import passed.

## Official Models

| baseline | model id | role |
|---|---|---|
| `ml_flextok` | `EPFL-VILAB/flextok_d12_d12_in1k` | official pretrained FlexTok tokenizer/reconstruction |
| `titok_1d_tokenizer` | `yucornetto/tokenizer_titok_l32_imagenet` | official pretrained TiTok tokenizer/reconstruction |

## Smoke Results

| baseline | dataset | images | source res | eval res | device | metric summary |
|---|---|---:|---:|---:|---|---|
| `titok_1d_tokenizer` | `cifar10` | 4 | 32 | 256 | `cuda` | PSNR `19.2483`, MSE `0.0475589`, lowres Frechet `1.7600` |
| `ml_flextok` | `cifar10` | 2 | 32 | 256 | `cuda` | PSNR `26.7195`, MSE `0.0085136`, lowres Frechet `0.2930` |

Report paths:

```text
artifacts/reports/baselines/titok_1d_tokenizer/smoke_cifar10_recon256_4img_gpu_2026-07-10/
artifacts/reports/baselines/ml_flextok/smoke_cifar10_recon256_2img_gpu_2026-07-10/
```

## Protocol Boundary

These runs should be reported as `completed_eval_only`, not as full same-dataset training baselines.

Reasons:

- The weights are official pretrained ImageNet-family tokenizers.
- Inputs from 32x32/64x64 datasets are resized to 256x256 before tokenization.
- The task is reconstruction, not unconditional generation or dense diffusion noise prediction.
- Metrics are useful for tokenizer-nearest related-method discussion, but not interchangeable with P0 pixel-diffusion generation rows.

## All-Dataset Eval-Only Pass

After smoke validation, a 64-image eval-only pass was run for all current matrix datasets:

```bash
MAX_IMAGES=64 TAG=eval64_2026-07-10 bash artifacts/runbooks/p1_tokenizer_reconstruction_eval_2026-07-10.sh
```

Generated table:

```text
artifacts/reports/baselines/p1_tokenizer_reconstruction_eval64_2026-07-10/p1_tokenizer_reconstruction_table.md
```

Latest guard matrix:

```text
artifacts/reports/paper_comparison_matrix_2026-07-10_p1_eval64_guard_latest/paper_comparison_matrix_audit.md
```

Status counts:

```text
completed=72
completed_eval_only=16
protocol_blocked=24
```
