# Multiscale Backbone 20k Validation

Date: 2026-07-08

Purpose: close the strict-audit gap that the stronger `multiscale_unet`
predictor only had 10k pilot-scale evidence while the main Tiny/HF baselines
use 20k training.

## Protocol

Entrypoint:

```bash
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/train_short.py
```

Configs:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json
```

Common settings:

- predictor: `multiscale_unet`
- token count: `K=8`
- token channels: `16`
- `S_k`: restricted, token-only, bias-free local synthesis
- loss: light denoise-path objective
- seed: `151`
- steps: `20000`
- batch size: `24`
- datasets: Tiny ImageNet-200 and ImageNet-1K 64x64 HF fallback

## Results

| dataset | steps | final clean MSE | path AUC | effective tokens | zero-token ratio | shuffled final ratio | elapsed seconds | report dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny ImageNet-200 | 10000 | 0.125013 | 0.043255 | 6.809599 | 0.0000 | 188.2605 | 182.2 | `train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda_2026-07-08` |
| Tiny ImageNet-200 | 20000 | 0.123434 | 0.034593 | 6.760604 | 0.0000 | 187.6608 | 343.1 | `train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_2026-07-08` |
| ImageNet-1K 64x64 HF | 10000 | 0.077563 | 0.036447 | 6.749266 | 0.0000 | 300.0311 | 181.1 | `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda_2026-07-08` |
| ImageNet-1K 64x64 HF | 20000 | 0.065420 | 0.025294 | 6.720911 | 0.0000 | 356.9997 | 334.1 | `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_2026-07-08` |

## Interpretation

The stronger predictor now has matched 20k train-scale validation on both main
datasets while keeping the restricted `S_k` contract intact. Both datasets
improve from the 10k pilot to the 20k run in endpoint clean MSE and path AUC.
The zero-token ratio remains exactly 0.0, so the stronger predictor did not
weaken the non-degenerate synthesis claim.

This closes the strict `longer_stronger_backbone_validation` audit gap as a
train-scale backbone check. It does not turn the multiscale path into a new
headline generation-quality claim; 20k quality/sampling rows can still be added
later if the final paper chooses to center backbone scaling.
