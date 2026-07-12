# CoFiTok Ablation Seed Repeats

Date: 2026-07-08

Purpose: strengthen the MVP validation beyond single-seed ablations by adding
ImageNet-64 HF seed repeats for the two most important non-default ablations:
deep `S_k` and simultaneous token prediction.

## New Runs

| run type | config | seed | output |
|---|---|---:|---|
| training | `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_seed2_cuda.json` | 103 | `/root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_seed2_2026-07-08` |
| training | `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_seed2_cuda.json` | 103 | `/root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_seed2_2026-07-08` |
| order eval | deep `S_k`, ordered/random/reverse | 103 | `eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_seed2_*_2026-07-08` |
| order eval | simultaneous, ordered/random/reverse | 103 | `eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_seed2_*_2026-07-08` |

## Training Summary

| variant | seed | final MSE | path AUC | effective K | zero ratio |
|---|---:|---:|---:|---:|---:|
| deep `S_k` | 139 | 0.0977 | 0.0366 | 6.792 | 0.0522 |
| deep `S_k` | 103 | 0.1327 | 0.0433 | 6.748 | 0.0450 |
| simultaneous predictor | 139 | 0.0995 | 0.0600 | 6.845 | 0.0000 |
| simultaneous predictor | 103 | 0.1024 | 0.0723 | 6.801 | 0.0000 |

## Order Evaluation Summary

| variant | seed | order | final MSE | path AUC | zero ratio |
|---|---:|---|---:|---:|---:|
| deep `S_k` | 139 | ordered | 0.1022 | 0.0382 | 0.0521 |
| deep `S_k` | 139 | random | 0.1022 | 0.1361 | 0.0521 |
| deep `S_k` | 139 | reverse | 0.1022 | 0.6431 | 0.0521 |
| deep `S_k` | 103 | ordered | 0.1371 | 0.0441 | 0.0450 |
| deep `S_k` | 103 | random | 0.1371 | 0.1599 | 0.0450 |
| deep `S_k` | 103 | reverse | 0.1371 | 0.6698 | 0.0450 |
| simultaneous predictor | 139 | ordered | 0.1025 | 0.0611 | 0.0000 |
| simultaneous predictor | 139 | random | 0.1025 | 0.2242 | 0.0000 |
| simultaneous predictor | 139 | reverse | 0.1025 | 0.6505 | 0.0000 |
| simultaneous predictor | 103 | ordered | 0.1053 | 0.0723 | 0.0000 |
| simultaneous predictor | 103 | random | 0.1053 | 0.3341 | 0.0000 |
| simultaneous predictor | 103 | reverse | 0.1053 | 0.7523 | 0.0000 |

## Interpretation

- The deep `S_k` ablation fails the zero-token diagnostic in both seeds. This
  reinforces the claim that a stronger synthesis operator can carry
  token-independent priors and should remain an ablation, not the default
  CoFiTok operator.
- The simultaneous predictor remains order-sensitive in both seeds: random and
  reverse component accumulation increase path AUC while endpoint MSE is
  unchanged by construction.
- These repeats reduce the previous single-seed weakness for two major
  ablations, but they do not replace full multi-seed error bars across every
  dataset and metric.
