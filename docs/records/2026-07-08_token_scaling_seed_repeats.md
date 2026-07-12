# CoFiTok Token-Scaling Seed Repeats

Date: 2026-07-08

Purpose: strengthen the K-scaling evidence for the ImageNet-64 HF fallback by
repeating the K4 and K16 light denoise-path runs with seed 103. This closes the
previous audit gap where K4/K16 order sensitivity was only shown for seed 139.

## New Runs

| run type | config | seed | output |
|---|---|---:|---|
| training | `configs/train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_seed2_cuda.json` | 103 | `/root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_seed2_2026-07-08` |
| training | `configs/train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_seed2_cuda.json` | 103 | `/root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_seed2_2026-07-08` |
| order eval | K4 ordered/random/reverse | 103 | `eval_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_seed2_*_2026-07-08` |
| order eval | K16 ordered/random/reverse | 103 | `eval_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_seed2_*_2026-07-08` |

## Training Summary

| K | seed | final MSE | path AUC | effective K | zero ratio | shuffle ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 139 | 0.0988 | 0.0665 | 3.318 | 0.0000 | 242.6 |
| 4 | 103 | 0.0954 | 0.0571 | 3.323 | 0.0000 | 245.0 |
| 16 | 139 | 0.0992 | 0.0674 | 13.560 | 0.0000 | 241.4 |
| 16 | 103 | 0.1014 | 0.0691 | 13.780 | 0.0000 | 234.0 |

## Order Evaluation Summary

| K | seed | order | final MSE | path AUC | zero ratio |
|---:|---:|---|---:|---:|---:|
| 4 | 139 | ordered | 0.1029 | 0.0674 | 0.0000 |
| 4 | 139 | random | 0.1029 | 0.0727 | 0.0000 |
| 4 | 139 | reverse | 0.1029 | 0.7516 | 0.0000 |
| 4 | 103 | ordered | 0.0980 | 0.0575 | 0.0000 |
| 4 | 103 | random | 0.0980 | 0.0652 | 0.0000 |
| 4 | 103 | reverse | 0.0980 | 0.6975 | 0.0000 |
| 16 | 139 | ordered | 0.1021 | 0.0680 | 0.0000 |
| 16 | 139 | random | 0.1021 | 0.4680 | 0.0000 |
| 16 | 139 | reverse | 0.1021 | 0.7020 | 0.0000 |
| 16 | 103 | ordered | 0.1056 | 0.0701 | 0.0000 |
| 16 | 103 | random | 0.1056 | 0.5294 | 0.0000 |
| 16 | 103 | reverse | 0.1056 | 0.7032 | 0.0000 |

## Interpretation

- K4 remains a compact cost baseline: it has clean zero-token diagnostics and
  reverse-order damage in both seeds, but random order is only mildly worse
  because four components leave limited room for a rich prefix curriculum.
- K16 uses many components effectively in both seeds, but at 5k steps it does
  not improve ordered path AUC over K8; random/reverse order damages path AUC
  strongly, so the component order is still meaningful.
- The scaling evidence now supports the MVP K choice more cleanly: K8 remains
  the practical sweet spot, K4 is cheaper but less expressive, and K16 likely
  needs longer training or stronger shaping before its extra token budget pays
  off.
