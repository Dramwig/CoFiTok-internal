# ImageNet-256 20k Repeat Protocol - 2026-07-11

## Purpose

Strengthen the top-tier scaling claim beyond the current ImageNet-256 5k
short-budget row. This is a matched CoFiTok-versus-dense repeat, not a broad
hyperparameter search.

## Locked Runs

```text
dataset: imagenet_256 full train + val
resolution: 256
methods: K4 CoFiTok light denoise-path; endpoint-only factorized; parameter-matched monolithic dense head
seeds: 103, 139
train steps: 20,000
batch size: 4
quality: 1,024 val images at t=500 with every prefix 1,2,3,4; LPIPS and Inception features
generation: 4,096 DDIM-50 samples; 10,000 real images
order test: ordered/random/reverse plus all 24 K4 permutations at both seeds
```

Configs are generated only by:

```text
scripts/make_imagenet256_long_budget_configs.py
```

Runbook:

```text
artifacts/runbooks/imagenet256_long_budget_repeat_2026-07-11.sh
```

## Confirmatory Predictions

1. CoFiTok path AUC is lower than the endpoint-only factorized control at both seeds.
2. Mean CoFiTok endpoint MSE remains within +5% of the monolithic dense head.
3. Random (fixed seed 1, permutation `[1,3,2,0]`) and reverse accumulation
   have worse path AUC than ordered CoFiTok at both seeds while preserving the
   same endpoint sum.
4. Across all 23 non-identity K4 permutations, the paired per-image mean path
   AUC difference and the reverse-order difference have a positive 95% bootstrap
   confidence interval at both seeds, while all 24 endpoint epsilon sums remain
   equal up to MSE `1e-12`. This exhaustive test is fixed before any confirmatory
   order result is evaluated.
5. Restricted CoFiTok keeps zero-token component energy ratio at zero.

Generated Frechet-style metrics are exploratory controls. No CoFiTok generation
win is preregistered; a loss or mixed result must be retained in the report.

## Decision Rule

The ImageNet-256 scaling evidence supports the scoped paper claim only if all
confirmatory gates pass. If path AUC improves but endpoint cost
exceeds 5%, report a scaling tradeoff rather than claiming preserved endpoint
quality.

## Queue Dependency

The run starts only after MAR official 50K sampling, packing, and ADM evaluation
finish successfully. The dependency wrapper is:

```text
artifacts/runbooks/after_mar_ema_imagenet256_path_eval_2026-07-11.sh
```
