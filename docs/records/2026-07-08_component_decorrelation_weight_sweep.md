# Component Decorrelation Weight Sweep

Date: 2026-07-08

Purpose: follow up the `component_decorrelation_weight=0.05` probe, which
reduced component correlation but damaged denoise-path ordering. This sweep
tests smaller CIFAR-10 weights while keeping the K8 light denoise-path baseline
unchanged.

## Configs

All runs use CIFAR-10, K=8, seed 137, 3000 steps, restricted `S_k`, and the
light denoise-path objective.

| variant | config | weight | remote train dir |
|---|---|---:|---|
| baseline | `configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json` | 0.0 | `/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08` |
| decor 0.005 | `configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_3k_cuda.json` | 0.005 | `/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_decorw0005_3k_2026-07-08` |
| decor 0.01 | `configs/train_cifar10_k8_denoisepath_p150_light_decorw001_3k_cuda.json` | 0.01 | `/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_decorw001_3k_2026-07-08` |
| decor 0.05 | `configs/train_cifar10_k8_denoisepath_p150_light_decor_3k_cuda.json` | 0.05 | `/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_decor_3k_2026-07-08` |

Local report mirror:

```text
artifacts/reports/component_decorrelation_weight_sweep_2026-07-08/
artifacts/reports/component_decorrelation_weight_sweep_2026-07-08/component_decorrelation_weight_sweep_summary.json
```

## Training And Quality Summary

Lower is better for path AUC, quality MSE AUC, and component cosine. Higher is
better for effective token count.

| variant | train path AUC | quality final MSE | quality MSE AUC | effective K | mean abs cosine | delta cosine | delta path AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1003 | 0.1531 | 6.7221 | 6.8335 | 0.6345 | 0.0000 | 0.0000 |
| decor 0.005 | 0.1148 | 0.1530 | 6.9887 | 6.8810 | 0.5165 | -0.1180 | +0.0142 |
| decor 0.01 | 0.1322 | 0.1516 | 7.2381 | 6.7756 | 0.4309 | -0.2036 | +0.0315 |
| decor 0.05 | 0.2429 | 0.1494 | 8.0704 | 5.9885 | 0.2050 | -0.4295 | +0.1412 |

## Order Diagnostics

| variant | ordered path AUC | random path AUC | reverse path AUC | ordered effective K |
|---|---:|---:|---:|---:|
| baseline | 0.0994 | 0.2563 | 0.6742 | 6.8345 |
| decor 0.005 | 0.1136 | 0.2774 | 0.7370 | 6.8819 |
| decor 0.01 | 0.1308 | 0.3013 | 0.8103 | 6.7798 |
| decor 0.05 | 0.2406 | 0.5799 | 1.0400 | 5.9839 |

All decor weights preserve order sensitivity: random and reverse orders remain
worse than ordered. The penalty on the ordered path grows monotonically with the
decor weight.

## Decision

`component_decorrelation_weight=0.005` is the best current probe. It gives a
measurable correlation reduction, from `0.6345` to `0.5165`, with the smallest
ordered path-AUC penalty and no effective-token loss in this CIFAR 3k run.

Do not promote it to the default yet. This is one seed and one small dataset.
Promotion requires at least Tiny ImageNet confirmation and preferably a second
CIFAR seed repeat.

Recommended next run:

```text
Tiny ImageNet K8 light, 5k or 10k steps, component_decorrelation_weight=0.005
```

