# Component Decorrelation Tiny Probe

Date: 2026-07-08

Purpose: check whether the best CIFAR decor candidate
`component_decorrelation_weight=0.005` transfers to Tiny ImageNet-200.

## Setup

Baseline:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08
```

Decor candidate:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_cuda.json
component_decorrelation_weight = 0.005
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_2026-07-08
```

Both runs use Tiny ImageNet-200, K=8, seed 139, 5000 steps, restricted `S_k`,
and the light denoise-path objective.

Local report mirror:

```text
artifacts/reports/component_decorrelation_tiny_probe_2026-07-08/
artifacts/reports/component_decorrelation_tiny_probe_2026-07-08/component_decorrelation_tiny_probe_summary.json
```

## Result

Lower is better for path AUC, quality MSE AUC, and component cosine. Higher is
better for effective token count.

| variant | train path AUC | quality final MSE | quality MSE AUC | effective K | mean abs cosine | delta cosine | delta path AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0759 | 0.1487 | 6.7288 | 6.8043 | 0.7223 | 0.0000 | 0.0000 |
| decor 0.005 | 0.0955 | 0.1481 | 7.0983 | 6.7977 | 0.5904 | -0.1319 | +0.0197 |

Order diagnostics:

| variant | ordered path AUC | random path AUC | reverse path AUC | ordered effective K |
|---|---:|---:|---:|---:|
| baseline | 0.0766 | 0.2570 | 0.7140 | 6.8033 |
| decor 0.005 | 0.0963 | 0.3016 | 0.8128 | 6.7984 |

The Tiny result matches the CIFAR direction:

- Component mean absolute cosine decreases from `0.7223` to `0.5904`.
- Final quality MSE improves slightly from `0.1487` to `0.1481`.
- Ordered path AUC worsens from `0.0766` to `0.0963`.
- Effective token count is essentially unchanged: `6.8043` to `6.7977`.
- Order sensitivity remains intact: random and reverse are worse than ordered.

## Decision

`component_decorrelation_weight=0.005` is now a cross-dataset useful ablation
candidate on CIFAR-10 and Tiny ImageNet-200. It should still not become the
default objective term because the path-AUC penalty appears on both datasets.

Next recommended refinement:

```text
repeat decor 0.005 with a second seed, or schedule decor after early prefix/order
training stabilizes, then compare path AUC and component cosine again.
```

