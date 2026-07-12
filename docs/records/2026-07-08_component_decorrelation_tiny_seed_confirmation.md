# Component Decorrelation Tiny Seed Confirmation

Date: 2026-07-08

Purpose: confirm whether the conservative Tiny ImageNet component-decorrelation
schedule remains useful on a second seed.

## Setup

Schedule:

```text
component_decorrelation_weight = 0.003
component_decorrelation_start_step = 3000
component_decorrelation_warmup_steps = 1500
```

Seed 139 reference:

```text
baseline:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08

scheduled decor:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_2026-07-08
```

Seed 103 reference:

```text
baseline:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_2026-07-08

scheduled decor:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_seed2_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_seed2_2026-07-08
```

All runs use Tiny ImageNet-200, K=8, restricted `S_k`, 5000 steps, and the light
denoise-path objective. Quality uses the 256-image/t=500 probe.

Local report mirror:

```text
artifacts/reports/component_decorrelation_tiny_conservative_seed_confirm_2026-07-08/
artifacts/reports/component_decorrelation_tiny_conservative_seed_confirm_2026-07-08/component_decorrelation_tiny_conservative_seed_confirm_summary.json
```

## Result

Lower is better for ordered path AUC, component cosine, and final MSE. Higher is
better for effective token count.

| seed | variant | ordered path AUC | mean abs cosine | final MSE | effective K | random path AUC | reverse path AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 139 | baseline | 0.0766 | 0.7223 | 0.1487 | 6.8043 | 0.2570 | 0.7140 |
| 139 | scheduled decor 0.003 | 0.0804 | 0.6890 | 0.1486 | 6.8316 | 0.4477 | 0.7305 |
| 103 | baseline | 0.0740 | 0.6967 | 0.1503 | 6.8430 | 0.4090 | 0.6941 |
| 103 | scheduled decor 0.003 | 0.0776 | 0.6644 | 0.1501 | 6.8829 | 0.4209 | 0.7086 |

Mean delta versus baseline:

| metric | mean delta |
|---|---:|
| train path AUC | +0.0037 |
| ordered path AUC | +0.0036 |
| quality final MSE | -0.0002 |
| quality MSE AUC | +0.0486 |
| effective token count | +0.0336 |
| mean abs component cosine | -0.0328 |
| random path AUC | +0.1013 |
| reverse path AUC | +0.0155 |

## Decision

The conservative schedule is confirmed as a useful decorrelation ablation
candidate:

- Component cosine decreases on both seeds.
- Ordered path AUC penalty stays below `0.004` on both seeds.
- Final MSE slightly improves on both 256-image/t=500 probes.
- Random/reverse order remain worse than ordered, so the ordered-prefix
  interpretation is not broken.

This schedule should be used in decorrelation ablation tables. The headline
CoFiTok objective should still remain the no-decorrelation light denoise-path
objective because it has the cleanest prefix-order interpretation.
