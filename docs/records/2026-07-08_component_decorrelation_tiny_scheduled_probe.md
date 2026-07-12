# Component Decorrelation Tiny Scheduled Probe

Date: 2026-07-08

Purpose: check whether delayed/warmup component decorrelation improves the
tradeoff observed in the Tiny ImageNet static decorrelation probe.

## Setup

Baseline:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08
```

Static decor candidate:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_cuda.json
component_decorrelation_weight = 0.005
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_2026-07-08
```

Scheduled decor candidate:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_cuda.json
component_decorrelation_weight = 0.005
component_decorrelation_start_step = 2500
component_decorrelation_warmup_steps = 1500
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_2026-07-08
```

All runs use Tiny ImageNet-200, K=8, seed 139, restricted `S_k`, 5000 steps,
and the light denoise-path objective.

Local report mirror:

```text
artifacts/reports/component_decorrelation_tiny_scheduled_probe_2026-07-08/
artifacts/reports/component_decorrelation_tiny_scheduled_probe_2026-07-08/component_decorrelation_tiny_scheduled_probe_summary.json
```

Schedule sanity from the training log:

```text
step 2000 decor_w=0.00000
step 3000 decor_w=0.00167
step 4000 decor_w=0.00500
step 5000 decor_w=0.00500
```

## Result

Lower is better for path AUC, quality MSE AUC, quality final MSE, and component
cosine. Higher is better for effective token count.

| variant | train path AUC | quality final MSE | quality MSE AUC | effective K | mean abs cosine | delta cosine | delta path AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0759 | 0.1487 | 6.7288 | 6.8043 | 0.7223 | 0.0000 | 0.0000 |
| static decor 0.005 | 0.0955 | 0.1481 | 7.0983 | 6.7977 | 0.5904 | -0.1319 | +0.0196 |
| scheduled decor 0.005 | 0.0865 | 0.1480 | 6.9266 | 6.9175 | 0.6208 | -0.1015 | +0.0105 |

Order diagnostics:

| variant | ordered path AUC | random path AUC | reverse path AUC | ordered effective K |
|---|---:|---:|---:|---:|
| baseline | 0.0766 | 0.2570 | 0.7140 | 6.8033 |
| static decor 0.005 | 0.0963 | 0.3016 | 0.8128 | 6.7984 |
| scheduled decor 0.005 | 0.0872 | 0.4737 | 0.7680 | 6.9192 |

Tiny confirms the CIFAR schedule direction:

- Static decor gives the strongest component-correlation reduction:
  `0.7223 -> 0.5904`, but worsens ordered path AUC by about `+0.0197`.
- Scheduled decor still reduces component correlation:
  `0.7223 -> 0.6208`, while roughly halving the ordered path-AUC penalty:
  `+0.0105` instead of `+0.0197`.
- Scheduled decor improves final MSE slightly over both baseline and static
  decor at the 256-image/t=500 probe scale.
- Scheduled decor increases effective token count from `6.8043` to `6.9175`.
- Order sensitivity remains intact: random and reverse are both worse than
  ordered.

## Decision

Delayed/warmup decorrelation is a better refinement knob than static
decorrelation on both CIFAR-10 and Tiny ImageNet-200. It should still not become
the default objective term because baseline retains the best ordered path AUC on
Tiny, and the schedule still adds a measurable path penalty.

Next recommended refinement:

```text
run a second seed for scheduled decor, or try a later/lower schedule such as
start=3000, warmup=1500, weight=0.003 to preserve ordered path AUC more tightly.
```
