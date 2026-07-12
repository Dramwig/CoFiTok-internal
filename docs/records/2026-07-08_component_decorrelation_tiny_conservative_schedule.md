# Component Decorrelation Tiny Conservative Schedule

Date: 2026-07-08

Purpose: test a later/lower component-decorrelation schedule that prioritizes
preserving Tiny ImageNet prefix ordering while still reducing component
correlation.

## Setup

Reference runs:

```text
baseline:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08

static decor 0.005:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_2026-07-08

scheduled decor 0.005:
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_2026-07-08
```

New conservative candidate:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_cuda.json
component_decorrelation_weight = 0.003
component_decorrelation_start_step = 3000
component_decorrelation_warmup_steps = 1500
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_2026-07-08
```

All runs use Tiny ImageNet-200, K=8, seed 139, restricted `S_k`, 5000 steps,
and the light denoise-path objective. Quality uses the same 256-image/t=500
probe as the previous decorrelation probes.

Local report mirror:

```text
artifacts/reports/component_decorrelation_tiny_conservative_schedule_2026-07-08/
artifacts/reports/component_decorrelation_tiny_conservative_schedule_2026-07-08/component_decorrelation_tiny_conservative_schedule_summary.json
```

Schedule sanity from the training log:

```text
step 3000 decor_w=0.00000
step 4000 decor_w=0.00200
step 5000 decor_w=0.00300
```

## Result

Lower is better for path AUC, quality MSE AUC, quality final MSE, and component
cosine. Higher is better for effective token count.

| variant | train path AUC | quality final MSE | quality MSE AUC | effective K | mean abs cosine | ordered path AUC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0759 | 0.1487 | 6.7288 | 6.8043 | 0.7223 | 0.0766 |
| static decor 0.005 | 0.0955 | 0.1481 | 7.0983 | 6.7977 | 0.5904 | 0.0963 |
| scheduled decor 0.005 | 0.0865 | 0.1480 | 6.9266 | 6.9175 | 0.6208 | 0.0872 |
| scheduled decor 0.003 | 0.0797 | 0.1486 | 6.7836 | 6.8316 | 0.6890 | 0.0804 |

Order diagnostics:

| variant | ordered path AUC | random path AUC | reverse path AUC | ordered effective K |
|---|---:|---:|---:|---:|
| baseline | 0.0766 | 0.2570 | 0.7140 | 6.8033 |
| static decor 0.005 | 0.0963 | 0.3016 | 0.8128 | 6.7984 |
| scheduled decor 0.005 | 0.0872 | 0.4737 | 0.7680 | 6.9192 |
| scheduled decor 0.003 | 0.0804 | 0.4477 | 0.7305 | 6.8323 |

Key deltas versus baseline:

- Static decor 0.005: component cosine `-0.1319`, ordered path AUC `+0.0197`.
- Scheduled decor 0.005: component cosine `-0.1015`, ordered path AUC `+0.0105`.
- Scheduled decor 0.003: component cosine `-0.0333`, ordered path AUC `+0.0037`.

## Decision

`component_decorrelation_weight=0.003`, `start=3000`, `warmup=1500` is the best
schedule so far when prefix-order preservation is the priority. It keeps the
ordered path close to baseline while still reducing component correlation. The
stronger `0.005` schedules remain useful stress-test ablations when the goal is
to show component disentanglement pressure.

This should still not become the default objective term until a second seed
confirms the same path/cosine tradeoff.
