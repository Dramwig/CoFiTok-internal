# Scheduled Component Decorrelation Probe

Date: 2026-07-08

Purpose: test whether delaying `component_decorrelation_weight=0.005` can
reduce the path-ordering penalty observed in static decor runs.

## Setup

Scheduled decor config:

```text
configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_sched1500w1000_3k_cuda.json
component_decorrelation_weight = 0.005
component_decorrelation_start_step = 1500
component_decorrelation_warmup_steps = 1000
```

Remote output:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_decorw0005_sched1500w1000_3k_2026-07-08
```

Local report mirror:

```text
artifacts/reports/component_decorrelation_scheduled_probe_2026-07-08/
artifacts/reports/component_decorrelation_scheduled_probe_2026-07-08/component_decorrelation_scheduled_probe_summary.json
```

Schedule sanity from training logs:

```text
step 1500: decor_w=0.00000
step 2000: decor_w=0.00250
step 2500: decor_w=0.00500
step 3000: decor_w=0.00500
```

## Summary

Lower is better for path AUC, quality MSE AUC, and component cosine. Higher is
better for effective token count.

| variant | ordered path AUC | quality final MSE | quality MSE AUC | effective K | mean abs cosine | delta cosine | delta path AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0994 | 0.1531 | 6.7221 | 6.8335 | 0.6345 | 0.0000 | 0.0000 |
| static decor 0.005 | 0.1136 | 0.1530 | 6.9887 | 6.8810 | 0.5165 | -0.1180 | +0.0142 |
| scheduled decor 0.005 | 0.1076 | 0.1529 | 6.8585 | 6.9495 | 0.5463 | -0.0881 | +0.0083 |

Order diagnostics:

| variant | ordered path AUC | random path AUC | reverse path AUC |
|---|---:|---:|---:|
| baseline | 0.0994 | 0.2563 | 0.6742 |
| static decor 0.005 | 0.1136 | 0.2774 | 0.7370 |
| scheduled decor 0.005 | 0.1076 | 0.2723 | 0.7189 |

## Decision

Scheduling decor after step 1500 and warming it over 1000 steps is a better
refinement knob than static decor when preserving prefix ordering matters:

- It reduces the ordered path-AUC penalty from `+0.0142` to `+0.0083`.
- It keeps a real component-correlation reduction: `0.6345 -> 0.5463`.
- It improves effective token use in this probe: `6.8335 -> 6.9495`.
- It slightly improves endpoint MSE: `0.1531 -> 0.1529`.

The tradeoff is that static decor separates components more strongly
(`0.5165` mean cosine) than scheduled decor (`0.5463`). Do not make scheduled
decor a default yet; run Tiny ImageNet or seed-repeat confirmation first.

