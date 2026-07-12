# D-AR Official ImageNet-256 50K Launch - 2026-07-10

## Purpose

Run D-AR official pretrained ImageNet-256 evaluation at 50K samples as a
secondary related-method row. This does not belong to the P0 same-dataset,
same-budget generation table.

## Status

Status at launch monitor: `running`

Superseded by completion record:

```text
CoFiTok-internal/docs/records/2026-07-10_d_ar_official_50k_completion.md
```

Remote process:

```text
pid: 990280
started: 2026-07-10T06:16:54+08:00
host: pro6000
env: pf-vlm
method: d_ar
num_images: 50000
per_proc_batch_size: 64
gpu: 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Run status file:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/runbooks/d_ar_official_imagenet256_50k_2026-07-10.status
```

Run log:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/runbooks/d_ar_official_imagenet256_50k_2026-07-10.log
```

Expected sample NPZ:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.npz
```

Reference NPZ:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz
```

## First Monitor

```text
sample loop: 2 / 782 iterations observed
png count: 128
gpu utilization: ~96%
gpu memory: ~7.2GB
```

## Notes

- The earlier 16-sample smoke artifacts were moved to:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples_smoke16_2026-07-10
```

- The 50K run is eligible only for a secondary official-checkpoint related
  methods table after it completes and the evaluator writes metrics.
- Do not merge this result into the P0 same-budget all-dataset table.
