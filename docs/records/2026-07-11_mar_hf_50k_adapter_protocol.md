# MAR HF Safetensors 50K Adapter Protocol - 2026-07-11

## Purpose

Complete the missing MAR official ImageNet-256 eval-only row using the already
verified Hugging Face safetensors. This is secondary related-method evidence and
must not enter the P0 same-budget retraining table.

## Locked Assets

```text
HF code/assets:
/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/hf_repo

MAR-B weights:
mar-base.safetensors

KL-VAE weights:
kl16.safetensors

Project adapter:
CoFiTok-internal/scripts/baselines/sample_mar_hf_official.py
```

The adapter loads both safetensors with `strict=True` and does not modify the
external MAR clone.

## Locked Sampling Protocol

The settings reproduce the official MAR-B evaluation command:

```text
dataset: ImageNet-256 class-conditional
images: 50,000, exactly 50 per ImageNet class
model: MAR-B
autoregressive iterations: 256
DiffLoss sampling steps: 100
CFG: 2.9
CFG schedule: linear
temperature: 1.0
precision: fp16 autocast
global seed: 0
```

For interruption-safe resumption, each output batch uses
`batch_seed = global_seed + batch_start_index`. The batch size is fixed in the
final run report and must not change when resuming that run.

## Pilot Gate

Before launching 50K:

1. Strict MAR-B and KL-VAE loading must pass.
2. A real `num_iter=256`, CFG 2.9 batch must complete without NaN, OOM, or
   malformed PNG output.
3. Peak memory and images/second must be recorded.
4. Select the largest stable batch size that leaves a practical memory margin.

## Final Gate

After generation:

1. Verify exactly 50,000 numbered 256x256 RGB PNGs.
2. Pack them as `arr_0` with shape `[50000, 256, 256, 3]`, dtype `uint8`.
3. Evaluate against
   `/root/autodl-tmp/CoFiTok/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz`
   using the same ADM evaluator used for D-AR and ReTok.
4. Record FID, sFID, IS, precision, recall, hashes, runtime, and exact command.

## Interpretation Boundary

A successful run closes the MAR official-checkpoint metric row only. It does not
prove same-budget all-dataset MAR retraining and does not change any
`protocol_blocked` cell in the P0 comparison matrix.
