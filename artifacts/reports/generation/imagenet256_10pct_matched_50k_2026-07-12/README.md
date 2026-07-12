# ImageNet-256 10% matched 50K gate

Runbook: `artifacts/runbooks/generation_10pct_matched_50k_2026-07-12.sh`

The queue trains CoFiTok K8 first and the same-backbone dense epsilon control
second. Both use class-conditional ImageNet-256, effective batch 64, 50K
optimizer steps, cosine diffusion, bf16, EMA, and the same optimizer/schedule.

Large checkpoints and logs remain on `pro6000` under:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_cofitok_k8_50k_2026-07-12
/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_dense_50k_2026-07-12
```

This directory receives only small config, training, sampling, and evaluation
reports after each stage completes.
