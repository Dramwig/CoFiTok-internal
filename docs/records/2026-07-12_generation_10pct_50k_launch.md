# ImageNet-256 10% matched 50K launch (2026-07-12)

The resumable matched queue was launched on `pro6000` from clean commit
`781a01444fddbf0d48a427ba58bdeed50167b5be`.

```text
runbook PID: 28952
CoFiTok training PID: 29127
log: /root/autodl-tmp/CoFiTok/checkpoints/generation/logs/imagenet256_10pct_matched_50k_2026-07-12.log
CoFiTok run: /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_cofitok_k8_50k_2026-07-12
dense run: /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_dense_50k_2026-07-12
```

The queue runs CoFiTok first and launches the matched dense control only after
CoFiTok exits successfully. Each stage writes atomic checkpoints every 5,000
steps and keeps the latest three. Re-running the same runbook resumes from
`latest.json` when present.

Initial live evidence:

```text
GPU memory: 20,425 MiB
GPU utilization: 99%
power: 428 W
step 1 total loss: 1.16019395
step 1 epsilon loss: 1.13034904
step 1 gradient norm before clipping: 8.16974926
```

This launch is not a completion claim. Promotion still requires both 50K runs,
EMA sampling, matched FID, and prefix diagnostics.
