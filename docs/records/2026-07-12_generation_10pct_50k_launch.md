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

At step 50, elapsed training time was 120.44 seconds, total loss had fallen to
0.71358 and epsilon loss to 0.68842 with finite gradient norm 3.33986. This is
about 2.41 seconds per optimizer step, implying roughly 33.5 hours per 50K run
before checkpoint/evaluation overhead and about 67 hours for the serial pair.

At step 1,500, elapsed training time was 3,591.81 seconds. Total loss was
0.029425, epsilon loss was 0.026842, and the gradient norm was 0.1915. The GPU
reported 22,979 MiB allocated, 98% utilization, and 63 C. No checkpoint exists
before the configured 5,000-step interval; at the observed 2.39 seconds per
step, the first atomic recovery point is expected about 2.3 hours after this
snapshot.

The remote `781a014` runbook relies on `set -e`: a non-zero training exit stops
the queue before dense starts. The local branch later added an explicit
`training_complete` report guard, but that newer revision is intentionally not
synchronized while this matched pair is active.

At step 4,000, a standalone read-only progress audit ran on `pro6000` from
`/tmp` without changing the active repository or loading the model. The
synchronized snapshot is:

```text
artifacts/reports/generation/training_progress_2026-07-12/cofitok_step_004000.json
```

The audit reports `healthy` with no issues, `2.3916888 s/step`, recent 10-record
mean total loss `0.0325722`, recent mean epsilon loss `0.0306235`, and recent
mean pre-clip gradient norm `0.1435504`. Checkpoint status is `not_due`, and the
first configured validation/checkpoint remains at step 5,000. The maximum
logged gradient norm `8.1697` is the early step-1 pre-clipping norm; the loop
applies the configured `clip_grad_norm=1.0` before each optimizer update. Five
logged pre-clip norms exceeded 1.0 in the first 4,000 steps.

At step 5,000, the first atomic recovery checkpoint completed successfully:

```text
checkpoint: checkpoint_step_00005000.pt
bytes: 1,008,218,658
latest.json: {"checkpoint": "checkpoint_step_00005000.pt", "step": 5000}
temporary files remaining: 0
```

The original training PID remained active and reached step 5,050 after the
save. A second read-only audit reports `healthy`, `issues=[]`, checkpoint status
`available`, `2.3917204 s/step`, and a remaining CoFiTok ETA of approximately
107,508 seconds at that snapshot. The synchronized audit is:

```text
artifacts/reports/generation/training_progress_2026-07-12/cofitok_step_005050.json
```

The audit's `validation_event_count=0` is a logging limitation of commit
`781a014`, not evidence that validation code was skipped. That revision writes
the JSONL row before running its periodic validation, so intermediate
`validation_epsilon_mse` values are not persisted in `train_metrics.jsonl`.
Final checkpoint evaluation and the matched 10K EMA generation gate remain the
authoritative quality evidence. This limitation is fixed on the local upgrade
branch but is intentionally not synchronized during the active matched pair.
