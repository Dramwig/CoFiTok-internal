# Stability-scaling 50K exact resume from step 36,545

Date: 2026-08-01 CST

## Outcome

The user-requested GPU pause completed through the trainer's supported
`SIGTERM` path. The trainer finished the active optimizer step, wrote an atomic
production checkpoint, integrity sidecar, `latest.json`, and an incomplete
training report, then released the GPU. Training has now resumed from that
exact checkpoint rather than falling back to the scheduled 35K recovery point.

```text
checkpoint: checkpoint_step_00036545.pt
bytes: 1006321642
sha256: 3b02917393ac00d49ece5c61ac357b6e638096919e17c835bdbef6c810544c2e
revision: 2c2c1f5166b73d4f28df93b276901671ac1a7836
branch: scale/generation-stability-50k-preflight
```

The pause report records `completed_steps=36545`, `training_complete=false`,
`stop_requested=true`, and `stop_signal=15`. Its checkpoint binding agrees with
the integrity sidecar and `latest.json`.

## Resume integrity

The matched 50K runbook reused its frozen `64x1` runtime selection and launched
the trainer with `--resume auto`. The new run manifest binds the exact 36,545
checkpoint and the same clean Git identity. Recursive config, runtime,
dataset, checkpoint-integrity, optimizer/scheduler/scaler, EMA, RNG, and sampler
state validation therefore remained on the production resume path.

Metrics reconciliation returned:

```text
status: unchanged
resume step: 36545
retained rows: 731
orphaned rows: 0
orphan archive: null
```

The first resumed metric is step 36,546. The local evidence snapshot extends
through step 36,700, so the resumed trajectory is demonstrably advancing past
the checkpoint rather than only loading it.

## Queue restoration

At the snapshot boundary there is exactly one live instance of each queue
role:

```text
matched runbook: 918098
pair monitor: 918502
training watchdog: 918584 (invocation 2)
CoFiTok trainer: 918650
post-eval waiter: 919562
full-readiness waiter: 919564
```

A manually launched duplicate post-eval waiter, PID 919739, was detected before
it had any child and terminated. The earlier authoritative waiter remains.
There is one trainer and one GPU compute PID.

The queue remains pinned to:

```text
training: 2c2c1f5166b73d4f28df93b276901671ac1a7836 / scale/generation-stability-50k-preflight
evaluation: c1efb12c6640f2d2d62ac7e9982c8804d96e7289 / scale/generation-stability-50k-posteval-v4
readiness: 5dd3488ac9b30274f4960195e252cc9fdb161002 / scale/generation-large-capacity
```

The readiness waiter still reports `full_training_launch_allowed=false`.
Resuming this 10% stability pair does not authorize full ImageNet-256 300K.

## Storage handling

At resume preflight, free/required/headroom were:

```text
181533130752 / 118385312804 / 63147817948 bytes
```

Before resume, an unverified redundant staging tar was removed only after all
three original sample roots were confirmed present, releasing 3,423,416,320
bytes. A byte-identical archive tar was then recreated with low CPU/I/O
priority; its size/SHA256 are
`3423416320 / 7d34b3c24f40c5c426a10c42286feaadcd391c787d238f2b3d61c8ae8557ecf6`.
The prior partial local transfer is being resumed in the background. No source
sample tree will be removed until the complete local tar and all six sample-set
digests pass the tracked archive auditor.

## Scope

This record proves an exact, reproducible training resume and restored waiting
chain. It does not prove improved sample quality, a matched dense result,
post-eval success, a passing promotion gate, or authorization for full 300K.

Small evidence is stored under
`artifacts/reports/generation/stability_scaling_50k_2026-08-01/resume_step_00036545/`.
