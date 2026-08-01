# CoFiTok stability 50K step-40K audit waiter

Date: 2026-08-02 CST

## Purpose

The resumed CoFiTok member of the stability-scaling matched 50K queue is still
training. A bounded observer now waits for the 40K milestone and automatically
runs the tracked progress auditor as soon as the atomic checkpoint publication
is complete. This removes manual polling while preserving the fixed training
and evaluation identities.

The observer is read-only with respect to the training run. It reads the
canonical JSONL, Git identity, checkpoint bytes, integrity sidecar, and
`latest.json`; it does not load the model, allocate GPU memory, signal a
process, change a revision, or authorize the next pipeline stage.

## Recovery-aware contract

The exact-resume checkpoint at step 36,545 changes the three-point rolling
recovery set. At 40K the expected set is:

```text
checkpoint_step_00035000.pt
checkpoint_step_00036545.pt
checkpoint_step_00040000.pt
```

The observer therefore permits a non-5K recovery point but still requires the
new milestone itself to align to the configured 5K checkpoint interval. The
tracked progress audit must also prove:

- required checkpoint integrity with SHA256 replay;
- `latest.json`, sidecar, training revision, branch, and clean-state binding;
- at least 40 scheduled validation events with complete provenance;
- config-bound rollout and EMA-teacher schedules over every canonical metric
  row, with finite nonzero loss for every active row;
- empty issues and warnings.

## Implementation and validation

The observer implementation is commit
`8bf4bb73ec05543a6e0317ef00a72fed5ac2a3b4`. Its exact deployed source is:

```text
path: /tmp/cofitok_stability_40k_milestone_waiter_84c04e9.py
bytes: 15043
sha256: 84c04e91dfb846f1ef3eeabe88e7be1b5d6fd6b02121c0fb2908163f886af99e
```

Validation at the exact commit passed:

```text
targeted waiter + progress-auditor tests: 33 passed
full pytest: 893 collected / 887 passed / 6 skipped
git diff --check: pass
```

The observer calls the auditor from the clean checkout
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` on
`scale/generation-stability-50k-posteval-v4`. The training checkpoint must
remain bound to `2c2c1f5166b73d4f28df93b276901671ac1a7836` on
`scale/generation-stability-50k-preflight`. The resolved config SHA256 is
`4b9bf0d89f639983df6f5c046deb36ad6bd65dac8ec8f9b587721dbcef688f02`.

## Launch state

The observer launched as PID `939556`, with a 60-second poll interval, a
5-hour bounded timeout, and an 1,800-second metrics-staleness threshold. Its
initial locked snapshot was:

```text
observer status: waiting
detail: waiting for milestone metrics
last step: 38050
audit attempts: 0
snapshot sha256: e89347a1fc8de93d9987907b1f256e5238654bbc8cf50efb0724ba9774c1867a
```

At capture time the trainer was alive at PID `918650`, pair monitor was
`running/issues=[]`, all logged float values were finite, GPU allocation was
85,294 MiB, and filesystem free space was `191,443,877,888` bytes. The 40K
progress report was correctly absent because the milestone had not yet been
reached.

The authoritative live paths are:

```text
status: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/cofitok_step_00040000/milestone_waiter.json
report: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/cofitok_step_00040000/progress_audit.json
log: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/cofitok_step_00040000/milestone_waiter.log
```

Machine-readable launch evidence and the initial status snapshot are stored in
`artifacts/reports/generation/stability_scaling_50k_2026-08-01/cofitok_step_00040000/`.

## Completion boundary

Passing the 40K audit will prove recovery and schedule integrity only. It does
not prove sample quality, train the dense matched member, authorize post-eval,
pass the promotion gate, or authorize full 300K training.
