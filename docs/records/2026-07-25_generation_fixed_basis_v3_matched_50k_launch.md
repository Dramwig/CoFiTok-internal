# Fixed-basis v3 matched 50K launch (2026-07-25)

## Purpose

Launch the fresh ImageNet-256 10% matched training pair selected by the
terminal v8 fixed-basis probe. This run is the only current path to a new
promotion decision. It does not itself authorize full ImageNet-256 300K
training.

## Deployed revision

- Branch: `scale/generative-system`
- Revision: `58d83bfce2770eab2565b8c89a5f9a06201a0c86`
- Tracked worktree at launch: clean
- Previous immutable training source:
  `781a01444fddbf0d48a427ba58bdeed50167b5be`

The target-specific immutable deployment receipt is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/generation-upgrade-58d83bfce2770eab2565b8c89a5f9a06201a0c86.receipt.json
```

Receipt identity:

- status: `pass`
- bytes: `2,305`
- SHA256:
  `3a85d03023dd66223edb2ee46913b0e48f88e197a430a38650b09708d5b5285a`
- pytest: `647` tests, `0` failures, `0` errors, `0` skipped
- runbook syntax: `51/51` passed
- untracked target conflicts: `0`
- deployment bundle bytes: `12,117,313`
- deployment bundle SHA256:
  `9476e72e259d0c336b96f2e6bf13c100ba07703c7b6514856c93c519ef927594`

The receipt binds the source pair validation, source and target revisions,
bundle prerequisite/head, complete pytest report, runbook syntax report, and
untracked-file conflict scan.

## Authoritative v3 identities

```text
CoFiTok config:
  configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json
Dense config:
  configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json

CoFiTok run:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3
Dense run:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_dense_50k_v3
Report root:
  /root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3
Promotion gate:
  /root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/promotion_gate.json
```

The generated pair contract at launch is `11,663` bytes with SHA256
`e90c24f6fe45aaa0c17c93df7daf3ab4f799e734367df6ea7a4e017729ba3384`.
The formal parameter counts are CoFiTok `62,836,011` and dense `62,824,707`,
a relative gap of `+0.017993%`.

## Launch state

- Supervisor PID: `502309`
- Supervisor:
  `/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/runbooks/generation_completion_supervisor.sh`
- Supervisor status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_completion_supervisor.status.json`
- Pipeline status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_complete_pipeline_after_10pct.status.json`
- Supervisor log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_completion_supervisor.log`

At the first post-launch audit, both status files were `running`, the pipeline
stage was `scaling_training`, and the runtime selector benchmarked all shared
effective-batch-64 candidates on the RTX PRO 6000. It selected
`micro_batch_size=64` and `gradient_accumulation_steps=1`:

- selection policy: minimize the slower method's mean optimizer-step time
- selected score: `2.216154` seconds
- selected maximum memory fraction: `0.550936`
- estimated speedup over `16x4`: `1.019377x`
- runtime environment SHA256:
  `51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57`

The selected runtime report is:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/runtime_selection.json
```

Formal CoFiTok training then entered the authoritative run directory with
`64x1`. Its first logged updates through step 50 were finite. One supervisor,
one completion pipeline, one matched-pair runbook, one training watchdog, and
one formal trainer held the execution chain. The apparent multiple
`train_generation.py` processes were data-loader workers belonging to that
single trainer.

The remote tracked revision must remain fixed at `58d83bf` until the v3
CoFiTok and dense 50K reports and the subsequent promotion decision are
terminal.

The first authoritative monitor refresh observed:

- monitor PID: `504563`
- training watchdog PID: `504584`
- trainer PID: `504648`
- monitor state: `running/cofitok_training`
- monitor issues and run health issues: empty
- step: `100/50,000`
- images seen: `6,400`
- progress: `0.2%`
- cumulative elapsed: `223.34` seconds
- total loss: `1.007648`
- epsilon MSE: `0.992196`
- gradient norm: `2.570774`
- GPU: `69,893` MiB, `100%` utilization
- remote tracked revision: clean `58d83bf`

These are launch-health observations, not promotion evidence. The first
scheduled validation and recovery checkpoint remain due at steps 1,000 and
5,000 respectively.

## Mandatory decision sequence

1. Select one shared effective-batch-64 runtime from real checkpoint-free
   CoFiTok/dense benchmarks.
2. Train CoFiTok to exactly 50,000 steps and 3,200,000 images.
3. Train dense to the same steps, images, data order contract, optimizer, and
   runtime selection.
4. Validate both final checkpoints, integrity sidecars, training reports,
   clean Git provenance, runtime environment, and matched pair contract.
5. Generate exact matched 10,000-image EMA DDIM-100 sample sets and run the
   unchanged scaling promotion gate.
6. Start full matched ImageNet-256 300K only if the promotion gate is
   `pass/promote`.

The gate remains fail closed. In particular, CoFiTok must have finite
distribution metrics, FID no worse than 5% relative to dense, absolute FID at
most 100, endpoint MSE no worse than 5%, ordered rank 1, sufficient coarse
energy, exact zero-token behavior, valid shuffle diagnostics, and complete
provenance. A hold or failure must stop before full 300K.

## Local archive policy

The local archive branch was rebased onto the deployed implementation:

```text
58d83bf  Promote fixed-basis synthesis to formal generation v3
7d9dcbb  Archive 10% generation promotion hold evidence
361b52e  Archive fixed-basis v8 generation evidence
```

This launch record and later small reports remain local evidence-only changes
while formal training is active. No large checkpoint, full sample tree,
feature cache, or long log is copied into the local repository.

The bounded launch pack is:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/
```

It contains the deployment receipt, pair contract, runtime selection,
supervisor/pipeline/monitor/watchdog states, run manifest, the first five
metrics rows, and a tracked progress audit. The progress audit was `healthy`
at step 200 with `0` issues, `0` warnings, `2.219099` seconds per step, and
an ETA of `110,511` seconds to CoFiTok step 50,000. `README.md` binds every
copied file by byte count and SHA256.

## Read-only milestone observer

A bounded read-only observer was launched after formal training reached step
400:

- PID: `507430`
- local source:
  `artifacts/operations/generation/fixed_basis_v3_milestone_waiter.py`
- remote source:
  `/tmp/cofitok_fixed_basis_v3_milestone_waiter.py`
- source SHA256:
  `7a9798d6af3f0c484374f8307ad3136209f2eabf3fafac2a3a74608c33333745`
- status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_fixed_basis_v3_milestone_waiter.status.json`
- log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_fixed_basis_v3_milestone_waiter.log`
- bounded timeout: `18,000` seconds
- polling interval: `120` seconds

At step 1,000 it calls the deployed tracked progress auditor and requires a
healthy report, no issues or warnings, and complete scheduled-validation
logging. At step 5,000 it additionally requires the recovery checkpoint,
integrity sidecar, `latest.json` binding, and recomputed checkpoint integrity
to verify. Its milestone reports are written under the authoritative v3 report
root as:

```text
cofitok_progress_step_00001000.json
cofitok_progress_step_00005000.json
```

The observer only reads metrics, Git identity, and checkpoint bytes. It does
not import or load the model, allocate GPU memory, signal processes, alter the
pipeline decision, or move the remote revision. Its status was `waiting` with
last step 400 immediately after launch.
