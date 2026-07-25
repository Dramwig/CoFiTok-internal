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
supervisor/pipeline/monitor/watchdog states, run manifest, bounded metrics
snapshots through step 10,750, scheduled-validation audits, and the 5K and
10K checkpoint trust-boundary metadata. The launch progress audit was
`healthy` at step 200 with `0` issues and `0` warnings. `README.md` binds
every copied file by byte count and SHA256.

## Read-only milestone observer

A bounded read-only observer was initially launched after formal training
reached step 400:

- initial PID: `507430`
- local source:
  `artifacts/operations/generation/fixed_basis_v3_milestone_waiter.py`
- remote source:
  `/tmp/cofitok_fixed_basis_v3_milestone_waiter.py`
- initial source SHA256:
  `7a9798d6af3f0c484374f8307ad3136209f2eabf3fafac2a3a74608c33333745`
- status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_fixed_basis_v3_milestone_waiter.status.json`
- log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_fixed_basis_v3_milestone_waiter.log`
- bounded timeout: `18,000` seconds
- polling interval: `120` seconds

At each 1K validation milestone it calls the deployed tracked progress auditor
and requires a healthy report, no issues or warnings, and complete
scheduled-validation logging. At step 5,000 it additionally requires the
recovery checkpoint, integrity sidecar, `latest.json` binding, and recomputed
checkpoint integrity to verify. Its milestone reports are written under the
authoritative v3 report root as:

```text
cofitok_progress_step_00001000.json
cofitok_progress_step_00002000.json
cofitok_progress_step_00003000.json
cofitok_progress_step_00004000.json
cofitok_progress_step_00005000.json
```

The observer only reads metrics, Git identity, and checkpoint bytes. It does
not import or load the model, allocate GPU memory, signal processes, alter the
pipeline decision, or move the remote revision. Its status was `waiting` with
last step 400 immediately after launch.

Observer acceptance logic is covered by
`tests/test_fixed_basis_v3_milestone_waiter.py`. Seven targeted tests cover a
complete 1K validation, missing validation, warnings, a verified 5K
checkpoint, a missing integrity manifest, invalid integrity, and a mismatched
checkpoint step. The targeted suite passed `7/7`; the complete local project
suite then passed `651` tests with the existing `2` skips (`653` collected).

After the step-1,000 result showed that later validation trend matters, the
observer was upgraded to cover every 1K event through the first checkpoint.
The upgraded source passed `11/11` targeted tests and has SHA256:

```text
10fe73ca0f86835fab52e23263896f8c06111cd2900406e27485ef3415661f6c
```

Only the observer PID was terminated; the formal trainer, watchdog,
supervisor, and GPU workload remained alive. The upgraded observer was
launched as PID `511498`, reused the existing 2,054-byte step-1,000 report
without rewriting it, and exposed pending slots for 2K, 3K, 4K, and 5K. Its
initial upgraded status read formal training step 1,200.

## Early trajectory alignment

The formal v3 metrics through step 650 were compared with the selected v8
probe at all 14 common logged steps. Source byte identities and the complete
result are bound in:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/early_trajectory_vs_v8.json
```

Windowed mean relative changes, formal v3 versus v8:

| Steps | Epsilon | Total loss |
|---:|---:|---:|
| 1-200 | -0.032% | -0.054% |
| 201-400 | +0.545% | +0.621% |
| 401-600 | +1.000% | +1.037% |

The maximum absolute window delta was `1.037%`, below the diagnostic `2%`
boundary. At step 650, formal epsilon and total loss were respectively
`6.676%` and `3.451%` below v8. This establishes that the formal run preserved
the selected probe's early optimization trajectory despite the formal random
flip and longer schedule. It is not a sample-quality result, promotion gate,
or authorization for full 300K.

## Step-1,000 scheduled validation

The milestone observer independently completed the first required audit:

- observer milestone status: `pass`
- tracked progress status: `healthy`
- issues/warnings: empty
- metric rows: `21`
- validation events: `1/1`, logging complete
- checkpoint: correctly `not_due`
- seconds per step: `2.219961`
- ETA to CoFiTok 50K: `108,778` seconds

The exact step-1,000 row reported:

- training epsilon: `0.03296850`
- total loss: `0.04680164`
- validation epsilon MSE: `0.04005979`
- gradient norm: `0.25228009`
- samples seen: `64,000`

Against v8 at the same step, formal training epsilon and total loss were
`15.10%` and `14.39%` lower, while the first scheduled validation MSE was
`11.63%` higher. The run remains finite, fully logged, and operationally
healthy, so one validation event does not justify stopping or modifying the
formal trajectory. Equally, the lower training objective must not hide the
validation increase. The locked decision is to continue unchanged and require
the 2K-5K validation trend plus formal generation metrics.

Bound evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00001000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step1000_validation_vs_v8.json
```

## Terminal completion audit snapshot

The deployed terminal auditor was run after the 1K validation with:

```text
deployment source: 781a01444fddbf0d48a427ba58bdeed50167b5be
10% revision:       58d83bfce2770eab2565b8c89a5f9a06201a0c86
full revision:      58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

It correctly exited nonzero and reported:

- status: `in_progress`
- complete: `false`
- failed checks: `0`
- warnings: `0`
- missing checks: `17`
- passed terminal check: `controlled_revision_transition`

The passing transition evidence binds the target revision, source validation,
deployment bundle, conflict scan, 647-test report, and all 51 runbooks. The
missing checks explicitly include the 10% matched pair, promotion gate, full
300K pair, full checkpoint reproducibility, milestone evaluations, formal 50K
generation, visual audit, EMA artifacts, final gate, and comparison report.

Snapshot:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/completion_audit_in_progress_after_step1000.json
```

It is `3,635` bytes with SHA256
`ab4c258638e300e41b575e46cc665cbaa30e0dd5b99fc1d81a3d919559ffbefc`.
This is evidence that the completion contract recognizes the new deployment
and still refuses premature completion.

## Early storage risk check

The exact formal 10% post-evaluation storage parameters were evaluated early,
without starting sampling:

```text
stage: 10pct_posteval
sample count: 20,256
estimated sample size: 256 KiB
additional reserve: 16 GiB
safety margin: 32 GiB
```

Result:

- status: `pass`
- required free bytes: `56,849,596,416`
- observed free bytes: `408,803,782,656`
- headroom bytes: `351,954,186,240`
- revision: clean `58d83bf`

Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/storage_preflight_early_after_step1000.json
```

It is `930` bytes with SHA256
`5e16f3b7c5dee38b9c699c0288762ae189d6dd78c3ac9051c9687116cd0351c7`.
This removes an immediate capacity risk but does not satisfy the terminal
`generation_storage_capacity` check. The formal runbook must rerun the same
preflight against post-training disk state before creating samples.

## Step-2,000 scheduled validation

The read-only milestone observer and tracked progress auditor independently
accepted the second scheduled validation:

- observer milestone status: `pass`
- tracked progress status: `healthy`
- issues/warnings: empty
- metric rows: `41`
- validation events: `2/2`, logging complete
- checkpoint: correctly `not_due`
- seconds per step: `2.220870`
- ETA to CoFiTok 50K: `106,602` seconds

The exact step-2,000 row reported:

- training epsilon: `0.02923447`
- total loss: `0.04426858`
- validation epsilon MSE: `0.03190126`
- gradient norm: `0.15392414`
- samples seen: `128,000`

Against v8 at the same step, formal training epsilon, total loss, and
validation MSE were respectively `5.94%`, `2.25%`, and `15.07%` lower.
Formal validation MSE also improved `20.37%` from step 1,000 to step 2,000,
resolving the isolated first-event increase in the favorable direction.

GPU memory measured by `nvidia-smi` immediately after the step-1,000 and
step-2,000 validation events was `77,983 MiB` both times, out of
`97,887 MiB`. The zero delta supports a caching-allocator plateau rather than
repeated validation-event growth across these two observations. This remains a
point-measurement diagnostic, not a broad memory-safety proof.

Bound evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00002000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_train_metrics_through_step_00002050.jsonl
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/milestone_waiter_after_step_00002000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step2000_validation_vs_v8.json
```

This second validation remains a short-horizon health diagnostic. The formal
run continues unchanged and still requires the 3K-5K validation trend, verified
step-5,000 checkpoint integrity, both 50K trainings, and the unchanged formal
10K promotion gate before full 300K can be authorized.

## Step-3,000 scheduled validation

The milestone observer accepted the third scheduled validation after the
tracked progress auditor reported:

- status: `healthy`
- issues/warnings: empty
- validation events: `3/3`, logging complete
- checkpoint: correctly `not_due`
- seconds per step: `2.221382`
- ETA to CoFiTok 50K: `104,294` seconds

The exact step-3,000 validation row was:

- training epsilon: `0.03181857`
- total loss: `0.04608585`
- validation epsilon MSE: `0.03765950`
- gradient norm: `0.17540261`
- samples seen: `192,000`

This event is an adverse diagnostic and is not hidden by the healthy
operational status. Relative to the selected v8 probe at step 3,000, formal
training epsilon, total loss, and validation MSE were respectively `29.15%`,
`21.40%`, and `61.65%` higher. Formal validation MSE also rose `18.05%` from
step 2,000, although it remained `5.99%` below the formal step-1,000 value.

The run continues unchanged because a single non-monotonic scheduled
validation event does not determine 50K generation quality, the formal recipe
has a different augmentation and schedule horizon from the 5K probe, and all
losses, gradients, logging, Git, and process health remain valid. The adverse
comparison strengthens the requirement to inspect step 4K and 5K and to let
the unchanged formal sample gate decide promotion.

GPU memory remained `77,983 MiB` immediately after the third validation,
matching the step-1,000 and step-2,000 measurements. No repeated validation
memory growth has been observed across these three point measurements.

Bound evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00003000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_train_metrics_through_step_00003050.jsonl
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/milestone_waiter_after_step_00003000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step3000_validation_vs_v8.json
```

## Step-4,000 scheduled validation

The fourth scheduled validation passed the observer and tracked progress
auditor with:

- status: `healthy`
- issues/warnings: empty
- validation events: `4/4`, logging complete
- checkpoint: correctly `not_due`
- seconds per step: `2.221167`
- ETA to CoFiTok 50K: `102,174` seconds

The exact step-4,000 row reported:

- training epsilon: `0.03061404`
- total loss: `0.04433357`
- validation epsilon MSE: `0.02938104`
- gradient norm: `0.12236804`
- samples seen: `256,000`

The within-run validation trend improved: step 4,000 was `21.98%` below step
3,000, `7.90%` below step 2,000, and `26.66%` below step 1,000. The cross-run
comparison remains adverse, however. Relative to the selected v8 probe at the
same step, formal training epsilon, total loss, and validation MSE were
`40.57%`, `27.25%`, and `58.77%` higher.

This mixed result shows that the step-3,000 increase was partly non-monotonic,
but it does not erase the persistent gap to the short probe. The formal run
remains finite, fully logged, and frozen, so it proceeds to the first
checkpoint. Step-5,000 integrity and the eventual formal sample gate, rather
than operational health or a selected validation point, remain authoritative.

Post-validation GPU memory was again `77,983 MiB`; all four point
measurements after scheduled validation events are identical.

Bound evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00004000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_train_metrics_through_step_00004050.jsonl
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/milestone_waiter_after_step_00004000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step4000_validation_vs_v8.json
```

## Step-5,000 validation and checkpoint integrity

The bounded milestone observer completed all requested milestones and reached
terminal `pass`. The tracked progress auditor reported:

- status: `healthy`
- issues/warnings: empty
- validation events: `5/5`, logging complete
- checkpoint status: `available`
- required checkpoint steps: `[5000]`
- missing required steps: empty
- latest integrity status: `verified`
- seconds per step: `2.221132`
- ETA to CoFiTok 50K: `99,951` seconds

The exact step-5,000 row reported:

- training epsilon: `0.02518125`
- total loss: `0.03765042`
- validation epsilon MSE: `0.03096321`
- gradient norm: `0.06627424`
- samples seen: `320,000`

Relative to v8 at step 5,000, formal training epsilon and total loss were
`3.97%` and `6.31%` lower, while validation MSE was `36.31%` higher. Within
the formal run, validation MSE was `5.38%` above step 4,000, `2.94%` below
step 2,000, and `22.71%` below step 1,000. Lower training loss therefore still
must not be substituted for sample quality.

The first formal checkpoint trust boundary is:

```text
checkpoint: checkpoint_step_00005000.pt
bytes:      1,006,351,466
sha256:     8daedd38f44f719cbd488bc1d528187d1e8799dba6e879eed42bdc3f6bd7ce44
format:     1
step:       5,000
revision:   58d83bfce2770eab2565b8c89a5f9a06201a0c86
dataset:    97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741
runtime:    51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57
```

The integrity sidecar, `latest.json`, and progress report matched on all ten
audited identity fields. Their copied local byte identities are:

```text
checkpoint_step_00005000.pt.integrity.json
  572 bytes
  18e0ecd2b507169c4bc5911621c04d7e28fcde462c52e7a00557af8834687554
latest_after_step_00005000.json
  642 bytes
  a3d12182bd038ad7e49d6cba94dd5d4e1391b2bb1de4cfb434a80ae4016ede97
cofitok_progress_step_00005000.json
  3,106 bytes
  30329026393a158f69f6dd93b2b5207a6811b0dedc87367d944582d0547d3e0e
```

The 1 GB checkpoint payload remains only on the server. The local archive
contains its verified identity and trust-boundary metadata, not the weight
file. GPU memory after validation and checkpoint publication remained
`77,983 MiB`, matching all four prior validation measurements.

Bound summary:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step5000_validation_and_checkpoint_vs_v8.json
```

This establishes a reproducible first recovery point but does not satisfy the
10% matched-pair, promotion, full 300K, formal 50K, or release-artifact gates.
The frozen CoFiTok 50K run continues.

The tracked terminal completion auditor was repeated after this checkpoint.
It intentionally exited nonzero and remained:

```text
status:   in_progress
complete: false
pass:     1
missing:  17
failed:   0
```

`controlled_revision_transition` remained the only pass. The report was
byte-identical to the step-1,000 audit at `3,635` bytes and SHA256
`ab4c258638e300e41b575e46cc665cbaa30e0dd5b99fc1d81a3d919559ffbefc`.
This proves that the newly verified 5K recovery point does not accidentally
satisfy the dual 50K, storage, promotion, full 300K, milestone, formal
sampling, visual audit, EMA export, final gate, or comparison checks.

Snapshot:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/completion_audit_in_progress_after_step5000.json
```

## Long-horizon observer

The bounded 1K-5K observer exited normally after terminal `pass`. Its source
was then generalized for long-horizon checkpoints:

- milestones are CLI-controlled and validated as unique scheduled-validation
  steps within the training horizon;
- each checkpoint-aligned milestone requires the exact current checkpoint,
  integrity sidecar, and `latest.json` binding;
- a stale step-5,000 latest pointer is rejected at step 10,000;
- the final step 50,000 milestone requires progress status `complete`, not
  merely `healthy`.

Verification:

```text
targeted tests: 18 passed
full pytest:    664 collected, 662 passed, 2 skipped, exit 0
source SHA256:  47a24e56604b8720ff689f7a9d98e2e54a876faabeec2c2fd4d97ab2cf7f210c
remote syntax:  pass
```

The exact source was copied to:

```text
/tmp/cofitok_fixed_basis_v3_long_horizon_waiter.py
```

Local and remote SHA256 matched. It was launched as PID `531838` with:

```text
milestones:             10,000 / 25,000 / 50,000
expected steps:         50,000
checkpoint interval:    5,000
evaluation interval:    1,000
require complete final: true
poll:                   240 seconds
hard timeout:           129,600 seconds
```

Initial status was `waiting` at step 5,450 with all three milestones pending.
The observer reads metrics, Git identity, and checkpoint metadata and invokes
the frozen tracked progress auditor. It does not load a checkpoint, use the
GPU, control training, or modify the remote tracked worktree.

At step 10K, the observer status was still pending even though the checkpoint
had been published. Its own log exposed the cause: the subprocess invoking
`scripts/audit_generation_training_progress.py` did not inherit the project
`src` directory in `PYTHONPATH`, so it could not import `cofitok`. This was an
observer-runtime fault only; the formal trainer, watchdog, supervisor,
metrics, and checkpoint publisher continued uninterrupted.

Only observer PID `531838` was terminated. The same SHA-bound `/tmp` source
was restarted as PID `555237` with:

```text
PYTHONPATH=/root/autodl-tmp/CoFiTok/CoFiTok-internal/src
```

It immediately audited the already-written 10K state and recorded milestone
`pass`. No tracked remote file changed, no checkpoint was loaded, no GPU was
used, and no training process was signalled. Runtime repair provenance is
bound in:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/long_horizon_waiter_restart_after_10k.json
```

The local successor source was then hardened so
`run_progress_audit` constructs the child environment itself: it prepends
`<project_root>/src` to `PYTHONPATH` and preserves inherited entries. Two
regression cases cover absent and existing parent paths. Verification passed:

```text
targeted: 20/20 passed
full:     666 collected, 664 passed, 2 skipped
source:   13,222 bytes
SHA256:   9f9d9768f09b9a35977244244408f7df9c6e402b179decc087b24afc6aaa5988
```

The active remote observer remains on the earlier source SHA with the
corrected launch environment because it is healthy. This local hardening is
not deployed into the frozen formal worktree and does not alter the active
observer mid-run. Bound evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/observer_self_contained_pythonpath_hardening.json
```

The complete local pytest entrypoint had the same class of hidden environment
dependency: `pyproject.toml` declared only `pythonpath = ["src"]`, so a clean
`uv run pytest -q` could import `cofitok` but produced 51 collection errors
for repository-level `scripts` imports. The pytest configuration now declares:

```toml
pythonpath = [".", "src"]
```

With the parent `PYTHONPATH` explicitly removed, the observer plus tracked
progress-auditor subset passed `31/31`, and the complete suite passed:

```text
666 collected
664 passed
2 skipped
0 failed
0 collection errors
```

This is a local reproducibility fix only; the formal remote worktree remains
frozen. Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/pytest_pythonpath_reproducibility_hardening.json
```

## Post-training transition preflight

A separate read-only audit checked the frozen 50K-to-promotion transition
before either training finishes. Local and remote SHA256 matched for the
completion pipeline, 10% post-eval runbook, workspace path resolver, and gate
builder. The frozen resolver returned only the authoritative v3 paths:

```text
CoFiTok:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3
Dense:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_dense_50k_v3
Reports:
  /root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3
Gate:
  /root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/promotion_gate.json
```

No historical v2 run or failed gate is referenced. Each formal runbook
activates `pf-vlm`, changes to the project root, and exports
`PYTHONPATH=src` before invoking downstream scripts, so the external observer
environment failure cannot propagate into the formal pair validator,
sampling, metrics, or gate stages.

The frozen post-eval remains bound to:

- exact step-50K checkpoints and required integrity audits;
- shared sampling-batch selection;
- 10,000 EMA DDIM-100 samples per method;
- CFG `1.5`, batched CFG, bf16, and exact minimum count 10,000;
- 1,024-image `t=500` mechanism evaluation;
- 64-image prefix diagnostic at budgets `1/2/4/8`;
- visual audit and the scaling gate;
- maximum relative FID and endpoint regressions of `5%`;
- maximum absolute FID `100`;
- ordered rank 1, exact zero-token, shuffle mismatch, and coarse-energy
  requirements.

The post-eval runbook may write a failed gate report, but the parent pipeline
then validates it and cannot enter full 300K without an explicit pass. No GPU
work or post-eval output was started by this preflight. Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/posteval_transition_preflight_after_step11000.json
```

## Dense transition readiness

After CoFiTok step 12K, the future dense transition was checked without
starting a second GPU workload:

- authoritative dense directory does not exist;
- no stale metrics, report, checkpoint, or resume state is present;
- the frozen config validator returned `pass`;
- the recomputed config-pair bytes matched the launch report at SHA256
  `e90c24f6fe45aaa0c17c93df7daf3ab4f799e734367df6ea7a4e017729ba3384`;
- matched data/diffusion/runtime/optimization sections had zero mismatch;
- shared backbone fields had zero mismatch;
- dense auxiliary loss list was empty;
- CoFiTok/dense parameters were `62,836,011 / 62,824,707`, a
  `0.017993%` gap.

The selected `64x1` runtime report at SHA256
`cb125609a28bf8a531c495b365c6e22a8ee9d2eb7e377ba4c2d538be1b63df0d`
contains completed real benchmarks for both methods under the same dataset
and runtime environment:

| Method | Mean optimizer step | Peak VRAM | Effective batch |
|---|---:|---:|---:|
| CoFiTok | `2.216154 s` | `56,182,542,336` bytes | `64` |
| Dense identity | `2.020304 s` | `55,393,513,984` bytes | `64` |

Neither benchmark wrote a checkpoint. Selection uses the slower method's
`2.216154 s` as the shared score, so the dense baseline does not receive a
different runtime choice. The supervisor remains unchanged and must still
wait for the exact CoFiTok 50K completion report before creating the dense
directory. Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/dense_transition_readiness_after_step12000.json
```

## Final evaluation and release chain preflight

The full-scale delivery path was also inspected before authorization or GPU
work. Local and frozen-remote SHA256 matched for the full post-eval runbook,
EMA export runbook and implementation, large-scale comparison builder, and
terminal completion auditor. Their inference-artifact, final-gate, and
completion-audit tests passed `120/120`.

After full matched 300K training, the formal evaluation is fixed to:

- step-300K CoFiTok and dense checkpoints;
- shared sampling-batch selection;
- exact 50,000 EMA samples per method;
- DDIM-250, CFG `1.5`, batched CFG, and bf16;
- 1,024-image EMA mechanism evaluation at `t=500`;
- 64-image CoFiTok prefix diagnostic at `1/2/4/8`;
- formal visual audit and large-scale comparison.

The full gate is stricter than the 10% promotion gate:

- relative FID regression no more than `5%`;
- absolute FID no more than `20`;
- endpoint regression no more than `5%`;
- precision and recall each at least `0.30`;
- precision and recall regressions each no more than `5%`;
- ordered rank 1, exact zero-token, shuffle mismatch, and coarse energy.

Only after the parent pipeline validates this gate does it export CoFiTok and
dense EMA-only inference artifacts. Each export binds the final release gate,
step-300K source checkpoint SHA, integrity sidecar, runtime environment and
training authorization. It must then pass:

- real-forward preflight with warmup and measured forwards;
- `infer_generation.py` smoke generation through the stable API;
- CoFiTok prefix budgets `1/8` and dense budget `1`;
- canonical RGB 256x256 PNG paths and SHA256.

Finally, the terminal completion auditor reopens the source reports and files.
It verifies exact 50K counts, immutable sampling manifests, atomic progress,
batch-invariant random streams, matched sampling environments, stable
inference API provenance, visual audit, comparison, final gate, artifact
integrity, preflight, smoke reports and PNG bytes. The pipeline writes
terminal `complete/pass` only after that audit succeeds.

Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/final_release_chain_preflight_after_step12000.json
```

Bound launch evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/long_horizon_waiter_launch.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/long_horizon_waiter_launch_status.json
```

## Step-6,000 post-checkpoint diagnostic

The first scheduled validation after checkpoint publication was inspected
because its value moved adversely:

```text
validation epsilon MSE: 0.03626135
relative to step 5K:    +17.11%
relative to step 4K:    +23.42%
relative to step 1K:     -9.48%
```

The tracked progress auditor separated this quality diagnostic from execution
health:

- status: `healthy`
- issues/warnings: empty
- validation events: `6/6`, logging complete
- latest checkpoint: step 5,000
- required step 5,000 checkpoint: present
- latest integrity: `verified`
- Git revision: clean `58d83bf`

Evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00006000_diagnostic.json
```

It is `3,111` bytes with SHA256
`93e74f800896705429200bf3c769b98cff9204977ba2aeb97d0100736c2a5f20`.
The adverse value does not justify hiding the event or declaring quality
success. It also does not by itself determine the 50K endpoint, so the frozen
run and unchanged 10K/25K/50K observer continue.

## Steps 7,000-10,000 and second checkpoint integrity

The next four fixed-validation events were:

| Step | Validation epsilon MSE | Relative to previous event |
|---:|---:|---:|
| 7,000 | `0.03119027` | `-13.98%` |
| 8,000 | `0.02415371` | `-22.56%` |
| 9,000 | `0.02835677` | `+17.40%` |
| 10,000 | `0.03104844` | `+9.49%` |

Step 8K was the lowest of the first ten scheduled events. Across the larger
window, the last-five-event mean was `0.03020211`, which was `11.15%` below
the first-five-event mean `0.03399296`. Step 10K was nearly unchanged from
step 5K (`+0.28%`) but `28.55%` above the 8K minimum. The series is therefore
finite and improved in its windowed mean, but plainly non-monotonic. It cannot
replace generated-sample evaluation.

After the observer runtime repair, the deployed tracked progress auditor
reported:

- status: `healthy`
- issues/warnings: empty
- validation events: `10/10`, logging complete
- required checkpoint steps: `[10000]`
- missing required steps: empty
- latest integrity: `verified`
- audit last step: `10,650`
- seconds per step: `2.221821`
- ETA at audit: `87,429` seconds

The exact step-10,000 training row was:

- training epsilon: `0.02715974`
- total loss: `0.04189920`
- validation epsilon MSE: `0.03104844`
- gradient norm: `0.05060176`
- samples seen: `640,000`

The second formal checkpoint trust boundary is:

```text
checkpoint: checkpoint_step_00010000.pt
bytes:      1,006,351,466
sha256:     1310dffc22b927946b0fd402c3abdc01623a710fa50b4176374c5168052e33e3
format:     1
step:       10,000
revision:   58d83bfce2770eab2565b8c89a5f9a06201a0c86
dataset:    97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741
runtime:    51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57
```

The integrity sidecar, `latest.json`, and tracked progress report matched on
all ten audited identity fields. The 1 GB checkpoint payload remains only on
the server. Local evidence:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00010000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/checkpoint_step_00010000.pt.integrity.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/latest_after_step_00010000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/long_horizon_waiter_after_step_00010000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step10000_validation_and_checkpoint.json
```

This proves a second reproducible recovery point. It does not satisfy the
matched 50K pair or the downstream formal 10K-sample promotion gate, so the
frozen CoFiTok run continues toward 25K and 50K.

## Step-15,000 third checkpoint integrity

The frozen formal run reached step 15,000 without a supervisor, watchdog,
monitor, or tracked-worktree fault. A fresh invocation of the tracked progress
auditor used:

```text
expected steps:              50,000
checkpoint interval:         5,000
evaluation interval:         1,000
required checkpoints:        5,000 / 10,000 / 15,000
integrity policy:             required
```

It returned:

- status: `healthy`
- issues/warnings: empty
- validation events: `15/15`, logging complete
- checkpoint steps: `[5000, 10000, 15000]`
- missing required checkpoints: empty
- latest integrity: `verified`
- seconds per step: `2.221710`
- remaining CoFiTok ETA at audit: `77,760s`

The exact step-15,000 row recorded `960,000` images seen, training epsilon
`0.02757408`, total loss `0.04086439`, gradient norm `0.07614650`, and fixed
validation epsilon MSE `0.03080699`. The 11K-15K validation values were:

```text
11K  0.03187243
12K  0.03147752
13K  0.02831916
14K  0.03288554
15K  0.03080699
```

Their mean was `0.03107233`, `8.59%` below the first-five mean. Step 15K was
`0.78%` below step 10K and `6.32%` below step 14K. The series remains finite
and non-monotonic; this is execution-health evidence, not a generated-sample
quality conclusion.

The third formal checkpoint trust boundary is:

```text
checkpoint: checkpoint_step_00015000.pt
bytes:      1,006,351,466
sha256:     29ad6fc611b93b431e9fdb9c589107edfd6254d044b115f8f4a3234b4e0e42d0
format:     1
step:       15,000
revision:   58d83bfce2770eab2565b8c89a5f9a06201a0c86
dataset:    97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741
runtime:    51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57
```

The checkpoint payload remains only on the server. Its sidecar,
`latest.json`, exact 301-row metrics snapshot, tracked progress report, and
structured interpretation were copied into the bounded local evidence pack:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00015000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/checkpoint_step_00015000.pt.integrity.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/latest_after_step_00015000.json
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_train_metrics_through_step_00015000.jsonl
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step15000_validation_and_checkpoint.json
```

This establishes 30% progress and a third reproducible recovery point for the
CoFiTok half of the 10% matched pair. Dense 50K, dual formal 10K sampling,
promotion, full matched 300K, dual formal 50K sampling, final gate, EMA
exports, and the terminal completion audit all remain mandatory.
