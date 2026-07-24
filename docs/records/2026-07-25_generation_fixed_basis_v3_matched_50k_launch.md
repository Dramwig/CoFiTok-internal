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
