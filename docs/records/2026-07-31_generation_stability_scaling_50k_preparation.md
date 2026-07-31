# Generation stability scaling 50K preparation

Date: 2026-07-31

## Scope

This record prepares, but does not authorize or launch, the next matched 50K
ImageNet-256 10% run for the rollout-stability correction. The active matched
5K qualification remains pinned to
`59db142fc45d69dc92bb0333be5ac2d0162d9dc4`. Its final raw n=8 and two-seed
n=64 decision must pass before the successor runbook can execute.

The official server repository remains clean and unchanged at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`. The active 5K checkout also remains
clean and unchanged. No 50K or 300K process was started.

## Gap found

The repository previously contained only the old formal 50K recipe: the
two-channel final-tail layout, squared-Hellinger objective, and no rollout or
EMA-teacher consistency. Reusing those configs after the stability probe would
silently discard the mechanism that is currently being qualified.

The correction therefore needs a separately versioned recipe and run path. It
must preserve the immutable v3 stages while preventing a short-probe decision
from directly launching expensive training.

## Implementation

Commit `c6075159eb04fc855ba4df8bb56e9fb0463fef12` adds:

- Recipe schema `cofitok_generation_training_recipe_v4`.
- New `stability_scaling` and `stability_full` stages; historical
  `legacy_scaling`, `scaling`, and `full` semantics remain unchanged.
- Matched formal 50K configs:
  `configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json`
  and
  `configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json`.
- A `rgbtail3` CoFiTok layout with channel schedule `[4,4,8,8,8,1,1,1]`
  and spatial strides `[16,16,8,8,4,1,1,1]`.
- Shared two-step clipped-`x0` rollout consistency:
  `weight=0.1`, `start=0`, `warmup=10000`, `delta=10`,
  `batch_fraction=0.125`.
- Shared late EMA-teacher consistency:
  `weight=0.25`, `start=30000`, `warmup=10000`,
  `batch_fraction=0.0625`.
- CoFiTok-only Hellinger-stable energy and low-SNR high-frequency objectives.
- `scripts/validate_generation_stability_scaling_decision.py`, which verifies
  the expected decision SHA, rehashes all qualification sources, rebuilds the
  decision, and requires exact equality with the declared decision.

The formal parameter audit passes with:

```text
CoFiTok:       62,834,083
dense:         62,824,707
relative gap:  +0.014924%
recipe stage:  stability_scaling
recipe schema: cofitok_generation_training_recipe_v4
```

Commit `c8e3def25d179ffe325b329a2d9685302be62f1f` adds:

- Dormant gated runbook
  `artifacts/runbooks/generation_stability_ema_teacher_matched_50k_after_gate.sh`.
- Audited pair-summary builder
  `scripts/build_generation_stability_50k_summary.py`.

The runbook requires all of the following before CUDA benchmarking or training:

- Exact clean `EXPECTED_TARGET_REVISION`.
- Branch `scale/generation-stability`.
- Explicit `EXPECTED_STABILITY_DECISION_SHA256`.
- Source decision revision exactly `59db142...`.
- Successful source-rehashed and rebuilt 5K decision validation.
- Successful recipe-v4 `stability_scaling` pair validation.
- Storage headroom based on both final 5K checkpoints.
- Idle GPU.

It selects one matched effective-batch-64 runtime from `16x4`, `32x2`, and
`64x1`; starts a required-integrity pair monitor at 5K checkpoint cadence; and
allows exact resume only under the same target revision. The final pair summary
requires exactly 50,000 steps and 3,200,000 images per method, but writes
`formal_300k_authorization_allowed=false` and
`formal_ema_sampling_gate_required=true`.

## Verification

The incremental bundle from monitor revision `164c96e` to target `c8e3def` is:

```text
bytes:  19,223
sha256: dd2fc45d0eb3835eb1cd219b713abe72879aba37135ef20c1f98d0c7cc1f6d33
base:   164c96e71dd97f0ac85a4f906c5a86cf768b7390
head:   c8e3def25d179ffe325b329a2d9685302be62f1f
```

Remote `git bundle verify` confirmed the prerequisite and advertised only the
target HEAD. An isolated detached Linux checkout at
`/tmp/cofitok-stability-scaling-c8e3def` passed:

```text
targeted pytest: 35 passed
tracked runbook bash -n: 89/89 passed
tracked status: clean
```

This rehearsal did not fetch into or move the official repository, did not
modify the active 5K checkout, and did not execute the 50K runbook.

## Live qualification state

At the preparation snapshot, the active CoFiTok 5K member had reached direct
JSONL step 2,175 / 139,200 images. Both authoritative monitor and strict
provenance observer reported `running`, stage `cofitok_training`, with no
issues. The protected step-1,250 checkpoint, sidecar, `latest.json`, run
manifest, formal dataset identity, runtime environment, revision, and float32
rollout/teacher schedules remained verified. EMA teacher scale correctly
remained zero before start step 3,000.

The next evidence boundaries are protected checkpoints 2,500, 3,750, and 5,000;
3,750 is the first protected point after EMA-teacher activation. Dense training
then runs under the same pinned 5K revision, followed by raw n=8 and two-seed
n=64 qualification. Until that final decision exists and passes, formal 50K and
full 300K remain unauthorized.

## Final 5K qualification

The fresh matched 5K pair and its post-evaluation completed after the initial
preparation snapshot. The pair passed all four protected checkpoints, both
members completed exactly 5,000 steps and 320,000 images, and the n8 plus two
n64 seeds passed every configured mechanism gate. The final decision is:

```text
status:                pass
decision:              authorize_fresh_matched_50k_preparation
authorized next stage: fresh_matched_50k_preparation
source revision:       59db142fc45d69dc92bb0333be5ac2d0162d9dc4
decision SHA256:       d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4
```

Target revision `2c2c1f5` rehashed all qualification files and rebuilt the
decision exactly. The complete source evidence is recorded in:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/completion/raw_gate/
```

## Preparation-only preflight

The executable training target is fixed at
`2c2c1f5166b73d4f28df93b276901671ac1a7836`. All later local commits through
the evidence archive change only documentation and small reports; no
`src/`, `scripts/`, `configs/`, `artifacts/runbooks/`, tests, or packaging file
differs from this target.

An independent server worktree was created at:

```text
path:   /tmp/cofitok-stability-50k-preflight-2c2c1f5
branch: scale/generation-stability-50k-preflight
HEAD:   2c2c1f5166b73d4f28df93b276901671ac1a7836
```

Preparation-only checks passed:

```text
decision rehash/rebuild:          pass
recipe schema/stage:              v4 / stability_scaling
pair config contract:             pass
parameters CoFiTok/dense:         62,834,083 / 62,824,707
parameter gap:                    +0.014924%
storage free/required/headroom:   230,249,512,960 /
                                  118,385,312,804 /
                                  111,864,200,156 bytes
generation tests:                 746 passed, 2 existing skips
tracked runbook syntax:           90/90 passed
```

Four AAAI LaTeX layout tests were excluded because a standalone `/tmp`
worktree has no sibling `paper/` directory; the complete generation test set
otherwise passed. The official repository remained at `1ebcc152`, the active
5K checkout remained clean at `59db142`, the GPU was idle at final inspection,
and `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher`
was not created.

The preparation receipt and source reports are stored in:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_preflight/
```

This receipt authorizes preparation only. It does not launch or authorize
matched 50K training by itself, and it does not authorize full 300K.

## Actual matched 50K launch

After the preparation receipt, the final 5K decision, target revision, clean
worktree, idle GPU, output-root absence, config contract, and storage headroom
were rechecked. The gated runbook was then launched from the isolated target
worktree at `2026-07-31 10:59:34 CST`:

```text
output root:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/
    stability_scaling_50k_ema_teacher
target revision:
  2c2c1f5166b73d4f28df93b276901671ac1a7836
target branch:
  scale/generation-stability-50k-preflight
runbook PID: 315094
monitor PID: 319121
watchdog PID: 319138
trainer PID: 319202
```

The runbook benchmarked the three matched effective-batch-64 candidates and
selected `micro_batch_size=64`, `gradient_accumulation_steps=1`. The selected
score was `2.77796852` seconds with a measured peak-memory fraction of
`0.55226109`; the runtime environment SHA256 is
`d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`.

The first authoritative monitor refresh reported:

```text
status/stage:          running / cofitok_training
issues:                []
manifest:              verified
monitor step/rows:     100 / 3
metrics snapshot:      step 150 / 9,600 images / 4 rows
rollout schedule:      3/3 checked, no mismatch or missing step
EMA-teacher schedule:  3/3 checked, no mismatch or missing step
GPU:                   76,043 / 97,887 MiB, 99% utilization
```

At step 150 the rollout scale was the expected float32 warmup value
`0.014999999664723873`; EMA-teacher scale was correctly zero before step
30,000. The watchdog reported `running / child_and_monitor_active`. No
checkpoint was expected or available yet because the first required-integrity
boundary is step 5,000.

The official repository remained clean and unchanged at `1ebcc152`; the
training worktree remained clean at `2c2c1f5`. No second launch was attempted.
The launch receipt and all small source snapshots are stored in:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_launch/
```

This is an early-running receipt, not a completion or quality result. CoFiTok
must finish 50,000 steps before the matched dense member starts. Both members
must then pass checkpoint integrity, formal EMA sampling, and the mechanism
gate before any full ImageNet-256 scaling decision. Full 300K remains
unauthorized.
