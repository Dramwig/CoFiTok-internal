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
