# Formal generation training recipe contract (2026-07-13)

## Finding

The matched pair contract proves that CoFiTok and dense share data, diffusion,
runtime, optimization, and U-Net fields. It did not prove that those shared
values were strong enough for the formal run. Both configs could therefore be
changed together, for example to disable class-dropout CFG training or weaken
EMA, and still appear fair before consuming the full 300K budget.

## Contract

`cofitok_generation_training_recipe_v1` defines separate `scaling` and `full`
recipes. It locks:

- ImageNet dataset stage, 256 resolution, class conditioning, and full-data
  horizontal-flip augmentation;
- cosine 1,000-step epsilon diffusion;
- the 128-base-channel `[1,2,3,4]` U-Net, attention, class dropout, and gradient
  checkpointing;
- bf16/TF32, exact horizon and checkpoint/evaluation cadence;
- effective batch 64 while allowing the measured `16x4/32x2/64x1` runtime
  selector to choose micro-batch and accumulation;
- AdamW schedule, clipping, EMA decay/warmup;
- K8 restricted synthesis and the formal denoise-path auxiliary objective;
- the K1 dense-identity control with no factorization auxiliary loss.

Pinned 10% reports predate explicit horizontal-flip and protected-checkpoint
fields. The scaling recipe accepts only their historical implicit defaults
(`0.0` and an empty protected-step list); the full recipe requires explicit
augmentation and four protected milestones.

## Enforcement

Checked-in config validation writes the recipe result before full training.
Predeployment 10% pair validation uses the scaling recipe. Completed full pair
validation and terminal completion audit use the full recipe, so two identically
weakened reports fail `full_matched_training` even when their fairness contract
still matches.

Tests cover the checked-in scaling/full configs, runtime-selected `32x2`, pinned
legacy defaults, simultaneous EMA/class-dropout weakening, changed CoFiTok
auxiliary loss, runbook CLI wiring, and terminal completion rejection.
