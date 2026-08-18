# Conditioning-ranking matched 5K training confirmation preparation

Date: 2026-08-19

## Purpose

The source-bound four-arm 5K generated-sample validation can recommend
`prepare_separately_bound_matched_5k_training_recipe_confirmation` only when
both CoFiTok and dense identity show a paired requested-class improvement
without an excessive FID regression. Previously, that recommendation had no
implemented next-stage preparation contract.

This change adds the preparation layer for a fresh four-arm 5K training
confirmation. It deliberately does not add a GPU launch or claim authorization.

## Frozen stage

```text
stage: conditioning_ranking_four_arm_train5k_confirmation_v1
output root:
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_train5k_confirmation_v1
```

The four fresh runs are:

1. control CoFiTok K8;
2. ranked CoFiTok K8;
3. control dense identity;
4. ranked dense identity.

Every run uses `imagenet_256_10pct`, 5,000 optimizer steps, effective batch 64,
seed 2027, bf16, and protected checkpoints at 1,250, 2,500, and 5,000. The
ranked objective keeps weight `0.05`, margin `0.01`, wrong-label offset `500`,
and minimum timestep `500`; its start and warmup scale to steps 500 and 1,000.

Fresh training is mandatory. The step-1K probe checkpoints cannot be resumed
because their optimizer schedule and fully resolved runtime horizon are bound
to 1,000 steps. Reusing them would violate exact-resume config equality and
would change the intended 5K learning-rate trajectory.

## Source gate

Preparation accepts only the exact completed sampling-validation report from
revision `f77e311546beba697020debd35e26888096ca258` when:

- both method gates pass;
- shared generated class-alignment recovery is true;
- the recommended next action exactly selects matched 5K training recipe
  confirmation;
- every source claim and authorization flag remains false;
- the sample count is exactly 5,000 per arm.

Asymmetric, failed, malformed, authorizing, wrong-revision, or wrong-output
reports fail closed.

## Implemented files

```text
src/cofitok/generation/conditioning_ranking_training_confirmation.py
scripts/prepare_generation_conditioning_ranking_training_confirmation.py
configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classrank_k8_confirm5k.json
configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classrank_dense_confirm5k.json
tests/test_generation_conditioning_ranking_training_confirmation.py
```

The preparation binds the physical source report, standing authorization,
four exact config identities, parameter counts, clean builder revision, stage,
and output root. It records `gpu_execution_authorized=false` and
`training_allowed=false`; a separate source-bound execution receipt and
supervisor remain required before any GPU work can start.

## Latest Linux rehearsal

The hardened preparation commit was rehearsed on `pro6000` without touching
the formal checkout or exposing CUDA:

```text
revision: 9ec0d0bd3a714be2b7fa6f3f2008852c2d761329
tree: 8dbc8a31318068fe09b1e074afb6a2a1f9984b9e
prerequisite: f77e311546beba697020debd35e26888096ca258
bundle: /tmp/conditioning-confirm5k-9ec0d0b.bundle
bundle bytes: 11,663
bundle SHA256: 976889e4f6c7ddd11e1bddb91103c757cbc32bbb55d007f5aa2d7cf8bf7e60a2
isolated checkout: /tmp/cofitok-conditioning-confirm5k-rehearsal-9ec0d0b.Ygws4Q/CoFiTok-internal
CUDA_VISIBLE_DEVICES: -1
CPU/IO priority: nice 15, ionice idle class
related tests: 79 passed
post-test tracked/untracked status bytes: 0
verified at: 2026-08-19T03:11:30+08:00
```

The tested set covers the training-conditioning and ranking losses, the shared
pair contract, the 1K probe and post-evaluation chain, the matched 5K sampling
execution and validation chain, and the new 5K training-confirmation
preparation. Python compilation also passed for the new module, CLI, and test.

The formal remote checkout remained unchanged at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on `scale/generative-system` with
zero tracked-status bytes. The only GPU compute process remained the active
full-data 100K bridge training process, PID `79894`; the rehearsal performed no
GPU work and created no training output root.
