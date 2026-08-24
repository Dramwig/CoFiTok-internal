# Shared semantic residual-alignment objective (2026-08-25)

## Scientific basis

The matched full-data 100K terminal evaluation remains a hold. CoFiTok improves
matched FID relative to `dense_identity`, but both systems fail absolute quality,
recall, and class fidelity. The canonical terminal screen reports CoFiTok FID
`115.26217262363417` versus dense FID `123.02103114594166`, while recall is only
`0.008320000022649765` and `0.009999999776482582`, respectively.
`generation_advantage_proven` therefore remains false.

The completed matched four-arm class-conditioning ranking screen also failed for
both methods. Its canonical postevaluation recommends
`revise_training_time_semantic_alignment_objective`, disallows a CoFiTok-specific
advantage claim, and does not support shared recovery from the ranking hinge. The
source report is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_probe1k_terminal_rebind_v2/reports/conditioning_ranking_posteval_v1/postevaluation.json
SHA256: 97f33bf774ad4537a674a2d37af3e50f7f03b6e1283f7644f6c629be430a08bb
```

This change prepares a different shared training objective. It is an engineering
candidate only, not new generation evidence.

## Objective

For a deterministic high-timestep subset of each micro-batch, the model predicts
`x0` under the correct class, two wrong classes, and the classifier-free null
condition. Define:

```text
target_residual = clean_x0 - stopgrad(null_x0)
correct_update  = correct_x0 - stopgrad(null_x0)
wrong_update_i  = stopgrad(wrong_x0_i - null_x0)
```

The objective averages three low-frequency scales. It combines:

- cosine direction loss between `correct_update` and `target_residual`;
- a temperature-smoothed contrastive margin against both wrong-label updates;
- an optional normalized correct-`x0` reconstruction term.

Only the correct-label branch is differentiable. Null and wrong-label branches
run under `torch.no_grad()` and are detached. All auxiliary forwards use model
evaluation mode so random class dropout cannot alter the compared conditions;
the prior model mode is restored in `finally`, including exception paths. The
eligible subset is deterministic and consumes no additional RNG, preserving
exact-resume trajectories.

Class and timestep information remain inputs to the predictor `T_k` only. The
restricted synthesis operator `S_k` is architecturally unchanged, receives only
the current token, remains condition-free, bias-free, linear, local, and
zero-preserving.

## Implementation identity

```text
branch: analysis/generation-semantic-residual-alignment-v2-20260825
base revision: 67a45d7961b1d3101cb6404afd5c13987636fe11
implementation revision: 6d1ac45035fcea203b92d715abd488c8ecdfd864
implementation tree: bbcc2ff4f45fd1e4762ed6691116b7f1d9513a43
```

The implementation adds strict configuration validation, loss accounting,
finite/range monitoring, progress-auditor schedule replay, exact-resume defaults,
and shared matched-pair fields. Existing formal recipes explicitly bind the
main residual-alignment weight to `0.0`; legacy checkpoints may omit the fields
only when all resolved values equal their disabled defaults. Ranking and
residual-alignment objectives cannot be enabled simultaneously.

## Matched 1K candidate configs

The prepared candidates are:

```text
configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classresidualalign_k8_probe1k.json
configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classresidualalign_dense_probe1k.json
```

Each candidate differs from its existing no-ranking 1K control only by its
descriptive name and these fields:

```text
weight: 0.05
start_step: 100
warmup_steps: 200
batch_fraction: 0.0625
margin: 0.1
temperature: 0.1
wrong_label_offsets: [1, 500]
min_timestep: 500
pooling_factors: [8, 16, 32]
reconstruction_weight: 0.25
```

The two candidates use the same data, augmentation, diffusion, optimizer,
runtime, seed, effective batch, rollout consistency, EMA-teacher consistency,
and semantic residual-alignment settings. Their pair contract passes. The only
method-specific differences remain the existing CoFiTok factorization fields
versus `dense_identity`.

## CPU verification

Validation used the project-specific `.venv` in the isolated D-drive worktree,
with pytest cache disabled and a D-drive `--basetemp`.

```text
candidate config + pair-contract suite: 15 passed
full collection: 1337 passed, 6 skipped, 4 failed in 560.57s
compileall: pass
git diff --check: pass
direct training-config validation for both candidates: pass
```

The four full-collection failures are the existing
`tests/test_aaai27_experiment_structure.py` cases. This standalone D-drive
worktree resolves their repository-external sibling paths as `D:/paper/...`,
which does not exist. All four fail only with `FileNotFoundError`; there are no
assertion failures. A proposed temporary junction was rejected by the local
safety policy before execution, so no junction or paper file was created or
removed. The other 1,343 outcomes completed normally.

Tests cover the numerical residual direction, correct-only gradients, detached
references, connected zero, mode restoration after failure, strict type/range
validation, loss scaling and metrics, exact segmented resume, legacy checkpoint
normalization, matched-pair mismatch detection, immutable formal-recipe
exclusion, monitor schedule replay, and progress-auditor fail-closed behavior.

## Non-authorizing boundary

This change did not create a runbook, preparation report, execution sentinel,
remote checkout, output root, controller, trainer, sampler, or evaluator. It did
not move any formal checkout or signal any process. No GPU experiment was
launched.

At the read-only 2026-08-25 check, `pro6000` had zero GPU compute processes,
`0 MiB` GPU memory in use, and `278641922048` bytes free under
`/root/autodl-tmp`. Both 100K training reports, all six 90K/95K/100K physical
audits, all three dense historical replays, runtime fairness, and runtime claim
guard still passed. The terminal hold and all authorization prohibitions remain
unchanged.

Any execution of these candidate configs requires a new exact-source-bound
execution boundary. This implementation commit alone authorizes no training,
sampling, 300K stage, promotion, export, or release.
