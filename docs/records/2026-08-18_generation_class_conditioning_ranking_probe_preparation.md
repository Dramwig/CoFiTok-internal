# Class-conditioning ranking objective and four-arm probe preparation

Date: 2026-08-18

## Scientific decision

The frozen matched conditioning diagnostics show that class information reaches
all conditioned U-Net blocks, but the final epsilon prediction does not
consistently prefer the correct label over the classifier-free null condition.
Increasing inference gain amplifies class-dependent directions without recovering
semantic denoising.  The next intervention therefore belongs in training rather
than CFG or inference gain.

This change adds a shared final-epsilon ranking objective to both CoFiTok and
`dense_identity`.  It does not alter token synthesis and does not pass class,
time, image, or previous-token information into `S_k`.

## Objective

For a deterministic high-timestep subset of each micro-batch, the model is
evaluated with the correct label, an offset wrong label, and the
classifier-free null label on the same noisy image and noise target.  Per-sample
epsilon MSE is computed for all three conditions.  The wrong and null branches
run under `torch.no_grad()` and are detached.  Only the correct-label branch
retains gradients.

The relative hinge is:

```text
0.5 * [
  relu((MSE_correct - stopgrad(MSE_wrong)) / stopgrad(MSE_wrong) + margin)
  +
  relu((MSE_correct - stopgrad(MSE_null)) / stopgrad(MSE_null) + margin)
]
```

The auxiliary forwards temporarily use model evaluation mode so random class
dropout cannot change the compared conditions.  The prior training/evaluation
mode is restored in `finally`.  The subset is the first deterministic eligible
indices with `t >= min_timestep`; the auxiliary path consumes no RNG, preserving
exact-resume trajectories.

Implemented fields:

```text
class_conditioning_ranking_weight
class_conditioning_ranking_start_step
class_conditioning_ranking_warmup_steps
class_conditioning_ranking_batch_fraction
class_conditioning_ranking_margin
class_conditioning_ranking_wrong_label_offset
class_conditioning_ranking_min_timestep
```

All seven fields are shared fields in `generation_pair_contract`.  Existing
formal recipes require the new objective to remain disabled.  Legacy checkpoint
configs may omit these fields only when the resolved values equal the exact
disabled defaults; any non-default value still fails exact resume.

## Metrics and auditing

Training metrics now include:

```text
class_conditioning_ranking
class_conditioning_ranking_scale
class_conditioning_correct_mse
class_conditioning_wrong_mse
class_conditioning_null_mse
class_conditioning_correct_better_wrong_fraction
class_conditioning_correct_better_null_fraction
class_conditioning_ranking_selected_fraction
```

The pair monitor, progress auditor, and matched trajectory builder validate the
new schedule.  Old runs without the disabled fields remain readable.  A zero
active hinge is allowed because it can mean that both margins are already
satisfied; this differs from rollout/EMA consistency audits that require a
nonzero active loss somewhere.

## Proposed GPU probe

The prepared experiment is a fresh four-arm ImageNet-256 10% probe:

```text
control CoFiTok   1,000 steps
ranked CoFiTok    1,000 steps
control dense     1,000 steps
ranked dense      1,000 steps
```

Within each method, control and ranked configs are identical after removing the
seven ranking fields and the descriptive run name.  Every arm uses seed 2027,
effective batch 64, the same data/augmentation/diffusion/runtime/optimizer, and
the existing rollout plus EMA-teacher stability recipe.  The ranked arms use:

```text
weight: 0.05
start_step: 100
warmup_steps: 200
batch_fraction: 0.0625
margin: 0.01
wrong_label_offset: 500
min_timestep: 500
```

The four-arm design is required so a shared semantic-training improvement is not
misreported as a CoFiTok-specific factorization advantage.

## Authorization boundary

The runbook is prepared but has not been launched.  It refuses any active GPU
compute process, requires an isolated fully clean checkout (including no
untracked files) at the exact approved
revision, recomputes the immutable preparation report, validates its SHA256,
and requires an exact execution-only approval sentinel.  The exact approval
text encoded by the validator is:

```text
Approve the non-authorizing four-arm 1000-step class-conditioning-ranking probe only.
```

That approval, if later supplied, authorizes only the four fresh 1K training
runs.  It does not authorize sampling, checkpoint promotion, follow-up 5K/50K
training, full training, release, or any modification of the active quality
bridge.

Prepared entry points:

```text
scripts/prepare_generation_conditioning_ranking_probe.py
scripts/validate_generation_conditioning_ranking_probe_approval.py
artifacts/runbooks/generation_conditioning_ranking_four_arm_probe1k_v1.sh
```

The active full-data quality bridge remains untouched and authoritative.
