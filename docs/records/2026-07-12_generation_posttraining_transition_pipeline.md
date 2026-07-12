# Generation post-training transition pipeline (2026-07-12)

## Purpose

The active matched ImageNet-256 10% queue must remain pinned to commit
`781a01444fddbf0d48a427ba58bdeed50167b5be` until both 50K-step trainings
finish. Sampling and evaluation require later integrity, preflight, and
provenance hardening commits. The transition therefore cannot be an in-place
code update while either training process is active.

## Deployment contract

`scripts/deploy_generation_posttraining_pipeline.ps1` is the only supported
transition command. It creates a bundle from the clean local
`scale/generative-system` HEAD and delegates the state-changing work to
`artifacts/runbooks/deploy_generation_posttraining_pipeline_remote.sh`. The
local command also transfers `scripts/validate_generation_training_pair.py`
to `/tmp`, so pre-deployment and post-deployment checks use the same testable
training-report contract even while the remote repository is still pinned.

The remote helper refuses deployment unless:

- remote HEAD is either the pinned training commit for a first deployment or
  the exact target commit for an idempotent retry;
- no tracked remote file is modified (historical untracked reports are
  permitted);
- both 10% training reports reached exactly 50,000 steps, identify the pinned
  revision and upgrade branch, report a clean tracked worktree, use matched
  data/diffusion/runtime/optimization sections, remain within the 2% parameter
  matching tolerance, and identify the exact step-50K checkpoint;
- neither the matched runbook nor either 10% training command is still active;
- the bundle resolves to the exact local target commit and can be applied with
  a fast-forward-only merge.
- no remote untracked file has the same path as a file tracked by the target
  revision; the helper computes this intersection before merge and reports all
  conflicts without moving HEAD.

After the fast-forward, the helper runs the full test suite and shell syntax
checks before launching one background completion pipeline. If deployment was
already fast-forwarded but validation or launch stopped, rerunning the same
command resumes from the target revision. A live PID file makes a repeated
command return successfully without launching a duplicate.

After tests and shell checks pass, the helper atomically writes
`generation_upgrade_deployment_receipt.json` under the generation output root.
The receipt binds the pinned training revision, deployed target revision,
upgrade bundle bytes/SHA256, training-pair validation SHA256, clean tracked Git
state, and successful pytest/runbook verification. The final completion audit
requires this receipt, so a merely changed remote HEAD is not accepted as
evidence of a controlled transition.

## Completion pipeline

`artifacts/runbooks/generation_complete_pipeline_after_10pct.sh` holds an
exclusive `flock` and advances through these stages:

1. validate the completed matched 10% pair;
2. migrate legacy checkpoint integrity, preflight inference, sample 10K images
   per method, compute metrics, and build the scaling gate;
3. require `promote_to_full_imagenet256`;
4. run alternating matched 50K/100K/200K/300K full-data training and milestone
   diagnostics;
5. run the formal matched 50K-sample DDIM-250 evaluation;
6. require the final `large_scale_generation_ready` decision.

Every stage atomically updates
`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_complete_pipeline_after_10pct.status.json`.
An exit trap records the failed stage and exit code. A scaling or final gate
hold is therefore visible and cannot be mistaken for system readiness.

The deployment command must not be run while the active matched 50K queue is
still training.
