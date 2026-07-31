# Stability consistency schedule audit and post-eval waiter v4

Date: 2026-07-31

## Purpose

The active stability 50K recipe uses two scheduled stabilization losses:

- rollout consistency starts at step 0 and warms up over 10,000 steps;
- EMA-teacher consistency starts at step 30,000 and warms up over 10,000 steps.

The live monitor already checked scale fields against the immutable run
manifest. The post-training progress auditor did not independently bind those
fields and losses to the resolved training config. A future post-evaluation
could therefore miss schedule drift or a consistency loss that stayed zero
after activation.

## Implementation

Commit `c1efb12c6640f2d2d62ac7e9982c8804d96e7289` adds config-bound schedule
auditing to `scripts/audit_generation_training_progress.py`:

- `--config` loads the resolved experiment config and records its SHA256.
- Rollout and EMA-teacher scale fields must be finite and match the trainer's
  `consistency_weight_scale` function.
- A zero configured weight requires an effective logged scale of zero.
- Loss values must be finite and exactly zero while their schedule is disabled.
- Once a schedule is active, at least one audited row must have a nonzero loss;
  the report records active, verified, and nonzero-loss row counts.
- Legacy callers remain supported when `--config` is omitted.

The stability 50K post-evaluation, full 300K training, and full 50K
post-evaluation runbooks now pass the CoFiTok and dense configs to their own
training audits. Tests require the method/config pairing rather than only
counting `--config` occurrences.

## Validation

Local validation:

- schedule auditor and stability runbook tests: 34 passed;
- full suite passed twice with the existing 3 local environment skips;
- all 95 shell runbooks passed `bash -n`;
- `git diff --check` passed.

The incremental bundle:

- requires `08b67cc58026adf3e052a3eeab48b4bc0834cfd4`;
- advertises only `c1efb12c6640f2d2d62ac7e9982c8804d96e7289`;
- is 16,155 bytes;
- has SHA256
  `9c16b936b93daab2db8fdf3733d5da59a222f8feb8e7ac93c01ec51d40f647c7`.

Linux validation used an isolated checkout at
`/tmp/cofitok-stability-schedule-audit-c1efb12` with
`CUDA_VISIBLE_DEVICES=-1`:

- code suite excluding the sibling-paper layout file: 796 collected,
  794 passed, 2 CUDA skips;
- the 4 sibling-paper layout tests passed in a separate checkout with the real
  parent-directory layout;
- all 95 runbooks passed `bash -n`;
- the checkout remained tracked-clean;
- the official repository stayed at
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

## Live audit

The new auditor read the active CoFiTok metrics and checkpoint in place but
wrote only `/tmp/cofitok-c1efb12-live-schedule-audit.json`. It did not load the
checkpoint onto the GPU and did not mutate the run.

At step 6,700:

- overall status: `healthy`;
- checkpoint integrity: `verified`;
- metric rows: 135;
- validation events: 6 with complete provenance;
- rollout schedule: 135/135 active rows had nonzero loss, expected scale 0.67;
- EMA-teacher schedule: 0 active rows and scale/loss remained zero before its
  step-30,000 start;
- issues and warnings: empty.

The live audit SHA256 is
`385f3735374f7c52491fe6408af19e0fff0bcfd074adbb8e1c3d1224cd6b9bf2`.

## Controlled waiter switch

Waiter v3 PID `25861` was verified in
`waiting_for_completed_training_pair` with no child process. Its status was
archived as `posteval_waiter_v3_before_c1efb12.json` before termination.

Waiter v4 was launched from the clean isolated checkout as PID `281834`. Its
status binds:

- training revision/branch:
  `2c2c1f5166b73d4f28df93b276901671ac1a7836` /
  `scale/generation-stability-50k-preflight`;
- evaluation revision/branch:
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289` /
  `scale/generation-stability-50k-posteval-v4`;
- stability decision SHA256:
  `d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4`;
- `formal_300k_allowed=false`.

The old and new waiters had no child process. GPU compute membership was only
trainer PID `319202` before and after the switch. The official repository HEAD
and active training checkout were not moved or modified.

## Completion boundary

This change makes schedule activation and loss activity part of the
post-training evidence. It does not complete the matched 50K pair, run formal
EMA sampling, pass the mechanism/promotion gate, or authorize full ImageNet-256
300K training.

Evidence:

`artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_posteval_waiter_v4_schedule_audit/`