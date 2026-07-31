# Stability 50K post-eval waiter v3

Date: 2026-07-31

## Problem

The immutable stability 50K training revision
`2c2c1f5166b73d4f28df93b276901671ac1a7836` runs
`build_generation_stability_50k_summary.py` after both matched training jobs.
That revision predates the direct-entry import fallback, so the final summary
command can fail under the runbook's `PYTHONPATH=src` environment before
creating `reports/pair_summary.json`.

The v2 post-evaluation waiter required that summary to exist. Even if the
authoritative monitor reached clean `pass/complete`, the waiter could
therefore wait forever after the training runbook failed at this CPU-only
reporting step.

## Recovery

Commit `08b67cc58026adf3e052a3eeab48b4bc0834cfd4` adds a bounded recovery to
the isolated waiter:

- No summary command runs until the exact training monitor reports
  `pass`, stage `complete`, and an empty issue list.
- A missing summary is built only with the clean evaluation checkout's bound
  `scripts/build_generation_stability_50k_summary.py`.
- CoFiTok and dense training reports, decision validation, config validation,
  expected training revision, and expected training branch are explicit
  arguments.
- The generated or pre-existing summary is accepted only when all four source
  paths, byte counts, and SHA256 values match the current source reports.
- Builder failure, malformed output, source drift, monitor failure, checkout
  drift, or a non-idle GPU fails closed.
- The waiter still cannot launch full 300K training.

The active training checkout and revision were not modified or moved.

## Validation

Local validation:

- Targeted waiter and post-evaluation tests: 15 passed.
- Full suite: passed with the existing 3 skips.

Linux isolated validation at
`/tmp/cofitok-stability-50k-posteval-08b67cc`:

- Branch: `scale/generation-stability-50k-posteval-v3`.
- Direct waiter and summary-builder CLI checks passed.
- Targeted tests: 15 passed.
- Full code suite excluding the sibling-paper layout file:
  786 collected, 783 passed, 3 skipped.
- All 95 shell runbooks passed `bash -n`.
- Checkout remained tracked-clean.
- Official repository HEAD remained
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

The incremental bundle required `aed2014` and advertised only `08b67cc`; it
was 19,603 bytes with SHA256
`7c8c24c72db0f9035d827d13b2c0791a3983af6a559e9c0aadeee2f179f8a7d7`.

## Controlled switch

Waiter v2 PID `900875` was verified in
`waiting_for_completed_training_pair` with no child process. Its status was
archived before it was terminated with `SIGTERM`.

Waiter v3 was then launched as PID `25861`. Its status binds:

- training:
  `2c2c1f5166b73d4f28df93b276901671ac1a7836` /
  `scale/generation-stability-50k-preflight`;
- evaluation:
  `08b67cc58026adf3e052a3eeab48b4bc0834cfd4` /
  `scale/generation-stability-50k-posteval-v3`;
- stability decision SHA256:
  `d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4`;
- `formal_300k_allowed=false`.

GPU process membership was exactly trainer PID `319202` before and after the
switch. The source-bound pair summary did not exist at switch time, and v3
correctly remained idle because the monitor was still
`running/cofitok_training/issues=[]`.

The archived receipt captured CoFiTok at step 2,950 and 188,800 images. A
follow-up read observed step 3,000 and 192,000 images, total loss
`0.05975177`, epsilon MSE `0.03922706`, rollout consistency scale `0.30`, and
scheduled validation epsilon MSE `0.03815802`.

## Completion boundary

This handoff closes a reporting deadlock; it is not a scientific promotion.
The matched 50K training pair, formal EMA sampling, mechanism gate, and
promotion gate remain incomplete. Full ImageNet-256 300K training remains
unauthorized.

Evidence:

`artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_posteval_waiter_v3/`
