# Stability 50K post-evaluation provenance handoff

Date: 2026-07-31

## Problem found

The matched stability 50K training was correctly isolated on
`scale/generation-stability-50k-preflight`, but the shared generation gate
builder still hard-coded `scale/generative-system` for four provenance checks:

- Matched training revision and branch.
- Sampling code identity.
- Distribution-metric evaluator code identity.
- Checkpoint evaluator code identity.

Consequently, the formal stability post-evaluation could complete every
scientific operation and still fail provenance solely because it used the
intended isolated stability branches. Running from `scale/generative-system`
would not solve the issue because the stability pair summary must also match
the original training branch and revision.

## Correction

Commit `caab51348d546e98858d1203f2958d9e396e2d18` separates and explicitly
binds two identities:

```text
training:
  revision 2c2c1f5166b73d4f28df93b276901671ac1a7836
  branch   scale/generation-stability-50k-preflight
evaluation:
  revision caab51348d546e98858d1203f2958d9e396e2d18
  branch   scale/generation-stability-50k-posteval
```

`build_generation_gate_report.py` now accepts exact expected training and
evaluation revisions and branches. The old `scale/generative-system` defaults
remain unchanged for the locked legacy/full path. When explicit values are
provided, all matched training, sampling, metrics-evaluator, and
checkpoint-evaluator gates must use those exact clean identities.

The stability post-evaluation runbook now rebuilds the 50K pair summary against
the training identity while binding newly produced sampling and evaluation
reports to the evaluation identity. This allows report-only hardening after a
long training run without claiming that the checkpoint was trained by the
newer evaluation commit.

## Bound waiter

The new
`generation_stability_ema_teacher_50k_posteval_waiter.sh` delegates to a
structured Python waiter. It:

- Verifies the evaluation worktree remains at the exact clean revision and
  branch before waiting and immediately before launch.
- Accepts only the named training monitor at the exact clean training identity.
- Fails on `failed`, `stalled`, malformed, mismatched, or stale monitor state.
- Requires `pass/complete`, no monitor issues, and a source-bound pair summary
  for exactly 50,000 steps and 3,200,000 images per method.
- Requires the pair summary to retain
  `formal_300k_authorization_allowed=false` and
  `formal_ema_sampling_gate_required=true`.
- Waits for an idle GPU, then executes only the formal EMA 10K DDIM-100
  post-evaluation runbook.
- Atomically records waiting, running, pass, or failed state in
  `reports/posteval_waiter.json`.

The waiter cannot start training or invoke the full 300K runbook. Its `pass`
state means only that the post-evaluation runbook exited successfully. A
scientific scaling authorization still requires `promotion_gate.json` itself
to be `pass` and to pass the complete gate validator.

## Rehearsal and launch

The incremental bundle was:

```text
base:   2c2c1f5166b73d4f28df93b276901671ac1a7836
head:   caab51348d546e98858d1203f2958d9e396e2d18
bytes:  482,968
sha256: 08783631747bcc4eae186443f5c923a94afaa0c1b4098f675092314858b4075e
```

Remote bundle verification proved the prerequisite and a single advertised
head. The isolated Linux worktree
`/tmp/cofitok-stability-50k-posteval-caab513` passed:

```text
pytest:                 756 passed, 2 skipped
excluded layout tests:  4
tracked runbook bash-n: 91/91
tracked status:         clean
```

The four excluded tests require a sibling `paper/` directory that a standalone
inner-repository `/tmp` worktree does not contain. The complete local suite
passed `759` tests with `3` existing skips.

The waiter was launched as PID `442981`. Its first status was
`waiting / waiting_for_completed_training_pair`, with all training/evaluation
identities and the 5K decision SHA bound. At the concurrent snapshot, CoFiTok
training was at step 850, the authoritative monitor reported
`running/cofitok_training/issues=[]`, and the run manifest remained verified.

The active training worktree stayed at `2c2c1f5`; the official repository
stayed at `1ebcc152`. No checkpoint, training process, or full-scaling
authorization was modified by this handoff.

Small evidence is stored in:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/completion/
    stability_50k_posteval_handoff/
```
