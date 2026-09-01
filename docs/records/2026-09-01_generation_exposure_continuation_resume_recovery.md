# Exposure continuation interrupted-recovery hardening (2026-09-01)

## Scope

This record covers CPU-only control-plane hardening for the bounded 100K to
110K exposure continuation. It does not authorize a GPU stage, create a stage
authorization, modify the locked 100K evidence, or permit 300K escalation.

## Recovery contract

The continuation controller now accepts `--resume` only when all of the
following are true:

- the output root already exists as a real directory;
- `controller_status.json` is inside that root, is bound to the exact
  execution-authorization identity, and is still `running` or `failed`;
- the root has no terminal result or unexpected top-level entries;
- no matching controller, trainer, sampler, or evaluator is active; and
- every existing training `latest.json` resolves to a verified checkpoint in
  the bounded `[100000, 110000]` step interval.

An absent method starts only from its immutable 100K source checkpoint. A
partial method resumes with `--resume auto`; a completed method is skipped only
when its final report and execution Git identity match. Rollout output has no
resume contract, so an incomplete rollout is rejected rather than rerun.

The shell runbook keeps recovery opt-in through
`EXPOSURE_CONTINUATION_RESUME=true`; the default remains a fresh-root launch.

## Gate binding

The candidate execution gate schema is now `v2` and the exposure contract
binds `preserve_source_scheduler_horizon`, with source and effective horizon
100,000 and target horizon 110,000. Existing v1 gate artifacts must not be
reused; a fresh source-compatible candidate gate is required.

## Verification

- Exposure continuation, authorization, and exact-resume tests pass.
- Python compilation and `git diff --check` pass.
- The continuation runbook passes `bash -n` with Git Bash.
- No remote process, GPU stage, formal checkout, or locked artifact was
  changed.
