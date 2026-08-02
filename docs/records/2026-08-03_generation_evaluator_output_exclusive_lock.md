# Generation evaluator output exclusive lock

Date: 2026-08-03

## Problem

The formal checkpoint diagnostic and torch-fidelity metrics evaluator already
have immutable manifests, completed-result replay, and source validation.  Those
contracts reject drift, but without component-level single-writer ownership two
matching invocations can both pass validation and race on one output report.

The cost is material:

- checkpoint evaluation can load a roughly 1 GB model and occupy the GPU before
  it observes output evidence;
- metrics evaluation can enumerate and hash the 50K real/generated trees and
  initialize FID/IS/precision/recall computation before discovering the race.

The top-level formal post-evaluation controller lock protects the current
runbook, but direct evaluator use and future orchestration must enforce the same
invariant at the component boundary.

## Change

`scripts/evaluate_generation_checkpoint.py` and
`scripts/evaluate_generation_metrics.py` now acquire the shared
`cofitok.output_lock.exclusive_output_lock` immediately after parsing the
request and before validation, directory creation, checkpoint loading, image
enumeration, hashing, runtime capture, or metric calculation.

Contention is non-blocking and raises `OutputLockError`; it never waits for,
signals, kills, or takes over the live writer.  The lock file remains adjacent
to the target output tree and is therefore outside the immutable evaluator
outputs and stage-receipt output declarations.  The existing manifest/report
schemas and completed-result replay behavior are unchanged.

## Verification

- Checkpoint-evaluator contention test proves the model loader is not reached
  and the output directory is not created.
- Metrics-evaluator contention test proves neither image enumeration nor
  torch-fidelity calculation is reached and the output directory is not
  created.
- Existing completed replay, provenance drift, symlink, source hashing,
  gate/comparison, and terminal completion-audit tests remain authoritative.
- Focused evaluator/output-lock/gate suite: `51 passed`.
- Full repository suite: `954 collected`, `948 passed`, `6 skipped`, exit code
  `0`.
- `py_compile` and `git diff --check`: passed.

This is execution-integrity evidence, not a sample-quality result.  It cannot
substitute for the formal 50K EMA samples or authorize full 300K training.
