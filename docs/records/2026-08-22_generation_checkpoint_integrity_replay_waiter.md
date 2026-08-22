# Generation checkpoint integrity replay waiter

Date: 2026-08-22

## Purpose

The full-data matched 100K quality bridge already has independent source-bound
physical checkpoint waiters queued for dense steps 90K, 95K, and 100K. Their
reports are strongest at creation time, but `train_metrics.jsonl` continues to
append and `latest.json` advances at the next checkpoint.

`scripts/wait_generation_checkpoint_integrity_audit_replays.py` waits for each
original audit/status pair to pass and then invokes the independently committed
checkpoint replay verifier. It publishes one immutable replay receipt per
milestone:

```text
dense_checkpoint_step_00090000_physical_integrity_audit_replay.json
dense_checkpoint_step_00095000_physical_integrity_audit_replay.json
dense_checkpoint_step_00100000_physical_integrity_audit_replay.json
```

## Safety and provenance

The waiter:

- verifies its own exact clean Git revision, tree, branch, and source SHA on
  every poll;
- verifies the exact clean training and original-auditor checkouts;
- requires the original waiter source SHA and canonical audit/status paths;
- rejects stale, failed, malformed, or mismatched source-waiter states;
- launches the exact replay verifier with CUDA hidden;
- inherits low CPU/I/O priority from its launcher;
- holds a non-blocking output lock so only one waiter instance can own the
  status target;
- validates already existing replay receipts by independently rerunning the
  immutable historical checks, allowing safe waiter restart;
- never loads a checkpoint through PyTorch and never signals training.

Every status and receipt remains permanently non-authorizing:

```text
sampling_authorization_allowed=false
promotion_authorization_allowed=false
release_authorization_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
```

This is checkpoint-integrity and reproducibility evidence only. It is not
generation-quality evidence and cannot change the terminal scientific decision.

## Local validation

The implementation was validated in the isolated worktree
`D:/cofitok-checkpoint-audit-replay` with a project-local Python 3.10
environment and CUDA unused:

- `py_compile`: pass for the waiter and its focused test module;
- focused waiter tests: `11/11` pass, including missing/live/stale/failed source
  states, detached auditor branch propagation, restart validation, output-lock
  contention, and `--once` waiting behavior;
- original auditor/verifier plus checkpoint/exact-resume regression group:
  `86/86` pass;
- full Windows collection: `1,106` tests. After rerunning the four paper-layout
  tests from the canonical project-root layout, the adjusted result is `1,099
  passed + 6 skipped + 1 known Windows CRLF deployment-receipt failure`.

The CRLF receipt failure predates this waiter and compares a Linux-deployed
source byte identity against the Windows checkout. The waiter, verifier,
checkpoint, resume, and reporting regressions all pass.
