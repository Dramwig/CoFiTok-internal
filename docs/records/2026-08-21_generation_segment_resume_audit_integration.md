# Segment-resume training audit integration

Date: 2026-08-21 (Asia/Shanghai)

## Scope

The post-bridge integration branch incorporates commit
`3d85b9d31f641476cd18966ec42c00c48333bf82`, which adds a strict audit route for
segmented exact-resume training. It addresses the observed condition in which a
prior segment's `training_report.json` still names its segment endpoint while
canonical `train_metrics.jsonl` has advanced after an exact checkpoint resume.

The route is opt-in (`--allow-stale-segment-resume-report`) and accepts the
stale report only when all of the following are source-bound:

- the newer resume checkpoint is integrity-verified;
- the reconciliation filename and embedded `resume_step` agree;
- the reconciliation points to canonical `train_metrics.jsonl`;
- retained-row count matches the canonical prefix;
- the orphan archive exists and its SHA256 and row count match;
- the stale report is incomplete, targets the same run horizon, and points to
  the older checkpoint.

Without this exact binding, the audit remains `invalid`. The change is audit
and observability hardening only; it does not alter model weights, training
losses, sampling, promotion, release, or any 300K authorization.

## Validation

The clean integration worktree is:

```text
worktree: C:/qbintegration
branch: integration/generation-postbridge-hardening-v1
commit: 3d85b9d31f641476cd18966ec42c00c48333bf82
tree: 87144f47c74fb319b7a28ee574adc58e4db05f04
```

Targeted resume/audit tests passed (`34 passed`). The full CPU-only regression
suite was started from this exact clean commit with `CUDA_VISIBLE_DEVICES=''`
and `PYTHONPATH=.;src`; it reached 100% and exited with code 0 (no failed tests;
the test runner reported four skips). This is a local validation record only;
the integration commit was not deployed to the active remote execution.

The active remote quality-bridge execution remains pinned to
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` and was not modified.
