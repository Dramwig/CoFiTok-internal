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

The live bridge later exposed an additional boundary case after the resumed
segment wrote its next checkpoint: the exact reconciliation remained bound to
the step-80,000 resume checkpoint, while `latest.json` correctly advanced to
step 85,000. Requiring `resume_step == latest_checkpoint_step` therefore made
the opt-in route invalid again even though both checkpoints and the complete
canonical metrics chain were healthy. The integration now accepts a verified
reconciliation when:

```text
stale_report_step < reconciliation.resume_step <= latest_verified_checkpoint_step
```

This preserves the exact resume boundary while allowing later atomic
checkpoints from the same canonical resumed segment. It does not infer a
resume from a checkpoint alone: the content-addressed reconciliation artifact,
canonical metrics prefix, orphan archive, SHA256, and stale-report bindings are
still all mandatory.

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

The post-85K boundary fix added positive and negative regression tests. The
focused auditor file returned `28 passed`; the
auditor/quality-bridge/runbook group returned `45 passed` with
`CUDA_VISIBLE_DEVICES=-1`. A read-only server audit
against the live bridge then reported `healthy`, no issues, canonical step
86,050, verified latest checkpoint step 85,000, and accepted reconciliation
step 80,000. The snapshot SHA256 is
`c1df033c329f475a9302d1c89ba81b6ea18398cd8f4e201402406c5d82652879`.
