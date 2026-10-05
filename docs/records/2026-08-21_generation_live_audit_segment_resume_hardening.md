# Live training audit: bound segment-resume report hardening

Date: 2026-08-21 (Asia/Shanghai)

## Finding

The active full-data quality bridge is intentionally segmented. After the
CoFiTok run resumed from the verified step-80,000 checkpoint, its canonical
`train_metrics.jsonl` advanced beyond the older segment's `training_report.json`
(`completed_steps=50000`, `training_complete=false`). The report itself is
not corrupt, but the generic live auditor classified the temporary mismatch as
`invalid` because the old report did not set `stop_requested=true`.

The active run remains healthy: the newest checkpoint sidecar, `latest.json`,
metrics schedule, validation provenance, and exact-resume reconciliation report
all verify. No training process or active execution checkout was changed to
address this observability issue.

## Isolated fix

An isolated worktree was created from the active bridge parent revision:

```text
worktree: C:/qblive
branch:   fix/generation-live-audit-segment-report-20260821
commit:   056073337efe7bf089d28bb7a7d0553e18c33d6a
```

The auditor now supports an explicit
`--allow-stale-segment-resume-report` mode. It accepts a stale incomplete
segment report only when all of the following are true:

- the newest checkpoint integrity sidecar is verified;
- the stale report's target step and checkpoint pointer are internally bound;
- a `metrics_resume_reconciliation_*.json` artifact names the exact newer
  resume step;
- the reconciliation artifact points to the canonical metrics file;
- its orphan archive exists, has the recorded SHA256, and has the recorded
  row count.

Unbound or malformed stale reports remain invalid. The existing
`--allow-stale-incomplete-training-report` behavior and its negative tests are
unchanged.

## Verification

Using the server's `pf-vlm` Python 3.10 environment, with CUDA hidden:

```text
test_audit_generation_training_progress.py: 26 passed
test_generation_quality_bridge.py:           15 passed
test_generation_runbook_entrypoints.py:       1 passed
total:                                      42 passed
py_compile:                                 pass
```

The first isolated live audit of the current bridge remains preserved as
`reports/live_training_audit_2026-08-21_1232/`. It showed dense `healthy` and
CoFiTok `invalid` only for the stale report mismatch; all checkpoint, schedule,
validation, and integrity checks passed. The new mode is ready for a future
revision-bound deployment, but it is deliberately not copied into the active
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` execution checkout during training.

Against the live bridge (read-only, server `pf-vlm` runtime), the candidate
mode returned:

```text
status: healthy
canonical step: 81450
issues: []
training_report: stale_segment_resume_report
resume_reconciliation: accepted
accepted resume step: 80000
```

The live candidate snapshot is mirrored as
`artifacts/reports/generation/stability_full_data_100k_base128_quality_bridge_v1/live_training_audit_2026-08-21_1232/cofitok_segment_bound_candidate.json`
with SHA256
`0ffcbabfa60ba70d9cfbc758820d2585e686e21c7660574459d10dabfbccd5492`.
