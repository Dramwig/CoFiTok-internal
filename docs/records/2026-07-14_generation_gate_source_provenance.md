# Generation gate source provenance (2026-07-14)

## Finding

Promotion and final gate JSON files were semantically fail-closed, but they did
not bind the files used to build them. The completion pipeline skipped 10K
post-evaluation whenever `promotion_gate.json` existed and skipped full
post-evaluation when gate/comparison/visual files existed. A source training,
metrics, or mechanism report could be replaced after gate creation while the
old gate remained internally valid. Terminal cross-checks caught some full-stage
differences, but only after expensive downstream work and at a non-retryable
completion stage.

## Source contract

Every CLI-generated scaling or full gate now binds exactly six authoritative
reports:

- CoFiTok and dense training reports;
- CoFiTok and dense distribution-metrics reports;
- CoFiTok and dense checkpoint-mechanism evaluation reports.

Each binding records an absolute authoritative path, positive byte count, and
lowercase SHA256. Stage-specific suffixes prevent substituting a report from the
wrong run, sample budget, checkpoint evaluator, or method.

`validate_generation_gate_report.py` reopens and hashes all six files before
scientific authorization. Its `--sources-only` mode does not require a passing
quality decision. The completion pipeline uses that distinction as follows:

- missing or source-stale gate evidence reruns the recoverable post-evaluation
  stage;
- a source-valid `fail/hold` skips redundant sampling and stops at the
  non-retryable scientific gate;
- a source-valid `pass` may authorize the next stage.

The terminal completion audit receives a fresh source verification for both the
scaling and final gates and requires it to equal each gate's embedded identities.

The source verifier was subsequently moved from the report-builder script into
`cofitok.generation_gate_sources` so authorization consumers can enforce the
same contract without depending on a CLI module. Formal training capture now
rehashes the scaling sources at every start/resume, final release capture does
the same before EMA export deserializes a checkpoint, and full 50K post-eval
revalidates both the gate and matched training pair before GPU evaluation. See
`docs/records/2026-07-14_generation_authorization_source_freshness.md`.

## Verification

Tests cover six-source construction, authoritative suffix rejection, byte-level
tampering, source-only acceptance of a quality hold, transition-runbook reuse,
and terminal scaling/final verification drift. Full local and isolated Linux
suite counts are recorded after final validation.

- implementation commit: `e34e523223e9e04399b6032638e9b6ad0356dfc3`;
- complete local suite: `563/563` passed;
- complete isolated Linux suite in the real sibling-`paper/` layout: `563/563`
  passed;
- tracked Linux runbook syntax audit: `40/40` passed;
- formal remote target-added-path scan: `152` paths, `0` conflicts;
- pinned `781a01444fddbf0d48a427ba58bdeed50167b5be` to implementation
  bundle: `426,122` bytes, SHA256
  `3af69c5726cec701191db75caec172fbc6fa3d87f6d7074cd0742cda09ead087`;
- formal remote HEAD remained pinned at
  `781a01444fddbf0d48a427ba58bdeed50167b5be` with a clean tracked
  worktree before and after verification.
