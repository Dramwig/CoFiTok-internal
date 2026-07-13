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

## Verification

Tests cover six-source construction, authoritative suffix rejection, byte-level
tampering, source-only acceptance of a quality hold, transition-runbook reuse,
and terminal scaling/final verification drift. Full local and isolated Linux
suite counts are recorded after final validation.
