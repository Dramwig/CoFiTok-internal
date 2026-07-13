# Final comparison source provenance (2026-07-13)

## Finding

The schema-v4 comparison carried the key training, sampling, checkpoint,
real-set, evaluator, and protocol values. Completion independently recomputed
most cost and provenance fields, but the comparison did not content-address its
five matched source JSON files. Completion also did not compare the displayed
matched IS, precision, and recall values back to the metrics reports.

## Contract

Comparison schema v5 records authoritative path, byte count, and SHA256 for:

- CoFiTok and dense full 300K training reports;
- CoFiTok and dense formal 50K generation metrics reports;
- the final full-stage generation gate.

`verify_comparison_source_reports` rereads every bound file. The terminal
completion audit requires that verification and independently checks every
matched distribution metric, evaluator identity, sampling invocation count,
training and sampling cost field, checkpoint/sample/real-set hashes, formal
sampling protocol, and derived FID summary. Official D-AR/MAR/ReTok rows remain
in the separate contextual tier with cross-tier numeric ranking disabled.

## Verification

Targeted comparison and completion-audit tests cover out-of-range matched
metrics, changed source files, altered displayed metrics, stale protocol fields,
misreported compute, contextual-row identity drift, and cross-tier policy drift.
