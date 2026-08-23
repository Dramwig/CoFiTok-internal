# Terminal single-stream replication boundary: downstream consumers

Date: 2026-08-23

## Evidence interpretation

The full-data 100K terminal uncertainty analysis reuses the exact bound 10K
CoFiTok and dense sample sets. FID, paired block KID, bootstrap intervals, and
the sign test are multiple analyses of one terminal generation stream, not an
independently generated replication.

The upstream language and terminal-system guards record this as:

- `bound_terminal_stream_count=1`;
- `independent_replication_count=0`;
- `independent_replication_supported=false`; and
- `interpretation=paired_reanalysis_of_one_exact_bound_terminal_sample_stream`.

## Downstream changes

The strong-baseline comparison consumer now requires and propagates the exact
replication scope, forbids independent-replication and multiple-stream claim
permissions, and states the limitation in both JSON and Markdown output.

The terminal completion audit now independently checks the same terminal guard,
comparison JSON, comparison waiter status, and rendered comparison outputs. It
propagates the replication scope into the final audit and waiter status. Its
limitation text no longer uses the ambiguous phrase `independent pass`; an
operational audit pass only preserves the upstream scientific `pass` or `hold`.

All changed artifacts remain permanently non-authorizing. They do not authorize
training, sampling, 300K scaling, promotion, export, release, or process
signaling.

## Isolated commits

- Comparison branch:
  `analysis/generation-comparison-replication-boundary-v1`, commit
  `d478e85252019e01b4034f1498bd5f8493bc8088`.
- Completion branch:
  `analysis/generation-completion-replication-boundary-v1`, implementation
  commit `7ddfd59` (record commit follows this file).

The completion branch is based on the authoritative completion consumer base
`a253d56e49b78e1c8ddb10c3bd07af5aa22e5919` and also contains the comparison
classifier-integrity and replication-boundary consumer commits required for
exact comparison replay.

## Verification

Tests used an existing D:-hosted virtual environment with `TEMP`, `TMP`, and
`PYTHONPATH` redirected to D: because C: had no free space. No GPU was used.

```text
tests/test_generation_quality_bridge_comparison.py
tests/test_generation_quality_bridge_comparison_waiter.py
tests/test_generation_quality_bridge_terminal_completion_audit.py

68 passed
```

Python compilation and `git diff --check` also passed before commit.

## Live-chain boundary

Neither isolated branch has been deployed. No live controller, trainer, waiter,
lock, checkpoint, report, or terminal route was replaced or signaled. Canonical
runtime decisions remain governed by the existing remote chain, and
`generation_advantage_proven` remains false until the matched terminal evidence
and all claim guards complete.
