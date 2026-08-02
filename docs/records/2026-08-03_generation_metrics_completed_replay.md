# Generation metrics completed replay (2026-08-03)

## Problem

Formal stability and full post-evaluation compute torch-fidelity FID,
Inception Score, precision, and recall after generating 10K or 50K samples.
Sampling is exactly resumable, but the metrics entrypoint previously invoked
torch-fidelity again on every runbook restart and overwrote its only report.
This made a failure in a later gate, visual audit, comparison, or completion
stage unnecessarily repeat one of the most expensive non-training phases.

## Completed-replay contract

`evaluate_generation_metrics.py --resume` now distinguishes fresh,
completed, and invalid evidence:

- An empty metrics directory runs torch-fidelity normally and atomically
  publishes `generation_metrics_report.json` schema 3.
- A completed schema 3 report is reused only after the entrypoint independently
  re-enumerates and validates every generated PNG, recomputes the generated
  sample-set SHA256 and the physical real-set tree SHA256, and revalidates the
  immutable sampling report/manifest/progress chain.
- A report from an older schema, a rerun without `--resume`, an unexpected
  output file, a symlinked evidence path, or any source/request/environment
  drift fails closed instead of overwriting evidence.

The report binds:

- evaluator Git and captured runtime-environment SHA256;
- torch-fidelity package version and CUDA/CPU selection;
- canonical real, generated, sampling-report, output, and report paths;
- real/generated counts, real-set tree digest, and generated sample-set digest;
- sampling report, manifest, and completed progress file identities;
- batch size, PRC batch size, minimum sample count, seed, cache root and
  content-addressed real-cache name, and PRC enablement;
- finite metric domains and positive runtime provenance.

This is completed-result replay, not an unsupported claim that torch-fidelity
can resume inside its feature-extraction or PRC batches. If the evaluator dies
before the report is atomically published, the same validated request reruns
the metrics computation from the beginning.

## Formal callers

The following future formal paths now always pass `--resume`:

- stability matched 50K post-eval (10K DDIM-100 sample sets);
- full matched 300K post-eval (50K DDIM-250 sample sets);
- shared full-training milestone evaluation (2,048 DDIM-50 warning sets).

Locked legacy 10% outputs are not migrated, renamed, or rerun. Existing gate
and completion-audit source binding remains in force; a later gate still binds
the exact metrics-report bytes and SHA256.

## Scope and deployment boundary

This change improves evaluation recoverability and provenance. It does not
alter training, EMA weights, generated images, torch-fidelity metric formulas,
quality thresholds, or full-300K authorization. The active stability dense
trainer and its immutable post-eval/readiness waiters were not modified. The
new contract applies only after deployment through a separately bound clean
checkout and receipt.

## Verification

Tests cover fresh `--resume`, byte/mtime-preserving completed reuse, refusal to
overwrite without resume, request drift, physical real-set drift, malformed
metric domains, sampling-evidence identities, and all three formal runbook
callers. Gate, comparison, and terminal completion-audit regression tests also
remain green. Full repository pytest and Linux shell syntax results are
recorded in the implementing commit.
