# Matched Min-SNR Terminal Physical Replay Guard

Date: 2026-08-25

## Scope

This record defines the one-shot, CPU-only physical replay guard for the
matched Min-SNR gamma=5 50K pilot. The guard is intentionally separate from
the active controller and is not a waiter. It must not be run until the
controller has produced the completed four-arm terminal result.

Canonical output:

```text
<pilot-output>/reports/terminal_physical_guard_v1/terminal_physical_guard.json
```

The output is permanently non-authorizing. Training, sampling, evaluation,
continuation beyond 50K, full 300K, promotion, export, release, process
signals, and GPU execution remain false. A successful replay does not change
`terminal_status=hold` or `generation_advantage_proven=false` and cannot make
a continuation decision.

## Evidence replayed

The guard fails closed unless all of the following replay twice to identical
evidence:

- `min_snr_pilot_result.json` is rebuilt exactly from every bound source
  report.
- The preparation is rebuilt exactly from its ten immutable sources, and the
  five constrained training-semantic files are physically rehashed.
- Both frozen legacy gamma=0 controls are replayed through their immutable 50K
  audit, checkpoint payload and sidecar. The audit-embedded historical
  `latest.json` bytes and 50K metrics target row are verified without falsely
  requiring the later 100K canonical files to remain at their 50K state.
- Both pilot gamma=5 controls are replayed through their 50K physical audit,
  resolved config, checkpoint payload and sidecar, `latest.json`, and metrics.
- Every metric row is exposure-bound and every numeric field is finite. Pilot
  rows additionally require `epsilon <= epsilon_unweighted` throughout.
- All four 10K DDIM-100 arms replay the immutable sampling manifest, completed
  progress, flat numbered PNG set and sample-set SHA256, physical checkpoint,
  generation metrics, class-fidelity report, and checkpoint-evaluation
  manifest/report.
- All four arms use the canonical ImageNet-256 validation tree and the same
  physically hashed ResNet50 ImageNet1K-v2 classifier weights.
- Legacy arms match the frozen preparation controls; pilot arms match their
  corresponding physical training audits.
- Unique checkpoint payloads, terminal reports, classifier weights, and guard
  source files are rehashed after replay to detect source drift.

The implementation binds the unchanged training replay modules to their exact
Git blobs at training revision
`842a34130e82f241330707118a05bf6ed01e263e`. The guard implementation itself
must run from a committed, clean descendant revision. Its CLI requires:

```text
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

## Verification

Local verification in the isolated D: worktree includes:

- Python compilation for the guard, builder, verifier, and focused test file.
- 11 focused tests covering authorization invariants, replay drift, source
  drift, checkpoint and classifier tampering, preparation-to-legacy-control
  binding, all-field numeric finiteness, sampling progress finiteness, and PNG
  sample-set tampering.
- Related protocol regressions for the Min-SNR result, generation metrics,
  class fidelity, and checkpoint evaluation.
- Builder and verifier CLI help checks with the isolated worktree on
  `PYTHONPATH`.
- A read-only remote check reconstructed both legacy 50K historical
  `latest.json` identities byte-for-byte and SHA256-for-SHA256. Their current
  canonical files are one byte larger because they correctly advanced to
  100K, confirming why the guard must distinguish immutable milestone
  evidence from later mutable pointers.

No guard process has been deployed and no terminal evidence has been replayed:
the four terminal arms do not exist yet. The active pilot controller and its
single GPU trainer were not modified, signaled, paused, or restarted during
this work.
