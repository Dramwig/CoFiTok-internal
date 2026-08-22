# Exact-resume history integration for terminal completion

Date: 2026-08-23

## Scope

This change closes a provenance gap in the future large-scale generation
completion path. Before this integration, the trajectory auditor consumed the
append-only `metrics_resume_history`, while the final large-scale and stability
completion auditors could accept canonical final metrics and physically verified
final checkpoints without proving that every exact-resume reconciliation event
was durably represented.

The active full-data 100K quality bridge remains immutable legacy evidence. This
change was developed only in the isolated D: worktree and was not deployed into
the active training checkout.

## Implementation

Code commit:

```text
revision: 1fad3c5d8fcdd26359e0745f9f13e3c0cfb5aee1
tree:     39e1cd341d675da82c8c213657cf01ed49dd93df
branch:   integration/generation-resume-history-trajectory-v1-20260823
```

The shared `verify_metrics_resume_history_evidence` verifier now checks:

- the embedded history digest and schema;
- exact equality between `run_manifest.json` and the independent
  `metrics_resume_history.json` journal;
- exact equality with the final `training_report.json` history;
- immutable reconciliation-report bytes and SHA256;
- orphan-archive bytes and SHA256;
- canonical metrics-prefix bytes, SHA256, and retained-row counts;
- checkpoint filename/bytes/SHA and integrity-sidecar bindings for every
  checkpoint-bound resume event;
- exact binding of the current manifest and training report to the latest
  history event;
- explicit `legacy_history_complete` handling.

The verifier deliberately records
`checkpoint_payload_sha256_recomputed=false`. Large payload hashing remains the
responsibility of the independent physical checkpoint auditors; this verifier
does not duplicate multi-gigabyte checkpoint reads.

Both `audit_large_scale_generation_completion.py` and
`audit_generation_stability_completion.py` now fail closed unless both future
full 300K runs have complete checkpoint-bound histories. Legacy or absent history
is reported as incomplete and cannot be upgraded into a complete recovery-chain
claim.

`build_generation_matched_training_trajectory.py` now uses the same shared
verifier instead of maintaining a second copy of the history-resolution logic.

Checkpoint pruning now preserves the small integrity sidecar for any checkpoint
referenced by the append-only resume history, even if the large payload itself is
rotated out. This keeps the durable checkpoint identity auditable without
retaining every recovery payload.

## Verification

Targeted local regression:

```text
tests/test_generation_system.py
tests/test_generation_exact_resume.py
tests/test_generation_metrics_resume.py
tests/test_build_generation_matched_training_trajectory.py
tests/test_large_scale_generation_completion_audit.py
tests/test_generation_stability_completion_audit.py

result: pass (two existing environment-dependent skips)
```

Full local suite:

```text
1113 passed, 6 skipped, 4 failed in 328.15s
```

All four failures are the existing AAAI paper-layout tests resolving the paper
root as `D:/paper/...` because this isolated worktree is located directly under
`D:/`. The missing paths are outside `CoFiTok-internal`; no failure touched the
generation, training, checkpoint, resume-history, trajectory, or completion
code.

`python -m compileall -q src scripts` and `git diff --check` both passed.

## Authorization boundary

- No active checkout was modified.
- No GPU process was started, stopped, signaled, or duplicated.
- No 300K training, promotion, or release was authorized.
- `generation_advantage_proven=false` remains unchanged.
- The current 100K bridge is legacy and must not be described as having a
  complete append-only resume history merely because this future code exists.

