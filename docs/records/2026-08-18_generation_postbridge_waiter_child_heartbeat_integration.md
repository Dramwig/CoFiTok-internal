# Post-bridge waiter child-heartbeat integration (2026-08-18)

## Outcome

The clean post-bridge integration target now retains fresh, source-bound status
while the frozen post-evaluation, frozen supplemental, and full-readiness
waiters own a long-running child stage. This closes the previously observed
coordination failure in which a healthy child could outlive the downstream
status-silence window even though the child itself continued normally.

This is future-facing control-plane hardening only. It was not deployed into,
and did not modify, the active full-data matched 100K execution.

## Exact implementation

```text
worktree: C:/qbintegration
branch: integration/generation-postbridge-hardening-v1
code revision: 9a90c4c629b53fafc87707105817fe6cebebf713
code tree: 9e0a0a44ee01c25f8bd5a6f1b454e1f9d42adac8
subject: Keep generation waiter child stages live
```

The implementation adds
`cofitok.process_monitoring.wait_for_child_with_heartbeat`. The helper waits
with a finite positive timeout and invokes the waiter's existing atomic status
writer after every `TimeoutExpired`. Non-finite, zero, and negative poll
intervals fail closed.

The helper is used by:

- `scripts/run_generation_stability_50k_posteval_waiter.py`;
- `scripts/run_generation_stability_frozen_supplemental_waiter.py`;
- `scripts/run_generation_stability_full_readiness_waiter.py`.

Each heartbeat republishes the same running detail, exact source identities,
non-authorizing boundary, and owned child PID as the initial running record.
Child exit-code handling and all scientific, launch, promotion, release, and
process-signaling decisions are unchanged.

## Validation

Using `C:/qbintegration/.venv/Scripts/python.exe` with CUDA hidden and one
OMP/MKL thread:

- focused process-monitoring and three-waiter suite: `31 passed`;
- complete repository suite: `1,646 collected`, `1,637 passed`, `9 skipped`,
  `0 failed`, `0 errors` in `738.882` seconds;
- Python `compileall`: pass;
- `git diff --check`: pass.

The full-suite JUnit file was `248,572` bytes with SHA256
`85c8b23b8669cd46ab3ca25628f35958899ce2a641d68a6b0bb54e8665a0f2c0`.

Machine-readable receipt:
`artifacts/reports/generation/postbridge_waiter_heartbeat_integration_2026-08-18/validation_receipt.json`.

## Active-run preservation

The read-only `pro6000` snapshot at `2026-08-18T16:03:31+08:00` showed:

```text
pair: running / cofitok_training
issues: []
CoFiTok metrics: 48,700 / 100,000
samples seen: 3,116,800
latest bound checkpoint: step 45,000 / metadata_verified
step-50K integrity sidecar: absent
dense run: absent
quality bridge result: absent
GPU compute: only trainer PID 619775, 85,286 MiB
unrelated GPU processes in the bound monitor: []
free bytes: 321,599,881,216
```

No trainer, controller, monitor, waiter, checkpoint, report, lock, remote
checkout, or GPU process was signaled, restarted, replaced, or modified.

## Deployment boundary

The earlier two-stage preflight for target `6cd02bb...` remains a historical,
non-authorizing receipt. It must not be reused for this newer target. Only after
the active 100K execution and controller are terminal may a fresh two-stage
preflight build bundles for a descendant containing `9a90c4c...`, re-evaluate
the formal checkout and its untracked conflicts, and require the explicit
external identical-conflict archive for any remaining byte-identical paths.

This change does not authorize training, sampling, evaluation, promotion,
release, export, full 300K, or any process signal.
