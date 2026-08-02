# Stability dense live-audit hardening (2026-08-02)

## Motivation

The active dense-identity 50K recovery already has a pair monitor, training
watchdog, recovery controller, post-evaluation waiter, and readiness waiter.
However, the recurring operational check still had to combine their reports
manually with `/proc`, `nvidia-smi`, the controller lock, and two different
storage questions:

- whether the active matched 50K run still has enough runway; and
- whether the larger downstream completion reserve remains available.

Those quantities diverged after an unrelated project created a roughly 9.72 GB
tar file. The active matched run remained safe while aggregate completion
headroom became negative. A single unlabelled `disk.free_bytes` field cannot
express that distinction.

## Read-only auditor

`scripts/audit_generation_stability_live.py` builds one bounded snapshot and
does not control any process. It:

- reads the authoritative pair monitor, dense metrics and manifest, watchdog,
  recovery status, post-evaluation waiter, and readiness waiter;
- computes each small source identity from the exact bytes it parsed, avoiding a
  mixed snapshot if an atomically replaced status file changes mid-audit;
- reuses `cofitok.monitoring.inspect_run` with required, metadata-only
  checkpoint integrity, so it does not load or hash checkpoint payloads;
- independently checks strict finite metrics and
  `samples_seen = step * effective_batch`;
- binds the clean controller, immutable training, evaluation, and readiness Git
  identities;
- requires exactly one root recovery controller, pair monitor, watchdog, GPU
  trainer leader, post-evaluation waiter, and readiness waiter;
- intersects trainer argv/cwd identity with the GPU compute PID set, so
  DataLoader children that inherit trainer argv are not counted as additional
  GPU trainers;
- verifies controller fd 6 and a nonblocking `flock` probe for
  `dense_recovery.lock`;
- records unrelated GPU compute identities and memory/SM observations without
  signalling them; and
- reports matched-50K and aggregate-completion storage headroom separately.

Aggregate runway failure is a warning while matched runway remains positive;
it never authorizes deletion or full training. A negative matched runway,
process multiplicity, identity drift, stale/invalid metrics, missing lock, or a
readiness waiter that sets `full_training_launch_allowed` to true is a failed
audit.

The report itself always records:

```text
read_only=true
process_control_performed=false
checkpoint_payload_hashes_performed=false
full_training_launch_allowed=false
```

## Verification

Focused tests cover:

- positive matched runway with negative aggregate runway;
- DataLoader-child exclusion from the GPU trainer count;
- duplicate GPU trainer and unheld-lock failure;
- PID change as a material warning only when the full role identity remains
  valid;
- finite/monotonic/exact-sample metric enforcement; and
- fail-closed rejection of a readiness waiter that authorizes full training.

Verification completed locally in the project `uv` environment:

```text
uv run pytest tests/test_generation_stability_live_audit.py \
  tests/test_generation_stability_dense_recovery_runbook.py \
  tests/test_monitor_generation_10pct_pair.py
32 passed

uv run pytest
904 passed, 6 skipped
```

This hardening does not change or deploy into the active immutable training
checkout, does not modify the active dense run, and does not authorize formal
post-evaluation or full 300K training.
