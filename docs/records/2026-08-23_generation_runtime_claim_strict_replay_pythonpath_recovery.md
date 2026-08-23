# Runtime-claim strict replay Python-path recovery (2026-08-23)

## Incident

The first `runtime_compute_claim_guard_strict_replay_v2` wrapper reached the
pinned strict runner only after the canonical runtime waiter had exited, but the
wrapper omitted the project root from `PYTHONPATH`. The runner therefore failed
before argument parsing and before it could create a status, guard, lock, PID
file, or deployment receipt:

```text
ModuleNotFoundError: No module named 'scripts'
```

The canonical runtime guard remained `pass` with the observational-only policy,
and the terminal GPU sampling chain was not affected. The missing strict replay
is supplementary evidence and cannot authorize quality, training, sampling,
promotion, release, or full 300K execution.

## Recovery boundary

`scripts/recover_generation_runtime_claim_guard_strict_replay.py` provides one
versioned, fail-closed recovery path. It requires:

- the exact original wrapper receipt and failure-log SHA256;
- the original wrapper PID to be absent;
- the original strict status and guard to be absent before recovery;
- the canonical runtime waiter status to remain the exact completed source;
- the strict checkout, runner, builder, and recovery checkout to be clean and
  pinned by revision, tree, branch, and source SHA256;
- a new `runtime_compute_claim_guard_strict_replay_v3` output root with no
  status, guard, PID, or deployment artifacts;
- detached PPID 1, empty `CUDA_VISIBLE_DEVICES`, OMP/MKL=1, nice=10, and
  `ionice=idle`.

Before the recovery execs the strict runner, it materializes the original v2
import failure as a non-authorizing failure status. This lets the v1 comparison
and terminal-conjunct waiters exit naturally without signals. It then writes a
new wrapper receipt that binds the old receipt, failure log, failure status,
canonical status, recovery source, strict runner/builder, exact runtime
identity, and runner argv SHA256. The process uses `os.execve`, so the wrapper
PID in the receipt is exactly the strict waiter PID.

The v2 runtime comparison must use `--require-recovery-binding`. Its builder
physically reopens and hashes every recovery source, checks the old guard is
still absent, and rejects any PID, target, source, or policy drift.

## Authorization boundary

This recovery is CPU-only and permanently non-authorizing. It sends no process
signals, does not overwrite v1/v2 evidence, and cannot enable direct wall-clock,
throughput, FLOP, GPU-hour, cost-efficiency, quality, promotion, release, or
full-300K claims. Recovery-adjusted elapsed values remain observational physical
lower bounds including orphaned recovery compute.
