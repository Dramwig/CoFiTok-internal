# Quality-bridge downstream guard source rebind v2

Date: 2026-08-21 (Asia/Shanghai)

## Incident

The active full-data matched 100K quality bridge remained healthy after the
server restart, but the original runtime claim guard was still bound to the
superseded runtime-fairness waiter PID `7736` and its old deployment receipt.
The authoritative runtime-fairness waiter had been replaced by PID `75555`
after a second physical orphan archive was discovered at the 80K resume.

This exact-identity change correctly caused the original downstream chain to
fail closed:

- `reports/runtime_compute_claim_guard_v1/waiter_status.json` failed with
  `runtime fairness source waiter contract differs`;
- `reports/terminal_system_claim_guard_v1/waiter_status.json` then failed with
  `upstream_terminal_evidence_failed`.

The two failed status files remain immutable historical evidence. Neither was
overwritten or repurposed.

## Runtime claim guard rebind

The replacement uses the lower-bound-aware runtime claim implementation:

```text
checkout:
  /root/autodl-tmp/CoFiTok/checkouts/
  quality-bridge-runtime-claim-guard-rebind-0d1fa8d/CoFiTok-internal
revision: 0d1fa8db971c9b33b16f943f511efaf5e51b0ace
tree: 02bed3345634098b598f534721bd3c336eb41058
branch: analysis/generation-quality-bridge-runtime-claim-guard-rebind-v2
```

The implementation prevents direct wall-clock, throughput, or cost-efficiency
ranking whenever either method's recovery-adjusted elapsed time is only a
physical lower bound, even if GPU observation coverage is otherwise complete.

Source bundle:

```text
path: /tmp/cofitok-runtime-claim-guard-rebind-0d1fa8d.bundle
bytes: 12,242
SHA256: 74504e0c02531421c0309581bd05d4aa822873225fca1cc0104a20dabf65de91
```

The exact source was validated with `19 passed` on Windows and `19 passed` on
the Linux server with CUDA hidden. Python compile validation also passed.

Active CPU-only waiter:

```text
PID: 80962
parent PID: 1
nice: 10
ionice: idle
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
status:
  reports/runtime_compute_claim_guard_v1/
  waiter_status.2026-08-21_source_rebind_v2.json
deployment receipt:
  reports/runtime_compute_claim_guard_v1/
  deployment_receipt.2026-08-21_source_rebind_v2.json
receipt SHA256:
  0579e9c91bc36d797eb3f1d71178b9a9e657d918c39dc6a0a2791c15d373af43
canonical output:
  reports/runtime_compute_claim_guard_v1/runtime_compute_claim_guard.json
```

The waiter is bound to runtime-fairness PID `75555` and receipt SHA256
`032e15155c8d1309b622692ee7ec105bbae3045b29fa0d43810232f28fa0bb3d`.

## Terminal system guard rebind

The terminal system guard reuses its exact tested checkout:

```text
checkout:
  /root/autodl-tmp/CoFiTok/checkouts/
  terminal-system-guard-8ec9a09/CoFiTok-internal
revision: 8ec9a09ddcd981c1ffbd06b6bda81386b33321de
tree: 344f6365d2e962e350b73f5ce4bc18c005f6dff9
branch: analysis/generation-terminal-system-claim-guard-v1
```

The exact terminal guard suite passed `7 passed` on Windows and `7 passed` on
the Linux server with CUDA hidden; Python compile validation also passed. The
replacement waiter reads the versioned runtime-claim
status above while preserving the canonical terminal output expected by the
already-running downstream factorization supervisor.

```text
PID: 81499
parent PID: 1
nice: 10
ionice: idle
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
status:
  reports/terminal_system_claim_guard_v1/
  waiter_status.2026-08-21_runtime_rebind_v2.json
deployment receipt:
  reports/terminal_system_claim_guard_v1/
  deployment_receipt.2026-08-21_runtime_rebind_v2.json
receipt bytes: 14,310
receipt SHA256:
  e5f7eca693cf1156b5b1fbc5373c7f454fc14243709c077075232ddb9121337a
canonical output:
  reports/terminal_system_claim_guard_v1/terminal_system_claim_guard.json
```

At deployment both waiters were in a valid `waiting` state. Neither appeared
in `nvidia-smi`; the only GPU compute process remained the active CoFiTok
trainer.

## Live boundary

At `2026-08-21T18:53:58+08:00`:

```text
CoFiTok: 89,250 / 100,000
samples_seen: 5,712,000
dense: 50,000 / 100,000
90K checkpoint waiter: waiting / checkpoint_missing
```

This repair restores the terminal evidence path only. It does not establish a
generation-quality advantage, authorize sampling beyond the locked pipeline,
authorize promotion or release, or permit full 300K training.
