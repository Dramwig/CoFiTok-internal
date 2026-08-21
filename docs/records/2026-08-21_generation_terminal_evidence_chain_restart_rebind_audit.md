# Generation terminal evidence-chain restart rebind audit

Date: 2026-08-21 (Asia/Shanghai)

## Outcome

The active full-data matched 100K terminal evidence chain is correctly rebound
after the server-side execution restart. The two earlier failed status files
remain preserved as historical fail-closed evidence, but no live process reads
them. All active statistical, runtime, visual, exposure, distribution-support,
and terminal-system waiters are alive and waiting on their intended current
sources.

This is a chain-integrity pass, not a scientific quality pass. Terminal quality,
uncertainty, runtime, and system-claim outputs remain absent, so generation
advantage, runtime-efficiency, release, and full-300K claims remain disallowed.

## Live training boundary

At the audit snapshot CoFiTok had continued beyond its independently verified
90K checkpoint:

```text
CoFiTok step:      90,100 / 100,000
images seen:       5,766,400
dense step:        50,000 / 100,000
pair status/stage: running / cofitok_training
pair issues:       []
```

The 90K physical checkpoint audit was terminal `pass`; it bound payload SHA256
`81a919859df79bf7fdef6750602b877512ba683d32c1b3e8586d7f2ca7361954`
to the exact sidecar, `latest.json`, Git/data/runtime identities, canonical
metrics, and 5,760,000 images seen. The only GPU compute process was the active
CoFiTok trainer PID 3773.

## Statistical quality chain

The active chain remains serial and correctly source-bound:

| role | PID | revision | live state |
|---|---:|---|---|
| preceding matched uncertainty | 7817 | `f161fe3` | waiting for terminal bridge result |
| terminal 100K uncertainty | 7878 | `e9a424b` | waiting for terminal bridge result |
| claim qualification | 7937 | `c42ac96` | waiting for quality + uncertainty |
| claim-language guard | 7960 | `5fab079` | waiting for qualification |

The visual audit, terminal exposure audit, exposure-aware follow-up, and
distribution-support waiters are likewise alive on PIDs 7767, 7777, 7791, and
7804. Their CUDA visibility is disabled and they do not signal training or
other processes.

## Runtime and terminal-system rebind

The current runtime chain is:

```text
runtime fairness PID 75555 / revision 85bf1d6
  -> runtime claim PID 80962 / revision 0d1fa8d
  -> terminal system PID 81499 / revision 8ec9a09
```

Their deployment receipts are bound by SHA256:

```text
runtime fairness: 032e15155c8d1309b622692ee7ec105bbae3045b29fa0d43810232f28fa0bb3d
runtime claim:    0579e9c91bc36d797eb3f1d71178b9a9e657d918c39dc6a0a2791c15d373af43
terminal system:  e5f7eca693cf1156b5b1fbc5373c7f454fc14243709c077075232ddb9121337a
```

All three processes have parent PID 1, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, nice level 10, idle I/O priority, and no GPU compute row.
The runtime fairness source is the post-restart `2/2` orphan-aware waiter.

The deprecated unsuffixed runtime-claim and terminal-system status files remain
terminal failed, as expected. A `/proc` command-line audit found zero live
references to either old status, the old ENOSPC statuses, or the superseded
runtime receipt. The active terminal-system guard explicitly reads
`waiter_status.2026-08-21_source_rebind_v2.json` and writes
`waiter_status.2026-08-21_runtime_rebind_v2.json`.

The factorization follow-up supervisor waits on the canonical future
`terminal_system_claim_guard.json`, so it cannot mistake either preserved old
failure for a current terminal decision.

## Remaining evidence

The following outputs do not yet exist and remain mandatory:

- exact paired 100K quality result;
- terminal 10K matched uncertainty report;
- statistical claim qualification and language guard;
- runtime fairness final report with physical orphan coverage `2/2`;
- runtime claim guard;
- visual/class-fidelity terminal audit;
- terminal system claim guard.

Until those exact artifacts exist and pass, the allowed conclusion remains
limited to the established mechanism and prefix-control claims.

Machine-readable snapshot:

```text
artifacts/reports/generation/
terminal_evidence_chain_restart_rebind_audit_2026-08-21.json
```
