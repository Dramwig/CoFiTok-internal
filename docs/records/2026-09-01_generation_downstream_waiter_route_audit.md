# Downstream waiter route audit (2026-09-01)

## Scope

This is a CPU-only, read-only audit of the downstream waiters attached to the
completed full-data 100K quality bridge. It does not send signals, restart a
waiter, create a checkout, create an output root, launch training or sampling,
or change locked evidence.

## Live source state

At `2026-09-01T17:04:48+08:00`, `pro6000` was reachable and the target GPU had
no compute applications. The standing authorization remained bound to
SHA256 `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.
The authoritative bridge checkout remained:

```text
revision cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree     6cef27723196fd363379bca2e7b85b1678ebd777
branch   scale/generation-stability-quality-bridge-100k
```

The terminal comparison and completion audit still report a scientific
`terminal_status=hold` and `generation_advantage_proven=false`.

## Waiter state

The following live processes are PPID 1, CPU-only (`CUDA_VISIBLE_DEVICES=-1`),
single-threaded, and low priority. Each has `child_pid=null` and remains in a
fail-closed waiting state:

| process | PID | status detail |
| --- | ---: | --- |
| terminal distribution support waiter | 7804 | `waiting_for_quality_bridge_followup_decision` |
| exposure-aware follow-up waiter | 644674 | `waiting_for_terminal_training_exposure_report` |
| factorization regression supervisor | 761338 | `waiting_for_quality_bridge_followup_decision` |
| conditioning ranking supervisor | 765582 | `waiting_for_quality_bridge_followup_decision` |
| random-token wrapper | 773804 | lock-held waiter; no child launched |

The factorization and conditioning supervisors still reference
`reports/followup_experiment_decision_exposure_aware_v2.json`,
`execution_authorization.json`, and `source_binding.json` paths that are
absent. The newer versioned follow-up decision exists separately at
`reports/followup_decision_authoritative_verifier_v3_20260824/followup_experiment_decision.json`
(23074 bytes, SHA256
`ae989b2e0b0f06a2b28346aae629f5368079a45a9a5b53d90c2d6a20c137175a`).

Their intended factorization, conditioning, and random-token output roots are
also absent. No downstream child has therefore started, and no new GPU work
is inferable from the live waiting processes.

## Decision

The old waiters must not be rebound by aliasing paths, restarted, or signaled
based on this audit. A future rebind would require a new exact source-bound
deployment receipt and the existing authorization boundary. The current
scientific hold, all training/sampling/300K/promotion/release permissions, and
the locked 100K evidence remain unchanged.
