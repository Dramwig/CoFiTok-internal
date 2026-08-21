# Quality-bridge runtime fairness orphan-set rebinding

Date: 2026-08-21

## Outcome

The CPU-only runtime/compute fairness waiter for the active full-data 100K
quality bridge was rebound before terminal evaluation so that its eventual
compute adjustment will cover every physical exact-resume orphan archive.
The active trainer, controller, checkpoint payloads, metrics trajectory, GPU
schedule, and downstream scientific decisions were not changed.

The replacement waiter is currently:

```text
PID:        75555
parent:     1
checkout:   /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-runtime-compute-fairness-85bf1d6/CoFiTok-internal
revision:   85bf1d6ae5b44f401153d0d1e390fc3973ff4bed
tree:       7900c5617c28c18e328c93aba4c547a3889f1aa7
source SHA: 0caba8d7af93c83c8862e18cc1e77e0a7d497d4d8289173942fa5f2118efdf57
status:     waiting_for_both_exact_100k_training_reports
```

It has `CUDA_VISIBLE_DEVICES=""`, OMP/MKL thread counts of one, `nice=10`,
and idle I/O priority. The only GPU process after rebinding remained the
quality-bridge trainer, PID `3773`.

## Why the old binding was insufficient

The previous waiter, PID `7736`, was correctly source-bound but pointed its
CoFiTok adjustment argument at the immutable report created after the 20K
recovery incident:

```text
reports/recovery_incident_2026-08-17_pin_memory/resume_compute_adjustment.json
SHA256: 7e4c2362ee38b4b973034dbe492a8ac64044c17f805e6f3cd66b7421c48af740
```

After the later server restart, exact resume from step 80K created a second
physical orphan archive. The locked auditor discovers all files matching
`train_metrics_orphaned_at_resume_*.jsonl` and requires the adjustment event
set to equal that discovered set. Because the old report contains only the
20K event and is immutable, it would have failed closed when the terminal
100K reports became available.

The two physical archives are:

| Resume step | Logged rows | Orphan step span | SHA256 |
|---:|---:|---:|---|
| 20,000 | 4 | 20,050–20,200 | `6b3c4bca8ffa99e814ee77559d140f3a0db6cf7b78a8d78e58ef1b716e2f1cec` |
| 80,000 | 69 | 80,050–83,450 | `ea0f91f8fde6a2c04fbf3578d9184709503c5833645cb9ee7f3ecb4cd26e89c1` |

## Safe rebinding

PID `7736` was checked against its start ticks, executable, cwd, complete
command-line digest, locked Git identity, and source digest before receiving
`SIGTERM`. It had no training role and did not own a GPU process. Its old
adjustment and deployment receipt remain unchanged.

The replacement uses the same audited source and the same launch/config/runtime
contracts, but targets a new method-specific path:

```text
reports/runtime_compute_fairness/cofitok/resume_compute_adjustment.json
```

The new deployment receipt is:

```text
reports/runtime_compute_fairness/deployment_receipt.2026-08-21_post_restart_orphan_set_v2.json
bytes:  4,396
SHA256: 032e15155c8d1309b622692ee7ec105bbae3045b29fa0d43810232f28fa0bb3d
```

The formal adjustment is intentionally absent while training is incomplete.
The waiter will create it only after both exact 100K training reports exist.

Two short launch attempts exited before becoming persistent because the
manually detached process initially lacked the checkout import paths. The
first reported a missing `cofitok` module; the second reported a missing
`scripts` module. The final launch explicitly sets
`PYTHONPATH=<checkout>:<checkout>/src`. Neither failed attempt used a GPU or
modified training state.

## Physical replay

The exact locked builder and verifier were run in memory against the current
canonical metrics through step 88,450 and both orphan archives. No formal
adjustment file was written. The replay passed with:

```text
event count:                           2
physical archives discovered/covered: 2 / 2
extra compute lower bound:             10,321.792349338531 seconds
extra compute lower bound:             2.8671645414829254 hours
orphaned optimizer steps:              3,650
orphaned images:                       233,600
physical verification:                verified
```

This adjustment changes cost accounting only. It does not alter the model,
quality metrics, training trajectory, checkpoint trust, promotion decision,
or release state.

## Monitoring and terminal requirements

The existing heartbeat automation `cofitok-100k-bridge-monitor` was updated,
not duplicated. It now checks the replacement waiter and requires the final
CoFiTok adjustment to bind both orphan archives with
`discovered_orphan_archive_count == covered_orphan_archive_count == 2`.

At the post-rebinding snapshot, training was still healthy:

```text
CoFiTok step:     88,450
samples seen:     5,660,800
pair status:      running / cofitok_training
pair issues:      []
GPU process:      trainer PID 3773 only
GPU memory:       89,398 MiB
```

The terminal runtime-fairness report remains pending. This work does not
authorize full 300K, promotion, release, or a positive generation-quality
claim.

Machine-readable evidence:

```text
artifacts/reports/generation/quality_bridge_runtime_fairness_orphan_set_rebinding_2026-08-21.json
```
