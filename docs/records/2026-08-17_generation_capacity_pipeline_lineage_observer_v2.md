# Capacity pipeline lineage observer v2

Date: 2026-08-17

## Outcome

The active full-data matched 100K quality bridge now has a second, independently
deployed read-only lineage observer that understands the exact-resume recovery
supersession introduced after the 20K recovery incident. It binds the obsolete
failure record to the live recovery-v2 supervisor, the 20K metrics
reconciliation, the physically audited 25K checkpoint, and the active 50K
watchdog.

The observer does not launch training, sampling, evaluation, promotion, export,
or release. It does not allocate CUDA memory or signal any process. Its only
persistent outputs are its atomically replaced status report, lock, and log.

## Exact code and deployment identity

- branch: `scale/generation-capacity-pipeline-observer-v1`
- deployed code revision:
  `1ff6bb3db932ef9e43ddfafc067db779dab797d9`
- deployed tree: `bf0d2e8208ea3bce0ab04c7f1991ea17882a3ea2`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-lineage-recovery-586fe30/CoFiTok-internal`
- incremental-bundle prerequisite:
  `0bbd1cc43b9bb713ab2da8d4da6404f304f88ffe`
- incremental bundle: `20,764` bytes
- incremental bundle SHA256:
  `afa62c3010eba2fd6cb45f3939ca5d4f98c646ab2712597906ece401197771b5`

The bundle advertised exactly one branch/ref at the target revision. Only the
isolated observer checkout was fast-forwarded. The formal checkout and active
training checkout were not fetched, merged, checked out, or modified.

## Validation

- targeted local observer/recovery tests: `20 passed`;
- broad local capacity/quality-bridge suite: `254 passed` across `41` files;
- exact Linux CPU-only observer/recovery/entrypoint suite: `22 passed`;
- Linux `py_compile`, runbook `bash -n`, and `git diff --check`: passed.

The real-source one-shot report was `24,603` bytes with SHA256
`890aec9a4819bd5f2a3e94090b385ae1109b5e9142a8561a52b50ceab042bfc9`.
It reported `running`, no issues, recovery supersession `pass`, exact resume at
20K, trusted checkpoint at 25K, live CoFiTok step 25,550, and watchdog target
50K.

## Persistent observer

- PID: `765120`
- poll interval: `60` seconds
- stale threshold: `600` seconds, covering two complete 300-second pair-monitor
  intervals
- output:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_generation_pipeline_lineage_observer_v2.json`
- lock:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/capacity_generation_pipeline_lineage_observer_v2.lock`
- log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_generation_pipeline_lineage_observer_v2.log`

Five polls had completed when the immutable deployment receipt was written.
The first three report identities were distinct and monotonically updated; all
reported `running`, the same recovery stage, and no issue. At receipt time the
observer reported live CoFiTok step 25,650.

The remote deployment receipt is:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_generation_pipeline_lineage_observer_v2_deployment_receipt.json`

- bytes: `12,867`
- SHA256:
  `925e62a420bb9b6f5d84e161cc31f62ae7def0f69ec8582794ccdd4066f9de9a`
- independent replay validation: passed

The local machine-readable mirror of the immutable identity and deployment
boundary is
`artifacts/reports/generation/capacity_pipeline_lineage_observer_v2_2026-08-17.json`.

## Preserved state

The old observer PID `12692` remained alive and was not signaled. Active
training PID `619775`, watchdog PID `619618`, and recovery-v2 PID `630988`
remained alive. The sole GPU compute process remained the active CoFiTok
trainer; no new GPU stage was launched.

The formal checkout snapshot remained:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- branch: `scale/generative-system`
- tracked changes: `0`
- porcelain count: `89`
- porcelain SHA256:
  `18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`
- `-uall` count: `301`

## Scientific boundary

This deployment improves the reproducibility and auditability of the active
matched 100K experiment. It is not evidence that the final generation-quality
gate has passed. CoFiTok and dense training plus matched post-evaluation must
still complete before any full-data generation advantage is claimed.

The first guarded persistent-launch attempt exited before creating an output,
lock, log, or process. Read-only verification confirmed the absence of all four
effects before a second guarded launch with explicit diagnostics succeeded.
