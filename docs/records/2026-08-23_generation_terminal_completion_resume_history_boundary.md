# Terminal completion audit: legacy resume-history claim boundary

Date: 2026-08-23

## Purpose

The active ImageNet-256 matched 100K quality bridge was trained by the legacy
trainer at revision `cf0e5faa94bf4ab38d947b921935b3b765b5537a`. Its
`run_manifest.json` and `training_report.json` do not contain the newer
append-only `metrics_resume_history` journal. They retain only the most recent
`metrics_resume_reconciliation` object. The physical run directories retain two
canonical CoFiTok reconciliation/orphan pairs at 20K and 80K and one dense pair
at 50K. Consequently, the terminal completion audit can enumerate and
physically verify every currently discoverable legacy pair, but directory
discovery still cannot prove that every exact-resume event in the historical
training recovery chain is represented.

This change tightens that claim boundary without changing the upstream terminal
guard's independent `pass` or `hold` decision.

## Implementation

`scripts/build_generation_quality_bridge_terminal_completion_audit.py` now:

- Requires both canonical legacy manifests and final training reports to agree.
- Confirms that append-only resume history is absent rather than inventing it.
- Enumerates every canonical `metrics_resume_reconciliation_*.json` and matching
  `train_metrics_orphaned_at_resume_*.jsonl` in each run directory.
- Physically binds every discovered report, orphan archive, canonical metrics
  file, and event-specific retained metrics prefix.
- Rejects malformed names, unpaired files, duplicate-step divergence, report or
  orphan drift, invalid row counts/digests, and a manifest-embedded latest event
  absent from the discovered set.
- Revalidates Git, dataset, runtime, effective-batch, step, and `samples_seen`
  identities.
- Does not require pruned 50K/80K resume checkpoint payloads; the independent
  90K/95K/100K checkpoint physical-integrity chain remains authoritative.
- Replays the legacy boundary twice and rejects source drift.
- Emits explicit policy fields:
  - `all_discovered_legacy_evidence_physically_verified=true`
  - `discovery_proves_no_missing_resume_events=false`
  - `final_checkpoint_and_terminal_sample_reproducibility_claim_allowed=true`
  - `complete_exact_resume_history_claim_allowed=false`
  - `training_recovery_chain_reproducibility_claim_allowed=false`
- Preserves all non-authorization guards, including
  `full_300k_launch_allowed=false` and no promotion or release authority.

## Verification

Local targeted suite:

```text
tests/test_generation_quality_bridge_terminal_completion_audit.py
34 passed
```

Broader local related suite:

```text
118 collected; 117 passed, one Windows-only skip
```

The added tests cover inclusion of the earlier CoFiTok 20K event, earlier-event
report/orphan/prefix drift, unpaired files, duplicate resume-step divergence,
the explicit legacy completeness limitation, manifest/training-report
divergence, missing pruned resume payload tolerance, double source
revalidation, preservation of terminal `pass`/`hold`, and the permanent
300K/promotion prohibition.

## Initial Linux CUDA-hidden rehearsal

- Code commit: `df87d0419f964c228e24da7c0b5e837e120587de`
- Tree: `4d89a42348acee7c8f0cfb60aec2cf1f7ba1f185`
- Incremental bundle prerequisite:
  `a253d56e49b78e1c8ddb10c3bd07af5aa22e5919`
- Bundle:
  `D:/cofitok-bundles/terminal-completion-resume-history-boundary-df87d04.bundle`
- Bundle bytes: `14,449`
- Bundle SHA256:
  `d4f7adf7ee350bc2224817fcda8e4d45d41fd152e0568996060d7dc47f05442c`
- Isolated remote checkout:
  `/tmp/cofitok-terminal-completion-resume-history-boundary-df87d04/CoFiTok-internal`
- Environment: `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `nice=10`, `ionice=idle`
- Result: `106 passed`
- Git status after rehearsal: clean
- GPU before and after rehearsal: only the pre-existing dense trainer PID `219593`
  at `79,132 MiB`; the rehearsal created no GPU process.

## Discovery-set Linux CUDA-hidden rehearsal

- Code commit: `90514bf46edbb43a414dd55d4a3fd6f4ea5982b9`
- Tree: `154f6f421d2fab6d0d5352ab069f469b88e89e4a`
- Incremental bundle prerequisite:
  `a253d56e49b78e1c8ddb10c3bd07af5aa22e5919`
- Bundle:
  `D:/cofitok-bundles/terminal-completion-resume-history-discovery-90514bf-from-a253d56.bundle`
- Bundle bytes: `20,018`
- Bundle SHA256:
  `511c9ee8da8f754e381037d44c41e68e5f7521b22ec0f94fa74678813421680d`
- Bundle advertised exactly one ref, `HEAD`, at the code commit above; remote
  `git bundle verify` passed before checkout.
- Isolated remote checkout:
  `/tmp/cofitok-terminal-resume-discovery-90514bf/CoFiTok-internal`
- Environment: `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `nice=10`, `ionice=idle`
- Result: `118 passed`
- Git status after rehearsal: clean
- GPU before and after rehearsal: only the pre-existing dense trainer PID
  `219593` at `79,132 MiB`; the rehearsal created no GPU process.

Live-source CPU-only replay during the active dense continuation behaved
fail-closed as intended: the complete two-method verifier rejected dense because
its final `training_report.json` still describes the completed 50K segment while
the canonical manifest has already advanced to the active 80K resume. A scoped
CoFiTok-only replay physically verified both discoverable events at 20K and 80K,
including their reports, orphan archives, and retained prefixes. It reported
`all_discovered_legacy_evidence_physically_verified=true` while preserving
`complete_recovery_chain_verified=false` and
`discovery_proves_no_missing_resume_events=false`. The full matched replay must
remain pending until dense reaches 100K and writes its final report; no active
waiter was replaced or modified.

## Deployment status

This branch is analysis-only and is not deployed into the active training
checkout or the existing terminal completion waiter. Rebinding that waiter, if
ever needed, remains a separate fail-closed decision requiring exact process,
lock, source, receipt, and duplicate checks after the active serial GPU chain is
complete.
