# Generation current-state recheck (2026-09-01 16:30 CST)

## Scope

This is a CPU-only, read-only state recheck. It does not launch training or
sampling, modify the remote checkout, change locked evidence, promote a
checkpoint, export an artifact, release a model, or send process signals.

## Live remote state

The authoritative `pro6000` checkout remains:

```text
/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
revision=cf0e5faa94bf4ab38d947b921935b3b765b5537a
branch=scale/generation-stability-quality-bridge-100k
tree=6cef27723196fd363379bca2e7b85b1678ebd777
tracked_clean=true
```

The pair monitor reports `pair_status=pass`, `pair_stage=complete`, and
`issues=[]`. Its observational GPU-coverage record is incomplete, so it does
not authorize direct wall-clock compute comparisons. The RTX PRO 6000 was
idle at the recheck (`0 MiB`, no compute application). Only CPU-only
waiters/supervisors were present.

Both matched runs are complete at 100,000 steps and 6,400,000 images seen:

| method | checkpoint bytes | checkpoint SHA256 |
| --- | ---: | --- |
| CoFiTok K8 | 1,010,937,514 | `b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e` |
| dense identity | 1,010,735,510 | `b6586cc906a9c38bbf9d592f5f6169b7a37a8d3aa485ca4879021fa840943d0b` |

Both `latest.json` and `training_report.json` bind the same dataset identity,
runtime identity, revision, branch, step, and checkpoint sidecar. The final
validation epsilon MSEs are `0.0292560` (CoFiTok) and `0.0292407` (dense).

## Scientific and execution boundary

The authoritative terminal comparison and completion audit remain operational
passes with `terminal_status=hold`; the claim guard decision is
`terminal_system_evidence_complete_without_qualified_matched_advantage`.
The terminal rows remain CoFiTok FID `115.2622`, precision `0.7556`, recall
`0.00832`, versus dense FID `123.0210`, precision `0.6653`, recall `0.01000`.
Absolute FID, recall floor, and class-fidelity checks fail; scoped mechanism
checks pass. Therefore `generation_advantage_proven=false`.

The matched 1,000-sample recovery screen selected
`no_shared_sampling_recovery_candidate`. The matched Min-SNR gamma-5 50K
pilot selected `no_shared_min_snr_candidate_at_50k`; neither is a formal
quality claim or a continuation authorization.

The current exposure/capacity preparation is validated locally with SHA256
`b34d0e0a057bed34511c88f83671c80039b86ea45cf548636daff02f91dcb074` and
`source_count=10`. Its two candidate arms remain `status=prepared` and
`execution_ready=false`. The remote execution directory contains the standing
authorization SHA256
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`, but no
candidate-specific exact-stage authorization. The standing instruction does
not bypass that source-compatible gate.

All training, sampling, continuation, full-training, 300K, promotion, export,
release, and process-signal permissions remain false. No new GPU stage is
running or selected.

## Verification

```text
52 focused tests passed in 11.86s
compileall: PASS
preparation validator: PASS
```

The next permitted action is another source-bound gate revalidation or a
bounded stage only after a valid candidate-specific exact-stage authorization
appears. The terminal hold and all locked evidence remain unchanged.
