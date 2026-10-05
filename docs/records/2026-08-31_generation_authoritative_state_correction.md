# Generation authoritative state correction (2026-08-31)

## Scope

This is a CPU-only evidence record. It does not launch training or sampling,
modify the remote checkout, promote checkpoints, export inference artifacts, or
change any locked report. It supersedes the status interpretation in
`2026-08-31_generation_downstream_visual_audit_reconciliation.md`; that file
remains a historical record of the older canonical v1 waiter failures.

## Rechecked authoritative state

The source-bound full-data 100K bridge is complete at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

The pair monitor is `status=pass`, `stage=complete`, and `issues=[]` (SHA256
`78306a02a47d96be78f325ec2cd708ff4741f248c544fbc861dc9acfa829a2eb`). Both
methods are at step 100,000 with 6,400,000 images seen, reconciled metrics,
and the target revision `cf0e5faa94bf4ab38d947b921935b3b765b5537a`.

The current authoritative downstream chain is:

| artifact | operational status | scientific status | SHA256 |
| --- | --- | --- | --- |
| `terminal_system_claim_guard_v3_authoritative/terminal_system_claim_guard.json` | hold | `terminal_system_evidence_complete_without_qualified_matched_advantage` | `cbb1c20512c9ae980dcf44537dd56cd73a13eb0be78eda7ff37b92f0a76650a9` |
| `quality_bridge_comparison_v4_authoritative/quality_bridge_comparison.json` | hold | same decision | `991600ffddc66d3d3f81d7294f202b636f71acc9e7de52fb33418f16f4f882e8` |
| `terminal_completion_audit_v4_authoritative/terminal_completion_audit.json` | pass | `terminal_status=hold`, `generation_advantage_proven=false` | `2c4d924573f99e6fc62cf7bf83a37a942fa683bb6523b01e2cdd4b867fac088f` |

The v4 comparison and completion waiters are operationally `pass` while
retaining `terminal_status=hold`. The matched terminal rows remain CoFiTok K8
FID `115.2622`, precision `0.7556`, recall `0.00832`, versus dense identity
FID `123.0210`, precision `0.6653`, recall `0.01000`. Absolute FID, recall,
and class-fidelity checks fail; the scoped factorization/mechanism checks pass.

## Follow-up evidence

The source-bound cross-protocol replay completed with
`operational_status=pass`, `terminal_status=hold`, and
`generation_advantage_proven=false` (SHA256
`6fc639caec0320225c2e8d26316490cde2d765d49c6062a18e0e671f79384d19`). It
resolves the ranking reversal as a sampler-step effect: the DDIM-50 2,048
sample milestone favors dense, while the DDIM-100 results favor CoFiTok. The
terminal 10K DDIM-100 result remains authoritative.

The resulting epsilon-stability sampling diagnostic completed all 16 matched
1,000-sample arms. Its result is `status=pass`,
`scientific_status=screening_only`,
`selection_status=no_shared_sampling_recovery_candidate`, and
`generation_advantage_proven=false` (SHA256
`d6b5c3190a205f5be886c01e8543b9e2511988d5f0b9f2e642a33d0e8be1249f`). No arm
simultaneously passed the per-method FID, class-fidelity, and artifact
non-regression checks, so no independent 10K confirmation is authorized.

The Min-SNR gamma-5 matched 50K pilot is complete with
`selection_status=no_shared_min_snr_candidate_at_50k`,
`terminal_status=hold`, and `generation_advantage_proven=false` (result SHA256
`dde62640e84a461be3b571b228e7903785e42f118dd7103941e20592bd0059fc`). Its
next action is `stop_min_snr_route_and_reassess_training_objective`.

## Boundary

The standing authorization file still hashes to
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`, but the
source-compatible execution gate is not ready. Training, sampling, full-data
training, 300K, promotion, release, export, and process-signal permissions all
remain false. GPU state was rechecked at 0 MiB with no compute application;
only CPU-only waiters/supervisors remain. Existing waiting processes are left
untouched.

The next permitted action is static preparation of a new source-compatible
training-objective decision/gate. No new GPU experiment is started by this
record, and `generation_advantage_proven` remains false.
