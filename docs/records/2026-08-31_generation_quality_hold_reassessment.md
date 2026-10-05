# Generation quality hold reassessment (2026-08-31)

## Scope

This is a CPU-only evidence record. It does not launch training, sampling,
checkpoint promotion, inference export, or a 300K run. Existing locked paper
evidence and the full-data 100K quality bridge are not modified.

## Authoritative state

The source-bound 100K bridge remains complete at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

The pair monitor is `pass` with no issues. The terminal comparison and
completion-audit waiters are operationally `pass`, but both preserve
`terminal_status=hold`. The terminal claim guard decision is:

```text
terminal_system_evidence_complete_without_qualified_matched_advantage
```

Formal terminal metrics are:

| method | FID | precision | recall |
| --- | ---: | ---: | ---: |
| CoFiTok K8 | 115.2622 | 0.7556 | 0.00832 |
| dense identity | 123.0210 | 0.6653 | 0.01000 |

The matched FID and precision rows pass, but CoFiTok fails the absolute FID
ceiling (`100`), recall floor (`0.30`), and class-fidelity qualification. The
terminal completion audit records `generation_advantage_proven=false` and an
independent replication count of zero.

The final 100K training reports are source-bound to revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. The final validation epsilon MSE
is `0.0292560` for CoFiTok and `0.0292407` for dense, so the quality hold is
not explained by a large matched validation-loss divergence.

## Approved recovery evidence

The non-authorizing matched 1000-sample sampling-recovery diagnostic is
technically `pass`, but its selection is:

```text
no_shared_sampling_recovery_candidate
```

Across its 16 method/case observations, the best CoFiTok start-975 cases have
FID about `142.5` and class Top-1 `0.2%`; the corresponding dense cases remain
better in FID. The class signal remains near chance, so the diagnostic cannot
replace the terminal result or authorize another stage.

The matched Min-SNR gamma-5 50K pilot is `screening_only` with
`terminal_status=hold` and selection:

```text
no_shared_min_snr_candidate_at_50k
```

Neither result authorizes training, sampling, 300K, promotion, release, or
inference export.

## Current boundary and next decision

At the latest recheck, the RTX PRO 6000 had `0 MiB` used and no GPU compute
application. Only CPU-only waiters/supervisors were present. The standing
authorization SHA256 still matches
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`, while
all training, sampling, full-300K, promotion, and release switches remain
false.

The evidence supports only the scoped ordered-prefix/mechanism claim. The
standing authorization permits direct execution of future necessary
experiments without a new per-stage user prompt, but it does not bypass the
source-compatible execution gate. The authoritative Min-SNR pilot result
requires `stop_min_snr_route_and_reassess_training_objective`; no successor
execution gate is currently ready, so no new GPU stage is started by this
record.
