# Fresh rank-complete 10% scaling promotion hold

Date: 2026-07-24  
Branch: `scale/generative-system`  
Training/evaluation revision: `a8f43f4b8695ce1ac92ca9bd774309c62947396e`

## Outcome

The fresh true-compressed ImageNet-256 10% matched pair completed both 50,000-step
training runs and both formal 10,000-sample evaluations. The authoritative scaling
gate returned `fail/hold`; full ImageNet-256 300K training is not authorized.

This is a sample-quality failure, not a training-completion, checkpoint-integrity,
matched-protocol, or ordered-factorization failure.

## Matched training

| Field | CoFiTok K8 | Dense identity |
|---|---:|---:|
| Steps | 50,000 | 50,000 |
| Images seen | 3,200,000 | 3,200,000 |
| Effective batch | 64 | 64 |
| Parameters | 62,836,796 | 62,824,707 |
| Elapsed seconds | 111,606.90 | 101,108.81 |
| Peak VRAM bytes | 56,186,152,960 | 55,393,710,592 |
| Final checkpoint SHA256 | `4ca727feba29a12045da5aa6bbaf5030c622cf1f9cb4348e208a35da616de239` | `6bd395928fa35cb46f3689fea0664556ea50215a092217cb68f642ac6a214227` |

Both terminal progress audits report `complete`, validation `50/50`, no issues or
warnings, and `latest_integrity.status=verified`. The read-only pair monitor reports
`pass/complete`.

## Formal scaling evaluation

Both methods used exact matched sampling and evaluation conditions:

- 10,000 generated images and the same 50,000-image ImageNet-256 validation set.
- EMA weights, bf16, DDIM-100, CFG 1.5, guidance rescale 0, eta 0, and `clip_x0=true`.
- Balanced-modulo class labels, seed 0, and per-global-index deterministic RNG.
- The same selected batch size, evaluator revision, runtime environment, and
  content-addressed real-set cache.

| Metric | CoFiTok K8 | Dense identity |
|---|---:|---:|
| FID | 191.2347 | 115.0052 |
| Inception Score | 3.4265 | 9.2240 |
| Precision | 0.9760 | 0.6404 |
| Recall | 0.000060 | 0.0128 |
| t=500 endpoint MSE | 0.0171590 | 0.0163199 |

The failed gate checks are:

- `fid_within_tolerance`: CoFiTok regressed 66.2835%; the maximum is 5%.
- `absolute_fid_quality`: CoFiTok FID is 191.2347; the maximum is 100.
- `endpoint_within_tolerance`: CoFiTok regressed 5.1412%; the maximum is 5%.

The ordered path ranks first among 18 orders, first-six-token energy is 7.8087%,
zero-token maximum absolute output is exactly zero, and the shuffled/ordered endpoint
ratio is 117.6538. These mechanism checks do not compensate for the failed generation
quality checks.

## Visual interpretation

The fixed CoFiTok samples are dominated by high-frequency texture and weak object
structure. Dense samples are also substantially undertrained and often exhibit strong
color shifts, but they retain more recognizable object structure. CoFiTok prefix
diagnostics show that low budgets remain partial denoising states rather than
independent natural-image samples.

The extremely low recall for both methods is consistent with poor distribution
coverage under the current undertrained checkpoint and fixed CFG protocol. CoFiTok's
much worse FID and Inception Score show an additional factorization/optimization
problem beyond the shared baseline weakness.

## Raw model versus EMA

A non-promotion 1,024-image t=500 diagnostic found:

| Method | Raw model endpoint MSE | EMA endpoint MSE | Raw relative to EMA |
|---|---:|---:|---:|
| CoFiTok K8 | 0.0169092 | 0.0171590 | -1.456% |
| Dense identity | 0.0162941 | 0.0163199 | -0.158% |

Using raw weights would reduce the matched endpoint regression to approximately 3.78%,
but this one-step result does not establish iterative sample quality and does not
satisfy the system requirement for stable EMA inference. It is a diagnostic for EMA
lag, not a replacement scaling result.

## Observer postmortem

The temporary step-50,000 evidence waiter produced a false negative. It accepted only
the in-progress auditor status `healthy`, while the auditor correctly emits `complete`
at the exact terminal step. The waiter stopped before copying its remaining small
files, but did not affect training, checkpoints, post-evaluation, or the gate decision.

The independent completion chain is the training report, `latest.json`, integrity
sidecar, terminal pre-promotion audit, pair monitor, and successful EMA checkpoint
load/evaluation. The machine-readable postmortem is:

```text
artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/
  cofitok_50k_evidence_waiter_postmortem.json
```

## Recovery diagnostic

An isolated matched sampling sweep completed from the immutable 50K checkpoints:

- CoFiTok and dense use the same 128 seeds and DDIM-100 schedule.
- CFG values are 0.75, 1.0, 1.25, and 1.5.
- CFG 1.5 additionally tests guidance rescale 0.5 and 1.0.
- Small-sample FID/IS and fixed previews are configuration-selection diagnostics only.

The six CoFiTok FIDs span only 301.4799--302.2837, while the six dense FIDs span
265.2326--269.5873. Raw CoFiTok weights improve the t=500 endpoint by only 1.456%
relative to EMA; the dense improvement is 0.158%. CFG, guidance rescale, and EMA lag
therefore do not explain the failed matched gate. The bounded report is:

```text
artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/
  sampling_recovery_diagnostic/summary.json
```

The first summary attempt read the obsolete field `counts.generated` instead of
`counts.generated_image_count`. The idempotent repair reused all twelve completed
sample and metric reports and rebuilt only the bounded summary. Its authoritative
status is `pass/complete`.

The sweep cannot authorize full training. If it identifies a credible shared protocol,
that protocol must be frozen and confirmed with a fresh matched 10,000-sample
evaluation and a new authoritative gate. Otherwise, the CoFiTok training
recipe/architecture must be revised and the matched scaling pair rerun.

## Fixed-basis v8 probe

Revision `5a93c7061e4df237234b19b052331fbc3ec10425` adds a non-formal 5K
fixed-basis probe. It replaces the learned local synthesis operator with a
deterministic, bias-free, condition-free RGB channel projection while keeping the
v7 data, U-Net, token layout, Hellinger capacity-path objective, and training budget.
The synthesis bank has zero trainable parameters, preserves `S_k(0)=0`, and the last
two token projections jointly have RGB rank three.

The remote real-parent-layout pytest suite and all runbook syntax checks passed before
the server fast-forward. The active paths are:

```text
run:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/
    imagenet256_10pct_rankcomplete_fixed_basis_hellinger_k8_probe5k_v8
monitor:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/
    generation_rank_recovery_probe_v8_monitor.json
runbook log:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/
    generation_rank_recovery_probe_v8_runbook.log
launcher PID:
  256821
```

This probe cannot authorize 50K or 300K. A fresh same-revision matched 50K pair and
formal 10K gate are required if the probe is directionally successful.

## Evidence

The bounded local evidence pack is:

```text
artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/
```

Its terminal index is `evidence_manifest_postgate_hold.json`; the manifest status must
remain `promotion_hold`. Large checkpoints and complete sample sets remain only under
the server `checkpoints/generation/` tree.
