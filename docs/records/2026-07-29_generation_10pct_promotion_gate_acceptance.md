# Generation 10% Promotion Gate Acceptance

Date: 2026-07-29

## Outcome

The fixed-basis v3 matched ImageNet-256 10% experiment completed its training,
sampling, distribution metrics, checkpoint-mechanism evaluation, prefix
diagnostics, visual audit, and promotion report. It did not authorize full
ImageNet-256 training.

The authoritative terminal state is:

- completion supervisor: `failed`
- completion pipeline: `failed`
- terminal stage: `promotion_gate`
- promotion report: `status=fail`, `decision=hold`
- GPU: idle

This is a fail-closed scientific hold, not an operational crash.

## Accepted Evidence

Both methods completed 50,000 optimizer steps and 3,200,000 images seen at
revision `1ebcc15210e63a776a2ba448481cbd8bb94a4066`. The matched-pair contract
passed with 62,836,011 CoFiTok parameters and 62,824,707 dense parameters, a
0.017993% gap.

Both formal sample sets contain exactly 10,000 images and use the same EMA,
DDIM-100, CFG 1.5, bf16, balanced-modulo class, and per-index random-stream
protocol. Gate source verification passed for all six bound reports. The two
physical step-50K checkpoints were rehashed during acceptance:

| Method | Bytes | SHA256 |
|---|---:|---|
| CoFiTok K8 | 1,006,356,010 | `6591bd55418e0c7fa712047d29355703cba5da7026d76f5044589820def84ae4` |
| Dense identity | 1,006,119,062 | `c2d09aee0c979cf3bcf378f65d78c3d189b5efb3ddc583b6408d4031b4ea67f3` |

Both byte counts and hashes match their integrity sidecars.

## Gate Decision

| Metric | CoFiTok K8 | Dense identity |
|---|---:|---:|
| FID | 226.4845 | 117.1660 |
| Inception Score | 3.1244 | 9.0862 |
| Precision | 0.9534 | 0.6664 |
| Recall | 0.00002 | 0.01218 |
| Endpoint MSE at t=500 | 0.016461 | 0.016313 |

The only failed checks are:

1. `fid_within_tolerance`: CoFiTok regresses 93.30% against dense; the maximum
   allowed regression is 5%.
2. `absolute_fid_quality`: CoFiTok FID is 226.48; the required maximum is 100.

The mechanism checks pass:

- endpoint MSE regression: 0.91%
- ordered path rank: 1 of 18
- coarse-token energy ratio: 8.28%, above the 5% floor
- zero-token maximum absolute output: 0
- shuffled-to-ordered endpoint ratio: 124.21x

The evidence therefore supports ordered restricted factorization, but not a
usable large-scale generator.

## Visual Finding

CoFiTok endpoint samples retain severe high-frequency chromatic noise and weak
semantic structure. Prefixes 1 and 2 are predominantly low-frequency green
fields, prefix 4 introduces regular chromatic texture, and prefix 8 introduces
weak semantics without removing the high-frequency residual. Dense samples are
also weak, but are materially smoother and preserve more coarse semantic
structure.

The first six tokens carry only 8.28% of component energy; the last two carry
91.72%. This tail concentration is consistent with the observed rollout
instability.

## Terminal Audit

The terminal completion auditor was rerun independently. It returned
`status=failed`, with `scaling_promotion_gate` as the failed check. Full 300K
training and all downstream formal 50K, final-gate, export, and comparison
evidence are absent because the promotion gate correctly prevented those stages.

An auxiliary milestone waiter timed out after completed training stopped
updating metrics. It did not participate in gate authorization and was not the
cause of the hold.

## Decision

Do not bypass the gate and do not launch full 300K from this checkpoint. The
next qualification cycle must first diagnose iterative denoising error
amplification, tail-token energy concentration, fixed-basis/channel allocation,
feedback dynamics, and path-loss weighting.

The bounded local evidence pack is:

```text
artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/acceptance_2026-07-29/
```
