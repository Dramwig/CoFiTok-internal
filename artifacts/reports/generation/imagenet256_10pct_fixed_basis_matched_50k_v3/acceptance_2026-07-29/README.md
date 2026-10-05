# CoFiTok Fixed-Basis V3 Acceptance

Date: 2026-07-29

## Verdict

**HOLD. This run is not accepted as a completed large-scale generation
system.**

The matched ImageNet-256 10% training and evaluation stage completed with valid
provenance, but the scientific promotion gate rejected full ImageNet-256
training. The supervisor stopped fail-closed at `promotion_gate`; full 300K
training, formal 50K evaluation, inference export, and the final gate were not
run.

## Evidence Accepted

- Remote revision `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
  `scale/generative-system`, with a clean tracked worktree.
- CoFiTok and dense identity both completed exactly 50,000 optimizer steps and
  3,200,000 images seen.
- The pair contract passed. Parameter counts were 62,836,011 and 62,824,707,
  a relative gap of 0.017993%.
- Both 10K sample sets completed under the same DDIM-100, EMA, bf16, CFG 1.5,
  balanced-class, per-index random-stream protocol.
- Gate source verification passed for all six bound training, checkpoint
  evaluation, and generation reports.
- Both physical step-50K checkpoints were rehashed during acceptance. Their
  byte counts and SHA256 values matched their integrity sidecars.
- Endpoint and mechanism checks passed: endpoint MSE regression was 0.91%,
  ordered prefix path ranked 1 of 18, coarse-token energy ratio was 8.28%,
  `S(0)=0` held exactly, and shuffled tokens produced a 124.21x endpoint
  mismatch.

## Scientific Gate Failure

| Metric | CoFiTok K8 | Dense identity |
|---|---:|---:|
| FID | 226.4845 | 117.1660 |
| Inception Score | 3.1244 | 9.0862 |
| Precision | 0.9534 | 0.6664 |
| Recall | 0.00002 | 0.01218 |
| Endpoint MSE at t=500 | 0.016461 | 0.016313 |
| Training time | 30.91 h | 28.13 h |
| Sampling time, 10K | 5.84 h | 5.31 h |

The only failed promotion checks were:

1. `fid_within_tolerance`: CoFiTok FID regressed by 93.30% versus dense; the
   allowed regression was 5%.
2. `absolute_fid_quality`: CoFiTok FID was 226.48; the required maximum was
   100.

The high precision and near-zero recall do not indicate good generation. In
combination with FID 226.48 and IS 3.12, they indicate extremely narrow and
poor distribution coverage.

## Visual Acceptance

The fixed CoFiTok panel contains strong pixel-scale chromatic residual noise and
weak semantic structure. The prefix panel shows a consistent but unusable
trajectory: budgets 1 and 2 are mostly low-frequency green fields, budget 4
introduces regular chromatic texture, and budget 8 adds weak semantics while
retaining severe noise. The matched dense samples are also low quality, but are
substantially smoother and preserve more recognizable coarse structure.

This agrees with the component-energy profile: the first six coarse tokens
carry only 8.28% of energy, while the final two tokens carry 91.72%. Ordered
factorization exists, but the current capacity allocation and objective do not
produce a stable iterative generator.

## Completion Audit

The independently rerun terminal audit returned `failed` with
`scaling_promotion_gate` as the failed check. All full-scale evidence is missing
because the gate correctly prevented those stages from starting. GPU use is
zero and no generation process remains.

An auxiliary `/tmp` milestone waiter reported a stale-metrics timeout after
training stopped writing metrics. This observer was not part of the promotion
decision and was not the cause of failure.

## Required Next Step

Do not loosen or bypass the gate and do not launch full 300K from this
checkpoint. First diagnose:

1. Iterative error amplification across diffusion timesteps, since endpoint
   t=500 MSE is matched while 100-step generation diverges badly.
2. Tail-token concentration and the transition from low-frequency prefixes to
   high-frequency residual noise.
3. Fixed-basis/channel allocation, feedback dynamics, and path-loss weighting.
4. Small matched qualification runs with timestep-sweep and rollout diagnostics
   before repeating the 50K gate.

The authoritative copied sources and visual panels are stored alongside this
file.
