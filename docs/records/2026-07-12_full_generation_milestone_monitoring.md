# Full generation milestone monitoring (2026-07-12)

## Motivation

The original full ImageNet-256 runbook trained one method for all 300K steps
before evaluating EMA sample quality. Training epsilon MSE on one validation
batch is useful for numerical health, but it cannot establish whether DDIM/CFG
generation quality is improving. Discovering a severe sampling regression only
after both full runs would waste several GPU-days.

## Protocol

The full matched queue now alternates CoFiTok and `dense_identity` at optimizer
steps 50K, 100K, 200K, and 300K. Every segment preserves the original 300K LR
schedule and resumes exact optimizer, scheduler, EMA, sampler, and RNG state.

At each method/milestone pair:

- verify a real batch-32 bf16 batched-CFG EMA forward;
- evaluate 256 validation images at `t=500` for endpoint and ordering health;
- generate 2,048 deterministic EMA samples with DDIM-50 / CFG 1.5;
- compute torch-fidelity FID and Inception Score with the shared real cache;
- bind generation and mechanism rows to the same checkpoint and sample hashes;
- produce a paired trend report with quality alerts.

Milestone FID is deliberately labeled `training_quality_trend_only` and cannot
support a formal generation claim. Final readiness still requires the separate
50K-sample DDIM-250 full gate.

## Verification

- Local full suite: 283 tests passed.
- Focused milestone, checkpoint-evaluation, sampling, and preflight suite: 20
  tests passed.
- Python compileall: passed.
- Remote `bash -n` for the segmented full-training and milestone-evaluation
  runbooks: passed.

CUDA milestone execution remains gated on completion of the active 10% matched
50K queue; no monitoring code was synchronized into that running revision.
