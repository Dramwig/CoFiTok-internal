# Quality bridge strong-baseline comparison preflight

Date: 2026-08-23 04:35 CST

## Purpose

This CPU-only audit verifies that the pending terminal comparison cannot present
official pretrained D-AR, MAR, or ReTok results as compute-matched evidence for
CoFiTok. It also checks that the direct CoFiTok/dense panel remains bound to the
same full-data ImageNet-256 100K training and 10K terminal sampling protocol.

The comparison is downstream of the terminal claim guard and is currently
waiting. This audit did not create a comparison report early, launch any GPU
work, alter a waiter, or change a scientific or authorization decision.

## Authoritative waiter identity

- Checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-comparison-classifier-integrity-8d11f0a/CoFiTok-internal`
- Revision: `8d11f0aad652197c922454104353f88b4f09d46a`
- Tree: `9259e1db0471ae3929c9e9657019e95dcc0022bc`
- Branch: `analysis/generation-quality-bridge-comparison-classifier-integrity-v1-20260823`
- Full Git porcelain: empty
- Waiter source SHA256:
  `db8333674547b0aa5b6bea7489b9c884dde561d5d99f3b2acadd2c1fbb9dc150`
- Builder source SHA256:
  `3f3e7f7d99ba25ad41e93db3c357ca50d1ce41164aa5afc65b657d4595ad4a61`
- Waiter PID: `771583`; parent PID `1`; `CUDA_VISIBLE_DEVICES=''`;
  `OMP_NUM_THREADS=1`; `MKL_NUM_THREADS=1`; nice `10`; ionice `idle`.
- Current state: `waiting_for_exact_terminal_system_sources`.

## Locked contextual source

The official related-method table is bound by SHA256:

```text
14cd71cb72e7ee121f571503cd75491858840de806a97fba9cda188d034c748a
```

It contains exactly three completed eval-only ImageNet-256 rows:

| method | samples | FID | status | role |
|---|---:|---:|---|---|
| D-AR | 50,000 | 2.6281051734 | `completed_eval_only_50k` | `secondary related-method only` |
| MAR | 50,000 | 2.3384937047 | `completed_eval_only_50k` | `secondary related-method only` |
| ReTok | 50,000 | 2.2188685477 | `completed_eval_only_50k` | `secondary related-method only` |

The MAR row remains explicitly bound to the official LTH14 PTH `model_ema`
protocol. Every row retains its official metrics text/NPZ path and protocol
description in the locked table.

## Verified separation contract

The builder and waiter require:

- `matched_training_direct`: only `CoFiTok K=8` and `dense_identity`;
- `official_pretrained_contextual`: only D-AR, MAR, and ReTok;
- `cross_tier_numeric_ranking_allowed=false` in the comparison policy,
  authorization boundary, terminal guard, waiter scope, and exported rows;
- different Markdown panels with an explicit “not a direct ranking” warning;
- CSV columns `comparison_tier` and `directly_comparable_to_cofitok`;
- locked official aliases, method names, eval-only status, 50K sample count,
  ImageNet-256 resolution, secondary role, protocol, source metrics, and metric
  domains;
- matched rows with the same dataset identity, training Git/runtime identity,
  training steps, effective batch, training-image exposure, evaluator identity,
  real set, EMA/DDIM-100 sampling protocol, class schedule, random-stream
  semantics, and 10K sample count;
- physical final checkpoint and integrity-sidecar replay for both direct rows;
- sample-set, real-set, evaluator, class-fidelity, runtime-fairness, and terminal
  claim-guard provenance;
- wall-clock, GPU-hour, FLOP, cost-efficiency, and peak-VRAM advantage wording to
  remain prohibited unless the independent runtime guard explicitly allows the
  corresponding direct comparison.

Even if the terminal guard passes, the comparison remains non-authorizing and
continues to prohibit absolute usability, broad superiority, SOTA, promotion,
release, inference export, full training, and 300K claims.

## Verification

The live locked checkout was tested with GPU visibility disabled:

```text
tests/test_generation_quality_bridge_comparison.py
tests/test_generation_quality_bridge_comparison_waiter.py
37 passed
```

The active GPU remained owned only by dense trainer PID `219593` at
`79,132 MiB` before and after the test. The waiter checkout remained clean.

## Decision

The strong-baseline evidence path is correctly tiered and source-bound at the
current stage. The direct panel will answer only the matched CoFiTok-vs-dense
question; official D-AR/MAR/ReTok values will remain contextual. No comparison
output can be finalized until the terminal quality, statistical, visual, and
runtime guards complete.

Current scientific state remains:

```text
generation_advantage_proven=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
```
