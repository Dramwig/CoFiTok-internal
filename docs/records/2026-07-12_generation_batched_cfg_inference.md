# Batched CFG inference for large-scale generation

Date: 2026-07-12

Branch: `scale/generative-system`

## Motivation

Formal ImageNet-256 evaluation requires 10K samples at the scaling gate and
50K samples per method at the full gate. Sequential classifier-free guidance
performed separate conditional and unconditional model calls at every DDIM
step, making sampling the dominant evaluation cost.

## Change

- Batched CFG reuses the scalable predictor's existing null-class embedding
  index for the unconditional half of the combined inference batch.
- `predict_epsilon` can concatenate conditional and unconditional examples and
  evaluate both branches in one model call. `sequential` remains available as
  a memory-conservative fallback.
- `generate_samples.py` records `cfg_batch_mode` in the immutable sampling
  manifest. Matched gate reports therefore reject CoFiTok/dense evaluations
  that use different CFG execution modes.
- Scaling and full post-evaluation runbooks explicitly select `batched` mode.

This changes inference throughput only. It does not change training, checkpoint
contents, EMA weights, DDIM equations, guidance arithmetic, sample-index RNG
streams, prefix pairing, or the restricted synthesis contract.

## Verification

- Batched and sequential CFG outputs match elementwise in the deterministic
  unit model; batched mode performs one model call and sequential mode two.
- Explicit null labels match the existing `force_unconditional` model path.
- Full local test suite: 252 tests passed.
- `python -m compileall -q src scripts`: passed.
- Both modified post-evaluation runbooks passed `bash -n` on `pro6000` from
  copies under `/tmp`.

## Deployment boundary

The active 10% matched 50K training queue remains pinned to commit `781a014`.
This inference-only change must not be synchronized into the remote repository
until both CoFiTok and dense training runs have completed, preserving one code
revision for the matched training pair. It will be used for post-training
sampling and evaluation after that boundary.
