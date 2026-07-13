# Exact-resume horizontal flip for full generation (2026-07-13)

## Scope

The full ImageNet-256 300K CoFiTok and `dense_identity` configurations now use
the same random horizontal flip probability of `0.5`. The active pinned 10%
50K pair is unchanged and resolves the new configuration field to its legacy
default of `0.0`.

## Implementation

Training images are flipped after transfer to the training device and before
diffusion noise or timestep sampling. The operation draws one Bernoulli decision
per image from the training process PyTorch RNG. It does not use DataLoader
worker randomness, and validation images are never augmented.

This placement keeps recovery exact because production checkpoints already
capture and restore Python, NumPy, CPU Torch, and CUDA RNG states before the
train iterator is created. The fully resolved probability is also part of the
checkpoint config identity, so changing it across `--resume` is rejected.

The matched-pair contract compares the entire resolved data section. Full
CoFiTok and dense runs therefore cannot pass preflight with different flip
probabilities.

## Verification

Tests cover the `0.0` and `1.0` boundaries, deterministic replay from a restored
Torch RNG state, legacy configuration defaulting, and the checked-in full pair's
shared `0.5` setting. The existing subprocess trajectory test runs with
probability `0.5` and requires uninterrupted training to match segmented exact
resume for model, EMA, optimizer, scheduler, RNG, sampler, losses, gradient
norm, learning rate, and validation MSE.
