# Matched generation training runtime selection (2026-07-12)

## Motivation

The active 10% ImageNet-256 run uses about 23 GiB on a 96 GiB RTX PRO 6000
while processing an effective batch of 64 as microbatch 16 x accumulation 4.
The conservative setting is stable, but applying it unchanged to two 300K runs
would spend roughly 16.6 GPU days at the observed 2.39 seconds/optimizer step.

Runtime packing is therefore selected before full training without changing the
scientific protocol.

## Benchmark mode

`train_generation.py` supports a checkpoint-free benchmark mode. It executes
the real data loader, forward pass, CoFiTok losses, backward pass, gradient
clipping, optimizer, scheduler, and EMA. CUDA is synchronized around every
optimizer step. Warmup steps are excluded from mean/median/p95 throughput, and
the report records:

- actual resolved config and Git revision;
- canonical Python/PyTorch/CUDA/GPU/project-lock environment and SHA256;
- microbatch, accumulation, and effective batch;
- optimizer-step durations and images/second;
- peak allocated VRAM and total device memory;
- parameter count and last finite training metrics;
- an explicit `checkpoint_written=false` contract.

Benchmark mode cannot resume, shorten the configured LR horizon, or write a
training checkpoint.

## Shared selection policy

`select_generation_training_runtime.py` benchmarks `16x4`, `32x2`, and `64x1`
for both CoFiTok and `dense_identity`. A candidate is eligible only if:

- both methods complete the real benchmark;
- both retain effective batch 64;
- timing and throughput are finite and positive;
- peak VRAM is valid and no more than 90% of device memory.
- every completed method/candidate benchmark has a valid, identical runtime
  environment fingerprint.

The selected shared candidate minimizes the slower method's mean optimizer-step
time. The conservative `16x4` baseline must itself pass; otherwise selection
fails closed. OOM for a larger candidate is an expected ineligible result, not
permission to give the two methods different runtimes.

The full runbook passes the selected microbatch and accumulation to every
50K/100K/200K/300K training segment. Both final training reports must contain
that exact selection and the same runtime-environment SHA. Cached benchmark
reports are reusable only under the same clean Git revision, resolved config,
benchmark horizon, and canonical environment. The final completion audit
recomputes every completed benchmark fingerprint and verifies the binding.

## Evidence boundary

This procedure optimizes execution packing only. It does not tune model quality,
learning rate, loss weights, data, training steps, or sampling protocol. Runtime
benchmark outputs are operational evidence and cannot be reported as generation
quality results.

The environment hardening implementation and adversarial tests are recorded in
`2026-07-13_generation_training_runtime_selection_environment.md`.
