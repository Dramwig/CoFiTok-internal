# Generation sampling preflight (2026-07-12)

## Purpose

Formal 10K and 50K EMA sampling must fail before creating a sampling manifest
or partial image directory when the requested inference shape does not fit the
GPU or produces an invalid output.

## Implementation

- `src/cofitok/generation/runtime.py` owns checkpoint, model, EMA, and device
  loading for both formal sampling and preflight.
- `scripts/preflight_generation_sampling.py` executes one real epsilon forward
  at the highest diffusion timestep with the requested batch size, prefix
  budget, precision, guidance scale, and CFG batching mode.
- The atomic JSON report records checkpoint SHA256 and step, effective CFG model
  batch, output shape/dtype/finiteness, elapsed time, and CUDA baseline/peak
  allocated and reserved memory. Forward failures retain the same provenance.
- Both 10% and full post-evaluation runbooks run CoFiTok and dense preflights
  before checkpoint diagnostics or sample generation.

## Verification

- Local full test suite: 279 passed.
- Python compileall: passed.
- Remote `bash -n` for both modified post-evaluation runbooks: passed.
- CPU checkpoint/EMA/batched-CFG preflight: passed with exact output shape and
  finite epsilon values.

The CUDA batch-32 bf16 preflight is intentionally deferred until the active
matched 50K training queue releases the GPU. No training process was interrupted
or competed with during this change.
