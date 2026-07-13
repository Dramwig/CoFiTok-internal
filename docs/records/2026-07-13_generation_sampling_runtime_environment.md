# Formal sampling runtime environment (2026-07-13)

## Problem

Formal sampling already bound checkpoint bytes, Git state, protocol, progress,
and PNG-set hashes. It did not bind the process that actually produced the
samples. CoFiTok and dense could therefore run under different Python, PyTorch,
CUDA, driver, GPU, backend flags, or project locks while still appearing matched.

## Contract

`generate_samples.py` now captures the canonical runtime environment after the
checkpoint is loaded and inference backend flags are applied. Immutable sampling
manifest schema v2 and completed report schema v5 store both the full environment
and `runtime_environment_sha256`. Exact `--resume` compares the whole manifest,
so environment drift is rejected before additional PNGs are produced.

Sampling preflight records the same fingerprint. Shared batch selection accepts
a candidate only when CoFiTok and dense expose valid, identical fingerprints.
Formal metrics require the report and immutable manifest to agree, then preserve
the environment as sample provenance.

## Fail-closed evidence chain

Promotion/final quality reports include a named
`matched_sampling_runtime_environment` gate. The terminal completion audit
requires one fingerprint across selected preflights, both formal generation
reports, and the final gate. Tests cover malformed digests, manifest/report
drift, resume under changed environment variables, unmatched method preflights,
and a dense formal environment that diverges after selection.

This provenance is separate from the training-checkpoint environment and the
EMA deployment artifact source environment. All three are required: training
reproducibility, actual formal-sampling fairness, and deployable-model identity.
