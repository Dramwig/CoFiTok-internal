# Epsilon-stability post-diagnostic decision (2026-08-25)

## Evidence outcome

The authorized matched 1,000-sample epsilon-stability sampling diagnostic
completed all eight cases for both CoFiTok and dense identity. It selected no
shared sampling-recovery candidate. Several controls lowered FID, but every
candidate regressed class fidelity or distance to the matched real-image
artifact statistics for at least one method. The bounded result therefore does
not support a sampler-only repair and does not authorize a 10K confirmation.

The terminal system remains operationally complete but scientifically held:
both methods fail the absolute FID and recall requirements, class fidelity is
held, and `generation_advantage_proven=false`. The runtime replay includes both
CoFiTok restart-orphan archives and preserves matched compute accounting.

## Next scientific discriminator

The diagnostic explicitly records `min_snr_training_tested=false`. The codebase
already implements standard epsilon Min-SNR weighting as
`min(SNR, gamma) / SNR`, logs both weighted and unweighted epsilon losses, and
requires the gamma to match between CoFiTok and dense identity. This control
preserves the CoFiTok method boundary: the prediction target remains epsilon,
and each restricted synthesis operator still expands one compressed
negative-noise token without condition inputs.

The deterministic post-diagnostic decision therefore recommends preparation of
a fresh matched Min-SNR training pilot. This is a training-recipe intervention,
not an exposure-only continuation. A changed-config resume is invalid. The
separate preparation must fix a positive shared gamma, training and evaluation
budgets, legacy-control policy, initialization/data/random streams, isolated
outputs, physical checkpoint audits, and DDIM-100 quality/support/class and
mechanism gates. If the frozen legacy checkpoints cannot be proven to be an
exact control at the chosen milestone, a fresh gamma-zero control is required.

## Authorization boundary

The decision builder and verifier are CPU-only evidence tools. Their outputs
are permanently non-authorizing:

- `training_launch_allowed=false`
- `sampling_launch_allowed=false`
- `full_training_launch_allowed=false`
- `full_300k_launch_allowed=false`
- `promotion_allowed=false`
- `inference_export_allowed=false`
- `release_allowed=false`
- `process_signals_allowed=false`
- `gpu_execution_allowed=false`

A new versioned, source-bound execution gate is required before any GPU work.
The existing terminal hold remains authoritative and no legacy route consumer
is signaled or re-enabled by this decision.
