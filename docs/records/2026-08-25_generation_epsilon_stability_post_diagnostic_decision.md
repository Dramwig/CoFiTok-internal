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

## Deployment evidence

The decision code was committed and deployed from an isolated checkout:

- branch:
  `analysis/generation-epsilon-stability-post-diagnostic-decision-v1-20260825`
- revision: `139bcd91015e62a0fb8aa4cd1fea8b2ae177439f`
- tree: `07ba13c1fd4fc7ecbf01c1888c6bbe2f318d0ac0`
- remote checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/epsilon-stability-post-diagnostic-decision-139bcd9/CoFiTok-internal`
- incremental bundle bytes: `118663`
- incremental bundle SHA256:
  `fe7e7fbcc03c0a96b6f523dc8caec720630ab18ecf40b21355960a28f43df7f4`
- bundle prerequisite:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a`

The canonical CPU-only outputs are:

- decision: `reports/epsilon_stability_post_diagnostic_decision_v1_20260825/post_diagnostic_decision.json`
- decision bytes: `7964`
- decision SHA256:
  `b9ce02d19a01e48652277eecc82c12fa193b036b54e7d57edc52225105a34aff`
- replay verification:
  `reports/epsilon_stability_post_diagnostic_decision_v1_20260825/post_diagnostic_decision_verification.json`
- verification bytes: `4342`
- verification SHA256:
  `e00837848be5ef785009d0321a096403e73d132c242889e22723eb7d4ce119a9`
- verification status: `pass`

The builder directly rehashed nine authoritative sources, including the 1K
diagnostic, reconciliation and post-reconciliation decision, pair monitor,
terminal exposure, runtime fairness, terminal guard/completion, and legacy
route supersession receipt. Local verification completed `1124 passed, 6
skipped`; the Linux focused suite completed `23 passed`. Both remote commands
used `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
The post-build GPU check remained idle with no compute process.
