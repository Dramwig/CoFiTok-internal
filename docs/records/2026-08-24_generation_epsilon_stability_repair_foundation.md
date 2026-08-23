# Generation epsilon-stability repair foundation (2026-08-24)

## Status and boundary

This change is a code-only, CPU-tested preparation rooted at the immutable
quality-bridge training revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. It does not launch training,
sampling, evaluation, promotion, export, or release, and it does not modify
the running 100K terminal chain or any locked paper evidence.

The terminal quality evidence available while this change was prepared shows
that the current cosine epsilon system starts DDIM at the near-zero terminal
alpha, applies hard x0 clipping at every step, and has no standard Min-SNR
primary-loss weighting. The implementation therefore adds auditable controls
for those failure modes without silently changing the historical protocol.

## Preserved CoFiTok method boundary

A direct v-prediction conversion was considered and rejected for the
claim-bearing CoFiTok path. If the restricted synthesis operators directly
produced velocity components, `S_k(z_k)` would no longer be a compressed
negative-noise component. That would change the method rather than repair its
generation system. The code in this branch therefore keeps production
training restricted to `prediction_target="epsilon"` and keeps every token
component additive in epsilon space.

Dense v-prediction may still be studied later as a clearly separated,
non-claim-bearing baseline, but it must not be presented as the same CoFiTok
representation.

## Implemented controls

1. **Standard epsilon Min-SNR weighting**
   - `loss.min_snr_gamma=0.0` is the exact legacy path and retains the original
     `torch.nn.functional.mse_loss` reduction and dtype.
   - A positive gamma applies the per-sample weight
     `min(SNR, gamma) / SNR` to the primary epsilon MSE.
   - Training metrics retain `epsilon` as the optimized primary loss and add
     `epsilon_unweighted` plus `min_snr_weight_mean` for auditability.
   - `min_snr_gamma` is a mandatory matched field in the CoFiTok/dense pair
     contract; it cannot be enabled for only one arm.

2. **Explicit nonterminal DDIM start**
   - Sampling accepts an optional `start_timestep` and records both the
     requested and actual first timestep.
   - Initial Gaussian noise may remain unit scale (legacy behavior) or be
     multiplied by the selected schedule sigma.
   - Sampling manifests/reports bind the start, scale mode, and exact timestep
     sequence, so resume rejects protocol drift.

3. **Auditable x0 constraints**
   - The legacy hard clip remains the default.
   - New requests may select no constraint or per-sample dynamic thresholding.
   - Dynamic thresholding records its percentile and is implemented in fp32
     before restoring the input dtype.

4. **Fail-closed formal protocol compatibility**
   - Historical sampling reports that omit the new optional fields remain
     valid under the existing protocol contract.
   - Existing milestone/scaling/full formal contracts reject a nonterminal
     start, sigma-scaled initialization, or dynamic thresholding. A future
     experiment must therefore use a new versioned evaluator/gate rather than
     silently replacing locked DDIM evidence.

## Verification

The isolated worktree uses its own `uv` environment. Targeted CPU coverage
includes diffusion/SNR math, loss integration, pair fairness, exact resume,
sampling streams, inference provenance, formal protocol rejection, and legacy
loss behavior. The targeted suite passed. The full repository suite then
passed all `1,088` collected tests (with the repository's existing skips) from
a workspace layout that exposes the unchanged paper tree expected by the
paper-structure tests.

## Authorization

This record and branch are permanently non-authorizing by themselves:

- `training_launch_allowed=false`
- `sampling_launch_allowed=false`
- `full_300k_launch_allowed=false`
- `promotion_or_release_allowed=false`
- `process_signals_allowed=false`

Any GPU experiment requires the terminal machine route to finish first and a
new versioned execution preparation whose exact Git revision, configs, output
root, fairness contract, checkpoint policy, and evaluation gates are bound
after that route selection.
