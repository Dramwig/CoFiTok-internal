# 2026-07-08 Next Step Decision

## Current Status

The CoFiTok MVP is scoped-ready and strict-complete for the current audit.

Current evidence inventory:

- train rows: 111
- order-eval rows: 95
- quality rows: 58
- sampling rows: 22
- generated-quality rows: 29
- official-FID rows: 4
- datasets covered in experiments: CIFAR-10, Tiny ImageNet-200, and ImageNet-family 64x64 via `imagenet_1k_64x64_hf`
- strict source staged: `downsampled_imagenet_64` Academic Torrents payload, recorded separately from the HF fallback
- paper artifacts: venue-neutral LaTeX draft compiled to `paper/latex/main.pdf`; AAAI-27 anonymous-submission adaptation compiled to `paper/venues/aaai27/main.pdf`

The final remote recovery package was synchronized to `pro6000`, extracted under `/root/autodl-tmp/CoFiTok`, and revalidated there. Summary consistency, goal audit, full pytest, runbook syntax checks, and project symlink checks passed. The GPU was idle after verification.

## Main Interpretation

The method implementation and MVP validation are complete enough for the scoped claim:

> CoFiTok factorizes dense pixel-space diffusion noise prediction into ordered compressed denoising components. Restricted, condition-free synthesis operators keep `S_k` from becoming a decoder, and token prefixes provide controllable partial denoising.

The evidence supports prefix controllability, ordering diagnostics, restricted-`S_k` behavior, seed repeats, token-count scaling, loss ablations, and multi-dataset operation.

The evidence does not support claiming a robust unconditional generation-quality win. The matched multiscale 20k generated-quality controls improved the absolute sample metrics, but epsilon-only remained better than K8 light CoFiTok on the streamed Inception-Frechet proxy for both Tiny ImageNet and ImageNet-family 64x64. The external `pytorch-fid` image-directory protocol is mixed: HF ImageNet-family favors K8 light, while Tiny ImageNet-200 favors epsilon-only. Therefore the paper should position generated quality as measured but not central.

## Open Strict Gap

None for the current audit. The previous `exact_downsampled_imagenet64_source`
gap is closed by the 2026-07-09 strict source record and manifest.

## Decision

Do not spend the next cycle trying to force a generation-quality win. That would move the project toward a different claim and would likely require a stronger diffusion baseline and longer training rather than clarifying the CoFiTok contribution.

The next cycle should be final paperization and claim hardening:

1. Keep the central claim on restricted ordered noise-factorization and prefix controllability.
2. Use the AAAI-27 adaptation as the working submission build unless a different target venue is chosen.
3. Tighten figures/tables around the supported claims: prefix curves, order ablations, zero/random/shuffle diagnostics, deep-`S_k` risk, and matched multiscale endpoint quality.
4. Treat official FID as a reported diagnostic, not the primary claim.

If a later paper revision needs exact-source result rows rather than source
availability, rerun the selected HF fallback experiments on
`downsampled_imagenet_64` and keep both aliases separate in tables.
