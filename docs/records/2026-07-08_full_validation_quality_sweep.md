# Full-Validation Reconstruction and Prefix Quality Sweep

Date: 2026-07-08

Purpose: close the strict-audit gap for deterministic full-validation
reconstruction/prefix evidence on the two main 20k seed-103 rows:
epsilon-only and K8 light CoFiTok on Tiny ImageNet-200 and ImageNet-1K
64x64 HF fallback.

## Protocol

Entrypoint:

```bash
PYTHONPATH=src python scripts/evaluate_quality.py
```

Evaluation settings:

- split: `val`
- dataloader: `drop_last=False`
- timestep: `500`
- prefix budgets: `1,2,4,8`
- optional LPIPS/Inception: disabled for the full validation sweep
- deterministic metrics: final MSE, MSE AUC, low-res Frechet proxy AUC

The full validation sweep deliberately avoids calling these rows
publication-scale generated-sample FID. The existing 1024-slice quality rows
remain the source for LPIPS Alex and torchvision Inception Frechet. This record
only establishes full-validation deterministic reconstruction/prefix coverage.

## Results

| dataset | variant | val images | final MSE | MSE AUC | low-res Frechet proxy AUC | report dir |
|---|---|---:|---:|---:|---:|---|
| Tiny ImageNet-200 | epsilon-only | 10000 | 0.126565 | 7.955221 | 5.366480 | `quality_tiny_epsilononly_20k_seed2_fullval_t500_2026-07-08` |
| Tiny ImageNet-200 | K8 light | 10000 | 0.131721 | 6.584363 | 4.946370 | `quality_tiny_k8_light_20k_seed2_fullval_t500_2026-07-08` |
| ImageNet-1K 64x64 HF | epsilon-only | 50000 | 0.107975 | 8.069069 | 5.225075 | `quality_imagenet_hf_epsilononly_20k_seed2_fullval_t500_2026-07-08` |
| ImageNet-1K 64x64 HF | K8 light | 50000 | 0.114178 | 6.604900 | 4.873849 | `quality_imagenet_hf_k8_light_20k_seed2_fullval_t500_2026-07-08` |

## Interpretation

The full-validation endpoint MSE still slightly favors epsilon-only on both
datasets, matching the existing scoped claim that CoFiTok should not claim
unconditional endpoint-quality wins at this scale.

The prefix/path aggregate metrics favor K8 light CoFiTok on both full
validation sets: MSE AUC improves from 7.955221 to 6.584363 on Tiny
ImageNet-200 and from 8.069069 to 6.604900 on ImageNet-1K 64x64 HF. The
low-res Frechet proxy AUC also improves on both datasets. This strengthens the
prefix-controllability and ordered-denoising-component claim without changing
the generated-sample quality limitation.

## Audit Impact

`scripts/validate_goal_completion.py` now treats these four rows as sufficient
full-validation reconstruction/prefix evidence when they are present in
`artifacts/reports/summary_2026-07-08/experiment_summary.json`. The strict goal
audit should therefore remove `full_validation_reconstruction_sweep` from the
open-gap list, while keeping generation-quality, stronger-backbone, exact
ImageNet-64-source, and venue-template gaps open.
