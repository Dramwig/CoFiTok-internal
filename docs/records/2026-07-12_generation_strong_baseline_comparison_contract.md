# Strong-baseline comparison contract for large-scale generation

Date: 2026-07-12

Branch: `scale/generative-system`

## Comparison tiers

The final ImageNet-256 generation report is split into two non-interchangeable
tiers.

### Matched-training direct comparison

- CoFiTok K=8 versus `dense_identity`;
- identical dataset, resolution, backbone family, optimizer, training steps,
  class conditioning, sample count, sampler settings, and torch-fidelity
  evaluator;
- parameter counts and checkpoint/sample-set SHA256 values are recorded;
- relative FID is computed only inside this tier.

### Official-pretrained contextual comparison

- D-AR, MAR PTH `model_ema`, and ReTok;
- completed official ImageNet-256 50K eval-only evidence;
- different pretrained model families and training budgets;
- metrics produced by the common pinned ADM TensorFlow evaluation graph;
- retained only as `secondary related-method only` context.

The machine-readable report sets `cross_tier_numeric_ranking_allowed=false`.
It rejects missing/incomplete external rows, wrong dataset/resolution/sample
count, non-finite metrics, or any external row relabeled as a direct comparison.

## Outputs

After full post-evaluation, the runbook writes atomic JSON, Markdown, and CSV
under:

```text
artifacts/reports/generation/imagenet256_full_matched_300k/comparison/
```

The report preserves a failed full gate as `hold`; producing a comparison table
does not itself declare the generation system ready.
