# CoFiTok Top-Tier Claim Audit

Scoped evidence ready: **True**
Broad generation superiority supported: **False**

## Required checks

| check | result |
| --- | --- |
| `broad_short_budget_path_auc` | PASS |
| `matched_20k_two_seed_path_auc` | PASS |
| `matched_20k_endpoint_cost_within_5pct` | PASS |
| `imagenet256_confirmatory_all_gates` | PASS |
| `restricted_zero_token_contract` | PASS |
| `defined_protocols_complete` | PASS |
| `official_related_50k_complete` | PASS |

## Residual risks

- The strongest evidence is mechanistic prefix control, not generation SOTA.
- Architecture-incompatible D-AR/MAR/ReTok rows are official eval-only, not matched-dataset/step retraining.
- The main repeated experiments still use small backbones and two seeds.
- ReTok's pinned repository does not declare a license; do not redistribute its code or weights.

A true scoped_top_tier_evidence_ready value means the experimental package is coherent enough to support the scoped submission claim. It is not a prediction or guarantee of venue acceptance.
