# Protocol Coverage Report

Scientific status: `official_eval_running`

| protocol view | observed | expected | complete | status counts |
| --- | --- | --- | --- | --- |
| primary_same_budget_training | 72 | 72 | True | {"completed": 72} |
| tokenizer_reconstruction_eval_only | 16 | 16 | True | {"completed_eval_only": 16} |
| official_imagenet256_eval_only | 3 | 3 | False | {"completed_eval_only_50k": 2, "sampling_or_eval_pending": 1} |

## Guarded Cartesian Product

The master matrix retains 24 cross-task cells as `protocol_blocked`. They must not be relabeled as completed fair training from official/eval-only evidence.

## Interpretation

Paper-facing completion should be judged within each defined protocol: same-budget generation, tokenizer reconstruction, and official ImageNet-256 related-method evaluation. Literal method-by-dataset Cartesian completion remains false by design.

