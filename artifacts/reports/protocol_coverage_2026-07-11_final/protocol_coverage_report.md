# Protocol Coverage Report

Scientific status: `completed_defined_protocols`

| protocol view | observed | expected | complete | status counts |
| --- | --- | --- | --- | --- |
| primary_matched_dataset_step_training | 80 | 80 | True | {"completed": 80} |
| tokenizer_reconstruction_eval_only | 16 | 16 | True | {"completed_eval_only": 16} |
| official_imagenet256_eval_only | 3 | 3 | True | {"completed_eval_only_50k": 3} |

## Guarded Cartesian Product

The master matrix retains 24 cross-task cells as `protocol_blocked`. They must not be relabeled as completed fair training from official/eval-only evidence.

## Interpretation

Paper-facing completion should be judged within each defined protocol: matched-dataset/optimizer-step generation, tokenizer reconstruction, and official ImageNet-256 related-method evaluation. Literal method-by-dataset Cartesian completion remains false by design.

