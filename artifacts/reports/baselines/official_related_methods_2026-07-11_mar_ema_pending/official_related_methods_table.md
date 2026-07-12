# Official Related-Method Table

This table is separate from the P0 same-dataset, same-budget generation table.

| method | dataset | protocol | status | samples | FID | sFID | IS | precision | recall | role |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D-AR | imagenet_256 | official pretrained eval-only | completed_eval_only_50k | 50000 | 2.6281 | 6.7195 | 285.8914 | 0.8029 | 0.5884 | secondary related-method only |
| MAR | imagenet_256 | community non-EMA conversion audit | community_non_ema_audit_completed_official_ema_pending | 50000 | 5.5610 | 6.2549 | 52.0020 | 0.7166 | 0.6248 | secondary related-method only |
| ReTok | imagenet_256 | official pretrained eval-only | completed_eval_only_50k | 50000 | 2.2189 | 5.8956 | 245.9392 | 0.8159 | 0.5994 | secondary related-method only |

Interpretation rule: completed rows here may be cited only as official-checkpoint related-method evidence. They are not fair retraining baselines for the all-dataset P0 matrix.
