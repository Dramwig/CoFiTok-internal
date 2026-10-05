# Official Related-Method Table

This table is separate from the P0 same-dataset, same-budget generation table.

| method | dataset | protocol | status | samples | FID | sFID | IS | precision | recall | role |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D-AR | imagenet_256 | official pretrained eval-only | sampling_or_eval_pending | 50048 | 2.6281 | 6.7195 | 285.8914 | 0.8029 | 0.5884 | secondary related-method only |
| MAR | imagenet_256 | official LTH14 PTH model_ema eval-only | official_ema_sampling_or_eval_pending | 24576 |  |  |  |  |  | secondary related-method only |
| ReTok | imagenet_256 | official pretrained eval-only | pilot128_metrics_completed_50k_running | 128 | 175.4510 | 598.2384 | 64.4521 | 0.7734 | 0.4237 | secondary related-method only |

Interpretation rule: completed rows here may be cited only as official-checkpoint related-method evidence. They are not fair retraining baselines for the all-dataset P0 matrix.
