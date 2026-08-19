# Generation training-exposure audit

This report makes dataset-normalized training exposure explicit and
source-bound. It prevents equal optimizer-step or equal image-count labels from
being interpreted as equal dataset exposure when the dataset aliases differ.

## Audited rows

| row | dataset | steps | images seen | equivalent epochs | status |
|---|---|---:|---:|---:|---|
| frozen 10% CoFiTok | `imagenet_256_10pct` | 50,000 | 3,200,000 | `24.9686` | complete |
| frozen 10% dense | `imagenet_256_10pct` | 50,000 | 3,200,000 | `24.9686` | complete |
| full-data CoFiTok milestone | `imagenet_256` | 50,000 | 3,200,000 | `2.4977` | partial, target 100K |

The active full-data 100K plan reaches only `4.9954` equivalent epochs. It
uses twice as many training images as the frozen 10% 50K source pair, but only
`20.0069%` of its dataset-normalized epoch exposure. The full-data 50K
milestone is `10.0035%` of the source pair's epoch exposure.

This does not make the 10% result a fairer or stronger generator result. It
shows why FID differences across those rows cannot be attributed to model
design or sampler changes without explicit dataset, checkpoint, evaluator,
sampling-protocol, and random-stream binding.

## Hardened contract

The quality-bridge preparation validator now verifies rather than merely
passes through:

- formal train-image counts;
- exact step, effective-batch, and images-seen arithmetic;
- exact formal dataset identity, rather than dataset alias alone;
- source and bridge equivalent epochs;
- both milestone equivalent-epoch values.

The terminal result builder also surfaces the validated exposure contract.
This implementation is intentionally not deployed into the active pinned
quality-bridge runbook while training is in progress. It is post-run audit and
future-pipeline hardening.

Machine-readable evidence: `training_exposure_report.json` (`6,298` bytes,
SHA256 `e1a557a4c54a0813af438a3237bfb4e540944198937062c4b26ea25bea7edaef`).
