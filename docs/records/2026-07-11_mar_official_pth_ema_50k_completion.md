# MAR official PTH EMA 50K completion (2026-07-11)

Status: `completed_eval_only_50k`. This is secondary ImageNet-256 evidence and
must not enter the matched-dataset/optimizer-step P0 table.

## Provenance

```text
repo: https://github.com/LTH14/mar
commit: c6d53f7fa6427634b5850ebed771b7c2d19ea21f
model state: checkpoint-last.pth['model_ema']
model sha256: 7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c
VAE state: kl16.ckpt['model']
VAE sha256: 34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f
```

Sampling follows the MAR-B README settings: 50,000 balanced ImageNet classes,
256 AR iterations, 100 diffusion sampling steps, CFG 2.9 with linear schedule,
temperature 1.0, fp16, and seed 0. Peak sampled CUDA allocation was
14,745,516,544 bytes.

## Packed samples

```text
PNG count: 50,000
NPZ key/shape/dtype: arr_0 / [50000, 256, 256, 3] / uint8
NPZ bytes: 9,830,400,360
NPZ sha256: 0bb2d567bfbf883e8087b943465c83cc3ff893bd3f41e927a3074591e264aa4c
```

## Unified ADM evaluator metrics

| FID | sFID | Inception Score | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 2.3385 | 4.6999 | 62.8979 | 0.8216 | 0.5716 |

The FID closely matches the MAR README value 2.31. The README reports
torch-fidelity IS 281.7, while the common ADM graph used for D-AR, MAR, and
ReTok gives IS 62.8979 for this sample set. These IS values are different
evaluation implementations and must not be silently interchanged. Paper tables
use the common ADM evaluator result for all three secondary methods.

Visual checks at class indices 0, 1, 2, 100, 500, and 999 showed the expected
class semantics (for example tench, goldfish, great white shark, black swan,
cliff dwelling, and toilet tissue), so there is no evidence of label offset or
RGB/BGR corruption.

Evaluator assets:

```text
reference: /root/autodl-tmp/CoFiTok/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz
graph sha256: 009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6
```

Runbook and standard report:

```text
artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.sh
artifacts/reports/baselines/mar/official_pth_50k_2026-07-11/baseline_eval_report.json
```
