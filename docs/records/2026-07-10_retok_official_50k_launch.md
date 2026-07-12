# ReTok Official ImageNet-256 50K Eval - 2026-07-10

## Decision

ReTok official ImageNet-256 eval-only is complete as a secondary related-method row. Do not move it into the P0 same-budget all-dataset completed table; it uses official pretrained checkpoints rather than our fair retraining protocol.

## Pilot Evidence

The official ReTok `GPT-XL` + `VQ_SB256` checkpoint path was first validated with a 128-sample end-to-end pilot: PNG generation, NPZ construction, and ADM evaluator metrics all completed.

Pilot sample directory:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_pilot128
```

Pilot metrics:

```text
Inception Score: 64.45208740234375
FID: 175.45103041378474
sFID: 598.2384348000761
Precision: 0.7734375
Recall: 0.4237
```

## 50K Run

Launch log:

```text
CoFiTok-internal/artifacts/reports/baselines/retok/official_50k_2026-07-10/retok_50k.log
```

Output directory:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k
```

Final artifacts:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k/GPT-XL-size-256-size-256-topk-0-topp-1.0-temperature-1.0-cfg-1.5-step-seed-0.npz
/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k/GPT-XL-size-256-size-256-topk-0-topp-1.0-temperature-1.0-cfg-1.5-step-seed-0.txt
CoFiTok-internal/artifacts/reports/baselines/retok/official_50k_2026-07-10/baseline_eval_report.json
```

Command:

```bash
METHOD=retok DRY_RUN=0 NUM_IMAGES=50000 \
RETOK_SAMPLE_DIR=/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k \
EVAL_BATCH_PER_GPU=64 GPUS=1 PORT=55642 EVAL_PYTHON_PATH=python \
bash artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
```

Run notes:

```text
sampled PNGs: 50,048
packed samples in NPZ: first 50,000
per-GPU batch size: 64
```

Final 50K metrics:

```text
Inception Score: 245.939208984375
FID: 2.2188685477057106
sFID: 5.895597453788696
Precision: 0.81586
Recall: 0.5994
```

## Paper Handling

- ReTok may be cited only in the secondary official-checkpoint related-method table.
- It is not a same-budget retrained baseline and must not be merged into the P0 all-dataset generation table.
- Matrix status for same-budget ReTok rows remains `protocol_blocked`.