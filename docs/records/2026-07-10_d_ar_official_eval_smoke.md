# D-AR Official Eval Smoke - 2026-07-10

## Decision

D-AR official ImageNet-256 eval-only is now runnable on `pro6000` at smoke
scale. This proves the official-checkpoint path can download/load weights,
sample images, build an NPZ, and run the ADM TensorFlow evaluator.

Do not report the smoke metrics in the paper; only a 50k run can be used as a
paper secondary related-method result.

## Artifacts

Report:

```text
CoFiTok-internal/artifacts/reports/baselines/d_ar/official_imagenet256_smoke_16_2026-07-10/baseline_eval_report.json
```

Remote sample outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.npz
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.txt
```

Weights:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/weights/D-AR-tokenizer_v1.pt
/root/autodl-tmp/CoFiTok/checkpoints/baselines/d_ar/official_imagenet256_eval_only/weights/D-AR-L-360K.pt
```

Reference batch:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz
```

## Smoke Metrics

`NUM_IMAGES=16`, `EVAL_BATCH_PER_GPU=4`.

```text
Inception Score: 12.787610054016113
FID: 314.33965908199684
sFID: 869.6633804653083
Precision: 1.0
Recall: 0.4626
```

These values are only pipeline diagnostics.

## Notes

- The first `wget` download of `D-AR-tokenizer_v1.pt` produced a corrupt file;
  it was replaced with `hf download showlab/D-AR D-AR-tokenizer_v1.pt`.
- `D-AR-L-360K.pt` and `D-AR-tokenizer_v1.pt` were verified with `torch.load`.
- `tensorflow-cpu==2.21.0` was installed into `pf-vlm` for the ADM evaluator.
- The runbook now uses `hf download` for D-AR weights.

## Next

For a paper-reportable secondary row, run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
METHOD=d_ar DRY_RUN=0 NUM_IMAGES=50000 EVAL_BATCH_PER_GPU=64 \
  bash artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
```

Keep the result separate from the P0 same-budget table.
