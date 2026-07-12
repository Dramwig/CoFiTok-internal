#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_CHECKPOINT="$OUTPUT_ROOT/imagenet256_full_cofitok_k8_300k/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$OUTPUT_ROOT/imagenet256_full_dense_300k/checkpoint_step_00300000.pt"
EXPORT_ROOT="$OUTPUT_ROOT/exports/imagenet256_full_300k"
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/exports"
COFITOK_ARTIFACT="$EXPORT_ROOT/cofitok_k8_ema_inference.pt"
DENSE_ARTIFACT="$EXPORT_ROOT/dense_identity_ema_inference.pt"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$EXPORT_ROOT" "$REPORT_ROOT"

test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"

python scripts/export_generation_inference_artifact.py \
  --checkpoint "$COFITOK_CHECKPOINT" --output "$COFITOK_ARTIFACT" \
  --report "$REPORT_ROOT/cofitok_export_report.json"

python scripts/export_generation_inference_artifact.py \
  --checkpoint "$DENSE_CHECKPOINT" --output "$DENSE_ARTIFACT" \
  --report "$REPORT_ROOT/dense_export_report.json"

python scripts/preflight_generation_sampling.py \
  --checkpoint "$COFITOK_ARTIFACT" \
  --output "$REPORT_ROOT/cofitok_export_preflight.json" \
  --batch-size 2 --prefix-budget 8 --guidance-scale 1.5 \
  --cfg-batch-mode batched --weights ema --precision bf16 \
  --warmup-forwards 1 --measured-forwards 2

python scripts/preflight_generation_sampling.py \
  --checkpoint "$DENSE_ARTIFACT" \
  --output "$REPORT_ROOT/dense_export_preflight.json" \
  --batch-size 2 --prefix-budget 1 --guidance-scale 1.5 \
  --cfg-batch-mode batched --weights ema --precision bf16 \
  --warmup-forwards 1 --measured-forwards 2

python scripts/infer_generation.py \
  --checkpoint "$COFITOK_ARTIFACT" \
  --output-dir "$EXPORT_ROOT/smoke/cofitok" \
  --report "$REPORT_ROOT/cofitok_export_inference_smoke.json" \
  --class-ids 0 --seeds 0,1 --prefix-budgets 1,8 \
  --batch-size 2 --sample-steps 10 --guidance-scale 1.5 \
  --weights ema --precision bf16 --overwrite

python scripts/infer_generation.py \
  --checkpoint "$DENSE_ARTIFACT" \
  --output-dir "$EXPORT_ROOT/smoke/dense_identity" \
  --report "$REPORT_ROOT/dense_export_inference_smoke.json" \
  --class-ids 0 --seeds 0,1 --prefix-budgets 1 \
  --batch-size 2 --sample-steps 10 --guidance-scale 1.5 \
  --weights ema --precision bf16 --overwrite
