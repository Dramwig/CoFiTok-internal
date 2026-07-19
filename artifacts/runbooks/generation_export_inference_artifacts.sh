#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
EXPORT_ROOT="$OUTPUT_ROOT/exports/imagenet256_full_300k"
COFITOK_ARTIFACT="$EXPORT_ROOT/cofitok_k8_ema_inference.pt"
DENSE_ARTIFACT="$EXPORT_ROOT/dense_identity_ema_inference.pt"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
eval "$(python scripts/print_generation_workspace_paths.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" --format shell)"
COFITOK_CHECKPOINT="$FULL_COFITOK_RUN/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$FULL_DENSE_RUN/checkpoint_step_00300000.pt"
REPORT_ROOT="$FULL_REPORT_ROOT/exports"
FINAL_GATE="$FULL_GATE"
mkdir -p "$EXPORT_ROOT" "$REPORT_ROOT"

test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"
test -f "$FINAL_GATE"

python scripts/export_generation_inference_artifact.py \
  --checkpoint "$COFITOK_CHECKPOINT" --output "$COFITOK_ARTIFACT" \
  --release-gate "$FINAL_GATE" \
  --report "$REPORT_ROOT/cofitok_export_report.json"

python scripts/export_generation_inference_artifact.py \
  --checkpoint "$DENSE_CHECKPOINT" --output "$DENSE_ARTIFACT" \
  --release-gate "$FINAL_GATE" \
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
