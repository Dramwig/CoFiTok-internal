#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"

CONFIG="${CONFIG:-configs/train_tiny_imagenet_k8_epsilononly_p150eval_5k_cuda.json}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08/checkpoint_final.pt}"
OUTPUT_ID="${OUTPUT_ID:-generated_quality_stream_tiny_epsilononly_5k_1024_ddim50_${DATE_TAG}}"
OUTPUT_DIR="$CHECKPOINT_ROOT/$OUTPUT_ID"
SUMMARY_DIR="${SUMMARY_DIR:-artifacts/reports/summary_2026-07-09_p0_formal64_tiny_dense_generated}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-artifacts/reports/baselines/summary_2026-07-09_plus_edm_tiny/baseline_summary.json}"
TABLE_DIR="${TABLE_DIR:-artifacts/reports/formal64_p0_horizontal_table_2026-07-09_tiny_external_dense_complete}"
MATRIX_DIR="${MATRIX_DIR:-artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_tiny_external_dense_complete}"
DAR_FEASIBILITY="${DAR_FEASIBILITY:-artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json}"

SAMPLE_COUNT="${SAMPLE_COUNT:-1024}"
REAL_COUNT="${REAL_COUNT:-4096}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"

if [ ! -f "$CHECKPOINT" ]; then
  printf "missing checkpoint: %s\n" "$CHECKPOINT" >&2
  exit 1
fi

if [ ! -f "$OUTPUT_DIR/generated_quality_report.json" ]; then
  "$PYTHON" scripts/evaluate_generated_samples_stream.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --split val \
    --sample-count "$SAMPLE_COUNT" \
    --max-real-images "$REAL_COUNT" \
    --batch-size "$BATCH_SIZE" \
    --sample-steps "$SAMPLE_STEPS" \
    --enable-inception-fid
else
  printf "[skip] %s\n" "$OUTPUT_DIR/generated_quality_report.json"
fi

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir "$SUMMARY_DIR"

"$PYTHON" scripts/build_formal64_paper_table.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-summary "$BASELINE_SUMMARY" \
  --dar-feasibility "$DAR_FEASIBILITY" \
  --output-dir "$TABLE_DIR"

"$PYTHON" scripts/build_paper_comparison_matrix.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-reports-root artifacts/reports/baselines \
  --output-dir "$MATRIX_DIR"

printf "p0_tiny_dense_generated_quality_done %s %s %s\n" "$SUMMARY_DIR" "$TABLE_DIR" "$MATRIX_DIR"
