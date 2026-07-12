#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-10_imagenet_hf_external}"

SUMMARY_DIR="${SUMMARY_DIR:-artifacts/reports/summary_2026-07-10_imagenet_hf_external}"
BASELINE_SUMMARY_DIR="${BASELINE_SUMMARY_DIR:-artifacts/reports/baselines/summary_2026-07-10_imagenet_hf_external}"
MATRIX_DIR="${MATRIX_DIR:-artifacts/reports/paper_comparison_matrix_2026-07-10_imagenet_hf_external}"

cd "$CODE_DIR"

DATASETS=imagenet_1k_64x64_hf \
DATE_TAG="$DATE_TAG" \
TRAIN_STEPS=5000 \
IMAGE_SIZE=64 \
PYTHON="$PYTHON" \
bash artifacts/runbooks/improved_diffusion_formal64_queue_2026-07-09.sh

DATASETS=imagenet_1k_64x64_hf \
DATE_TAG="$DATE_TAG" \
TRAIN_STEPS=5000 \
PYTHON="$PYTHON" \
bash artifacts/runbooks/edm_formal64_queue_2026-07-09.sh

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$PROJECT_ROOT/checkpoints" \
  --output-dir "$SUMMARY_DIR"

"$PYTHON" scripts/baselines/summarize_baseline_reports.py \
  --reports-root artifacts/reports/baselines \
  --output-dir "$BASELINE_SUMMARY_DIR"

"$PYTHON" scripts/build_paper_comparison_matrix.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-reports-root artifacts/reports/baselines \
  --output-dir "$MATRIX_DIR"

printf "p0_imagenet_hf_external_done %s %s %s\n" "$SUMMARY_DIR" "$BASELINE_SUMMARY_DIR" "$MATRIX_DIR"
