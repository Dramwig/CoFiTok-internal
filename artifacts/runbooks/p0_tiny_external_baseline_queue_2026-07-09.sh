#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"
SUMMARY_DIR="${SUMMARY_DIR:-artifacts/reports/summary_2026-07-09_p0_formal64_ablation_refresh}"
BASELINE_SUMMARY_DIR="${BASELINE_SUMMARY_DIR:-artifacts/reports/baselines/summary_2026-07-09_plus_edm_tiny}"
TABLE_DIR="${TABLE_DIR:-artifacts/reports/formal64_p0_horizontal_table_2026-07-09_tiny_external}"
DAR_FEASIBILITY="${DAR_FEASIBILITY:-artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json}"

cd "$CODE_DIR"

if [ "${RUN_IMPROVED_DIFFUSION:-1}" = "1" ]; then
  DATASETS=tiny_imagenet_200 \
  DATE_TAG="$DATE_TAG" \
  PYTHON="$PYTHON" \
  bash artifacts/runbooks/improved_diffusion_formal64_queue_2026-07-09.sh
fi

if [ "${RUN_EDM:-1}" = "1" ]; then
  DATASETS=tiny_imagenet_200 \
  DATE_TAG="$DATE_TAG" \
  PYTHON="$PYTHON" \
  bash artifacts/runbooks/edm_formal64_queue_2026-07-09.sh
fi

"$PYTHON" scripts/baselines/summarize_baseline_reports.py \
  --reports-root artifacts/reports/baselines \
  --output-dir "$BASELINE_SUMMARY_DIR"

"$PYTHON" scripts/build_formal64_paper_table.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-summary "$BASELINE_SUMMARY_DIR/baseline_summary.json" \
  --dar-feasibility "$DAR_FEASIBILITY" \
  --output-dir "$TABLE_DIR"

printf "p0_tiny_external_baseline_queue_done %s %s\n" "$BASELINE_SUMMARY_DIR" "$TABLE_DIR"
