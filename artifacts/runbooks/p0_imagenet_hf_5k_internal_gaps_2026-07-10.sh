#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-10_imagenet_hf_5k_internal_gaps}"

QUALITY_IMAGES="${QUALITY_IMAGES:-512}"
QUALITY_BATCHES="${QUALITY_BATCHES:-16}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-32}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1024}"
REAL_COUNT="${REAL_COUNT:-4096}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

SUMMARY_DIR="${SUMMARY_DIR:-artifacts/reports/summary_2026-07-10_imagenet_hf_5k_internal_gaps}"
BASELINE_SUMMARY_DIR="${BASELINE_SUMMARY_DIR:-artifacts/reports/baselines/summary_2026-07-10_imagenet_hf_5k_internal_gaps}"
MATRIX_DIR="${MATRIX_DIR:-artifacts/reports/paper_comparison_matrix_2026-07-10_imagenet_hf_5k_internal_gaps}"
P0_TABLE_DIR="${P0_TABLE_DIR:-artifacts/reports/p0_64_horizontal_table_2026-07-10_imagenet_hf_5k_complete}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"

run_if_missing() {
  local marker="$1"
  shift
  if [ -f "$marker" ]; then
    printf "[skip] %s\n" "$marker"
  else
    printf "[run] %s\n" "$*"
    "$@"
  fi
}

run_train_quality() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local train_dir="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}"
  local checkpoint="$train_dir/checkpoint_final.pt"
  local quality_id="quality_${label}_${QUALITY_IMAGES}_t500_lpips_inception_${DATE_TAG}"

  run_if_missing "$train_dir/report.json" \
    "$PYTHON" scripts/train_short.py \
      --config "$config" \
      --output-dir "$train_dir"

  run_if_missing "$CHECKPOINT_ROOT/$quality_id/quality_report.json" \
    "$PYTHON" scripts/evaluate_quality.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output-dir "$CHECKPOINT_ROOT/$quality_id" \
      --split val \
      --max-batches "$QUALITY_BATCHES" \
      --max-images "$QUALITY_IMAGES" \
      --timestep 500 \
      --enable-lpips \
      --enable-inception-fid
}

run_generated_quality() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local generated_quality_id="generated_quality_stream_${label}_${SAMPLE_COUNT}_ddim${SAMPLE_STEPS}_${DATE_TAG}"

  run_if_missing "$CHECKPOINT_ROOT/$generated_quality_id/generated_quality_report.json" \
    "$PYTHON" scripts/evaluate_generated_samples_stream.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output-dir "$CHECKPOINT_ROOT/$generated_quality_id" \
      --split val \
      --sample-count "$SAMPLE_COUNT" \
      --max-real-images "$REAL_COUNT" \
      --batch-size "$SAMPLE_BATCH_SIZE" \
      --sample-steps "$SAMPLE_STEPS" \
      --enable-inception-fid
}

run_train_quality \
  "imagenet_hf_nopathprefix_5k" \
  "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_5k_cuda"

run_train_quality \
  "imagenet_hf_cleanmono_5k" \
  "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_5k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_5k_cuda"

run_generated_quality \
  "imagenet_hf_epsilononly_5k_seed2" \
  "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_seed2_cuda.json" \
  "$CHECKPOINT_ROOT/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_seed2_2026-07-08/checkpoint_final.pt"

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir "$SUMMARY_DIR"

"$PYTHON" scripts/baselines/summarize_baseline_reports.py \
  --reports-root artifacts/reports/baselines \
  --output-dir "$BASELINE_SUMMARY_DIR"

"$PYTHON" scripts/build_paper_comparison_matrix.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-reports-root artifacts/reports/baselines \
  --output-dir "$MATRIX_DIR"

"$PYTHON" scripts/build_p0_64_paper_table.py \
  --summary "$SUMMARY_DIR/experiment_summary.json" \
  --baseline-summary "$BASELINE_SUMMARY_DIR/baseline_summary.json" \
  --dar-feasibility artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json \
  --output-dir "$P0_TABLE_DIR"

printf "p0_imagenet_hf_5k_internal_gaps_done %s %s %s %s\n" "$SUMMARY_DIR" "$BASELINE_SUMMARY_DIR" "$MATRIX_DIR" "$P0_TABLE_DIR"
