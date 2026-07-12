#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"
QUALITY_IMAGES="${QUALITY_IMAGES:-512}"
QUALITY_BATCHES="${QUALITY_BATCHES:-16}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-32}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1024}"
REAL_COUNT="${REAL_COUNT:-4096}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME" artifacts/reports artifacts/logs

run_if_missing() {
  local marker="$1"
  shift
  if [[ -f "$marker" ]]; then
    echo "[skip] $marker"
  else
    echo "[run] $*"
    "$@"
  fi
}

run_internal_suite() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local do_order="$4"
  local do_generated="$5"
  local train_dir="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}"
  local checkpoint="$train_dir/checkpoint_final.pt"

  run_if_missing "$train_dir/report.json" \
    "$PYTHON" scripts/train_short.py \
      --config "$config" \
      --output-dir "$train_dir"

  local quality_id="quality_${label}_${QUALITY_IMAGES}_t500_lpips_inception_${DATE_TAG}"
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

  if [[ "$do_order" == "1" ]]; then
    for component_order in ordered random reverse; do
      local order_id="order_${label}_${component_order}_${DATE_TAG}"
      run_if_missing "$CHECKPOINT_ROOT/$order_id/report.json" \
        "$PYTHON" scripts/evaluate_checkpoint.py \
          --config "$config" \
          --checkpoint "$checkpoint" \
          --output-dir "$CHECKPOINT_ROOT/$order_id" \
          --component-order "$component_order" \
          --random-order-seed 0
    done
  fi

  if [[ "$do_generated" == "1" ]]; then
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
  fi
}

run_internal_suite "downsampled_imagenet64_k8_light_5k" "configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json" "train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda" "1" "1"
run_internal_suite "downsampled_imagenet64_epsilononly_5k" "configs/train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda.json" "train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda" "0" "1"
run_internal_suite "downsampled_imagenet64_channelmask_5k" "configs/train_downsampled_imagenet64_k8_channelmask_p150eval_5k_cuda.json" "train_downsampled_imagenet64_k8_channelmask_p150eval_5k_cuda" "0" "0"

run_internal_suite "ffhq64_k8_light_5k" "configs/train_ffhq64_k8_denoisepath_p150_light_5k_cuda.json" "train_ffhq64_k8_denoisepath_p150_light_5k_cuda" "1" "1"
run_internal_suite "ffhq64_epsilononly_5k" "configs/train_ffhq64_k8_epsilononly_p150eval_5k_cuda.json" "train_ffhq64_k8_epsilononly_p150eval_5k_cuda" "0" "1"
run_internal_suite "ffhq64_channelmask_5k" "configs/train_ffhq64_k8_channelmask_p150eval_5k_cuda.json" "train_ffhq64_k8_channelmask_p150eval_5k_cuda" "0" "0"

run_internal_suite "afhqv2_64_k8_light_5k" "configs/train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda.json" "train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda" "1" "1"
run_internal_suite "afhqv2_64_epsilononly_5k" "configs/train_afhqv2_64_k8_epsilononly_p150eval_5k_cuda.json" "train_afhqv2_64_k8_epsilononly_p150eval_5k_cuda" "0" "1"
run_internal_suite "afhqv2_64_channelmask_5k" "configs/train_afhqv2_64_k8_channelmask_p150eval_5k_cuda.json" "train_afhqv2_64_k8_channelmask_p150eval_5k_cuda" "0" "0"

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-09_p0_formal64

"$PYTHON" scripts/build_paper_comparison_matrix.py \
  --matrix docs/experiment_conditions/paper_comparison_matrix_2026-07-09.json \
  --summary artifacts/reports/summary_2026-07-09_p0_formal64/experiment_summary.json \
  --output-dir artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64

echo "p0_formal64_queue_done"
