#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-08}"
SAMPLE_COUNT="${SAMPLE_COUNT:-50000}"
REAL_COUNT="${REAL_COUNT:-50000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME" artifacts/reports

run_stream() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local output_id="generated_quality_stream_${label}_${SAMPLE_COUNT}_ddim${SAMPLE_STEPS}_${DATE_TAG}"
  local checkpoint="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}/checkpoint_final.pt"
  local marker="$CHECKPOINT_ROOT/$output_id/generated_quality_report.json"

  if [[ -f "$marker" ]]; then
    echo "[skip] $marker"
  else
    echo "[run] $output_id"
    "$PYTHON" scripts/evaluate_generated_samples_stream.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output-dir "$CHECKPOINT_ROOT/$output_id" \
      --split val \
      --sample-count "$SAMPLE_COUNT" \
      --max-real-images "$REAL_COUNT" \
      --batch-size "$BATCH_SIZE" \
      --sample-steps "$SAMPLE_STEPS" \
      --prefix-budget 8 \
      --enable-inception-fid
  fi
}

run_stream \
  "imagenet_hf_epsilononly_20k_seed2" \
  "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda"

run_stream \
  "imagenet_hf_k8_light_20k_seed2" \
  "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda"

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-08
