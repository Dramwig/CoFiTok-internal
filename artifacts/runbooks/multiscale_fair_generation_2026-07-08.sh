#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-08}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TINY_SAMPLE_COUNT="${TINY_SAMPLE_COUNT:-10000}"
TINY_REAL_COUNT="${TINY_REAL_COUNT:-10000}"
HF_SAMPLE_COUNT="${HF_SAMPLE_COUNT:-50000}"
HF_REAL_COUNT="${HF_REAL_COUNT:-50000}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME" artifacts/reports

nvidia-smi

run_train() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local train_dir="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}"
  local marker="$train_dir/checkpoint_final.pt"

  if [[ -f "$marker" ]]; then
    echo "[skip-train] $label -> $marker"
  else
    echo "[train] $label -> $train_dir"
    "$PYTHON" scripts/train_short.py \
      --config "$config" \
      --output-dir "$train_dir"
  fi
}

run_stream() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local sample_count="$4"
  local real_count="$5"
  local output_id="generated_quality_stream_${label}_${sample_count}_ddim${SAMPLE_STEPS}_${DATE_TAG}"
  local checkpoint="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}/checkpoint_final.pt"
  local marker="$CHECKPOINT_ROOT/$output_id/generated_quality_report.json"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[missing-checkpoint] $checkpoint" >&2
    return 1
  fi

  if [[ -f "$marker" ]]; then
    echo "[skip-stream] $label -> $marker"
  else
    echo "[stream] $label -> $CHECKPOINT_ROOT/$output_id"
    "$PYTHON" scripts/evaluate_generated_samples_stream.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output-dir "$CHECKPOINT_ROOT/$output_id" \
      --split val \
      --sample-count "$sample_count" \
      --max-real-images "$real_count" \
      --batch-size "$BATCH_SIZE" \
      --sample-steps "$SAMPLE_STEPS" \
      --prefix-budget 8 \
      --enable-inception-fid
  fi
}

run_train \
  "tiny_epsilononly_multiscale_20k" \
  "configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k"

run_train \
  "imagenet_hf_epsilononly_multiscale_20k" \
  "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k"

run_stream \
  "tiny_epsilononly_multiscale_20k" \
  "configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k" \
  "$TINY_SAMPLE_COUNT" \
  "$TINY_REAL_COUNT"

run_stream \
  "tiny_k8_light_multiscale_20k" \
  "configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json" \
  "train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k" \
  "$TINY_SAMPLE_COUNT" \
  "$TINY_REAL_COUNT"

run_stream \
  "imagenet_hf_epsilononly_multiscale_20k" \
  "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k" \
  "$HF_SAMPLE_COUNT" \
  "$HF_REAL_COUNT"

run_stream \
  "imagenet_hf_k8_light_multiscale_20k" \
  "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k" \
  "$HF_SAMPLE_COUNT" \
  "$HF_REAL_COUNT"

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-08

"$PYTHON" scripts/validate_synthesis_contract.py --config-glob 'configs/*.json'
