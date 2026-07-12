#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-08}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TINY_COUNT="${TINY_COUNT:-10000}"
HF_COUNT="${HF_COUNT:-50000}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME"

nvidia-smi

run_export_and_fid() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local count="$4"
  local checkpoint="$CHECKPOINT_ROOT/${train_id}_${DATE_TAG}/checkpoint_final.pt"
  local export_dir="$CHECKPOINT_ROOT/official_fid_export_${label}_${count}_ddim${SAMPLE_STEPS}_${DATE_TAG}"
  local manifest="$export_dir/official_fid_export_manifest.json"
  local report="$export_dir/official_fid_report/official_fid_report.json"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[missing-checkpoint] $checkpoint" >&2
    return 1
  fi

  if [[ -f "$manifest" ]]; then
    echo "[skip-export] $label -> $manifest"
  else
    echo "[export] $label -> $export_dir"
    "$PYTHON" scripts/export_official_fid_dirs.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output-dir "$export_dir" \
      --split val \
      --real-count "$count" \
      --sample-count "$count" \
      --batch-size "$BATCH_SIZE" \
      --sample-steps "$SAMPLE_STEPS" \
      --prefix-budget 8
  fi

  if [[ -f "$report" ]]; then
    echo "[skip-fid] $label -> $report"
  else
    echo "[official-fid] $label -> $report"
    "$PYTHON" scripts/evaluate_official_fid_dirs.py \
      --real-dir "$export_dir/real" \
      --generated-dir "$export_dir/generated" \
      --output-dir "$export_dir/official_fid_report" \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      --require-pytorch-fid
  fi
}

run_export_and_fid \
  "tiny_epsilononly_multiscale_20k" \
  "configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k" \
  "$TINY_COUNT"

run_export_and_fid \
  "tiny_k8_light_multiscale_20k" \
  "configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json" \
  "train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k" \
  "$TINY_COUNT"

run_export_and_fid \
  "imagenet_hf_epsilononly_multiscale_20k" \
  "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k" \
  "$HF_COUNT"

run_export_and_fid \
  "imagenet_hf_k8_light_multiscale_20k" \
  "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json" \
  "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k" \
  "$HF_COUNT"
