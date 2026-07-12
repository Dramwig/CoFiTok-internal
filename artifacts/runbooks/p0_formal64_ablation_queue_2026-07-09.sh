#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"
QUALITY_IMAGES="${QUALITY_IMAGES:-512}"
QUALITY_BATCHES="${QUALITY_BATCHES:-16}"

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

run_ablation_suite() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local do_order="$4"
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
}

run_ablation_suite "tiny_imagenet_nopathprefix_5k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_5k_cuda" "0"
run_ablation_suite "tiny_imagenet_cleanmono_5k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_5k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_5k_cuda" "0"
run_ablation_suite "tiny_imagenet_simultaneous_5k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_simultaneous_5k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_simultaneous_5k_cuda" "1"
run_ablation_suite "tiny_imagenet_deepsk_5k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_deepsk_5k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_deepsk_5k_cuda" "0"

run_ablation_suite "downsampled_imagenet64_nopathprefix_5k" "configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json" "train_downsampled_imagenet64_k8_denoisepath_p150_light_nopathprefix_5k_cuda" "0"
run_ablation_suite "downsampled_imagenet64_cleanmono_5k" "configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_cleanmono_5k_cuda.json" "train_downsampled_imagenet64_k8_denoisepath_p150_light_cleanmono_5k_cuda" "0"
run_ablation_suite "downsampled_imagenet64_simultaneous_5k" "configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_simultaneous_5k_cuda.json" "train_downsampled_imagenet64_k8_denoisepath_p150_light_simultaneous_5k_cuda" "1"
run_ablation_suite "downsampled_imagenet64_deepsk_5k" "configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_deepsk_5k_cuda.json" "train_downsampled_imagenet64_k8_denoisepath_p150_light_deepsk_5k_cuda" "0"

run_ablation_suite "ffhq64_nopathprefix_5k" "configs/train_ffhq64_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json" "train_ffhq64_k8_denoisepath_p150_light_nopathprefix_5k_cuda" "0"
run_ablation_suite "ffhq64_cleanmono_5k" "configs/train_ffhq64_k8_denoisepath_p150_light_cleanmono_5k_cuda.json" "train_ffhq64_k8_denoisepath_p150_light_cleanmono_5k_cuda" "0"
run_ablation_suite "ffhq64_simultaneous_5k" "configs/train_ffhq64_k8_denoisepath_p150_light_simultaneous_5k_cuda.json" "train_ffhq64_k8_denoisepath_p150_light_simultaneous_5k_cuda" "1"
run_ablation_suite "ffhq64_deepsk_5k" "configs/train_ffhq64_k8_denoisepath_p150_light_deepsk_5k_cuda.json" "train_ffhq64_k8_denoisepath_p150_light_deepsk_5k_cuda" "0"

run_ablation_suite "afhqv2_64_nopathprefix_5k" "configs/train_afhqv2_64_k8_denoisepath_p150_light_nopathprefix_5k_cuda.json" "train_afhqv2_64_k8_denoisepath_p150_light_nopathprefix_5k_cuda" "0"
run_ablation_suite "afhqv2_64_cleanmono_5k" "configs/train_afhqv2_64_k8_denoisepath_p150_light_cleanmono_5k_cuda.json" "train_afhqv2_64_k8_denoisepath_p150_light_cleanmono_5k_cuda" "0"
run_ablation_suite "afhqv2_64_simultaneous_5k" "configs/train_afhqv2_64_k8_denoisepath_p150_light_simultaneous_5k_cuda.json" "train_afhqv2_64_k8_denoisepath_p150_light_simultaneous_5k_cuda" "1"
run_ablation_suite "afhqv2_64_deepsk_5k" "configs/train_afhqv2_64_k8_denoisepath_p150_light_deepsk_5k_cuda.json" "train_afhqv2_64_k8_denoisepath_p150_light_deepsk_5k_cuda" "0"

"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-09_p0_formal64_ablation_refresh

"$PYTHON" scripts/build_paper_comparison_matrix.py \
  --matrix docs/experiment_conditions/paper_comparison_matrix_2026-07-09.json \
  --summary artifacts/reports/summary_2026-07-09_p0_formal64_ablation_refresh/experiment_summary.json \
  --baseline-reports-root artifacts/reports/baselines \
  --output-dir artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_ablation_refresh

echo "p0_formal64_ablation_queue_done"
