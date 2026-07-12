#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-08}"
QUALITY_IMAGES="${QUALITY_IMAGES:-1024}"
QUALITY_BATCHES="${QUALITY_BATCHES:-64}"
SAMPLE_COUNT="${SAMPLE_COUNT:-2048}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
REAL_COUNT="${REAL_COUNT:-8192}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-32}"
REQUIRE_PUBLICATION_READY="${REQUIRE_PUBLICATION_READY:-1}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME" artifacts/reports

if [[ ! -x "$PYTHON" ]]; then
  echo "Configured PYTHON is not executable: $PYTHON" >&2
  exit 2
fi

write_queue_status() {
  local phase="$1"
  "$PYTHON" scripts/inspect_next_validation_queue.py \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --date-tag "$DATE_TAG" \
    --quality-images "$QUALITY_IMAGES" \
    --sample-count "$SAMPLE_COUNT" \
    --sample-steps "$SAMPLE_STEPS" \
    --output "artifacts/reports/next_validation_queue_status_${phase}_${DATE_TAG}.json"
}

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

run_suite() {
  local label="$1"
  local config="$2"
  local train_id="$3"
  local order_eval="$4"
  local sample_eval="$5"
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

  if [[ "$order_eval" == "1" ]]; then
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

  if [[ "$sample_eval" == "1" ]]; then
    local sample_id="generated_${label}_${SAMPLE_COUNT}_ddim${SAMPLE_STEPS}_${DATE_TAG}"
    run_if_missing "$CHECKPOINT_ROOT/$sample_id/sample_report.json" \
      "$PYTHON" scripts/sample_checkpoint.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --output-dir "$CHECKPOINT_ROOT/$sample_id" \
        --num-samples "$SAMPLE_COUNT" \
        --batch-size "$SAMPLE_BATCH_SIZE" \
        --sample-steps "$SAMPLE_STEPS" \
        --prefix-budgets 8 \
        --save-images

    local generated_quality_id="generated_quality_${label}_${SAMPLE_COUNT}_ddim${SAMPLE_STEPS}_${DATE_TAG}"
    run_if_missing "$CHECKPOINT_ROOT/$generated_quality_id/generated_quality_report.json" \
      "$PYTHON" scripts/evaluate_generated_samples.py \
        --config "$config" \
        --samples-dir "$CHECKPOINT_ROOT/$sample_id/samples_prefix_8" \
        --output-dir "$CHECKPOINT_ROOT/$generated_quality_id" \
        --split val \
        --max-real-images "$REAL_COUNT" \
        --max-sample-images "$SAMPLE_COUNT" \
        --batch-size "$QUALITY_BATCH_SIZE" \
        --enable-inception-fid
  fi
}

write_queue_status "start"

run_suite "tiny_epsilononly_20k_seed2" "configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda.json" "train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda" "0" "1"

run_suite "tiny_k8_light_20k_seed2" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda" "1" "1"

run_suite "imagenet_hf_epsilononly_20k_seed2" "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json" "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda" "0" "1"

run_suite "imagenet_hf_k8_light_20k_seed2" "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json" "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda" "1" "1"

run_suite "tiny_k8_light_multiscale_10k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda" "1" "1"

run_suite "imagenet_hf_k8_light_multiscale_10k" "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json" "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda" "1" "1"

run_suite "tiny_k8_light_nopathprefix_10k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda" "1" "0"

run_suite "tiny_k8_light_cleanmono_10k" "configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda.json" "train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda" "1" "0"

run_suite "imagenet_hf_k8_light_nopathprefix_10k" "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json" "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda" "1" "0"

run_suite "imagenet_hf_k8_light_cleanmono_10k" "configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json" "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda" "1" "0"


"$PYTHON" scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-08

write_queue_status "final"

"$PYTHON" scripts/validate_dataset_conditions.py \
  --conditions-dir docs/experiment_conditions \
  --require-ok

"$PYTHON" scripts/validate_summary_consistency.py \
  --summary-dir artifacts/reports/summary_2026-07-08 \
  --doc docs/reports/cofitok_mvp_report_2026-07-08.md \
  --doc docs/records/2026-07-08_completion_audit.md

"$PYTHON" scripts/validate_idea_requirements.py \
  --project-root . \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \
  --queue-manifest artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json \
  --output-json artifacts/reports/idea_requirements_2026-07-08.json \
  --output-md docs/records/2026-07-08_idea_requirements_matrix.md \
  --require-mvp-ok

"$PYTHON" scripts/validate_mvp_evidence.py \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json

if [[ "$REQUIRE_PUBLICATION_READY" == "1" ]]; then
  "$PYTHON" scripts/validate_publication_readiness.py \
    --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \
    --output-json artifacts/reports/publication_readiness_2026-07-08.json \
    --output-md docs/records/2026-07-08_publication_readiness_gap.md \
    --require-ready
else
  "$PYTHON" scripts/validate_publication_readiness.py \
    --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \
    --output-json artifacts/reports/publication_readiness_2026-07-08.json \
    --output-md docs/records/2026-07-08_publication_readiness_gap.md
fi

"$PYTHON" scripts/validate_synthesis_contract.py \
  --config-glob 'configs/*.json'

"$PYTHON" -m pytest -q
