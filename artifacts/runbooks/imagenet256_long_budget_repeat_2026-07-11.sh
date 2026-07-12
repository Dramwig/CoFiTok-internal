#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-11_imagenet256_20k_repeat}"
STATUS_PATH="${CODE_DIR}/artifacts/runbooks/imagenet256_long_budget_repeat_2026-07-11.status"
LOG_PATH="${CHECKPOINT_ROOT}/imagenet256_long_budget_repeat_2026-07-11.log"

QUALITY_IMAGES="${QUALITY_IMAGES:-1024}"
QUALITY_BATCHES="${QUALITY_BATCHES:-256}"
SAMPLE_COUNT="${SAMPLE_COUNT:-4096}"
REAL_COUNT="${REAL_COUNT:-10000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"

mkdir -p "${CHECKPOINT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

finish() {
  code=$?
  if [[ "${code}" == "0" ]]; then
    printf 'completed\n' > "${STATUS_PATH}"
  else
    printf 'failed:%s\n' "${code}" > "${STATUS_PATH}"
  fi
  date --iso-8601=seconds
}
trap finish EXIT

printf 'running\n' > "${STATUS_PATH}"
cd "${CODE_DIR}"
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-${CHECKPOINT_ROOT}/torch_cache}"

run_if_missing() {
  local marker="$1"
  shift
  if [[ -f "${marker}" ]]; then
    printf '[skip] %s\n' "${marker}"
  else
    printf '[run] %s\n' "$*"
    "$@"
  fi
}

run_suite() {
  local method="$1"
  local seed="$2"
  local order_eval="$3"
  local variant
  if [[ "${method}" == "cofitok" ]]; then
    variant="denoisepath_p150_light"
  else
    variant="epsilononly_p150eval"
  fi
  local stem="train_imagenet256_k4_${variant}_20k_seed${seed}_cuda"
  local config="configs/${stem}.json"
  local label="imagenet256_${method}_20k_seed${seed}"
  local train_dir="${CHECKPOINT_ROOT}/${stem}_${DATE_TAG}"
  local checkpoint="${train_dir}/checkpoint_final.pt"
  local quality_dir="${CHECKPOINT_ROOT}/quality_${label}_${QUALITY_IMAGES}_t500_lpips_inception_${DATE_TAG}"
  local generated_dir="${CHECKPOINT_ROOT}/generated_quality_stream_${label}_${SAMPLE_COUNT}_ddim${SAMPLE_STEPS}_${DATE_TAG}"

  run_if_missing "${train_dir}/report.json" \
    "${PYTHON}" scripts/train_short.py \
      --config "${config}" \
      --output-dir "${train_dir}"

  run_if_missing "${quality_dir}/quality_report.json" \
    "${PYTHON}" scripts/evaluate_quality.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${quality_dir}" \
      --split val \
      --max-batches "${QUALITY_BATCHES}" \
      --max-images "${QUALITY_IMAGES}" \
      --timestep 500 \
      --enable-lpips \
      --enable-inception-fid

  if [[ "${order_eval}" == "1" ]]; then
    for component_order in ordered random reverse; do
      local order_dir="${CHECKPOINT_ROOT}/order_${label}_${component_order}_${DATE_TAG}"
      run_if_missing "${order_dir}/report.json" \
        "${PYTHON}" scripts/evaluate_checkpoint.py \
          --config "${config}" \
          --checkpoint "${checkpoint}" \
          --output-dir "${order_dir}" \
          --component-order "${component_order}" \
          --random-order-seed 0
    done
  fi

  run_if_missing "${generated_dir}/generated_quality_report.json" \
    "${PYTHON}" scripts/evaluate_generated_samples_stream.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${generated_dir}" \
      --split val \
      --sample-count "${SAMPLE_COUNT}" \
      --max-real-images "${REAL_COUNT}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --sample-steps "${SAMPLE_STEPS}" \
      --enable-inception-fid
}

for seed in 103 139; do
  run_suite dense "${seed}" 0
  run_suite cofitok "${seed}" 1
done

"${PYTHON}" scripts/summarize_experiments.py \
  --reports-root "${CHECKPOINT_ROOT}" \
  --output-dir "artifacts/reports/summary_${DATE_TAG}"

printf 'imagenet256_long_budget_repeat_done\n'
