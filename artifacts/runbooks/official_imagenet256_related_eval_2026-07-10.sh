#!/usr/bin/env bash
set -euo pipefail

# Official-checkpoint ImageNet-256 related-method protocol.
# Safe default: DRY_RUN=1 prints commands only.
#
# Usage:
#   METHOD=d_ar DRY_RUN=1 bash artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
#   METHOD=mar DRY_RUN=1 bash artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
#   METHOD=retok DRY_RUN=1 bash artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
#
# Current status:
# - D-AR official 50K is already complete.
# - MAR community safetensors non-EMA audit is complete; official PTH EMA 50K is active.
# - ReTok official GPT+VQ 50K is complete.

METHOD="${METHOD:-all}"
DRY_RUN="${DRY_RUN:-1}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
PROJECT_ROOT="/root/autodl-tmp/CoFiTok"
INTERNAL_ROOT="${PROJECT_ROOT}/CoFiTok-internal"
REF_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
REF_NPZ="${PROJECT_ROOT}/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz"
EVAL_ROOT="${PROJECT_ROOT}/checkpoints/baselines/official_refs"
EVAL_GRAPH_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
EVAL_GRAPH="${EVAL_ROOT}/classify_image_graph_def.pb"
EVAL_SCRIPT="${PROJECT_ROOT}/baselines/repos/d_ar/evaluations/c2i/evaluator.py"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

prepare_common() {
  run_or_print "mkdir -p ${PROJECT_ROOT}/checkpoints/baselines/official_refs"
  run_or_print "test -f ${REF_NPZ} || wget -O ${REF_NPZ} ${REF_URL}"
  run_or_print "test -f ${EVAL_GRAPH} || wget -O ${EVAL_GRAPH} ${EVAL_GRAPH_URL}"
}

run_d_ar() {
  local repo="${PROJECT_ROOT}/baselines/repos/d_ar"
  local out="${PROJECT_ROOT}/checkpoints/baselines/d_ar/official_imagenet256_eval_only"
  local sample_npz="${out}/samples/GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.2,8.0-seed-0-None.npz"
  run_or_print "mkdir -p ${out}/weights ${out}/samples"
  run_or_print "test -f ${out}/weights/D-AR-tokenizer_v1.pt || hf download showlab/D-AR D-AR-tokenizer_v1.pt --local-dir ${out}/weights"
  run_or_print "test -f ${out}/weights/D-AR-L-360K.pt || hf download showlab/D-AR D-AR-L-360K.pt --local-dir ${out}/weights"
  run_or_print "cd ${repo} && bash scripts/autoregressive/sample_c2i.sh --gpt-model GPT-L --gpt-ckpt ${out}/weights/D-AR-L-360K.pt --tokenizer-config configs/tokenizer_v1.yaml --tokenizer-ckpt ${out}/weights/D-AR-tokenizer_v1.pt --cfg-scale 1.2,8.0 --top-p 1.0 --top-k 0 --temperature 1.0 --num-fid-samples ${NUM_IMAGES} --per-proc-batch-size \${EVAL_BATCH_PER_GPU:-64} --sample-dir ${out}/samples"
  run_or_print "cd ${EVAL_ROOT} && python ${EVAL_SCRIPT} ${REF_NPZ} ${sample_npz}"
}

run_mar() {
  local official_assets="${PROJECT_ROOT}/checkpoints/baselines/mar/official_imagenet256_eval_only/official_pth"
  local status_file="${INTERNAL_ROOT}/artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.status"
  printf '%s\n' "# MAR status: pinned LTH14 PTH model_ema runbook is available."
  printf '%s\n' "# Verified assets: ${official_assets}/checkpoint-last.pth and ${official_assets}/kl16.ckpt"
  if [[ -f "${status_file}" ]] && [[ "$(cat "${status_file}")" == "running" ]]; then
    printf '%s\n' "# MAR 50K is already running; refusing a duplicate launch."
    return 0
  fi
  run_or_print "bash ${INTERNAL_ROOT}/artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.sh"
}

run_retok() {
  local repo="${PROJECT_ROOT}/baselines/repos/retok"
  local out="${PROJECT_ROOT}/checkpoints/baselines/retok/official_imagenet256_eval_only"
  local weights="${out}/checkpoints"
  local sample_dir="${RETOK_SAMPLE_DIR:-${out}/samples_50k}"
  run_or_print "mkdir -p ${out}"
  run_or_print "test -f ${weights}/VQ_SB256_e250.pt"
  run_or_print "test -f ${weights}/GPT_XL256_e300_VQ_SB.pt"
  run_or_print "cd ${repo} && PYTHONPATH=${repo}:\${PYTHONPATH:-} GPUS=\${GPUS:-1} PORT=\${PORT:-55635} bash scripts/sample_c2i_search_cfg.sh --quant-way=vq --image-size=256 --image-size-eval=256 --sample-dir=${sample_dir} --vq-ckpt ${weights}/VQ_SB256_e250.pt --tok-config configs/vq/VQ_SB256.yaml --gpt-model GPT-XL --cfg-schedule step --cfg-scale 1.5 --top-k 0 --top-p 1.0 --temperature 1.0 --step-start-ratio 0.18 --gpt-ckpt ${weights}/GPT_XL256_e300_VQ_SB.pt --per-proc-batch-size \${EVAL_BATCH_PER_GPU:-1} --num-fid-samples ${NUM_IMAGES} --precision bf16 --eval-python-path \${EVAL_PYTHON_PATH:-python} --gt-npz-path ${REF_NPZ} --global-seed 0"
}

cd "${INTERNAL_ROOT}"
prepare_common

case "${METHOD}" in
  d_ar)
    run_d_ar
    ;;
  mar)
    run_mar
    ;;
  retok)
    run_retok
    ;;
  all)
    run_d_ar
    run_mar
    run_retok
    ;;
  *)
    echo "Unknown METHOD=${METHOD}; expected d_ar, mar, retok, or all" >&2
    exit 2
    ;;
esac
