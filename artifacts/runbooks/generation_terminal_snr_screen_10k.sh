#!/usr/bin/env bash
set -euo pipefail

# This entrypoint consumes immutable evidence only.  It cannot create an
# authorization and it cannot authorize frozen confirmation or full training.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
[[ "${TERMINAL_SNR_SCREEN_EXECUTION_ENABLED:-false}" == true ]] || {
  printf 'terminal-SNR screen disabled; exact execution authorization required\n' >&2
  exit 20
}

PREPARATION="${TERMINAL_SNR_SCREEN_PREPARATION:?set immutable preparation path}"
PREPARATION_SHA="${EXPECTED_TERMINAL_SNR_SCREEN_PREPARATION_SHA256:?set preparation SHA256}"
AUTHORIZATION="${TERMINAL_SNR_SCREEN_AUTHORIZATION:?set execution authorization path}"
AUTHORIZATION_SHA="${EXPECTED_TERMINAL_SNR_SCREEN_AUTHORIZATION_SHA256:?set authorization SHA256}"
LAUNCH_RECEIPT="${TERMINAL_SNR_SCREEN_LAUNCH_RECEIPT:?set launch receipt path}"
LAUNCH_SHA="${EXPECTED_TERMINAL_SNR_SCREEN_LAUNCH_RECEIPT_SHA256:?set launch receipt SHA256}"
RUNTIME_SELECTION="${TERMINAL_SNR_SCREEN_RUNTIME_SELECTION:?set runtime selection path}"
RUNTIME_SHA="${EXPECTED_TERMINAL_SNR_SCREEN_RUNTIME_SELECTION_SHA256:?set runtime selection SHA256}"
LIVE_SNAPSHOT="${TERMINAL_SNR_SCREEN_LIVE_SNAPSHOT:?set live snapshot path}"
LIVE_SHA="${EXPECTED_TERMINAL_SNR_SCREEN_LIVE_SNAPSHOT_SHA256:?set live snapshot SHA256}"

CONTROL_COFITOK_CONFIG="${TERMINAL_SNR_CONTROL_COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_reference_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json}"
CONTROL_DENSE_CONFIG="${TERMINAL_SNR_CONTROL_DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_reference_rollout_x0_u2_ema_teacher_dense_100k.json}"
ENDPOINT_COFITOK_CONFIG="${TERMINAL_SNR_ENDPOINT_COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_terminal_snr_endpoint0975_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json}"
ENDPOINT_DENSE_CONFIG="${TERMINAL_SNR_ENDPOINT_DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_terminal_snr_endpoint0975_rollout_x0_u2_ema_teacher_dense_100k.json}"
OUTPUT_ROOT="${TERMINAL_SNR_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1}"
REAL_DIR="${TERMINAL_SNR_SCREEN_REAL_DIR:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}"
CLASSIFIER="${TERMINAL_SNR_SCREEN_CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}"
CACHE_ROOT="${TERMINAL_SNR_SCREEN_CACHE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity}"

RESUME_ARGS=()
if [[ "${TERMINAL_SNR_SCREEN_RESUME:-false}" == true ]]; then
  RESUME_ARGS+=(--resume)
fi

cd "$PROJECT"
# Runtime identity is part of the immutable launch receipt.  Canonicalize every
# controller child (training, sampling, and evaluators) to the exact environment
# selected by the preflight benchmark.
unset CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES PYTHONHASHSEED
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/run_generation_terminal_snr_screen.py \
  --project-root "$PROJECT" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$PREPARATION_SHA" \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$AUTHORIZATION_SHA" \
  --launch-receipt "$LAUNCH_RECEIPT" \
  --expected-launch-receipt-sha256 "$LAUNCH_SHA" \
  --runtime-selection "$RUNTIME_SELECTION" \
  --expected-runtime-selection-sha256 "$RUNTIME_SHA" \
  --live-snapshot "$LIVE_SNAPSHOT" \
  --expected-live-snapshot-sha256 "$LIVE_SHA" \
  --control-cofitok-config "$CONTROL_COFITOK_CONFIG" \
  --control-dense-identity-config "$CONTROL_DENSE_CONFIG" \
  --endpoint0975-cofitok-config "$ENDPOINT_COFITOK_CONFIG" \
  --endpoint0975-dense-identity-config "$ENDPOINT_DENSE_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --real-dir "$REAL_DIR" \
  --classifier-checkpoint "$CLASSIFIER" \
  --cache-root "$CACHE_ROOT" \
  --python "$PYTHON" \
  "${RESUME_ARGS[@]}"
