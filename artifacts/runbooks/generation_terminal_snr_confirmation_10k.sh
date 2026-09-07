#!/usr/bin/env bash
set -euo pipefail

# This entrypoint only evaluates four frozen terminal-SNR screen checkpoints.
# It never creates authorization, trains, mutates checkpoints, or authorizes
# full-scale training. Every evidence object is supplied with an exact SHA256.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
EXECUTION_ENABLED="${TERMINAL_SNR_CONFIRMATION_EXECUTION_ENABLED:-false}"

[[ "$EXECUTION_ENABLED" == true ]] || {
  printf 'terminal-SNR confirmation is disabled; exact authorization is required\n' >&2
  exit 20
}

PREPARATION="${TERMINAL_SNR_CONFIRMATION_PREPARATION:?set immutable preparation path}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_PREPARATION_SHA256:?set preparation SHA256}"
AUTHORIZATION="${TERMINAL_SNR_CONFIRMATION_AUTHORIZATION:?set exact authorization path}"
EXPECTED_AUTHORIZATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_AUTHORIZATION_SHA256:?set authorization SHA256}"
LAUNCH_RECEIPT="${TERMINAL_SNR_CONFIRMATION_LAUNCH_RECEIPT:?set immutable launch receipt path}"
EXPECTED_LAUNCH_RECEIPT_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_LAUNCH_RECEIPT_SHA256:?set launch receipt SHA256}"
LIVE_SNAPSHOT="${TERMINAL_SNR_CONFIRMATION_LIVE_SNAPSHOT:?set immutable idle snapshot path}"
EXPECTED_LIVE_SNAPSHOT_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_LIVE_SNAPSHOT_SHA256:?set idle snapshot SHA256}"

SCREEN_ROOT="${TERMINAL_SNR_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1}"
CONTROL_COFITOK_SCREEN_VALIDATION="${TERMINAL_SNR_CONFIRMATION_CONTROL_COFITOK_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/control_cofitok.json}"
EXPECTED_CONTROL_COFITOK_SCREEN_VALIDATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_CONTROL_COFITOK_SCREEN_VALIDATION_SHA256:?set control CoFiTok validation SHA256}"
CONTROL_DENSE_SCREEN_VALIDATION="${TERMINAL_SNR_CONFIRMATION_CONTROL_DENSE_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/control_dense_identity.json}"
EXPECTED_CONTROL_DENSE_SCREEN_VALIDATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_CONTROL_DENSE_SCREEN_VALIDATION_SHA256:?set control dense validation SHA256}"
ENDPOINT_COFITOK_SCREEN_VALIDATION="${TERMINAL_SNR_CONFIRMATION_ENDPOINT_COFITOK_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/endpoint0975_cofitok.json}"
EXPECTED_ENDPOINT_COFITOK_SCREEN_VALIDATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_ENDPOINT_COFITOK_SCREEN_VALIDATION_SHA256:?set endpoint CoFiTok validation SHA256}"
ENDPOINT_DENSE_SCREEN_VALIDATION="${TERMINAL_SNR_CONFIRMATION_ENDPOINT_DENSE_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/endpoint0975_dense_identity.json}"
EXPECTED_ENDPOINT_DENSE_SCREEN_VALIDATION_SHA256="${EXPECTED_TERMINAL_SNR_CONFIRMATION_ENDPOINT_DENSE_SCREEN_VALIDATION_SHA256:?set endpoint dense validation SHA256}"

OUTPUT_ROOT="${TERMINAL_SNR_CONFIRMATION_OUTPUT_ROOT:-$SCREEN_ROOT/frozen_confirmation_10000}"
REAL_DIR="${TERMINAL_SNR_CONFIRMATION_REAL_DIR:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}"
CLASSIFIER_CHECKPOINT="${TERMINAL_SNR_CONFIRMATION_CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}"
CACHE_ROOT="${TERMINAL_SNR_CONFIRMATION_CACHE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity}"
RESUME="${TERMINAL_SNR_CONFIRMATION_RESUME:-false}"

RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  RESUME_ARGS+=(--resume)
fi

cd "$PROJECT"
unset CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES PYTHONHASHSEED
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT:$PROJECT/src"
exec "$PYTHON" scripts/run_generation_terminal_snr_confirmation.py \
  --project-root "$PROJECT" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_AUTHORIZATION_SHA256" \
  --launch-receipt "$LAUNCH_RECEIPT" \
  --expected-launch-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" \
  --live-snapshot "$LIVE_SNAPSHOT" \
  --expected-live-snapshot-sha256 "$EXPECTED_LIVE_SNAPSHOT_SHA256" \
  --control-cofitok-screen-validation "$CONTROL_COFITOK_SCREEN_VALIDATION" \
  --expected-control-cofitok-screen-validation-sha256 "$EXPECTED_CONTROL_COFITOK_SCREEN_VALIDATION_SHA256" \
  --control-dense-identity-screen-validation "$CONTROL_DENSE_SCREEN_VALIDATION" \
  --expected-control-dense-identity-screen-validation-sha256 "$EXPECTED_CONTROL_DENSE_SCREEN_VALIDATION_SHA256" \
  --endpoint0975-cofitok-screen-validation "$ENDPOINT_COFITOK_SCREEN_VALIDATION" \
  --expected-endpoint0975-cofitok-screen-validation-sha256 "$EXPECTED_ENDPOINT_COFITOK_SCREEN_VALIDATION_SHA256" \
  --endpoint0975-dense-identity-screen-validation "$ENDPOINT_DENSE_SCREEN_VALIDATION" \
  --expected-endpoint0975-dense-identity-screen-validation-sha256 "$EXPECTED_ENDPOINT_DENSE_SCREEN_VALIDATION_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --real-dir "$REAL_DIR" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --python "$PYTHON" \
  "${RESUME_ARGS[@]}"
