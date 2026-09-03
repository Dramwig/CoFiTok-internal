#!/usr/bin/env bash
set -euo pipefail

# This entrypoint evaluates only the four frozen step-10K checkpoints bound by
# a passing screen. It never creates authorization and never launches training.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
EXECUTION_ENABLED="${CAPACITY_CONFIRMATION_EXECUTION_ENABLED:-false}"

[[ "$EXECUTION_ENABLED" == true ]] || {
  printf 'capacity confirmation is disabled; enable it only with exact authorization\n' >&2
  exit 20
}

PREPARATION="${CAPACITY_CONFIRMATION_PREPARATION:?set the immutable confirmation preparation path}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_PREPARATION_SHA256:?set the preparation SHA256}"
AUTHORIZATION="${CAPACITY_CONFIRMATION_AUTHORIZATION:?set the exact confirmation authorization path}"
EXPECTED_AUTHORIZATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_AUTHORIZATION_SHA256:?set the authorization SHA256}"
LAUNCH_RECEIPT="${CAPACITY_CONFIRMATION_LAUNCH_RECEIPT:?set the immutable confirmation launch receipt path}"
EXPECTED_LAUNCH_RECEIPT_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_LAUNCH_RECEIPT_SHA256:?set the launch receipt SHA256}"
LIVE_SNAPSHOT="${CAPACITY_CONFIRMATION_LIVE_SNAPSHOT:?set the immutable idle live snapshot path}"
EXPECTED_LIVE_SNAPSHOT_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_LIVE_SNAPSHOT_SHA256:?set the live snapshot SHA256}"

SCREEN_ROOT="${CAPACITY_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1}"
BASE128_COFITOK_SCREEN_VALIDATION="${CAPACITY_CONFIRMATION_BASE128_COFITOK_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/base128_cofitok.json}"
EXPECTED_BASE128_COFITOK_SCREEN_VALIDATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_BASE128_COFITOK_SCREEN_VALIDATION_SHA256:?set the frozen base128 CoFiTok screen validation SHA256}"
BASE128_DENSE_SCREEN_VALIDATION="${CAPACITY_CONFIRMATION_BASE128_DENSE_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/base128_dense_identity.json}"
EXPECTED_BASE128_DENSE_SCREEN_VALIDATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_BASE128_DENSE_SCREEN_VALIDATION_SHA256:?set the frozen base128 dense screen validation SHA256}"
BASE256_COFITOK_SCREEN_VALIDATION="${CAPACITY_CONFIRMATION_BASE256_COFITOK_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/base256_cofitok.json}"
EXPECTED_BASE256_COFITOK_SCREEN_VALIDATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_BASE256_COFITOK_SCREEN_VALIDATION_SHA256:?set the frozen base256 CoFiTok screen validation SHA256}"
BASE256_DENSE_SCREEN_VALIDATION="${CAPACITY_CONFIRMATION_BASE256_DENSE_SCREEN_VALIDATION:-$SCREEN_ROOT/arm_validations/base256_dense_identity.json}"
EXPECTED_BASE256_DENSE_SCREEN_VALIDATION_SHA256="${EXPECTED_CAPACITY_CONFIRMATION_BASE256_DENSE_SCREEN_VALIDATION_SHA256:?set the frozen base256 dense screen validation SHA256}"

OUTPUT_ROOT="${CAPACITY_CONFIRMATION_OUTPUT_ROOT:-$SCREEN_ROOT/confirmation_10000}"
REAL_DIR="${CAPACITY_CONFIRMATION_REAL_DIR:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}"
CLASSIFIER_CHECKPOINT="${CAPACITY_CONFIRMATION_CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}"
CACHE_ROOT="${CAPACITY_CONFIRMATION_CACHE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity}"
RESUME="${CAPACITY_CONFIRMATION_RESUME:-false}"

RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  RESUME_ARGS+=(--resume)
fi

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/run_generation_capacity_confirmation.py \
  --project-root "$PROJECT" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_AUTHORIZATION_SHA256" \
  --launch-receipt "$LAUNCH_RECEIPT" \
  --expected-launch-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" \
  --live-snapshot "$LIVE_SNAPSHOT" \
  --expected-live-snapshot-sha256 "$EXPECTED_LIVE_SNAPSHOT_SHA256" \
  --base128-cofitok-screen-validation "$BASE128_COFITOK_SCREEN_VALIDATION" \
  --expected-base128-cofitok-screen-validation-sha256 "$EXPECTED_BASE128_COFITOK_SCREEN_VALIDATION_SHA256" \
  --base128-dense-identity-screen-validation "$BASE128_DENSE_SCREEN_VALIDATION" \
  --expected-base128-dense-identity-screen-validation-sha256 "$EXPECTED_BASE128_DENSE_SCREEN_VALIDATION_SHA256" \
  --base256-cofitok-screen-validation "$BASE256_COFITOK_SCREEN_VALIDATION" \
  --expected-base256-cofitok-screen-validation-sha256 "$EXPECTED_BASE256_COFITOK_SCREEN_VALIDATION_SHA256" \
  --base256-dense-identity-screen-validation "$BASE256_DENSE_SCREEN_VALIDATION" \
  --expected-base256-dense-identity-screen-validation-sha256 "$EXPECTED_BASE256_DENSE_SCREEN_VALIDATION_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --real-dir "$REAL_DIR" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --python "$PYTHON" \
  "${RESUME_ARGS[@]}"
