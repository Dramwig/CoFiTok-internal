#!/usr/bin/env bash
set -euo pipefail

# This entrypoint only consumes immutable, source-bound evidence. It never
# creates a stage authorization, execution authorization, or launch receipt.
# The four-arm screen remains diagnostic and cannot authorize confirmation,
# full training, 300K training, export, promotion, or release.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
EXECUTION_ENABLED="${CAPACITY_SCREEN_EXECUTION_ENABLED:-false}"

[[ "$EXECUTION_ENABLED" == true ]] || {
  printf 'capacity screen is disabled; set CAPACITY_SCREEN_EXECUTION_ENABLED=true only with exact authorization\n' >&2
  exit 20
}

PREPARATION="${CAPACITY_SCREEN_PREPARATION:?set the immutable capacity-screen preparation path}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_CAPACITY_SCREEN_PREPARATION_SHA256:?set the preparation SHA256}"
AUTHORIZATION="${CAPACITY_SCREEN_AUTHORIZATION:?set the exact capacity-screen execution authorization path}"
EXPECTED_AUTHORIZATION_SHA256="${EXPECTED_CAPACITY_SCREEN_AUTHORIZATION_SHA256:?set the execution authorization SHA256}"
LAUNCH_RECEIPT="${CAPACITY_SCREEN_LAUNCH_RECEIPT:?set the immutable capacity-screen launch receipt path}"
EXPECTED_LAUNCH_RECEIPT_SHA256="${EXPECTED_CAPACITY_SCREEN_LAUNCH_RECEIPT_SHA256:?set the launch receipt SHA256}"
RUNTIME_SELECTION="${CAPACITY_SCREEN_RUNTIME_SELECTION:?set the frozen capacity-screen runtime selection path}"
EXPECTED_RUNTIME_SELECTION_SHA256="${EXPECTED_CAPACITY_SCREEN_RUNTIME_SELECTION_SHA256:?set the runtime selection SHA256}"
LIVE_SNAPSHOT="${CAPACITY_SCREEN_LIVE_SNAPSHOT:?set the immutable idle prelaunch snapshot path}"
EXPECTED_LIVE_SNAPSHOT_SHA256="${EXPECTED_CAPACITY_SCREEN_LIVE_SNAPSHOT_SHA256:?set the live snapshot SHA256}"

BASE128_COFITOK_CONFIG="${CAPACITY_SCREEN_BASE128_COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_reference_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json}"
BASE128_DENSE_CONFIG="${CAPACITY_SCREEN_BASE128_DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_reference_rollout_x0_u2_ema_teacher_dense_100k.json}"
BASE256_COFITOK_CONFIG="${CAPACITY_SCREEN_BASE256_COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_qualification_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json}"
BASE256_DENSE_CONFIG="${CAPACITY_SCREEN_BASE256_DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_capacity_qualification_rollout_x0_u2_ema_teacher_dense_100k.json}"
OUTPUT_ROOT="${CAPACITY_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1}"
REAL_DIR="${CAPACITY_SCREEN_REAL_DIR:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}"
CLASSIFIER_CHECKPOINT="${CAPACITY_SCREEN_CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}"
CACHE_ROOT="${CAPACITY_SCREEN_CACHE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity}"
RESUME="${CAPACITY_SCREEN_RESUME:-false}"

RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  RESUME_ARGS+=(--resume)
fi

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/run_generation_capacity_screen.py \
  --project-root "$PROJECT" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_AUTHORIZATION_SHA256" \
  --launch-receipt "$LAUNCH_RECEIPT" \
  --expected-launch-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" \
  --runtime-selection "$RUNTIME_SELECTION" \
  --expected-runtime-selection-sha256 "$EXPECTED_RUNTIME_SELECTION_SHA256" \
  --live-snapshot "$LIVE_SNAPSHOT" \
  --expected-live-snapshot-sha256 "$EXPECTED_LIVE_SNAPSHOT_SHA256" \
  --base128-cofitok-config "$BASE128_COFITOK_CONFIG" \
  --base128-dense-identity-config "$BASE128_DENSE_CONFIG" \
  --base256-cofitok-config "$BASE256_COFITOK_CONFIG" \
  --base256-dense-identity-config "$BASE256_DENSE_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --real-dir "$REAL_DIR" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --python "$PYTHON" \
  "${RESUME_ARGS[@]}"
