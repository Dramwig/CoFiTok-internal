#!/usr/bin/env bash
set -euo pipefail

# This entrypoint never creates authorization.  It can run only after an
# operator has supplied the exact source-bound authorization and its digest.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
EXECUTION_ENABLED="${EXPOSURE_CONTINUATION_EXECUTION_ENABLED:-false}"

[[ "$EXECUTION_ENABLED" == true ]] || {
  printf 'exposure continuation is disabled; set EXPOSURE_CONTINUATION_EXECUTION_ENABLED=true only with explicit authorization\n' >&2
  exit 20
}

AUTHORIZATION="${EXPOSURE_CONTINUATION_AUTHORIZATION:?set the source-bound execution authorization path}"
EXPECTED_AUTHORIZATION_SHA256="${EXPECTED_EXPOSURE_CONTINUATION_AUTHORIZATION_SHA256:?set the authorization SHA256}"
GATE="${EXPOSURE_CONTINUATION_GATE:?set the prepared execution gate path}"
EXPECTED_GATE_SHA256="${EXPECTED_EXPOSURE_CONTINUATION_GATE_SHA256:?set the gate SHA256}"
PREPARATION="${EXPOSURE_CONTINUATION_PREPARATION:?set the immutable preparation path}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_EXPOSURE_CONTINUATION_PREPARATION_SHA256:?set the preparation SHA256}"
STANDING_AUTHORIZATION="${STANDING_AUTHORIZATION:?set the standing authorization path}"
EXPECTED_STANDING_AUTHORIZATION_SHA256="${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set the standing authorization SHA256}"
STAGE_AUTHORIZATION="${EXPOSURE_CONTINUATION_STAGE_AUTHORIZATION:?set the user-created exact-stage authorization path}"
EXPECTED_STAGE_AUTHORIZATION_SHA256="${EXPECTED_EXPOSURE_CONTINUATION_STAGE_AUTHORIZATION_SHA256:?set the exact-stage authorization SHA256}"

SOURCE_PROJECT_ROOT="${EXPOSURE_CONTINUATION_SOURCE_PROJECT_ROOT:-/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal}"
COFITOK_CONFIG="${EXPOSURE_CONTINUATION_COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_exposure_continuation_rgbtail3_rollout_x0_u2_ema_teacher_k8_110k.json}"
DENSE_CONFIG="${EXPOSURE_CONTINUATION_DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_exposure_continuation_rollout_x0_u2_ema_teacher_dense_110k.json}"
OUTPUT_ROOT="${EXPOSURE_CONTINUATION_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure_capacity_disambiguation_v1/exposure}"
REAL_DIR="${EXPOSURE_CONTINUATION_REAL_DIR:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}"
CLASSIFIER_CHECKPOINT="${EXPOSURE_CONTINUATION_CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}"
CACHE_ROOT="${EXPOSURE_CONTINUATION_CACHE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity}"
RESUME="${EXPOSURE_CONTINUATION_RESUME:-false}"

RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  RESUME_ARGS+=(--resume)
fi

export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$PROJECT/scripts/run_generation_exposure_capacity_continuation.py" \
  --project-root "$PROJECT" \
  --source-project-root "$SOURCE_PROJECT_ROOT" \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_AUTHORIZATION_SHA256" \
  --gate "$GATE" \
  --expected-gate-sha256 "$EXPECTED_GATE_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --stage-authorization "$STAGE_AUTHORIZATION" \
  --expected-stage-authorization-sha256 "$EXPECTED_STAGE_AUTHORIZATION_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --real-dir "$REAL_DIR" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --python "$PYTHON" \
  "${RESUME_ARGS[@]}"
