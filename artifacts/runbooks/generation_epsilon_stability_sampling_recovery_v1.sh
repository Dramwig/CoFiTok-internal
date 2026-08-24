#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${PYTHON:?set PYTHON}"
: "${EXPECTED_REVISION:?set EXPECTED_REVISION}"
: "${EXPECTED_TREE:?set EXPECTED_TREE}"
: "${EXPECTED_BRANCH:?set EXPECTED_BRANCH}"
: "${DESIGN:?set DESIGN}"
: "${EXPECTED_DESIGN_SHA256:?set EXPECTED_DESIGN_SHA256}"
: "${PREPARATION:?set PREPARATION}"
: "${EXPECTED_PREPARATION_SHA256:?set EXPECTED_PREPARATION_SHA256}"
: "${USER_AUTHORIZATION:?set USER_AUTHORIZATION}"
: "${EXPECTED_USER_AUTHORIZATION_SHA256:?set EXPECTED_USER_AUTHORIZATION_SHA256}"
: "${EXECUTION_AUTHORIZATION:?set EXECUTION_AUTHORIZATION}"
: "${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:?set EXPECTED_EXECUTION_AUTHORIZATION_SHA256}"
: "${REAL_DIR:?set REAL_DIR}"
: "${REAL_SET_CONTRACT:?set REAL_SET_CONTRACT}"
: "${EVAL_CACHE:?set EVAL_CACHE}"
: "${CLASSIFIER_CHECKPOINT:?set CLASSIFIER_CHECKPOINT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${STATUS_OUTPUT:?set STATUS_OUTPUT}"
: "${CONTROLLER_LOCK:?set CONTROLLER_LOCK}"

EXPECTED_OUTPUT_ROOT="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_epsilon_stability_sampling_recovery_v1"
if [[ "$OUTPUT_ROOT" != "$EXPECTED_OUTPUT_ROOT" ]]; then
  printf 'epsilon-stability output root differs\n' >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]] \
  || [[ -n "${OMP_NUM_THREADS+x}" ]] \
  || [[ -n "${MKL_NUM_THREADS+x}" ]] \
  || [[ -n "${PYTHONHASHSEED+x}" ]] \
  || [[ -n "${PYTORCH_ALLOC_CONF+x}" ]] \
  || [[ -n "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
  printf 'epsilon-stability runtime variables must be unset to match the bound runtime\n' >&2
  exit 3
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'epsilon-stability Python runtime is unavailable\n' >&2
  exit 4
fi
cd "$PROJECT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_REVISION" ]] \
  || [[ "$(git rev-parse HEAD^{tree})" != "$EXPECTED_TREE" ]] \
  || [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]] \
  || [[ -n "$(git status --porcelain)" ]]; then
  printf 'epsilon-stability execution checkout differs\n' >&2
  exit 5
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf 'epsilon-stability runbook requires an idle GPU\n' >&2
  exit 6
fi

export PYTHONPATH="$PROJECT/src:$PROJECT"
exec "$PYTHON" scripts/run_generation_epsilon_stability_sampling_recovery.py \
  --project "$PROJECT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --python "$PYTHON" \
  --design "$DESIGN" \
  --expected-design-sha256 "$EXPECTED_DESIGN_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --user-authorization "$USER_AUTHORIZATION" \
  --expected-user-authorization-sha256 "$EXPECTED_USER_AUTHORIZATION_SHA256" \
  --execution-authorization "$EXECUTION_AUTHORIZATION" \
  --expected-execution-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" \
  --real-dir "$REAL_DIR" \
  --real-set-contract "$REAL_SET_CONTRACT" \
  --eval-cache "$EVAL_CACHE" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --status-output "$STATUS_OUTPUT" \
  --lock "$CONTROLLER_LOCK" \
  --required-idle-polls 3 \
  --idle-poll-seconds 2
