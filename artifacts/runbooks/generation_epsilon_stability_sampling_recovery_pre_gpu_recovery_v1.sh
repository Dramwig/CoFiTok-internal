#!/usr/bin/env bash
set -euo pipefail

: "${RECOVERY_PROJECT:?set RECOVERY_PROJECT}"
: "${EXECUTION_PROJECT:?set EXECUTION_PROJECT}"
: "${PYTHON:?set PYTHON}"
: "${EXPECTED_RECOVERY_REVISION:?set EXPECTED_RECOVERY_REVISION}"
: "${EXPECTED_RECOVERY_TREE:?set EXPECTED_RECOVERY_TREE}"
: "${EXPECTED_RECOVERY_BRANCH:?set EXPECTED_RECOVERY_BRANCH}"
: "${RECOVERY_AUTHORIZATION:?set RECOVERY_AUTHORIZATION}"
: "${EXPECTED_RECOVERY_AUTHORIZATION_SHA256:?set EXPECTED_RECOVERY_AUTHORIZATION_SHA256}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${FAILED_LOCK:?set FAILED_LOCK}"
: "${RECOVERY_LOCK:?set RECOVERY_LOCK}"
: "${RECOVERY_STATUS:?set RECOVERY_STATUS}"

EXPECTED_OUTPUT_ROOT="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_epsilon_stability_sampling_recovery_v1"
if [[ "$OUTPUT_ROOT" != "$EXPECTED_OUTPUT_ROOT" ]]; then
  printf 'epsilon-stability recovery output root differs\n' >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]] \
  || [[ -n "${OMP_NUM_THREADS+x}" ]] \
  || [[ -n "${MKL_NUM_THREADS+x}" ]] \
  || [[ -n "${PYTHONHASHSEED+x}" ]] \
  || [[ -n "${PYTORCH_ALLOC_CONF+x}" ]] \
  || [[ -n "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
  printf 'epsilon-stability recovery runtime variables must be unset\n' >&2
  exit 3
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'epsilon-stability recovery Python runtime is unavailable\n' >&2
  exit 4
fi
if [[ "$(git -C "$RECOVERY_PROJECT" rev-parse HEAD)" != "$EXPECTED_RECOVERY_REVISION" ]] \
  || [[ "$(git -C "$RECOVERY_PROJECT" rev-parse HEAD^{tree})" != "$EXPECTED_RECOVERY_TREE" ]] \
  || [[ "$(git -C "$RECOVERY_PROJECT" branch --show-current)" != "$EXPECTED_RECOVERY_BRANCH" ]] \
  || [[ -n "$(git -C "$RECOVERY_PROJECT" status --porcelain)" ]]; then
  printf 'epsilon-stability recovery checkout differs\n' >&2
  exit 5
fi
if [[ ! -d "$EXECUTION_PROJECT" ]] \
  || [[ ! -d "$OUTPUT_ROOT" ]] \
  || [[ ! -f "$FAILED_LOCK/failure.json" ]] \
  || [[ -e "$RECOVERY_LOCK" ]] \
  || [[ -e "$RECOVERY_STATUS" ]]; then
  printf 'epsilon-stability recovery state differs\n' >&2
  exit 6
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf 'epsilon-stability recovery requires an idle GPU\n' >&2
  exit 7
fi

export PYTHONPATH="$RECOVERY_PROJECT/src:$RECOVERY_PROJECT"
cd "$RECOVERY_PROJECT"
exec "$PYTHON" scripts/recover_generation_epsilon_stability_sampling_recovery.py run \
  --authorization "$RECOVERY_AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_RECOVERY_AUTHORIZATION_SHA256" \
  --required-idle-polls 3 \
  --idle-poll-seconds 2
