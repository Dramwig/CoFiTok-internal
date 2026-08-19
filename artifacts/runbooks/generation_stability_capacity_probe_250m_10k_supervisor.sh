#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set exact capacity-probe execution checkout}
PREPARATION_PROJECT=${PREPARATION_PROJECT:?set exact capacity-probe preparation checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set exact standing authorization record}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set standing authorization SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set exact execution revision}
EXPECTED_TARGET_TREE=${EXPECTED_TARGET_TREE:?set exact execution tree}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set exact execution branch}
EXPECTED_PREPARATION_REVISION=${EXPECTED_PREPARATION_REVISION:?set exact preparation revision}
EXPECTED_PREPARATION_TREE=${EXPECTED_PREPARATION_TREE:?set exact preparation tree}
EXPECTED_PREPARATION_BRANCH=${EXPECTED_PREPARATION_BRANCH:?set exact preparation branch}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
PREPARATION="$REPORT_ROOT/preparation.json"
PREPARATION_STATUS="$REPORT_ROOT/preparation_waiter_status.json"
STATUS="$REPORT_ROOT/execution_supervisor_status.json"
LOCK="$OUTPUT_ROOT/capacity_probe_execution_supervisor.lock"
RUNTIME_TMP="$OUTPUT_ROOT/runtime_tmp"
RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_capacity_probe_250m_10k_execute.sh"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
mkdir -p "$REPORT_ROOT" "$RUNTIME_TMP"
export TMPDIR="$RUNTIME_TMP"
export TEMP="$RUNTIME_TMP"
export TMP="$RUNTIME_TMP"
exec flock --exclusive --nonblock --conflict-exit-code 75 --no-fork \
  "$LOCK" \
  "$PYTHON" scripts/run_generation_capacity_probe_execution_supervisor.py \
  --project "$PROJECT" \
  --preparation-project "$PREPARATION_PROJECT" \
  --preparation "$PREPARATION" \
  --preparation-waiter-status "$PREPARATION_STATUS" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --runbook "$RUNBOOK" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --status-output "$STATUS" \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-tree "$EXPECTED_TARGET_TREE" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --expected-preparation-revision "$EXPECTED_PREPARATION_REVISION" \
  --expected-preparation-tree "$EXPECTED_PREPARATION_TREE" \
  --expected-preparation-branch "$EXPECTED_PREPARATION_BRANCH" \
  --poll-seconds 60 \
  --required-idle-polls 5 \
  --max-attempts 4 \
  --retry-seconds 120
