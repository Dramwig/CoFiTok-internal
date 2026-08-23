#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the exact exposure-aware decision checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact waiter revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact waiter tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact waiter branch}
EXPECTED_WAITER_SHA256=${EXPECTED_WAITER_SHA256:?set the waiter script SHA256}
QUALITY_BRIDGE_ROOT=${QUALITY_BRIDGE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1}
POLL_SECONDS=${POLL_SECONDS:-60}

REPORTS="$QUALITY_BRIDGE_ROOT/reports"
RESULT=${RESULT:-"$REPORTS/quality_bridge_result.json"}
EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:?set the exact result SHA256}
EXPOSURE=${EXPOSURE:?set the exact versioned training exposure report}
EXPECTED_EXPOSURE_SHA256=${EXPECTED_EXPOSURE_SHA256:?set the exact exposure SHA256}
DECISION=${DECISION:?set the exact versioned follow-up decision output}
DECISION_LOCK=${DECISION_LOCK:?set the exact versioned decision lock}
STATUS=${STATUS:?set the exact versioned waiter status}
LOG=${LOG:?set the exact versioned waiter log}
PID_FILE=${PID_FILE:?set the exact versioned waiter PID file}
LOCK=${LOCK:?set the exact versioned waiter lock}

cd "$PROJECT"
mkdir -p \
  "$REPORTS" \
  "$(dirname "$STATUS")" \
  "$(dirname "$LOG")" \
  "$(dirname "$PID_FILE")" \
  "$(dirname "$LOCK")" \
  "$(dirname "$DECISION")" \
  "$(dirname "$DECISION_LOCK")"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing a concurrent exposure-aware follow-up waiter\n' >&2
  exit 75
}
pid_tmp="$PID_FILE.tmp.$$"
printf '%s\n' "$$" >"$pid_tmp"
mv "$pid_tmp" "$PID_FILE"

exec env \
  CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/wait_for_generation_quality_bridge_exposure_followup.py \
    --project "$PROJECT" \
    --quality-bridge-root "$QUALITY_BRIDGE_ROOT" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-tree "$EXPECTED_TREE" \
    --expected-branch "$EXPECTED_BRANCH" \
    --expected-self-sha256 "$EXPECTED_WAITER_SHA256" \
    --python "$PYTHON" \
    --result "$RESULT" \
    --expected-result-sha256 "$EXPECTED_RESULT_SHA256" \
    --exposure "$EXPOSURE" \
    --expected-exposure-sha256 "$EXPECTED_EXPOSURE_SHA256" \
    --decision "$DECISION" \
    --decision-lock "$DECISION_LOCK" \
    --status "$STATUS" \
    --log "$LOG" \
    --poll-seconds "$POLL_SECONDS"
