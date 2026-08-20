#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the exact terminal distribution-support checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact diagnostic revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact diagnostic tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact diagnostic branch}
EXPECTED_WAITER_SHA256=${EXPECTED_WAITER_SHA256:?set the waiter script SHA256}
EXPECTED_FOLLOWUP_REVISION=${EXPECTED_FOLLOWUP_REVISION:?set the canonical v2 follow-up revision}
EXPECTED_FOLLOWUP_BRANCH=${EXPECTED_FOLLOWUP_BRANCH:?set the canonical v2 follow-up branch}
QUALITY_BRIDGE_ROOT=${QUALITY_BRIDGE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1}
POLL_SECONDS=${POLL_SECONDS:-60}
NEAREST_CHUNK_SIZE=${NEAREST_CHUNK_SIZE:-128}

REPORTS="$QUALITY_BRIDGE_ROOT/reports"
DIAGNOSTIC_ROOT="$REPORTS/terminal_distribution_support_v1"
STATUS="$DIAGNOSTIC_ROOT/waiter_status.json"
LOG="$DIAGNOSTIC_ROOT/waiter.log"
PID_FILE="$DIAGNOSTIC_ROOT/waiter.pid"
LOCK="$QUALITY_BRIDGE_ROOT/terminal_distribution_support_waiter.lock"

cd "$PROJECT"
mkdir -p "$DIAGNOSTIC_ROOT"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing a concurrent terminal distribution-support waiter\n' >&2
  exit 75
}
pid_tmp="$PID_FILE.tmp.$$"
printf '%s\n' "$$" >"$pid_tmp"
mv "$pid_tmp" "$PID_FILE"

exec env \
  CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  nice -n 10 ionice -c 3 \
  "$PYTHON" scripts/wait_for_generation_terminal_distribution_support.py \
    --project "$PROJECT" \
    --quality-bridge-root "$QUALITY_BRIDGE_ROOT" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-tree "$EXPECTED_TREE" \
    --expected-branch "$EXPECTED_BRANCH" \
    --expected-self-sha256 "$EXPECTED_WAITER_SHA256" \
    --expected-followup-revision "$EXPECTED_FOLLOWUP_REVISION" \
    --expected-followup-branch "$EXPECTED_FOLLOWUP_BRANCH" \
    --python "$PYTHON" \
    --status "$STATUS" \
    --log "$LOG" \
    --poll-seconds "$POLL_SECONDS" \
    --nearest-chunk-size "$NEAREST_CHUNK_SIZE"
