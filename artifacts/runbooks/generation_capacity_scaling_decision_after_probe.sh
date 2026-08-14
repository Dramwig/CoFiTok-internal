#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact capacity-scaling decision checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set STANDING_AUTHORIZATION}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}
EXPECTED_SELF_REVISION=${EXPECTED_SELF_REVISION:?set EXPECTED_SELF_REVISION}
EXPECTED_SELF_TREE=${EXPECTED_SELF_TREE:?set EXPECTED_SELF_TREE}
EXPECTED_SELF_BRANCH=${EXPECTED_SELF_BRANCH:?set EXPECTED_SELF_BRANCH}
EXPECTED_CAPACITY_REVISION=${EXPECTED_CAPACITY_REVISION:?set EXPECTED_CAPACITY_REVISION}
EXPECTED_CAPACITY_BRANCH=${EXPECTED_CAPACITY_BRANCH:?set EXPECTED_CAPACITY_BRANCH}
POLL_SECONDS=${POLL_SECONDS:-60}

REPORT_ROOT="$OUTPUT_ROOT/reports"
RESULT="$REPORT_ROOT/capacity_probe_result.json"
DECISION="$REPORT_ROOT/capacity_scaling_decision.json"
STATUS="$REPORT_ROOT/capacity_scaling_decision_waiter_status.json"
LOCK="$OUTPUT_ROOT/capacity_scaling_decision_waiter.lock"

mkdir -p "$REPORT_ROOT"
cd "$PROJECT"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing concurrent capacity-scaling decision waiter\n' >&2
  exit 75
}

exec "$PYTHON" scripts/wait_for_generation_capacity_scaling_decision.py \
  --project "$PROJECT" \
  --capacity-probe-result "$RESULT" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --decision "$DECISION" \
  --status "$STATUS" \
  --expected-self-revision "$EXPECTED_SELF_REVISION" \
  --expected-self-tree "$EXPECTED_SELF_TREE" \
  --expected-self-branch "$EXPECTED_SELF_BRANCH" \
  --expected-capacity-revision "$EXPECTED_CAPACITY_REVISION" \
  --expected-capacity-branch "$EXPECTED_CAPACITY_BRANCH" \
  --poll-seconds "$POLL_SECONDS"
