#!/usr/bin/env bash
set -euo pipefail

# CPU-only consumer of the completed frozen-checkpoint confirmation. It writes
# a scientific preparation decision and cannot authorize or launch training.
PROJECT=${PROJECT:?set PROJECT to the exact capacity-scaling decision checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
SCREEN_ROOT=${CAPACITY_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1}
CONTROL_ROOT=${CAPACITY_SCALING_CONTROL_ROOT:-$SCREEN_ROOT/control/capacity_scaling_50000}
EXPECTED_SELF_REVISION=${EXPECTED_SELF_REVISION:?set EXPECTED_SELF_REVISION}
EXPECTED_SELF_TREE=${EXPECTED_SELF_TREE:?set EXPECTED_SELF_TREE}
EXPECTED_SELF_BRANCH=${EXPECTED_SELF_BRANCH:?set EXPECTED_SELF_BRANCH}
EXPECTED_CONFIRMATION_REVISION=${EXPECTED_CONFIRMATION_REVISION:?set EXPECTED_CONFIRMATION_REVISION}
EXPECTED_CONFIRMATION_TREE=${EXPECTED_CONFIRMATION_TREE:?set EXPECTED_CONFIRMATION_TREE}
EXPECTED_CONFIRMATION_BRANCH=${EXPECTED_CONFIRMATION_BRANCH:?set EXPECTED_CONFIRMATION_BRANCH}
POLL_SECONDS=${POLL_SECONDS:-60}

CONFIRMATION_RESULT=${CAPACITY_CONFIRMATION_RESULT:-$SCREEN_ROOT/confirmation_10000/capacity_confirmation_result.json}
DECISION=${CAPACITY_SCALING_DECISION:-$CONTROL_ROOT/capacity_scaling_decision.json}
STATUS=${CAPACITY_SCALING_DECISION_STATUS:-$CONTROL_ROOT/decision_waiter_status.json}
LOCK=${CAPACITY_SCALING_DECISION_LOCK:-$SCREEN_ROOT/.capacity_scaling_decision_waiter.lock}

mkdir -p "$CONTROL_ROOT"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing concurrent capacity-scaling decision waiter\n' >&2
  exit 75
}

exec "$PYTHON" scripts/wait_for_generation_capacity_scaling_decision.py \
  --project "$PROJECT" \
  --capacity-confirmation-result "$CONFIRMATION_RESULT" \
  --decision "$DECISION" \
  --status "$STATUS" \
  --expected-self-revision "$EXPECTED_SELF_REVISION" \
  --expected-self-tree "$EXPECTED_SELF_TREE" \
  --expected-self-branch "$EXPECTED_SELF_BRANCH" \
  --expected-confirmation-revision "$EXPECTED_CONFIRMATION_REVISION" \
  --expected-confirmation-tree "$EXPECTED_CONFIRMATION_TREE" \
  --expected-confirmation-branch "$EXPECTED_CONFIRMATION_BRANCH" \
  --poll-seconds "$POLL_SECONDS"
