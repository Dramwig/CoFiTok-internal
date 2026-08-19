#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONTROL_PROCESS_MIGRATION_ALLOWED:-false}" != "true" ]]; then
  echo "CONTROL_PROCESS_MIGRATION_ALLOWED must equal true" >&2
  exit 2
fi

PROJECT=${PROJECT:?set PROJECT to the exact hardened supervisor checkout}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
EXPECTED_FORMAL_REVISION=${EXPECTED_FORMAL_REVISION:?set EXPECTED_FORMAL_REVISION}
EXPECTED_FORMAL_TREE=${EXPECTED_FORMAL_TREE:?set EXPECTED_FORMAL_TREE}
EXPECTED_FORMAL_BRANCH=${EXPECTED_FORMAL_BRANCH:?set EXPECTED_FORMAL_BRANCH}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:-/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}
CAPACITY_SCALING_50K_PID=${CAPACITY_SCALING_50K_PID:?set CAPACITY_SCALING_50K_PID}
CAPACITY_COMPLETION_100K_PID=${CAPACITY_COMPLETION_100K_PID:?set CAPACITY_COMPLETION_100K_PID}
CAPACITY_FULL_READINESS_PID=${CAPACITY_FULL_READINESS_PID:?set CAPACITY_FULL_READINESS_PID}
CAPACITY_FULL_TRAINING_PID=${CAPACITY_FULL_TRAINING_PID:?set CAPACITY_FULL_TRAINING_PID}
CAPACITY_FULL_POSTEVAL_PID=${CAPACITY_FULL_POSTEVAL_PID:?set CAPACITY_FULL_POSTEVAL_PID}
CAPACITY_FULL_FINALIZATION_PID=${CAPACITY_FULL_FINALIZATION_PID:?set CAPACITY_FULL_FINALIZATION_PID}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CHECKPOINT_ROOT/control_plane_continuity/capacity_pipeline_supervisor_migration_v1}
PLAN=$OUTPUT_ROOT/plan.json
APPROVAL=$OUTPUT_ROOT/approval.json
EXECUTION=$OUTPUT_ROOT/execution.json
LOCK="$OUTPUT_ROOT.lock"

[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$EXPECTED_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain=v1)" ]]
[[ "$(git -C "$FORMAL_PROJECT" rev-parse HEAD)" == "$EXPECTED_FORMAL_REVISION" ]]
[[ "$(git -C "$FORMAL_PROJECT" rev-parse HEAD^{tree})" == "$EXPECTED_FORMAL_TREE" ]]
[[ "$(git -C "$FORMAL_PROJECT" branch --show-current)" == "$EXPECTED_FORMAL_BRANCH" ]]
[[ ! -e "$OUTPUT_ROOT" ]]

mkdir -p "$(dirname "$OUTPUT_ROOT")"
exec 9>"$LOCK"
flock -n 9
mkdir "$OUTPUT_ROOT"

env -C "$PROJECT" \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/build_generation_capacity_pipeline_supervisor_migration.py \
  --project "$PROJECT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --formal-project "$FORMAL_PROJECT" \
  --expected-formal-revision "$EXPECTED_FORMAL_REVISION" \
  --expected-formal-tree "$EXPECTED_FORMAL_TREE" \
  --expected-formal-branch "$EXPECTED_FORMAL_BRANCH" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --process "capacity_scaling_50k_execution=$CAPACITY_SCALING_50K_PID" \
  --process "capacity_completion_100k_execution=$CAPACITY_COMPLETION_100K_PID" \
  --process "capacity_full_300k_readiness=$CAPACITY_FULL_READINESS_PID" \
  --process "capacity_full_300k_training=$CAPACITY_FULL_TRAINING_PID" \
  --process "capacity_full_300k_posteval=$CAPACITY_FULL_POSTEVAL_PID" \
  --process "capacity_full_300k_finalization=$CAPACITY_FULL_FINALIZATION_PID" \
  --plan-output "$PLAN" \
  --approval-output "$APPROVAL"

APPROVAL_SHA256="$(sha256sum "$APPROVAL" | awk '{print $1}')"
env -C "$PROJECT" \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/execute_generation_capacity_pipeline_supervisor_migration.py \
  --approval "$APPROVAL" \
  --expected-approval-sha256 "$APPROVAL_SHA256" \
  --output "$EXECUTION" \
  --stop-grace-seconds 10 \
  --status-timeout-seconds 30
