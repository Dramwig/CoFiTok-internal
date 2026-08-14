#!/usr/bin/env bash
set -euo pipefail

EXECUTION_PROJECT=${EXECUTION_PROJECT:?set EXECUTION_PROJECT}
TRAINING_PROJECT=${TRAINING_PROJECT:?set TRAINING_PROJECT}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:-/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json}

EXPECTED_EXECUTION_REVISION=${EXPECTED_EXECUTION_REVISION:?set EXPECTED_EXECUTION_REVISION}
EXPECTED_EXECUTION_TREE=${EXPECTED_EXECUTION_TREE:?set EXPECTED_EXECUTION_TREE}
EXPECTED_EXECUTION_BRANCH=${EXPECTED_EXECUTION_BRANCH:?set EXPECTED_EXECUTION_BRANCH}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_READINESS_REVISION=${EXPECTED_READINESS_REVISION:?set EXPECTED_READINESS_REVISION}
EXPECTED_READINESS_TREE=${EXPECTED_READINESS_TREE:?set EXPECTED_READINESS_TREE}
EXPECTED_READINESS_BRANCH=${EXPECTED_READINESS_BRANCH:?set EXPECTED_READINESS_BRANCH}
EXPECTED_READINESS_DEPLOYMENT_SHA256=${EXPECTED_READINESS_DEPLOYMENT_SHA256:?set EXPECTED_READINESS_DEPLOYMENT_SHA256}
EXPECTED_READINESS_CLARIFICATION_SHA256=${EXPECTED_READINESS_CLARIFICATION_SHA256:?set EXPECTED_READINESS_CLARIFICATION_SHA256}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}

REPORT_ROOT="$FULL_OUTPUT_ROOT/reports"
SOURCE_REPORT_ROOT="$SOURCE_OUTPUT_ROOT/reports"
READINESS="$REPORT_ROOT/capacity_full_300k_readiness.json"
READINESS_SUPERVISOR_STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_readiness_supervisor_status.json"
READINESS_DEPLOYMENT_RECEIPT="$SOURCE_REPORT_ROOT/capacity_full_300k_readiness_supervisor_deployment_receipt.json"
READINESS_DEPLOYMENT_CLARIFICATION="$SOURCE_REPORT_ROOT/capacity_full_300k_readiness_supervisor_deployment_validation_clarification.json"
COFITOK_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
DENSE_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
LAUNCH_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_launch.json"
TRAINING_LAUNCH_RECEIPT="$REPORT_ROOT/capacity_full_300k_training_launch_receipt.json"
COFITOK_RUN="$FULL_OUTPUT_ROOT/cofitok"
DENSE_RUN="$FULL_OUTPUT_ROOT/dense_identity"
REFERENCE_COFITOK="$SOURCE_OUTPUT_ROOT/base256_cofitok/checkpoint_step_00100000.pt"
REFERENCE_DENSE="$SOURCE_OUTPUT_ROOT/base256_dense_identity/checkpoint_step_00100000.pt"
RUNBOOK="$EXECUTION_PROJECT/artifacts/runbooks/generation_capacity_full_300k_execute.sh"
STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_training_supervisor_status.json"
CHILD_LOG="$SOURCE_REPORT_ROOT/capacity_full_300k_training_child.log"
LOCK="$SOURCE_OUTPUT_ROOT/capacity_full_300k_training_supervisor.lock"

[[ -x "$PYTHON" ]]
[[ -f "$RUNBOOK" ]]
[[ -f "$READINESS_DEPLOYMENT_RECEIPT" ]]
[[ -f "$READINESS_DEPLOYMENT_CLARIFICATION" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
mkdir -p "$SOURCE_REPORT_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate capacity-full training supervisor\n' >&2
  exit 10
fi

cd "$EXECUTION_PROJECT"
export PYTHONPATH="$EXECUTION_PROJECT/src:$TRAINING_PROJECT/src"
exec nice -n 19 "$PYTHON" scripts/run_generation_capacity_full_300k_training_supervisor.py \
  --execution-project "$EXECUTION_PROJECT" \
  --training-project "$TRAINING_PROJECT" \
  --formal-project "$FORMAL_PROJECT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --source-output-root "$SOURCE_OUTPUT_ROOT" \
  --full-output-root "$FULL_OUTPUT_ROOT" \
  --readiness "$READINESS" \
  --readiness-supervisor-status "$READINESS_SUPERVISOR_STATUS" \
  --readiness-deployment-receipt "$READINESS_DEPLOYMENT_RECEIPT" \
  --expected-readiness-deployment-sha256 "$EXPECTED_READINESS_DEPLOYMENT_SHA256" \
  --readiness-deployment-clarification "$READINESS_DEPLOYMENT_CLARIFICATION" \
  --expected-readiness-clarification-sha256 "$EXPECTED_READINESS_CLARIFICATION_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --launch-storage-capacity "$LAUNCH_STORAGE_CAPACITY" \
  --training-launch-receipt "$TRAINING_LAUNCH_RECEIPT" \
  --cofitok-run-dir "$COFITOK_RUN" \
  --dense-run-dir "$DENSE_RUN" \
  --reference-cofitok "$REFERENCE_COFITOK" \
  --reference-dense "$REFERENCE_DENSE" \
  --runbook "$RUNBOOK" \
  --status-output "$STATUS" \
  --child-log "$CHILD_LOG" \
  --expected-execution-revision "$EXPECTED_EXECUTION_REVISION" \
  --expected-execution-tree "$EXPECTED_EXECUTION_TREE" \
  --expected-execution-branch "$EXPECTED_EXECUTION_BRANCH" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-tree "$EXPECTED_TRAINING_TREE" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION" \
  --expected-readiness-tree "$EXPECTED_READINESS_TREE" \
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH" \
  --poll-seconds 60 \
  --required-idle-polls 5 \
  --max-attempts 8 \
  --retry-seconds 120 \
  --timeout-seconds 31536000
