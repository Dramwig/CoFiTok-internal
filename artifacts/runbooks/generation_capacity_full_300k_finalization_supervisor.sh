#!/usr/bin/env bash
set -euo pipefail

FINALIZATION_PROJECT=${FINALIZATION_PROJECT:?set FINALIZATION_PROJECT}
TRAINING_PROJECT=${TRAINING_PROJECT:?set TRAINING_PROJECT}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}

EXPECTED_EVALUATION_REVISION=${EXPECTED_EVALUATION_REVISION:?set EXPECTED_EVALUATION_REVISION}
EXPECTED_EVALUATION_TREE=${EXPECTED_EVALUATION_TREE:?set EXPECTED_EVALUATION_TREE}
EXPECTED_EVALUATION_BRANCH=${EXPECTED_EVALUATION_BRANCH:?set EXPECTED_EVALUATION_BRANCH}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256=${EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256:?set EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256}
EXPECTED_POSTEVAL_SUPERVISOR_DEPLOYMENT_SHA256=${EXPECTED_POSTEVAL_SUPERVISOR_DEPLOYMENT_SHA256:?set EXPECTED_POSTEVAL_SUPERVISOR_DEPLOYMENT_SHA256}

SOURCE_REPORT_ROOT="$SOURCE_OUTPUT_ROOT/reports"
REPORT_ROOT="$FULL_OUTPUT_ROOT/reports"
EXPORT_ROOT="$CHECKPOINT_ROOT/exports/stability_capacity_full_300k_v1"
POSTEVAL_SUPERVISOR_STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_posteval_supervisor_status.json"
POSTEVAL_SUPERVISOR_DEPLOYMENT="$SOURCE_REPORT_ROOT/capacity_full_300k_posteval_supervisor_deployment_receipt.json"
TRAINING_SUPERVISOR_DEPLOYMENT="$SOURCE_REPORT_ROOT/capacity_full_300k_training_supervisor_deployment_receipt.json"
TRAINING_LAUNCH_RECEIPT="$REPORT_ROOT/capacity_full_300k_training_launch_receipt.json"
FINAL_GATE="$REPORT_ROOT/final_generation_gate.json"
COMPARISON="$REPORT_ROOT/comparison/large_scale_generation_comparison.json"
COMPLETION_AUDIT="$REPORT_ROOT/capacity_full_generation_completion_audit.json"
RELEASE_RECEIPT="$EXPORT_ROOT/release_receipt.json"
COFITOK_ARTIFACT="$EXPORT_ROOT/cofitok_k8_ema_inference.pt"
DENSE_ARTIFACT="$EXPORT_ROOT/dense_identity_ema_inference.pt"
RUNBOOK="$FINALIZATION_PROJECT/artifacts/runbooks/generation_capacity_full_300k_finalize_after_gate.sh"
STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_finalization_supervisor_status.json"
CHILD_LOG="$SOURCE_REPORT_ROOT/capacity_full_300k_finalization_child.log"
LOCK="$SOURCE_OUTPUT_ROOT/capacity_full_300k_finalization_supervisor.lock"

[[ -x "$PYTHON" ]]
[[ -f "$RUNBOOK" ]]
[[ -f "$TRAINING_SUPERVISOR_DEPLOYMENT" ]]
[[ -f "$POSTEVAL_SUPERVISOR_DEPLOYMENT" ]]
[[ "$FULL_OUTPUT_ROOT" == "$CHECKPOINT_ROOT/stability_capacity_full_300k_v1" ]]
mkdir -p "$SOURCE_REPORT_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate capacity-full finalization supervisor\n' >&2
  exit 10
fi

cd "$FINALIZATION_PROJECT"
export PYTHONPATH="$FINALIZATION_PROJECT/src:$TRAINING_PROJECT/src"
exec nice -n 19 "$PYTHON" scripts/run_generation_capacity_full_300k_finalization_supervisor.py \
  --finalization-project "$FINALIZATION_PROJECT" \
  --training-project "$TRAINING_PROJECT" \
  --formal-project "$FORMAL_PROJECT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --full-output-root "$FULL_OUTPUT_ROOT" \
  --posteval-supervisor-status "$POSTEVAL_SUPERVISOR_STATUS" \
  --posteval-supervisor-deployment "$POSTEVAL_SUPERVISOR_DEPLOYMENT" \
  --expected-posteval-supervisor-deployment-sha256 "$EXPECTED_POSTEVAL_SUPERVISOR_DEPLOYMENT_SHA256" \
  --training-supervisor-deployment "$TRAINING_SUPERVISOR_DEPLOYMENT" \
  --expected-training-supervisor-deployment-sha256 "$EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256" \
  --training-launch-receipt "$TRAINING_LAUNCH_RECEIPT" \
  --final-gate "$FINAL_GATE" \
  --comparison "$COMPARISON" \
  --completion-audit "$COMPLETION_AUDIT" \
  --release-receipt "$RELEASE_RECEIPT" \
  --cofitok-artifact "$COFITOK_ARTIFACT" \
  --dense-artifact "$DENSE_ARTIFACT" \
  --runbook "$RUNBOOK" \
  --status-output "$STATUS" \
  --child-log "$CHILD_LOG" \
  --expected-evaluation-revision "$EXPECTED_EVALUATION_REVISION" \
  --expected-evaluation-tree "$EXPECTED_EVALUATION_TREE" \
  --expected-evaluation-branch "$EXPECTED_EVALUATION_BRANCH" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-tree "$EXPECTED_TRAINING_TREE" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --poll-seconds 60 \
  --required-idle-polls 5 \
  --max-attempts 8 \
  --retry-seconds 120 \
  --timeout-seconds 31536000
