#!/usr/bin/env bash
set -euo pipefail

EVALUATION_PROJECT=${EVALUATION_PROJECT:?set EVALUATION_PROJECT}
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

SOURCE_REPORT_ROOT="$SOURCE_OUTPUT_ROOT/reports"
REPORT_ROOT="$FULL_OUTPUT_ROOT/reports"
TRAINING_SUPERVISOR_STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_training_supervisor_status.json"
TRAINING_SUPERVISOR_DEPLOYMENT="$SOURCE_REPORT_ROOT/capacity_full_300k_training_supervisor_deployment_receipt.json"
TRAINING_LAUNCH_RECEIPT="$REPORT_ROOT/capacity_full_300k_training_launch_receipt.json"
COFITOK_TRAINING="$FULL_OUTPUT_ROOT/cofitok/training_report.json"
DENSE_TRAINING="$FULL_OUTPUT_ROOT/dense_identity/training_report.json"
PAIR_MONITOR="$FULL_OUTPUT_ROOT/pair_monitor.json"
FINAL_GATE="$REPORT_ROOT/final_generation_gate.json"
COMPARISON="$REPORT_ROOT/comparison/large_scale_generation_comparison.json"
RUNBOOK="$EVALUATION_PROJECT/artifacts/runbooks/generation_capacity_full_300k_posteval_50k.sh"
STATUS="$SOURCE_REPORT_ROOT/capacity_full_300k_posteval_supervisor_status.json"
CHILD_LOG="$SOURCE_REPORT_ROOT/capacity_full_300k_posteval_child.log"
LOCK="$SOURCE_OUTPUT_ROOT/capacity_full_300k_posteval_supervisor.lock"

[[ -x "$PYTHON" ]]
[[ -f "$RUNBOOK" ]]
[[ -f "$TRAINING_SUPERVISOR_DEPLOYMENT" ]]
mkdir -p "$SOURCE_REPORT_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate capacity-full posteval supervisor\n' >&2
  exit 10
fi

cd "$EVALUATION_PROJECT"
export PYTHONPATH="$EVALUATION_PROJECT/src:$TRAINING_PROJECT/src"
exec nice -n 19 "$PYTHON" scripts/run_generation_capacity_full_300k_posteval_supervisor.py \
  --evaluation-project "$EVALUATION_PROJECT" \
  --training-project "$TRAINING_PROJECT" \
  --formal-project "$FORMAL_PROJECT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --full-output-root "$FULL_OUTPUT_ROOT" \
  --training-supervisor-status "$TRAINING_SUPERVISOR_STATUS" \
  --training-supervisor-deployment "$TRAINING_SUPERVISOR_DEPLOYMENT" \
  --expected-training-supervisor-deployment-sha256 "$EXPECTED_TRAINING_SUPERVISOR_DEPLOYMENT_SHA256" \
  --training-launch-receipt "$TRAINING_LAUNCH_RECEIPT" \
  --cofitok-training "$COFITOK_TRAINING" \
  --dense-training "$DENSE_TRAINING" \
  --pair-monitor "$PAIR_MONITOR" \
  --final-gate "$FINAL_GATE" \
  --comparison "$COMPARISON" \
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
