#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:?set the passing 5K decision SHA256}
EXPECTED_SCALING_GATE_SHA256=${EXPECTED_SCALING_GATE_SHA256:?set the passing stability 50K gate SHA256}
EXPECTED_FULL_READINESS_SHA256=${EXPECTED_FULL_READINESS_SHA256:?set the immutable full readiness SHA256}
EXPECTED_FULL_LAUNCH_RECEIPT_SHA256=${EXPECTED_FULL_LAUNCH_RECEIPT_SHA256:?set the externally authorized full launch receipt SHA256}
EXPECTED_SCALING_TRAINING_REVISION=${EXPECTED_SCALING_TRAINING_REVISION:?set the stability 50K training revision}
EXPECTED_SCALING_TRAINING_BRANCH=${EXPECTED_SCALING_TRAINING_BRANCH:?set the stability 50K training branch}
EXPECTED_SCALING_EVALUATION_REVISION=${EXPECTED_SCALING_EVALUATION_REVISION:?set the stability 50K evaluation revision}
EXPECTED_SCALING_EVALUATION_BRANCH=${EXPECTED_SCALING_EVALUATION_BRANCH:?set the stability 50K evaluation branch}
EXPECTED_FULL_TRAINING_REVISION=${EXPECTED_FULL_TRAINING_REVISION:?set the externally authorized full training revision}
EXPECTED_FULL_TRAINING_BRANCH=${EXPECTED_FULL_TRAINING_BRANCH:?set the externally authorized full training branch}
EXPECTED_EVALUATION_REVISION=${EXPECTED_EVALUATION_REVISION:?set the clean post-training evaluation revision}
EXPECTED_EVALUATION_BRANCH=${EXPECTED_EVALUATION_BRANCH:?set the clean post-training evaluation branch}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5184000}
POLL_SECONDS=${POLL_SECONDS:-300}
MAX_STAGE_ATTEMPTS=${MAX_STAGE_ATTEMPTS:-3}
RETRY_SECONDS=${RETRY_SECONDS:-120}

FULL_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
SCALING_GATE="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/reports/promotion_gate.json"
FULL_MONITOR="$FULL_ROOT/pair_monitor.json"
FULL_LAUNCH_RECEIPT="$FULL_ROOT/reports/full_training_launch_receipt.json"
STATUS_OUTPUT="$FULL_ROOT/reports/posttraining_supervisor.json"
POSTEVAL_RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_full_posteval_50k.sh"
EXPORT_RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_export_inference_artifacts.sh"
COMPLETION_RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_completion_audit.sh"

cd "$PROJECT"
export PYTHONPATH=src
exec "$PYTHON" scripts/run_generation_stability_posttraining_supervisor.py \
  --project "$PROJECT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --full-monitor "$FULL_MONITOR" \
  --full-launch-receipt "$FULL_LAUNCH_RECEIPT" \
  --expected-full-launch-receipt-sha256 "$EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --scaling-gate "$SCALING_GATE" \
  --expected-scaling-gate-sha256 "$EXPECTED_SCALING_GATE_SHA256" \
  --expected-full-readiness-sha256 "$EXPECTED_FULL_READINESS_SHA256" \
  --expected-scaling-training-revision "$EXPECTED_SCALING_TRAINING_REVISION" \
  --expected-scaling-training-branch "$EXPECTED_SCALING_TRAINING_BRANCH" \
  --expected-scaling-evaluation-revision "$EXPECTED_SCALING_EVALUATION_REVISION" \
  --expected-scaling-evaluation-branch "$EXPECTED_SCALING_EVALUATION_BRANCH" \
  --expected-full-training-revision "$EXPECTED_FULL_TRAINING_REVISION" \
  --expected-full-training-branch "$EXPECTED_FULL_TRAINING_BRANCH" \
  --expected-evaluation-revision "$EXPECTED_EVALUATION_REVISION" \
  --expected-evaluation-branch "$EXPECTED_EVALUATION_BRANCH" \
  --status-output "$STATUS_OUTPUT" \
  --posteval-runbook "$POSTEVAL_RUNBOOK" \
  --export-runbook "$EXPORT_RUNBOOK" \
  --completion-runbook "$COMPLETION_RUNBOOK" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --poll-seconds "$POLL_SECONDS" \
  --max-stage-attempts "$MAX_STAGE_ATTEMPTS" \
  --retry-seconds "$RETRY_SECONDS"
