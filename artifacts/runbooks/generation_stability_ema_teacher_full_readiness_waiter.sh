#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set the matched 50K training revision}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set the matched 50K training branch}
EXPECTED_EVALUATION_REVISION=${EXPECTED_EVALUATION_REVISION:?set the formal 50K post-evaluation revision}
EXPECTED_EVALUATION_BRANCH=${EXPECTED_EVALUATION_BRANCH:?set the formal 50K post-evaluation branch}
EXPECTED_READINESS_REVISION=${EXPECTED_READINESS_REVISION:?set the isolated large-capacity readiness revision}
EXPECTED_READINESS_BRANCH=${EXPECTED_READINESS_BRANCH:-scale/generation-large-capacity}
EXPECTED_DEPLOYMENT_RECEIPT_SHA256=${EXPECTED_DEPLOYMENT_RECEIPT_SHA256:?set the isolated deployment receipt SHA256}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1209600}
POLL_SECONDS=${POLL_SECONDS:-300}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
PROMOTION_GATE="$SCALING_ROOT/reports/promotion_gate.json"
POSTEVAL_STATUS="$SCALING_ROOT/reports/posteval_waiter.json"
DEPLOYMENT_RECEIPT="$CHECKPOINT_ROOT/deployment/large_capacity/deployments/$EXPECTED_READINESS_REVISION/deployment_receipt.json"
STATUS_OUTPUT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher/reports/readiness_waiter.json"
READINESS_RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_full_readiness_after_gate.sh"

cd "$PROJECT"
export PYTHONPATH=src
exec "$PYTHON" scripts/run_generation_stability_full_readiness_waiter.py \
  --project "$PROJECT" \
  --promotion-gate "$PROMOTION_GATE" \
  --posteval-status "$POSTEVAL_STATUS" \
  --deployment-receipt "$DEPLOYMENT_RECEIPT" \
  --expected-deployment-receipt-sha256 "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" \
  --status-output "$STATUS_OUTPUT" \
  --readiness-runbook "$READINESS_RUNBOOK" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-evaluation-revision "$EXPECTED_EVALUATION_REVISION" \
  --expected-evaluation-branch "$EXPECTED_EVALUATION_BRANCH" \
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION" \
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --poll-seconds "$POLL_SECONDS"
