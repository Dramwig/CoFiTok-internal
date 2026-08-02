#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:-2c2c1f5166b73d4f28df93b276901671ac1a7836}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:-scale/generation-stability-50k-preflight}
EXPECTED_FROZEN_EVALUATION_REVISION=${EXPECTED_FROZEN_EVALUATION_REVISION:-c1efb12c6640f2d2d62ac7e9982c8804d96e7289}
EXPECTED_FROZEN_EVALUATION_BRANCH=${EXPECTED_FROZEN_EVALUATION_BRANCH:-scale/generation-stability-50k-posteval-v4}
EXPECTED_READINESS_REVISION=${EXPECTED_READINESS_REVISION:-5dd3488ac9b30274f4960195e252cc9fdb161002}
EXPECTED_READINESS_BRANCH=${EXPECTED_READINESS_BRANCH:-scale/generation-large-capacity}
EXPECTED_SUPPLEMENTAL_REVISION=${EXPECTED_SUPPLEMENTAL_REVISION:?set the clean supplemental revision}
EXPECTED_SUPPLEMENTAL_BRANCH=${EXPECTED_SUPPLEMENTAL_BRANCH:-scale/generation-large-capacity}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1209600}
POLL_SECONDS=${POLL_SECONDS:-300}
STATUS_SILENCE_SECONDS=${STATUS_SILENCE_SECONDS:-900}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
FULL_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
POSTEVAL_STATUS="$SCALING_ROOT/reports/posteval_waiter.json"
READINESS_STATUS="$FULL_ROOT/reports/readiness_waiter.json"
SUPPLEMENTAL_ROOT="$SCALING_ROOT/reports/frozen_posteval_supplemental"
SUPPLEMENTAL_REPORT="$SUPPLEMENTAL_ROOT/supplemental_qualification.json"
STATUS_OUTPUT="$SUPPLEMENTAL_ROOT/supplemental_waiter.json"
SUPPLEMENTAL_RUNBOOK="$PROJECT/artifacts/runbooks/generation_stability_frozen_50k_supplemental_after_posteval.sh"

cd "$PROJECT"
export PYTHONPATH=src
exec "$PYTHON" scripts/run_generation_stability_frozen_supplemental_waiter.py \
  --project "$PROJECT" \
  --posteval-status "$POSTEVAL_STATUS" \
  --readiness-status "$READINESS_STATUS" \
  --supplemental-report "$SUPPLEMENTAL_REPORT" \
  --status-output "$STATUS_OUTPUT" \
  --supplemental-runbook "$SUPPLEMENTAL_RUNBOOK" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-frozen-evaluation-revision "$EXPECTED_FROZEN_EVALUATION_REVISION" \
  --expected-frozen-evaluation-branch "$EXPECTED_FROZEN_EVALUATION_BRANCH" \
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION" \
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH" \
  --expected-supplemental-revision "$EXPECTED_SUPPLEMENTAL_REVISION" \
  --expected-supplemental-branch "$EXPECTED_SUPPLEMENTAL_BRANCH" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --poll-seconds "$POLL_SECONDS" \
  --status-silence-seconds "$STATUS_SILENCE_SECONDS"
