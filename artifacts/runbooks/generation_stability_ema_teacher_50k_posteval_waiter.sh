#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
EXPECTED_STABILITY_DECISION_SHA256=${EXPECTED_STABILITY_DECISION_SHA256:?set the passing 5K stability decision SHA256}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set the matched 50K training revision}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set the matched 50K training branch}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean post-evaluation revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean post-evaluation branch}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-604800}
POLL_SECONDS=${POLL_SECONDS:-300}
MONITOR_SILENCE_SECONDS=${MONITOR_SILENCE_SECONDS:-900}

cd "$PROJECT"
export PYTHONPATH=src
exec "$PYTHON" scripts/run_generation_stability_50k_posteval_waiter.py \
  --project "$PROJECT" \
  --monitor-report "$OUTPUT_ROOT/pair_monitor.json" \
  --pair-summary "$OUTPUT_ROOT/reports/pair_summary.json" \
  --status-output "$OUTPUT_ROOT/reports/posteval_waiter.json" \
  --posteval-runbook \
    "$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_50k_posteval_after_training.sh" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --expected-monitor-name generation_stability_ema_teacher_matched_50k \
  --expected-stability-decision-sha256 "$EXPECTED_STABILITY_DECISION_SHA256" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-evaluation-revision "$EXPECTED_TARGET_REVISION" \
  --expected-evaluation-branch "$EXPECTED_TARGET_BRANCH" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --poll-seconds "$POLL_SECONDS" \
  --monitor-silence-seconds "$MONITOR_SILENCE_SECONDS"
