#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the exact terminal-exposure waiter checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_SELF_REVISION=${EXPECTED_SELF_REVISION:?set the exact waiter revision}
EXPECTED_SELF_TREE=${EXPECTED_SELF_TREE:?set the exact waiter tree}
EXPECTED_SELF_BRANCH=${EXPECTED_SELF_BRANCH:?set the exact waiter branch}

QUALITY_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
REPORT_ROOT="$QUALITY_ROOT/reports"
OUTPUT_ROOT="$REPORT_ROOT/training_exposure_terminal_100k"
STATUS="$REPORT_ROOT/training_exposure_terminal_100k_waiter_status.json"
PID_FILE="$REPORT_ROOT/training_exposure_terminal_100k_waiter.pid"
PREPARATION="$REPORT_ROOT/preparation.json"
EXPECTED_PREPARATION_SHA256=7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea

cd "$PROJECT"
mkdir -p "$REPORT_ROOT"
pid_tmp="$PID_FILE.tmp.$$"
printf '%s\n' "$$" >"$pid_tmp"
mv "$pid_tmp" "$PID_FILE"

exec env \
  CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/wait_for_generation_training_exposure_audit.py \
    --project "$PROJECT" \
    --cofitok-training "$QUALITY_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/training_report.json" \
    --dense-training "$QUALITY_ROOT/dense_rollout_x0_u2_ema_teacher/training_report.json" \
    --quality-bridge-preparation "$PREPARATION" \
    --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
    --milestone-report "$REPORT_ROOT/milestones/step_00100000.json" \
    --expected-milestone-step 100000 \
    --milestone-source-profile quality_bridge \
    --quality-bridge-terminal-result "$REPORT_ROOT/quality_bridge_result.json" \
    --quality-bridge-execution-status "$REPORT_ROOT/execution_status.json" \
    --output-root "$OUTPUT_ROOT" \
    --status "$STATUS" \
    --expected-self-revision "$EXPECTED_SELF_REVISION" \
    --expected-self-tree "$EXPECTED_SELF_TREE" \
    --expected-self-branch "$EXPECTED_SELF_BRANCH" \
    --poll-seconds 10
