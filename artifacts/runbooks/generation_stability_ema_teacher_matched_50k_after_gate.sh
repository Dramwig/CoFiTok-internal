#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STABILITY_ROOT="$CHECKPOINT_ROOT/stability_probe_2026-07-29"
SOURCE_PAIR="$STABILITY_ROOT/pair5k_rollout_x0_u2_ema_teacher"
STABILITY_DECISION=${STABILITY_DECISION:-"$STABILITY_ROOT/scaling_decision5k_rollout_x0_u2_ema_teacher_to_50k/scaling_decision.json"}
EXPECTED_SOURCE_REVISION=59db142fc45d69dc92bb0333be5ac2d0162d9dc4
EXPECTED_STABILITY_DECISION_SHA256=${EXPECTED_STABILITY_DECISION_SHA256:?set the passing 5K stability decision SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean deployed stability-scaling revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:-scale/generation-stability}

COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
DECISION_VALIDATION="$REPORT_ROOT/stability_decision_validation.json"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
STORAGE_VALIDATION="$REPORT_ROOT/storage_capacity.json"
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
PAIR_SUMMARY="$REPORT_ROOT/pair_summary.json"
MONITOR_REPORT="$OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/pair_monitor.pid"
MONITOR_NAME=generation_stability_ema_teacher_matched_50k
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
source "$PROJECT/artifacts/runbooks/lib/generation_exact_runbook_identity.sh"
cofitok_capture_runbook_monitor_identity "$PYTHON" "$PROJECT" "$$"
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

"$PYTHON" scripts/validate_generation_stability_scaling_decision.py \
  --decision "$STABILITY_DECISION" \
  --expected-decision-sha256 "$EXPECTED_STABILITY_DECISION_SHA256" \
  --expected-source-revision "$EXPECTED_SOURCE_REVISION" \
  --expected-next-stage fresh_matched_50k_preparation \
  --output "$DECISION_VALIDATION" >/dev/null

"$PYTHON" scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_scaling \
  --output "$CONFIG_VALIDATION" >/dev/null

"$PYTHON" scripts/check_generation_storage_capacity.py \
  --path "$CHECKPOINT_ROOT" \
  --output "$STORAGE_VALIDATION" \
  --stage stability_scaling_matched_50k \
  --reference-checkpoint "$SOURCE_PAIR/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00005000.pt" \
  --reference-checkpoint "$SOURCE_PAIR/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00005000.pt" \
  --checkpoint-count 10 \
  --sample-count 20000 \
  --estimated-sample-kib 256 \
  --additional-gib 32 \
  --safety-margin-gib 64 >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing stability 50K runtime selection while the GPU is busy\n' >&2
  exit 9
fi

require_complete() {
  "$PYTHON" scripts/validate_generation_training_completion.py \
    --training-report "$1" \
    --config "$2" \
    --expected-steps 50000 \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-branch "$EXPECTED_TARGET_BRANCH" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
    >/dev/null
}

monitor_report_passes() {
  "$PYTHON" - "$MONITOR_REPORT" "$MONITOR_NAME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
identity = report.get("runbook_identity", {})
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
    or identity.get("role") != "generation_exact_runbook_process_identity"
    or identity.get("mode") != "exact_process_identity"
    or identity.get("status") != "active"
    or identity.get("mismatches") != []
):
    raise SystemExit(1)
PY
}

start_monitor() {
  if monitor_report_passes; then
    printf 'stability matched 50K monitor already passed\n'
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      if ! cofitok_monitor_report_matches_bound_controller \
        "$PYTHON" "$MONITOR_REPORT" "$MONITOR_NAME"; then
        printf 'stability matched 50K monitor belongs to another controller\n' >&2
        exit 10
      fi
      printf 'stability matched 50K monitor already active as PID %s\n' \
        "$existing_pid"
      return
    fi
  fi
  nohup "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*ema_teacher.*50k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_matched_50k_after_gate.sh' \
    --runbook-process-pid "$COFITOK_RUNBOOK_PROCESS_PID" \
    --runbook-process-start-ticks "$COFITOK_RUNBOOK_PROCESS_START_TICKS" \
    --runbook-process-executable "$COFITOK_RUNBOOK_PROCESS_EXECUTABLE" \
    --runbook-process-cwd "$COFITOK_RUNBOOK_PROCESS_CWD" \
    --runbook-process-cmdline-sha256 "$COFITOK_RUNBOOK_PROCESS_CMDLINE_SHA256" \
    --checkpoint-interval 5000 \
    --checkpoint-grace-steps 250 \
    --checkpoint-integrity-policy required \
    --expected-checkpoint-revision "$EXPECTED_TARGET_REVISION" \
    --poll-seconds 300 \
    --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    >"$MONITOR_LOG" 2>&1 </dev/null &
  local monitor_pid=$!
  local temporary="${MONITOR_PID_FILE}.tmp.$$"
  printf '%s\n' "$monitor_pid" >"$temporary"
  mv "$temporary" "$MONITOR_PID_FILE"
  sleep 1
  if ! kill -0 "$monitor_pid" 2>/dev/null && ! monitor_report_passes; then
    printf 'stability matched 50K monitor failed during launch\n' >&2
    exit 10
  fi
}

snapshot_monitor() {
  "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*ema_teacher.*50k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_matched_50k_after_gate.sh' \
    --runbook-process-pid "$COFITOK_RUNBOOK_PROCESS_PID" \
    --runbook-process-start-ticks "$COFITOK_RUNBOOK_PROCESS_START_TICKS" \
    --runbook-process-executable "$COFITOK_RUNBOOK_PROCESS_EXECUTABLE" \
    --runbook-process-cwd "$COFITOK_RUNBOOK_PROCESS_CWD" \
    --runbook-process-cmdline-sha256 "$COFITOK_RUNBOOK_PROCESS_CMDLINE_SHA256" \
    --checkpoint-interval 5000 \
    --checkpoint-grace-steps 250 \
    --checkpoint-integrity-policy required \
    --expected-checkpoint-revision "$EXPECTED_TARGET_REVISION" \
    --poll-seconds 300 \
    --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    --once
}

run_training() {
  local config="$1"
  local run_dir="$2"
  if [[ -f "$run_dir/training_report.json" ]] \
    && require_complete "$run_dir/training_report.json" "$config"; then
    printf 'training already complete for %s\n' "$run_dir"
    return
  fi
  local resume_args=()
  if [[ -f "$run_dir/latest.json" ]]; then
    "$PYTHON" - "$run_dir/latest.json" "$EXPECTED_TARGET_REVISION" <<'PY'
import json
import sys
from pathlib import Path

latest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if latest.get("git_revision") != sys.argv[2] or latest.get("git_dirty") is not False:
    raise SystemExit("stability 50K resume checkpoint has another Git identity")
PY
    resume_args=(--resume auto)
  fi
  "$PYTHON" scripts/run_generation_training_watchdog.py \
    --monitor-report "$MONITOR_REPORT" \
    --monitor-pid-file "$MONITOR_PID_FILE" \
    --expected-monitor-name "$MONITOR_NAME" \
    --status-output "$run_dir/training_watchdog.json" \
    --poll-seconds 30 \
    --startup-grace-seconds 600 \
    --monitor-silence-seconds 900 \
    --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    "$PYTHON" scripts/train_generation.py \
      --config "$config" \
      --output-dir "$run_dir" \
      --micro-batch-size "$SELECTED_MICRO_BATCH" \
      --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
      "${resume_args[@]}"
}

runtime_selected="$("$PYTHON" scripts/select_generation_training_runtime.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --output-root "$RUNTIME_BENCHMARK_ROOT" \
  --output "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN" \
  --training-run-dir "$DENSE_RUN" \
  --candidates 16x4,32x2,64x1 \
  --effective-batch-size 64 \
  --benchmark-steps 8 \
  --warmup-steps 2 \
  --max-memory-fraction 0.90)"
read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION <<<"$runtime_selected"
if [[ ! "$SELECTED_MICRO_BATCH" =~ ^[0-9]+$ \
  || ! "$SELECTED_ACCUMULATION" =~ ^[0-9]+$ \
  || $((SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION)) -ne 64 ]]; then
  printf 'invalid selected runtime: %s\n' "$runtime_selected" >&2
  exit 11
fi

start_monitor
run_training "$COFITOK_CONFIG" "$COFITOK_RUN"
require_complete "$COFITOK_RUN/training_report.json" "$COFITOK_CONFIG"
snapshot_monitor

run_training "$DENSE_CONFIG" "$DENSE_RUN"
require_complete "$DENSE_RUN/training_report.json" "$DENSE_CONFIG"
snapshot_monitor
monitor_report_passes

"$PYTHON" scripts/build_generation_stability_50k_summary.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --decision-validation "$DECISION_VALIDATION" \
  --config-validation "$CONFIG_VALIDATION" \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --output "$PAIR_SUMMARY"
