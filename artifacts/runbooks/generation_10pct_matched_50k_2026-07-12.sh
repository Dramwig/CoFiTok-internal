#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_10pct_compressed_matched_50k"
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_10pct_compressed_dense_50k"
MONITOR_REPORT="$OUTPUT_ROOT/generation_10pct_compressed_pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_10pct_compressed_pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_10pct_compressed_pair_monitor.pid"
MONITOR_NAME=generation_10pct_compressed_matched_pair

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

require_complete() {
  python - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training did not reach its target: {path}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"training step mismatch: {path}")
PY
}

monitor_report_passes() {
  python - "$MONITOR_REPORT" "$MONITOR_NAME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
with path.open(encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
):
    raise SystemExit(1)
PY
}

start_compressed_monitor() {
  if monitor_report_passes; then
    printf 'compressed 10%% matched monitor already has a terminal pass report\n'
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      printf 'compressed 10%% matched monitor already active as PID %s\n' "$existing_pid"
      return
    fi
  fi
  nohup python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run imagenet256_10pct_compressed_cofitok_k8_50k \
    --dense-run imagenet256_10pct_compressed_dense_50k --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_10pct_compressed_' \
    --runbook-process-pattern '[g]eneration_10pct_matched_50k_2026-07-12.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 300 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    >"$MONITOR_LOG" 2>&1 </dev/null &
  local monitor_pid=$!
  local temporary="${MONITOR_PID_FILE}.tmp.$$"
  printf '%s\n' "$monitor_pid" >"$temporary"
  mv "$temporary" "$MONITOR_PID_FILE"
  sleep 1
  if ! kill -0 "$monitor_pid" 2>/dev/null; then
    if monitor_report_passes; then
      return
    fi
    printf 'compressed 10%% monitor exited during launch; inspect %s\n' \
      "$MONITOR_LOG" >&2
    exit 1
  fi
}

snapshot_compressed_monitor() {
  python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run imagenet256_10pct_compressed_cofitok_k8_50k \
    --dense-run imagenet256_10pct_compressed_dense_50k --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_10pct_compressed_' \
    --runbook-process-pattern '[g]eneration_10pct_matched_50k_2026-07-12.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 300 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 --once
}

run_training() {
  local config="$1"
  local run_dir="$2"
  local resume_args=()
  if [[ -f "$run_dir/latest.json" ]]; then
    resume_args=(--resume auto)
  fi
  python scripts/run_generation_training_watchdog.py \
    --monitor-report "$MONITOR_REPORT" \
    --monitor-pid-file "$MONITOR_PID_FILE" \
    --expected-monitor-name "$MONITOR_NAME" \
    --status-output "$run_dir/training_watchdog.json" \
    --poll-seconds 30 --startup-grace-seconds 600 \
    --monitor-silence-seconds 900 --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    python scripts/train_generation.py \
      --config "$config" --output-dir "$run_dir" "${resume_args[@]}"
}

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_10pct_compressed_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_compressed_dense_50k.json \
  --output "$REPORT_ROOT/config_pair.json"

start_compressed_monitor

run_training \
  configs/generation/imagenet256_10pct_compressed_cofitok_k8_50k.json \
  "$COFITOK_RUN"
require_complete "$COFITOK_RUN/training_report.json"
snapshot_compressed_monitor

run_training \
  configs/generation/imagenet256_10pct_compressed_dense_50k.json \
  "$DENSE_RUN"
require_complete "$DENSE_RUN/training_report.json"
snapshot_compressed_monitor
monitor_report_passes
