#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/imagenet256_10pct_fixed_basis_v3_50k"
COFITOK_COMPATIBLE_RESUME_SOURCE_REVISION=58d83bfce2770eab2565b8c89a5f9a06201a0c86

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
eval "$(python scripts/print_generation_workspace_paths.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" --format shell)"
REPORT_ROOT="$SCALING_REPORT_ROOT"
COFITOK_RUN="$SCALING_COFITOK_RUN"
DENSE_RUN="$SCALING_DENSE_RUN"
MONITOR_REPORT="$OUTPUT_ROOT/generation_10pct_fixed_basis_v3_pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_10pct_fixed_basis_v3_pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_10pct_fixed_basis_v3_pair_monitor.pid"
MONITOR_NAME=generation_10pct_fixed_basis_v3_matched_pair
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
RUNTIME_COMPATIBILITY="$REPORT_ROOT/runtime_selection_resume_compatibility.json"
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

require_complete() {
  python scripts/validate_generation_training_completion.py \
    --training-report "$1" --config "$2" --expected-steps 50000 \
    --expected-revision "$(git rev-parse HEAD)" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" >/dev/null
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
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_10pct_fixed_basis_' \
    --runbook-process-pattern '[g]eneration_10pct_matched_50k_2026-07-12.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 300 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    >"$MONITOR_LOG" 2>&1 </dev/null 8>&- 9>&- &
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
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_10pct_fixed_basis_' \
    --runbook-process-pattern '[g]eneration_10pct_matched_50k_2026-07-12.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 300 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 --once
}

run_training() {
  local config="$1"
  local run_dir="$2"
  local compatible_source_revision="${3:-}"
  if [[ -f "$run_dir/training_report.json" ]] \
    && require_complete "$run_dir/training_report.json" "$config"; then
    printf 'training already complete for %s\n' "$run_dir"
    return
  fi
  local resume_args=()
  if [[ -f "$run_dir/latest.json" ]]; then
    resume_args=(--resume auto)
    local checkpoint_revision
    checkpoint_revision="$(python - "$run_dir/latest.json" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    latest = json.load(handle)
revision = latest.get("git_revision")
if not isinstance(revision, str) or len(revision) != 40:
    raise SystemExit("latest checkpoint has no full Git revision")
print(revision)
PY
)"
    local current_revision
    current_revision="$(git rev-parse HEAD)"
    if [[ "$checkpoint_revision" == "$current_revision" ]]; then
      :
    elif [[ -n "$compatible_source_revision" \
      && "$checkpoint_revision" == "$compatible_source_revision" ]]; then
      resume_args+=(--resume-source-revision "$compatible_source_revision")
    else
      printf 'checkpoint revision %s cannot resume under %s\n' \
        "$checkpoint_revision" "$current_revision" >&2
      exit 1
    fi
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
      --config "$config" --output-dir "$run_dir" \
      --micro-batch-size "$SELECTED_MICRO_BATCH" \
      --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
      "${resume_args[@]}"
}

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json \
  --output "$REPORT_ROOT/config_pair.json"

runtime_selected="$(python scripts/select_generation_training_runtime.py \
  --cofitok-config configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json \
  --output-root "$RUNTIME_BENCHMARK_ROOT" --output "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN" --training-run-dir "$DENSE_RUN" \
  --candidates 16x4,32x2,64x1 --effective-batch-size 64 \
  --benchmark-steps 8 --warmup-steps 2 --max-memory-fraction 0.90 \
  --compatible-source-revision "$COFITOK_COMPATIBLE_RESUME_SOURCE_REVISION" \
  --compatibility-output "$RUNTIME_COMPATIBILITY")"
read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION <<<"$runtime_selected"
if [[ ! "$SELECTED_MICRO_BATCH" =~ ^[0-9]+$ || ! "$SELECTED_ACCUMULATION" =~ ^[0-9]+$ ]]; then
  printf 'invalid selected runtime: %s\n' "$runtime_selected" >&2
  exit 1
fi
if (( SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION != 64 )); then
  printf 'selected runtime changes effective batch: %s\n' "$runtime_selected" >&2
  exit 1
fi

start_compressed_monitor

run_training \
  configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  "$COFITOK_RUN" \
  "$COFITOK_COMPATIBLE_RESUME_SOURCE_REVISION"
require_complete "$COFITOK_RUN/training_report.json" \
  configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json
snapshot_compressed_monitor

run_training \
  configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json \
  "$DENSE_RUN"
require_complete "$DENSE_RUN/training_report.json" \
  configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json
snapshot_compressed_monitor
monitor_report_passes
