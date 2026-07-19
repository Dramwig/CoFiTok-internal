#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
REPORT_ROOT="$PROJECT/artifacts/reports/generation/rank_recovery_probe_2026-07-20_v2"
DENOISE_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_denoise_path_k8_probe5k_v2"
BAND_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_epsilon_band_k8_probe5k_v2"
LEGACY_EVAL="$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
MONITOR_REPORT="$OUTPUT_ROOT/generation_rank_recovery_probe_v2_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_rank_recovery_probe_v2_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_rank_recovery_probe_v2_monitor.pid"
MONITOR_NAME=generation_rank_recovery_probe_pair_v2

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'rank-recovery probe requires a clean tracked worktree\n' >&2
  exit 66
fi

require_complete() {
  python - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training did not reach its target: {path}")
if report.get("completed_steps") != 5000 or report.get("target_steps") != 5000:
    raise SystemExit(f"probe step mismatch: {path}")
if report.get("git", {}).get("dirty") not in {False, None}:
    raise SystemExit(f"training used a dirty tracked worktree: {path}")
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

start_monitor() {
  if monitor_report_passes; then
    printf 'rank-recovery monitor already has a terminal pass report\n'
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      printf 'rank-recovery monitor already active as PID %s\n' "$existing_pid"
      return
    fi
  fi
  nohup python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$DENOISE_RUN")" \
    --dense-run "$(basename "$BAND_RUN")" --expected-steps 5000 \
    --training-process-pattern '[s]cripts/train_generation.py.*rankcomplete_' \
    --runbook-process-pattern '[g]eneration_rank_recovery_probe_2026-07-19.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 120 --stall-seconds 1800 \
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
    printf 'rank-recovery monitor exited during launch; inspect %s\n' \
      "$MONITOR_LOG" >&2
    exit 1
  fi
}

snapshot_monitor() {
  python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$DENOISE_RUN")" \
    --dense-run "$(basename "$BAND_RUN")" --expected-steps 5000 \
    --training-process-pattern '[s]cripts/train_generation.py.*rankcomplete_' \
    --runbook-process-pattern '[g]eneration_rank_recovery_probe_2026-07-19.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 120 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 --once
}

run_training() {
  local config="$1"
  local run_dir="$2"
  if [[ -f "$run_dir/training_report.json" ]] \
    && require_complete "$run_dir/training_report.json"; then
    printf 'probe training already complete for %s\n' "$run_dir"
    return
  fi
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
    --monitor-silence-seconds 600 --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    python scripts/train_generation.py \
      --config "$config" --output-dir "$run_dir" "${resume_args[@]}"
}

evaluate_candidate() {
  local run_dir="$1"
  local checkpoint="$run_dir/checkpoint_step_00005000.pt"
  local checkpoint_eval="$run_dir/checkpoint_eval_ema_t500_512"
  local samples="$run_dir/samples_probe512_ddim50_cfg15"

  test -f "$checkpoint"
  if [[ ! -f "$checkpoint_eval/checkpoint_evaluation_report.json" ]]; then
    python scripts/evaluate_generation_checkpoint.py \
      --checkpoint "$checkpoint" --output-dir "$checkpoint_eval" \
      --num-images 512 --timestep 500 --random-orders 16 \
      --weights ema --precision bf16
  fi

  python scripts/generate_samples.py \
    --checkpoint "$checkpoint" --output-dir "$samples" \
    --num-samples 512 --batch-size 16 --sample-steps 50 \
    --prefix-budgets 1,2,4,8 --guidance-scale 1.5 \
    --cfg-batch-mode batched --weights ema --precision bf16 --resume

  if [[ ! -f "$samples/metrics/generation_metrics_report.json" ]]; then
    python scripts/evaluate_generation_metrics.py \
      --real-dir /root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val \
      --generated-dir "$samples/prefix_8" \
      --sampling-report "$samples/sampling_report.json" \
      --output-dir "$samples/metrics" \
      --cache-root "$OUTPUT_ROOT/eval_cache/torch_fidelity" \
      --min-samples 512 --skip-prc
  fi
}

test -f "$LEGACY_EVAL"
start_monitor

run_training \
  configs/generation/imagenet256_10pct_rankcomplete_denoise_path_k8_probe5k.json \
  "$DENOISE_RUN"
require_complete "$DENOISE_RUN/training_report.json"
snapshot_monitor

run_training \
  configs/generation/imagenet256_10pct_rankcomplete_epsilon_band_k8_probe5k.json \
  "$BAND_RUN"
require_complete "$BAND_RUN/training_report.json"
snapshot_monitor
monitor_report_passes

evaluate_candidate "$DENOISE_RUN"
evaluate_candidate "$BAND_RUN"

python scripts/build_generation_rank_recovery_probe.py \
  --candidate "denoise_path=$DENOISE_RUN" \
  --candidate "epsilon_band=$BAND_RUN" \
  --legacy-checkpoint-eval "$LEGACY_EVAL" \
  --output "$REPORT_ROOT/rank_recovery_probe.json"
