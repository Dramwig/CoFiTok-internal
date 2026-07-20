#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
CONFIG=configs/generation/imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k.json
RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k_v7"
DENOISE_REFERENCE="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_denoise_path_k8_probe5k_v2"
EQUAL_REFERENCE="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k_v3"
TARGET_REFERENCE="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_target_energy_k8_probe5k_v4"
CAPACITY_REFERENCE="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k_v5"
LIGHT_REFERENCE="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k_v6"
LEGACY_EVAL="$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
REPORT_ROOT="$PROJECT/artifacts/reports/generation/rank_recovery_probe_2026-07-21_v7"
MONITOR_REPORT="$OUTPUT_ROOT/generation_rank_recovery_probe_v7_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_rank_recovery_probe_v7_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_rank_recovery_probe_v7_monitor.pid"
MONITOR_NAME=generation_rank_recovery_capacity_path_hellinger_v7

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'Hellinger capacity-path probe requires a clean tracked worktree\n' >&2
  exit 66
fi

require_complete() {
  python scripts/validate_generation_training_completion.py \
    --training-report "$RUN/training_report.json" \
    --config "$CONFIG" --expected-steps 5000 \
    --expected-revision "$(git rev-parse HEAD)" >/dev/null
}

monitor_report_passes() {
  python - "$MONITOR_REPORT" "$MONITOR_NAME" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
):
    raise SystemExit(1)
PY
}

monitor_args=(
  --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT"
  --monitor-name "$MONITOR_NAME"
  --cofitok-run "$(basename "$RUN")"
  --dense-run "$(basename "$LIGHT_REFERENCE")"
  --expected-steps 5000
  --training-process-pattern '[s]cripts/train_generation.py.*capacity_path_hellinger'
  --runbook-process-pattern '[g]eneration_rank_recovery_capacity_path_hellinger_probe_2026-07-21.sh'
  --checkpoint-interval 5000 --checkpoint-grace-steps 250
  --poll-seconds 120 --stall-seconds 1800
  --idle-failure-grace-seconds 600
)

start_monitor() {
  if monitor_report_passes 2>/dev/null; then
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      return
    fi
  fi
  nohup python scripts/monitor_generation_pair.py "${monitor_args[@]}" \
    >"$MONITOR_LOG" 2>&1 </dev/null &
  local monitor_pid=$!
  local temporary="${MONITOR_PID_FILE}.tmp.$$"
  printf '%s\n' "$monitor_pid" >"$temporary"
  mv "$temporary" "$MONITOR_PID_FILE"
  sleep 1
  if ! kill -0 "$monitor_pid" 2>/dev/null && ! monitor_report_passes 2>/dev/null; then
    printf 'Hellinger capacity-path monitor exited during launch\n' >&2
    exit 67
  fi
}

snapshot_monitor() {
  python scripts/monitor_generation_pair.py "${monitor_args[@]}" --once
}

evaluate_candidate() {
  local checkpoint="$RUN/checkpoint_step_00005000.pt"
  local ema_eval="$RUN/checkpoint_eval_ema_t500_512"
  local model_eval="$RUN/checkpoint_eval_model_t500_512"
  local samples="$RUN/samples_probe512_ddim50_cfg15"
  test -f "$checkpoint"

  if [[ ! -f "$ema_eval/checkpoint_evaluation_report.json" ]]; then
    python scripts/evaluate_generation_checkpoint.py \
      --checkpoint "$checkpoint" --output-dir "$ema_eval" \
      --num-images 512 --timestep 500 --random-orders 16 \
      --weights ema --precision bf16
  fi
  if [[ ! -f "$model_eval/checkpoint_evaluation_report.json" ]]; then
    python scripts/evaluate_generation_checkpoint.py \
      --checkpoint "$checkpoint" --output-dir "$model_eval" \
      --num-images 512 --timestep 500 --random-orders 16 \
      --weights model --precision bf16
  fi
  for timestep in 50 250 750 950; do
    local timestep_eval="$RUN/checkpoint_eval_ema_t${timestep}_256_energy_scope"
    if [[ ! -f "$timestep_eval/checkpoint_evaluation_report.json" ]]; then
      python scripts/evaluate_generation_checkpoint.py \
        --checkpoint "$checkpoint" --output-dir "$timestep_eval" \
        --num-images 256 --timestep "$timestep" --random-orders 16 \
        --weights ema --precision bf16
    fi
  done

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
for reference in "$DENOISE_REFERENCE" "$EQUAL_REFERENCE" "$TARGET_REFERENCE" "$CAPACITY_REFERENCE" "$LIGHT_REFERENCE"; do
  test -f "$reference/checkpoint_eval_ema_t500_512/checkpoint_evaluation_report.json"
  test -f "$reference/checkpoint_eval_model_t500_512/checkpoint_evaluation_report.json"
done
start_monitor

if ! require_complete 2>/dev/null; then
  resume_args=()
  if [[ -f "$RUN/latest.json" ]]; then
    resume_args=(--resume auto)
  fi
  python scripts/run_generation_training_watchdog.py \
    --monitor-report "$MONITOR_REPORT" \
    --monitor-pid-file "$MONITOR_PID_FILE" \
    --expected-monitor-name "$MONITOR_NAME" \
    --status-output "$RUN/training_watchdog.json" \
    --poll-seconds 30 --startup-grace-seconds 600 \
    --monitor-silence-seconds 600 --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    python scripts/train_generation.py \
      --config "$CONFIG" --output-dir "$RUN" "${resume_args[@]}"
fi

require_complete
snapshot_monitor
monitor_report_passes
evaluate_candidate

python scripts/build_generation_rank_recovery_probe.py \
  --candidate "denoise_path=$DENOISE_REFERENCE" \
  --candidate "equal_progress=$EQUAL_REFERENCE" \
  --candidate "target_energy=$TARGET_REFERENCE" \
  --candidate "capacity_path_heavy=$CAPACITY_REFERENCE" \
  --candidate "capacity_path_light=$LIGHT_REFERENCE" \
  --candidate "capacity_path_hellinger=$RUN" \
  --legacy-checkpoint-eval "$LEGACY_EVAL" \
  --output "$REPORT_ROOT/rank_recovery_probe.json"

python scripts/audit_generation_probe_ema.py \
  --candidate "denoise_path=$DENOISE_REFERENCE" \
  --candidate "equal_progress=$EQUAL_REFERENCE" \
  --candidate "target_energy=$TARGET_REFERENCE" \
  --candidate "capacity_path_heavy=$CAPACITY_REFERENCE" \
  --candidate "capacity_path_light=$LIGHT_REFERENCE" \
  --candidate "capacity_path_hellinger=$RUN" \
  --output "$REPORT_ROOT/ema_lag_audit.json"
