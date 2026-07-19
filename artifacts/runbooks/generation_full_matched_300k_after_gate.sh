#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/full_imagenet256_300k"
MONITOR_REPORT="$OUTPUT_ROOT/generation_full_matched_300k_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_full_matched_300k_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_full_matched_300k_monitor.pid"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
eval "$(python scripts/print_generation_workspace_paths.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" --format shell)"
GATE="$SCALING_GATE"
COFITOK_RUN="$FULL_COFITOK_RUN"
DENSE_RUN="$FULL_DENSE_RUN"
REFERENCE_COFITOK="$SCALING_COFITOK_RUN/checkpoint_step_00050000.pt"
REFERENCE_DENSE="$SCALING_DENSE_RUN/checkpoint_step_00050000.pt"
RUNTIME_SELECTION="$FULL_REPORT_ROOT/runtime_selection.json"
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'full matched 300K training requires a clean tracked worktree\n' >&2
  exit 66
fi
mkdir -p "$FULL_REPORT_ROOT"

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_cofitok_k8_300k.json \
  --dense-config configs/generation/imagenet256_dense_300k.json \
  --stage full \
  --output "$FULL_REPORT_ROOT/config_recipe.json"

python scripts/validate_generation_gate_report.py \
  --gate "$GATE" --stage scaling

python scripts/check_generation_storage_capacity.py \
  --path "$OUTPUT_ROOT" \
  --output "$FULL_REPORT_ROOT/storage_preflight_training.json" \
  --stage full_training \
  --reference-checkpoint "$REFERENCE_COFITOK" \
  --reference-checkpoint "$REFERENCE_DENSE" \
  --checkpoint-count 16 --sample-count 16384 \
  --estimated-sample-kib 256 --additional-gib 16 --safety-margin-gib 64

runtime_selected="$(python scripts/select_generation_training_runtime.py \
  --cofitok-config configs/generation/imagenet256_cofitok_k8_300k.json \
  --dense-config configs/generation/imagenet256_dense_300k.json \
  --output-root "$RUNTIME_BENCHMARK_ROOT" --output "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN" --training-run-dir "$DENSE_RUN" \
  --candidates 16x4,32x2,64x1 --effective-batch-size 64 \
  --benchmark-steps 8 --warmup-steps 2 --max-memory-fraction 0.90)"
read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION <<<"$runtime_selected"
if [[ ! "$SELECTED_MICRO_BATCH" =~ ^[0-9]+$ || ! "$SELECTED_ACCUMULATION" =~ ^[0-9]+$ ]]; then
  printf 'invalid selected runtime: %s\n' "$runtime_selected" >&2
  exit 1
fi
if (( SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION != 64 )); then
  printf 'selected runtime changes effective batch: %s\n' "$runtime_selected" >&2
  exit 1
fi

monitor_report_passes() {
  python - "$MONITOR_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
with path.open(encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("status") != "pass" or report.get("stage") != "complete":
    raise SystemExit(1)
PY
}

start_full_monitor() {
  if monitor_report_passes; then
    printf 'full matched monitor already has a terminal pass report\n'
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      printf 'full matched monitor already active as PID %s\n' "$existing_pid"
      return
    fi
  fi
  nohup python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name generation_full_matched_300k \
    --cofitok-run imagenet256_full_cofitok_k8_300k \
    --dense-run imagenet256_full_dense_300k --expected-steps 300000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_.*300k' \
    --runbook-process-pattern '[g]eneration_full_matched_300k_after_gate.sh' \
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
      printf 'full matched monitor published pass and exited\n'
      return
    fi
    printf 'full matched monitor exited during launch; inspect %s\n' "$MONITOR_LOG" >&2
    exit 1
  fi
}

snapshot_full_monitor() {
  python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" --output "$MONITOR_REPORT" \
    --monitor-name generation_full_matched_300k \
    --cofitok-run imagenet256_full_cofitok_k8_300k \
    --dense-run imagenet256_full_dense_300k --expected-steps 300000 \
    --training-process-pattern '[s]cripts/train_generation.py.*imagenet256_.*300k' \
    --runbook-process-pattern '[g]eneration_full_matched_300k_after_gate.sh' \
    --checkpoint-interval 5000 --checkpoint-grace-steps 250 \
    --poll-seconds 300 --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 --once
}

require_complete() {
  python scripts/validate_generation_training_completion.py \
    --training-report "$1" --config "$2" --expected-steps 300000 \
    --expected-revision "$(git rev-parse HEAD)" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" >/dev/null
}

latest_step() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "latest.json"
if not path.is_file():
    print(0)
else:
    with path.open(encoding="utf-8") as handle:
        print(int(json.load(handle)["step"]))
PY
}

train_to_milestone() {
  local config="$1"
  local run_dir="$2"
  local target="$3"
  local current
  current="$(latest_step "$run_dir")"
  if (( current > target )); then
    local protected_checkpoint
    protected_checkpoint="$run_dir/checkpoint_step_$(printf '%08d' "$target").pt"
    if [[ ! -f "$protected_checkpoint" ]]; then
      printf 'run %s passed milestone %s without protected checkpoint at step %s\n' \
        "$run_dir" "$target" "$current" >&2
      exit 1
    fi
    return
  fi
  if (( current == target )); then
    return
  fi
  local delta=$((target - current))
  local resume_args=()
  if (( current > 0 )); then
    resume_args=(--resume auto)
  fi
  local watchdog_status
  watchdog_status="$run_dir/training_watchdog_step_$(printf '%08d' "$target").json"
  python scripts/run_generation_training_watchdog.py \
    --monitor-report "$MONITOR_REPORT" \
    --monitor-pid-file "$MONITOR_PID_FILE" \
    --expected-monitor-name generation_full_matched_300k \
    --status-output "$watchdog_status" \
    --poll-seconds 30 --startup-grace-seconds 600 \
    --monitor-silence-seconds 900 --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    python scripts/train_generation.py \
      --config "$config" --output-dir "$run_dir" \
      --authorization-gate "$GATE" \
      --micro-batch-size "$SELECTED_MICRO_BATCH" \
      --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
      --stop-after-steps "$delta" "${resume_args[@]}"
  local reached
  reached="$(latest_step "$run_dir")"
  if (( reached != target )); then
    printf 'run %s reached step %s instead of milestone %s\n' "$run_dir" "$reached" "$target" >&2
    exit 1
  fi
}

paired_milestone_complete() {
  local step="$1"
  local report
  report="$FULL_REPORT_ROOT/milestones/step_$(printf '%08d' "$step").json"
  if [[ ! -f "$report" ]]; then
    return 1
  fi
  local checkpoint_tag
  checkpoint_tag="checkpoint_step_$(printf '%08d' "$step").pt"
  for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
    if [[ ! -f "$run_dir/$checkpoint_tag" || ! -f "$run_dir/$checkpoint_tag.integrity.json" ]]; then
      return 1
    fi
  done
  python scripts/validate_generation_milestone_report.py \
    --report "$report" --expected-step "$step" >/dev/null
}

evaluate_milestone() {
  local method="$1"
  local run_dir="$2"
  local step="$3"
  local prefix_budget="$4"
  local random_orders="$5"
  local checkpoint
  checkpoint="$run_dir/checkpoint_step_$(printf '%08d' "$step").pt"
  test -f "$checkpoint"
  bash artifacts/runbooks/generation_full_milestone_eval.sh \
    "$method" "$run_dir" "$checkpoint" "$step" "$prefix_budget" "$random_orders"
}

build_paired_milestone() {
  local step="$1"
  local step_tag
  step_tag="step_$(printf '%08d' "$step")"
  local cofitok_milestone="$COFITOK_RUN/milestones/$step_tag"
  local dense_milestone="$DENSE_RUN/milestones/$step_tag"
  python scripts/build_generation_milestone_report.py \
    --cofitok-generation "$cofitok_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --dense-generation "$dense_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --cofitok-checkpoint-eval "$cofitok_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --dense-checkpoint-eval "$dense_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --milestone-step "$step" --expected-samples 2048 \
    --output "$FULL_REPORT_ROOT/milestones/$step_tag.json"
  python scripts/validate_generation_milestone_report.py \
    --report "$FULL_REPORT_ROOT/milestones/$step_tag.json" \
    --expected-step "$step" >/dev/null
}

start_full_monitor

for milestone in 50000 100000 200000 300000; do
  if paired_milestone_complete "$milestone"; then
    printf 'paired milestone %s already complete; skipping\n' "$milestone"
    continue
  fi
  train_to_milestone \
    configs/generation/imagenet256_cofitok_k8_300k.json "$COFITOK_RUN" "$milestone"
  snapshot_full_monitor
  evaluate_milestone cofitok "$COFITOK_RUN" "$milestone" 8 4

  train_to_milestone \
    configs/generation/imagenet256_dense_300k.json "$DENSE_RUN" "$milestone"
  snapshot_full_monitor
  evaluate_milestone dense_identity "$DENSE_RUN" "$milestone" 1 0

  build_paired_milestone "$milestone"
done

require_complete "$COFITOK_RUN/training_report.json" \
  configs/generation/imagenet256_cofitok_k8_300k.json
require_complete "$DENSE_RUN/training_report.json" \
  configs/generation/imagenet256_dense_300k.json
snapshot_full_monitor

python scripts/validate_generation_training_pair.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 --expected-revision "$(git rev-parse HEAD)" \
  --expected-dataset imagenet_256 --expected-recipe-stage full \
  --authorization-gate "$GATE" \
  >"$FULL_REPORT_ROOT/training_pair_validation.json"

python scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" --expected-steps 300000 \
  --checkpoint-interval 5000 --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$FULL_REPORT_ROOT/cofitok_training_audit.json"

python scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" --expected-steps 300000 \
  --checkpoint-interval 5000 --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$FULL_REPORT_ROOT/dense_training_audit.json"
