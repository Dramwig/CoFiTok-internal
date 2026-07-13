#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
GATE="$PROJECT/artifacts/reports/generation/imagenet256_10pct_compressed_matched_50k/promotion_gate.json"
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_full_cofitok_k8_300k"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_full_dense_300k"
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/full_imagenet256_300k"
RUNTIME_SELECTION="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/runtime_selection.json"
REFERENCE_COFITOK="$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k/checkpoint_step_00050000.pt"
REFERENCE_DENSE="$OUTPUT_ROOT/imagenet256_10pct_compressed_dense_50k/checkpoint_step_00050000.pt"
MONITOR_REPORT="$OUTPUT_ROOT/generation_full_matched_300k_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/generation_full_matched_300k_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/generation_full_matched_300k_monitor.pid"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'full matched 300K training requires a clean tracked worktree\n' >&2
  exit 66
fi

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_cofitok_k8_300k.json \
  --dense-config configs/generation/imagenet256_dense_300k.json \
  --stage full \
  --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/config_recipe.json"

python scripts/validate_generation_gate_report.py \
  --gate "$GATE" --stage scaling

python scripts/check_generation_storage_capacity.py \
  --path "$OUTPUT_ROOT" \
  --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/storage_preflight_training.json" \
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
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training did not reach its target: {sys.argv[1]}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"training step mismatch: {sys.argv[1]}")
PY
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
  report="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/milestones/step_$(printf '%08d' "$step").json"
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
    --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/milestones/$step_tag.json"
  python scripts/validate_generation_milestone_report.py \
    --report "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/milestones/$step_tag.json" \
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

require_complete "$COFITOK_RUN/training_report.json"
require_complete "$DENSE_RUN/training_report.json"
snapshot_full_monitor

python scripts/validate_generation_training_pair.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 --expected-revision "$(git rev-parse HEAD)" \
  --expected-dataset imagenet_256 --expected-recipe-stage full \
  --authorization-gate "$GATE" \
  >"$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/training_pair_validation.json"

python scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" --expected-steps 300000 \
  --checkpoint-interval 5000 --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/cofitok_training_audit.json"

python scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" --expected-steps 300000 \
  --checkpoint-interval 5000 --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/dense_training_audit.json"
