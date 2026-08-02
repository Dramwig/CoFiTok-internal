#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_SCALING_GATE_SHA256=${EXPECTED_SCALING_GATE_SHA256:?set the passing stability scaling gate SHA256}
EXPECTED_STABILITY_SUPPLEMENTAL_SHA256=${EXPECTED_STABILITY_SUPPLEMENTAL_SHA256:?set the passing frozen stability supplemental SHA256}
EXPECTED_READINESS_SHA256=${EXPECTED_READINESS_SHA256:?set the immutable stability-full readiness SHA256}
EXPECTED_READINESS_BRIDGE_SHA256=${EXPECTED_READINESS_BRIDGE_SHA256:?set the immutable readiness revision bridge SHA256}
EXPECTED_FULL_LAUNCH_RECEIPT_SHA256=${EXPECTED_FULL_LAUNCH_RECEIPT_SHA256:-}
EXPECTED_DEPLOYMENT_RECEIPT_SHA256=${EXPECTED_DEPLOYMENT_RECEIPT_SHA256:?set the isolated deployment receipt SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean stability-full training revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean stability-full training branch}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
GATE="$SCALING_ROOT/reports/promotion_gate.json"
STABILITY_SUPPLEMENTAL="$SCALING_ROOT/reports/frozen_posteval_supplemental/supplemental_qualification.json"
REFERENCE_COFITOK="$SCALING_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
REFERENCE_DENSE="$SCALING_ROOT/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
COFITOK_CONFIG=configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/training"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity.json"
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
READINESS="$REPORT_ROOT/full_training_readiness.json"
READINESS_BRIDGE="$REPORT_ROOT/full_training_readiness_bridge.json"
LAUNCH_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_launch.json"
CURRENT_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_current.json"
FULL_LAUNCH_RECEIPT="$REPORT_ROOT/full_training_launch_receipt.json"
DEPLOYMENT_RECEIPT="$CHECKPOINT_ROOT/deployment/large_capacity/deployments/$EXPECTED_TARGET_REVISION/deployment_receipt.json"
MONITOR_REPORT="$OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/pair_monitor.pid"
MONITOR_NAME=generation_stability_ema_teacher_full_matched_300k

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$GATE" ]]
[[ -f "$STABILITY_SUPPLEMENTAL" ]]
[[ -f "$REFERENCE_COFITOK" ]]
[[ -f "$REFERENCE_DENSE" ]]
[[ -f "$READINESS" ]]
[[ -f "$READINESS_BRIDGE" ]]
[[ -f "$DEPLOYMENT_RECEIPT" ]]
[[ "$(sha256sum "$GATE" | awk '{print $1}')" == "$EXPECTED_SCALING_GATE_SHA256" ]]
[[ "$(sha256sum "$STABILITY_SUPPLEMENTAL" | awk '{print $1}')" == "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256" ]]
[[ "$(sha256sum "$READINESS" | awk '{print $1}')" == "$EXPECTED_READINESS_SHA256" ]]
[[ "$(sha256sum "$READINESS_BRIDGE" | awk '{print $1}')" == "$EXPECTED_READINESS_BRIDGE_SHA256" ]]
[[ "$(sha256sum "$DEPLOYMENT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" ]]
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$GATE" \
  --stage scaling >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing stability full training launch while the GPU is busy\n' >&2
  exit 9
fi

runtime_selected="$("$PYTHON" scripts/validate_generation_full_readiness_bridge.py \
  --project-root "$PROJECT" \
  --bridge "$READINESS_BRIDGE" \
  --expected-bridge-sha256 "$EXPECTED_READINESS_BRIDGE_SHA256" \
  --readiness "$READINESS" \
  --expected-readiness-sha256 "$EXPECTED_READINESS_SHA256" \
  --source-deployment-receipt "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_deployment_receipt"]["path"])' "$READINESS_BRIDGE")" \
  --expected-source-deployment-receipt-sha256 "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_deployment_receipt"]["sha256"])' "$READINESS_BRIDGE")" \
  --target-deployment-receipt "$DEPLOYMENT_RECEIPT" \
  --expected-target-deployment-receipt-sha256 "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" \
  --expected-source-revision "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_git"]["revision"])' "$READINESS_BRIDGE")" \
  --expected-source-branch "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_git"]["branch"])' "$READINESS_BRIDGE")" \
  --expected-target-revision "$EXPECTED_TARGET_REVISION" \
  --expected-target-branch "$EXPECTED_TARGET_BRANCH" \
  --print-selected-runtime)"
read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION <<<"$runtime_selected"
if [[ ! "$SELECTED_MICRO_BATCH" =~ ^[0-9]+$ \
  || ! "$SELECTED_ACCUMULATION" =~ ^[0-9]+$ \
  || $((SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION)) -ne 64 ]]; then
  printf 'invalid readiness-selected stability-full runtime: %s\n' "$runtime_selected" >&2
  exit 10
fi

storage_preflight() {
  local output="$1"
  "$PYTHON" scripts/check_generation_storage_capacity.py \
    --path "$CHECKPOINT_ROOT" \
    --output "$output" \
    --stage full_training \
    --reference-checkpoint "$REFERENCE_COFITOK" \
    --reference-checkpoint "$REFERENCE_DENSE" \
    --checkpoint-count 16 \
    --checkpoint-size-multiplier 4.0 \
    --sample-count 116640 \
    --estimated-sample-kib 256 \
    --additional-gib 16 \
    --safety-margin-gib 64 >/dev/null
}

launch_receipt_args=(
  --project-root "$PROJECT"
  --receipt "$FULL_LAUNCH_RECEIPT"
  --deployment-receipt "$DEPLOYMENT_RECEIPT"
  --promotion-gate "$GATE"
  --stability-supplemental "$STABILITY_SUPPLEMENTAL"
  --expected-stability-supplemental-sha256 "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256"
  --full-readiness "$READINESS"
  --readiness-bridge "$READINESS_BRIDGE"
  --expected-readiness-sha256 "$EXPECTED_READINESS_SHA256"
  --cofitok-config "$COFITOK_CONFIG"
  --dense-config "$DENSE_CONFIG"
  --config-validation "$CONFIG_VALIDATION"
  --storage-capacity "$STORAGE_CAPACITY"
  --runtime-selection "$RUNTIME_SELECTION"
  --launch-storage-capacity "$LAUNCH_STORAGE_CAPACITY"
  --training-run-dir "$COFITOK_RUN"
  --training-run-dir "$DENSE_RUN"
  --benchmark-root "$RUNTIME_BENCHMARK_ROOT"
  --storage-path "$CHECKPOINT_ROOT"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
  --require-current-runtime-environment
)

if [[ -f "$FULL_LAUNCH_RECEIPT" ]]; then
  if [[ -z "$EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'set EXPECTED_FULL_LAUNCH_RECEIPT_SHA256 to resume stability full training\n' >&2
    exit 10
  fi
  "$PYTHON" scripts/validate_generation_full_launch_receipt.py \
    "${launch_receipt_args[@]}" \
    --expected-receipt-sha256 "$EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" >/dev/null
  storage_preflight "$CURRENT_STORAGE_CAPACITY"
else
  if [[ -n "$EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'expected stability full launch receipt does not exist: %s\n' \
      "$FULL_LAUNCH_RECEIPT" >&2
    exit 10
  fi
  for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
    if [[ -d "$run_dir" ]] \
      && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      printf 'refusing unreceipted stability full training state: %s\n' \
        "$run_dir" >&2
      exit 10
    fi
  done
  if [[ -e "$LAUNCH_STORAGE_CAPACITY" ]]; then
    printf 'refusing stale launch storage evidence without a receipt: %s\n' \
      "$LAUNCH_STORAGE_CAPACITY" >&2
    exit 10
  fi
  storage_preflight "$LAUNCH_STORAGE_CAPACITY"
  cleanup_unbound_launch_storage() {
    if [[ ! -f "$FULL_LAUNCH_RECEIPT" ]]; then
      rm -f -- "$LAUNCH_STORAGE_CAPACITY"
    fi
  }
  trap cleanup_unbound_launch_storage EXIT
  "$PYTHON" scripts/build_generation_full_launch_receipt.py \
    --project-root "$PROJECT" \
    --deployment-receipt "$DEPLOYMENT_RECEIPT" \
    --promotion-gate "$GATE" \
    --stability-supplemental "$STABILITY_SUPPLEMENTAL" \
    --expected-stability-supplemental-sha256 "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256" \
    --full-readiness "$READINESS" \
    --readiness-bridge "$READINESS_BRIDGE" \
    --expected-readiness-sha256 "$EXPECTED_READINESS_SHA256" \
    --cofitok-config "$COFITOK_CONFIG" \
    --dense-config "$DENSE_CONFIG" \
    --config-validation "$CONFIG_VALIDATION" \
    --storage-capacity "$STORAGE_CAPACITY" \
    --runtime-selection "$RUNTIME_SELECTION" \
    --launch-storage-capacity "$LAUNCH_STORAGE_CAPACITY" \
    --training-run-dir "$COFITOK_RUN" \
    --training-run-dir "$DENSE_RUN" \
    --benchmark-root "$RUNTIME_BENCHMARK_ROOT" \
    --storage-path "$CHECKPOINT_ROOT" \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-branch "$EXPECTED_TARGET_BRANCH" \
    --require-current-runtime-environment \
    --output "$FULL_LAUNCH_RECEIPT" >/dev/null
  trap - EXIT
  EXPECTED_FULL_LAUNCH_RECEIPT_SHA256="$(sha256sum "$FULL_LAUNCH_RECEIPT" | awk '{print $1}')"
  printf 'stability full launch receipt: %s  %s\n' \
    "$EXPECTED_FULL_LAUNCH_RECEIPT_SHA256" "$FULL_LAUNCH_RECEIPT"
fi
monitor_report_passes() {
  "$PYTHON" - "$MONITOR_REPORT" "$MONITOR_NAME" \
    "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
git = report.get("git", {})
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
    or report.get("issues") != []
    or git.get("revision") != sys.argv[3]
    or git.get("branch") != sys.argv[4]
    or git.get("tracked_dirty") is not False
):
    raise SystemExit(1)
PY
}

start_monitor() {
  if monitor_report_passes; then
    printf 'stability full monitor already passed\n'
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] \
      && kill -0 "$existing_pid" 2>/dev/null; then
      printf 'stability full monitor already active as PID %s\n' "$existing_pid"
      return
    fi
  fi
  nohup "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 300000 \
    --training-process-pattern '[s]cripts/train_generation.py.*stability.*300k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_full_matched_300k_after_gate.sh' \
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
    printf 'stability full monitor failed during launch\n' >&2
    exit 11
  fi
}

snapshot_monitor() {
  "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 300000 \
    --training-process-pattern '[s]cripts/train_generation.py.*stability.*300k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_full_matched_300k_after_gate.sh' \
    --checkpoint-interval 5000 \
    --checkpoint-grace-steps 250 \
    --checkpoint-integrity-policy required \
    --expected-checkpoint-revision "$EXPECTED_TARGET_REVISION" \
    --poll-seconds 300 \
    --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    --once
}

latest_step() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "latest.json"
print(0 if not path.is_file() else int(json.loads(path.read_text(encoding="utf-8"))["step"]))
PY
}

validate_resume_identity() {
  "$PYTHON" - "$1" "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

latest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    latest.get("git_revision") != sys.argv[2]
    or latest.get("git_branch") != sys.argv[3]
    or latest.get("git_dirty") is not False
):
    raise SystemExit("stability full resume checkpoint has another Git identity")
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
    [[ -f "$protected_checkpoint" ]]
    [[ -f "$protected_checkpoint.integrity.json" ]]
    return
  fi
  if (( current == target )); then
    return
  fi
  local delta=$((target - current))
  local resume_args=()
  if (( current > 0 )); then
    validate_resume_identity "$run_dir/latest.json"
    resume_args=(--resume auto)
  fi
  "$PYTHON" scripts/run_generation_training_watchdog.py \
    --monitor-report "$MONITOR_REPORT" \
    --monitor-pid-file "$MONITOR_PID_FILE" \
    --expected-monitor-name "$MONITOR_NAME" \
    --status-output "$run_dir/training_watchdog_step_$(printf '%08d' "$target").json" \
    --poll-seconds 30 \
    --startup-grace-seconds 600 \
    --monitor-silence-seconds 900 \
    --monitor-process-grace-seconds 120 \
    --termination-grace-seconds 60 \
    -- \
    "$PYTHON" scripts/train_generation.py \
      --config "$config" \
      --output-dir "$run_dir" \
      --authorization-gate "$GATE" \
      --micro-batch-size "$SELECTED_MICRO_BATCH" \
      --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
      --stop-after-steps "$delta" \
      "${resume_args[@]}"
  local reached
  reached="$(latest_step "$run_dir")"
  if (( reached != target )); then
    printf 'run %s reached step %s instead of milestone %s\n' \
      "$run_dir" "$reached" "$target" >&2
    exit 12
  fi
}

paired_milestone_complete() {
  local step="$1"
  local tag
  tag="step_$(printf '%08d' "$step")"
  local report="$REPORT_ROOT/milestones/$tag.json"
  [[ -f "$report" ]] || return 1
  for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
    [[ -f "$run_dir/checkpoint_${tag}.pt" ]] || return 1
    [[ -f "$run_dir/checkpoint_${tag}.pt.integrity.json" ]] || return 1
  done
  "$PYTHON" scripts/validate_generation_milestone_report.py \
    --report "$report" \
    --expected-step "$step" >/dev/null
}

evaluate_milestone() {
  local method="$1"
  local run_dir="$2"
  local step="$3"
  local prefix_budget="$4"
  local random_orders="$5"
  local checkpoint="$run_dir/checkpoint_step_$(printf '%08d' "$step").pt"
  PROJECT="$PROJECT" \
    PYTHON="$PYTHON" \
    OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    bash artifacts/runbooks/generation_full_milestone_eval.sh \
      "$method" "$run_dir" "$checkpoint" "$step" \
      "$prefix_budget" "$random_orders"
}

build_paired_milestone() {
  local step="$1"
  local tag
  tag="step_$(printf '%08d' "$step")"
  local cofitok_milestone="$COFITOK_RUN/milestones/$tag"
  local dense_milestone="$DENSE_RUN/milestones/$tag"
  "$PYTHON" scripts/build_generation_milestone_report.py \
    --cofitok-generation "$cofitok_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --dense-generation "$dense_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --cofitok-checkpoint-eval "$cofitok_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --dense-checkpoint-eval "$dense_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --milestone-step "$step" \
    --expected-samples 2048 \
    --output "$REPORT_ROOT/milestones/$tag.json"
  "$PYTHON" scripts/validate_generation_milestone_report.py \
    --report "$REPORT_ROOT/milestones/$tag.json" \
    --expected-step "$step" >/dev/null
}

require_complete() {
  "$PYTHON" scripts/validate_generation_training_completion.py \
    --training-report "$1" \
    --config "$2" \
    --expected-steps 300000 \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" >/dev/null
}

start_monitor
for milestone in 50000 100000 200000 300000; do
  if paired_milestone_complete "$milestone"; then
    printf 'stability full paired milestone %s already complete\n' "$milestone"
    continue
  fi
  train_to_milestone "$COFITOK_CONFIG" "$COFITOK_RUN" "$milestone"
  snapshot_monitor
  evaluate_milestone cofitok "$COFITOK_RUN" "$milestone" 8 4

  train_to_milestone "$DENSE_CONFIG" "$DENSE_RUN" "$milestone"
  snapshot_monitor
  evaluate_milestone dense_identity "$DENSE_RUN" "$milestone" 1 0

  build_paired_milestone "$milestone"
done

require_complete "$COFITOK_RUN/training_report.json" "$COFITOK_CONFIG"
require_complete "$DENSE_RUN/training_report.json" "$DENSE_CONFIG"
snapshot_monitor
monitor_report_passes

"$PYTHON" scripts/validate_generation_training_pair.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --expected-dataset imagenet_256 \
  --expected-recipe-stage stability_full \
  --authorization-gate "$GATE" \
  >"$REPORT_ROOT/training_pair_validation.json"

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" \
  --config "$COFITOK_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/cofitok_training_audit.json"

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" \
  --config "$DENSE_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/dense_training_audit.json"
