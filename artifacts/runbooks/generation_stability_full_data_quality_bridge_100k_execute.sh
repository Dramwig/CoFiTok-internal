#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean quality-bridge execution revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean quality-bridge execution branch}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set the immutable preparation SHA256}
QUALITY_BRIDGE_EXECUTION_APPROVAL=${QUALITY_BRIDGE_EXECUTION_APPROVAL:?set the explicit execution approval sentinel path}
EXPECTED_EXECUTION_APPROVAL_SHA256=${EXPECTED_EXECUTION_APPROVAL_SHA256:?set the immutable execution approval SHA256}
QUALITY_BRIDGE_EXECUTION_ALLOWED=${QUALITY_BRIDGE_EXECUTION_ALLOWED:-false}
EXPECTED_LAUNCH_RECEIPT_SHA256=${EXPECTED_LAUNCH_RECEIPT_SHA256:-}
EXPECTED_QUALITY_BRIDGE_RESULT_SHA256=${EXPECTED_QUALITY_BRIDGE_RESULT_SHA256:-}

COFITOK_CONFIG=configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
PREPARATION="$REPORT_ROOT/preparation.json"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
LAUNCH_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_launch.json"
CURRENT_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_current.json"
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/training"
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
LAUNCH_RECEIPT="$REPORT_ROOT/launch_receipt.json"
TRAINING_PAIR_VALIDATION="$REPORT_ROOT/training_pair_validation.json"
COFITOK_TRAINING_AUDIT="$REPORT_ROOT/cofitok_training_audit.json"
DENSE_TRAINING_AUDIT="$REPORT_ROOT/dense_training_audit.json"
CLASS_FIDELITY_ROOT="$REPORT_ROOT/class_fidelity"
CLASS_FIDELITY_QUALIFICATION="$CLASS_FIDELITY_ROOT/qualification_report.json"
QUALITY_BRIDGE_RESULT="$REPORT_ROOT/quality_bridge_result.json"
EXECUTION_STATUS="$REPORT_ROOT/execution_status.json"
EXECUTION_LOCK="$OUTPUT_ROOT/quality_bridge_execution.lock"
MONITOR_REPORT="$OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/pair_monitor.pid"
MONITOR_NAME=generation_stability_full_data_quality_bridge_100k
REAL_DATA=${REAL_DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
EVAL_CACHE="$CHECKPOINT_ROOT/eval_cache/torch_fidelity"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$QUALITY_BRIDGE_EXECUTION_ALLOWED" == true ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
source "$PROJECT/artifacts/runbooks/lib/generation_exact_runbook_identity.sh"
cofitok_capture_runbook_monitor_identity "$PYTHON" "$PROJECT" "$$"
[[ -f "$PREPARATION" ]]
[[ -f "$QUALITY_BRIDGE_EXECUTION_APPROVAL" ]]
[[ "$(sha256sum "$PREPARATION" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$QUALITY_BRIDGE_EXECUTION_APPROVAL" | awk '{print $1}')" == "$EXPECTED_EXECUTION_APPROVAL_SHA256" ]]
[[ -d "$REAL_DATA" ]]
command -v flock >/dev/null
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"
exec 6>"$EXECUTION_LOCK"
if ! flock -n 6; then
  printf 'refusing concurrent quality bridge execution controller\n' >&2
  exit 15
fi

write_status() {
  local status="$1"
  local detail="$2"
  local exit_code="${3:-}"
  "$PYTHON" - "$EXECUTION_STATUS" "$status" "$detail" "$exit_code" \
    "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" "$$" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "role": "stability_full_data_quality_bridge_execution",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "exit_code": int(sys.argv[4]) if sys.argv[4] else None,
    "git": {
        "revision": sys.argv[5],
        "branch": sys.argv[6],
        "tracked_dirty": False,
    },
    "pid": int(sys.argv[7]),
    "hostname": socket.gethostname(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "quality_bridge_only": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(path)
PY
}

completed=false
on_exit() {
  local exit_code=$?
  if [[ "$completed" != true && $exit_code -ne 0 ]]; then
    write_status failed "quality bridge execution exited before terminal evidence verification" "$exit_code" || true
  fi
}
trap on_exit EXIT

write_status running "validating immutable preparation and explicit execution approval"

"$PYTHON" scripts/validate_generation_quality_bridge_preparation.py \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --promotion-gate "$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/reports/promotion_gate.json" \
  --expected-promotion-gate-sha256 2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90 \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" >/dev/null

"$PYTHON" scripts/validate_generation_quality_bridge_execution_approval.py \
  --approval "$QUALITY_BRIDGE_EXECUTION_APPROVAL" \
  --expected-approval-sha256 "$EXPECTED_EXECUTION_APPROVAL_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --expected-output-root "$OUTPUT_ROOT" >/dev/null

"$PYTHON" scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_quality_bridge \
  --output "$CONFIG_VALIDATION" >/dev/null

if pgrep -af '[t]rain_generation.py.*stability_quality_bridge.*100k' >/dev/null; then
  printf 'refusing quality bridge controller launch while a matching trainer is active\n' >&2
  exit 12
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing quality bridge launch while the GPU is busy\n' >&2
  exit 9
fi

storage_preflight() {
  local output="$1"
  "$PYTHON" scripts/check_generation_storage_capacity.py \
    --path "$CHECKPOINT_ROOT" \
    --output "$output" \
    --stage stability_full_data_quality_bridge_100k_execution \
    --reference-checkpoint "$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt" \
    --reference-checkpoint "$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt" \
    --checkpoint-count 10 \
    --sample-count 30000 \
    --estimated-sample-kib 256 \
    --additional-gib 16 \
    --safety-margin-gib 64 >/dev/null
}

receipt_args=(
  --project-root "$PROJECT"
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --execution-approval "$QUALITY_BRIDGE_EXECUTION_APPROVAL"
  --expected-execution-approval-sha256 "$EXPECTED_EXECUTION_APPROVAL_SHA256"
  --cofitok-config "$COFITOK_CONFIG"
  --dense-config "$DENSE_CONFIG"
  --config-validation "$CONFIG_VALIDATION"
  --storage-capacity "$LAUNCH_STORAGE_CAPACITY"
  --runtime-selection "$RUNTIME_SELECTION"
  --training-run-dir "$COFITOK_RUN"
  --training-run-dir "$DENSE_RUN"
  --benchmark-root "$RUNTIME_BENCHMARK_ROOT"
  --output-root "$OUTPUT_ROOT"
  --storage-path "$CHECKPOINT_ROOT"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
  --require-current-runtime-environment
)

if [[ -f "$LAUNCH_RECEIPT" ]]; then
  if [[ -z "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'set EXPECTED_LAUNCH_RECEIPT_SHA256 to resume the quality bridge\n' >&2
    exit 10
  fi
  "$PYTHON" scripts/validate_generation_quality_bridge_launch_receipt.py \
    "${receipt_args[@]}" \
    --receipt "$LAUNCH_RECEIPT" \
    --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
  storage_preflight "$CURRENT_STORAGE_CAPACITY"
else
  if [[ -n "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'expected quality bridge launch receipt is absent\n' >&2
    exit 10
  fi
  for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
    if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      printf 'refusing unreceipted quality bridge training state: %s\n' "$run_dir" >&2
      exit 10
    fi
  done
  runtime_selected="$("$PYTHON" scripts/select_generation_training_runtime.py \
    --project-root "$PROJECT" \
    --cofitok-config "$COFITOK_CONFIG" \
    --dense-config "$DENSE_CONFIG" \
    --output-root "$RUNTIME_BENCHMARK_ROOT" \
    --output "$RUNTIME_SELECTION" \
    --training-run-dir "$COFITOK_RUN" \
    --training-run-dir "$DENSE_RUN" \
    --candidates 16x4,32x2,64x1 \
    --baseline-candidate 16x4 \
    --effective-batch-size 64 \
    --benchmark-steps 8 \
    --warmup-steps 2 \
    --max-memory-fraction 0.90)"
  read -r selected_micro selected_accumulation <<<"$runtime_selected"
  if [[ ! "$selected_micro" =~ ^[0-9]+$ \
    || ! "$selected_accumulation" =~ ^[0-9]+$ \
    || $((selected_micro * selected_accumulation)) -ne 64 ]]; then
    printf 'invalid quality bridge runtime selection: %s\n' "$runtime_selected" >&2
    exit 11
  fi
  storage_preflight "$LAUNCH_STORAGE_CAPACITY"
  "$PYTHON" scripts/build_generation_quality_bridge_launch_receipt.py \
    "${receipt_args[@]}" \
    --output "$LAUNCH_RECEIPT" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256="$(sha256sum "$LAUNCH_RECEIPT" | awk '{print $1}')"
  printf 'quality bridge launch receipt: %s  %s\n' \
    "$EXPECTED_LAUNCH_RECEIPT_SHA256" "$LAUNCH_RECEIPT"
fi

read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION < <(
  "$PYTHON" - "$LAUNCH_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = report["runtime_selection"]
print(selected["micro_batch_size"], selected["gradient_accumulation_steps"])
PY
)
if (( SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION != 64 )); then
  printf 'launch receipt changes effective batch size\n' >&2
  exit 11
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
identity = report.get("runbook_identity", {})
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
    or report.get("issues") != []
    or identity.get("role") != "generation_exact_runbook_process_identity"
    or identity.get("mode") != "exact_process_identity"
    or identity.get("status") != "active"
    or identity.get("mismatches") != []
    or git.get("revision") != sys.argv[3]
    or git.get("branch") != sys.argv[4]
    or git.get("tracked_dirty") is not False
):
    raise SystemExit(1)
PY
}

start_monitor() {
  if monitor_report_passes; then
    printf 'quality bridge pair monitor already passed\n'
    return
  fi
  local matching_monitor_pids=()
  mapfile -t matching_monitor_pids < <(
    pgrep -f '[m]onitor_generation_pair.py.*stability_full_data_100k_base128_quality_bridge_v1' || true
  )
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      local existing_argv
      local existing_cwd
      existing_argv="$(tr '\0' ' ' <"/proc/$existing_pid/cmdline")"
      existing_cwd="$(readlink -f "/proc/$existing_pid/cwd")"
      if [[ "$existing_argv" != *"monitor_generation_pair.py"* \
        || "$existing_argv" != *"$OUTPUT_ROOT"* \
        || "$existing_argv" != *"$EXPECTED_TARGET_REVISION"* \
        || "$existing_cwd" != "$PROJECT" \
        || ${#matching_monitor_pids[@]} -ne 1 \
        || "${matching_monitor_pids[0]}" != "$existing_pid" ]] \
        || ! cofitok_monitor_report_matches_bound_controller \
          "$PYTHON" "$MONITOR_REPORT" "$MONITOR_NAME"; then
        printf 'quality bridge monitor PID file points to another process\n' >&2
        exit 13
      fi
      printf 'quality bridge pair monitor already active as PID %s\n' "$existing_pid"
      return
    fi
  fi
  if (( ${#matching_monitor_pids[@]} > 0 )); then
    printf 'refusing an unreceipted or duplicate quality bridge pair monitor\n' >&2
    exit 13
  fi
  nohup "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 100000 \
    --training-process-pattern '[s]cripts/train_generation.py.*stability_quality_bridge.*100k' \
    --runbook-process-pattern '[g]eneration_stability_full_data_quality_bridge_100k_execute.sh' \
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
    printf 'quality bridge pair monitor failed during launch\n' >&2
    exit 13
  fi
}

snapshot_monitor() {
  "$PYTHON" scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 100000 \
    --training-process-pattern '[s]cripts/train_generation.py.*stability_quality_bridge.*100k' \
    --runbook-process-pattern '[g]eneration_stability_full_data_quality_bridge_100k_execute.sh' \
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

latest_step() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "latest.json"
print(0 if not path.is_file() else int(json.loads(path.read_text(encoding="utf-8"))["step"]))
PY
}

verify_checkpoint() {
  "$PYTHON" - "$1" "$2" "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.training.checkpointing import verify_training_checkpoint

run = Path(sys.argv[1])
step = int(sys.argv[2])
checkpoint = run / f"checkpoint_step_{step:08d}.pt"
integrity = verify_training_checkpoint(checkpoint)
latest_path = run / "latest.json"
if latest_path.is_file() and int(json.loads(latest_path.read_text(encoding="utf-8"))["step"]) == step:
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if (
        latest.get("git_revision") != sys.argv[3]
        or latest.get("git_branch") != sys.argv[4]
        or latest.get("git_dirty") is not False
        or latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
    ):
        raise SystemExit("quality bridge latest checkpoint binding differs")
if int(integrity.get("step", -1)) != step:
    raise SystemExit("quality bridge checkpoint step differs")
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
    or int(latest.get("step", -1)) < 1
    or int(latest.get("step", -1)) > 100000
):
    raise SystemExit("quality bridge resume checkpoint has another identity")
PY
}

train_to_milestone() {
  local config="$1"
  local run_dir="$2"
  local target="$3"
  local current
  current="$(latest_step "$run_dir")"
  if (( current > target )); then
    verify_checkpoint "$run_dir" "$target"
    return
  fi
  if (( current == target )); then
    verify_checkpoint "$run_dir" "$target"
    return
  fi
  local resume_args=()
  if (( current > 0 )); then
    validate_resume_identity "$run_dir/latest.json"
    verify_checkpoint "$run_dir" "$current"
    resume_args=(--resume auto)
  elif [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'refusing quality bridge fresh launch into non-empty state: %s\n' "$run_dir" >&2
    exit 14
  fi
  local delta=$((target - current))
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
      --micro-batch-size "$SELECTED_MICRO_BATCH" \
      --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
      --stop-after-steps "$delta" \
      "${resume_args[@]}"
  local reached
  reached="$(latest_step "$run_dir")"
  if (( reached != target )); then
    printf 'quality bridge run %s reached %s instead of %s\n' \
      "$run_dir" "$reached" "$target" >&2
    exit 14
  fi
  verify_checkpoint "$run_dir" "$target"
}

paired_milestone_complete() {
  local step="$1"
  local tag="step_$(printf '%08d' "$step")"
  local report="$REPORT_ROOT/milestones/$tag.json"
  [[ -f "$report" ]] || return 1
  verify_checkpoint "$COFITOK_RUN" "$step" || return 1
  verify_checkpoint "$DENSE_RUN" "$step" || return 1
  "$PYTHON" scripts/validate_generation_milestone_report.py \
    --report "$report" \
    --expected-step "$step" \
    --source-profile quality_bridge >/dev/null
}

evaluate_milestone() {
  local method="$1"
  local run_dir="$2"
  local step="$3"
  local prefix_budget="$4"
  local random_orders="$5"
  local checkpoint="$run_dir/checkpoint_step_$(printf '%08d' "$step").pt"
  PROJECT="$PROJECT" PYTHON="$PYTHON" OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    bash artifacts/runbooks/generation_full_milestone_eval.sh \
      "$method" "$run_dir" "$checkpoint" "$step" \
      "$prefix_budget" "$random_orders"
}

build_paired_milestone() {
  local step="$1"
  local tag="step_$(printf '%08d' "$step")"
  local report="$REPORT_ROOT/milestones/$tag.json"
  if [[ -e "$report" ]]; then
    printf 'refusing to overwrite invalid quality bridge milestone report: %s\n' "$report" >&2
    exit 16
  fi
  "$PYTHON" scripts/build_generation_milestone_report.py \
    --cofitok-generation "$COFITOK_RUN/milestones/$tag/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --dense-generation "$DENSE_RUN/milestones/$tag/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --cofitok-checkpoint-eval "$COFITOK_RUN/milestones/$tag/checkpoint_eval/checkpoint_evaluation_report.json" \
    --dense-checkpoint-eval "$DENSE_RUN/milestones/$tag/checkpoint_eval/checkpoint_evaluation_report.json" \
    --milestone-step "$step" \
    --expected-samples 2048 \
    --source-profile quality_bridge \
    --output "$report" >/dev/null
  "$PYTHON" scripts/validate_generation_milestone_report.py \
    --report "$report" \
    --expected-step "$step" \
    --source-profile quality_bridge >/dev/null
}

require_complete() {
  "$PYTHON" scripts/validate_generation_training_completion.py \
    --training-report "$1" \
    --config "$2" \
    --expected-steps 100000 \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-branch "$EXPECTED_TARGET_BRANCH" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" >/dev/null
}

start_monitor
for milestone in 50000 100000; do
  if paired_milestone_complete "$milestone"; then
    printf 'quality bridge paired milestone %s already complete\n' "$milestone"
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
  --expected-steps 100000 \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --expected-dataset imagenet_256 \
  --expected-recipe-stage stability_quality_bridge \
  >"$TRAINING_PAIR_VALIDATION"

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" \
  --config "$COFITOK_CONFIG" \
  --expected-steps 100000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 1000 \
  --required-checkpoint-steps 50000,100000 \
  --integrity-policy required \
  --output "$COFITOK_TRAINING_AUDIT" >/dev/null

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" \
  --config "$DENSE_CONFIG" \
  --expected-steps 100000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 1000 \
  --required-checkpoint-steps 50000,100000 \
  --integrity-policy required \
  --output "$DENSE_TRAINING_AUDIT" >/dev/null

terminal_eval() {
  local run_dir="$1"
  local prefix_budget="$2"
  local random_orders="$3"
  local terminal_root="$run_dir/terminal_100k"
  local samples="$terminal_root/samples_10000_ddim100_cfg15"
  local checkpoint="$run_dir/checkpoint_step_00100000.pt"
  "$PYTHON" scripts/preflight_generation_sampling.py \
    --checkpoint "$checkpoint" \
    --output "$terminal_root/sampling_preflight.json" \
    --batch-size 32 \
    --prefix-budget "$prefix_budget" \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --cfg-batch-mode batched \
    --weights ema \
    --precision bf16 >/dev/null
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$checkpoint" \
    --output-dir "$terminal_root/checkpoint_eval" \
    --num-images 256 \
    --timestep 500 \
    --random-orders "$random_orders" \
    --weights ema \
    --precision bf16 \
    --resume >/dev/null
  "$PYTHON" scripts/generate_samples.py \
    --checkpoint "$checkpoint" \
    --output-dir "$samples" \
    --num-samples 10000 \
    --batch-size 32 \
    --sample-steps 100 \
    --prefix-budgets "$prefix_budget" \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --cfg-batch-mode batched \
    --weights ema \
    --precision bf16 \
    --resume >/dev/null
  "$PYTHON" scripts/evaluate_generation_metrics.py \
    --real-dir "$REAL_DATA" \
    --generated-dir "$samples/prefix_$prefix_budget" \
    --sampling-report "$samples/sampling_report.json" \
    --output-dir "$samples/metrics" \
    --cache-root "$EVAL_CACHE" \
    --min-samples 10000 \
    --resume >/dev/null
  "$PYTHON" scripts/evaluate_generation_class_fidelity.py \
    --generated-dir "$samples/prefix_$prefix_budget" \
    --sampling-report "$samples/sampling_report.json" \
    --output-dir "$samples/class_fidelity" \
    --min-samples 10000 \
    --resume >/dev/null
}

result_args=(
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --launch-receipt "$LAUNCH_RECEIPT"
  --expected-launch-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256"
  --cofitok-training "$COFITOK_RUN/training_report.json"
  --dense-training "$DENSE_RUN/training_report.json"
  --training-pair-validation "$TRAINING_PAIR_VALIDATION"
  --cofitok-training-audit "$COFITOK_TRAINING_AUDIT"
  --dense-training-audit "$DENSE_TRAINING_AUDIT"
  --milestone-50000 "$REPORT_ROOT/milestones/step_00050000.json"
  --milestone-100000 "$REPORT_ROOT/milestones/step_00100000.json"
  --cofitok-sampling-preflight "$COFITOK_RUN/terminal_100k/sampling_preflight.json"
  --dense-sampling-preflight "$DENSE_RUN/terminal_100k/sampling_preflight.json"
  --cofitok-generation "$COFITOK_RUN/terminal_100k/samples_10000_ddim100_cfg15/metrics/generation_metrics_report.json"
  --dense-generation "$DENSE_RUN/terminal_100k/samples_10000_ddim100_cfg15/metrics/generation_metrics_report.json"
  --cofitok-checkpoint-eval "$COFITOK_RUN/terminal_100k/checkpoint_eval/checkpoint_evaluation_report.json"
  --dense-checkpoint-eval "$DENSE_RUN/terminal_100k/checkpoint_eval/checkpoint_evaluation_report.json"
  --class-fidelity-qualification "$CLASS_FIDELITY_QUALIFICATION"
  --cofitok-class-fidelity "$COFITOK_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json"
  --dense-class-fidelity "$DENSE_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
)

if [[ -f "$QUALITY_BRIDGE_RESULT" ]]; then
  if [[ -z "$EXPECTED_QUALITY_BRIDGE_RESULT_SHA256" ]]; then
    printf 'set EXPECTED_QUALITY_BRIDGE_RESULT_SHA256 to reuse the terminal result\n' >&2
    exit 17
  fi
  "$PYTHON" scripts/verify_generation_quality_bridge_result.py \
    "${result_args[@]}" \
    --result "$QUALITY_BRIDGE_RESULT" \
    --expected-result-sha256 "$EXPECTED_QUALITY_BRIDGE_RESULT_SHA256" >/dev/null
else
  if [[ -n "$EXPECTED_QUALITY_BRIDGE_RESULT_SHA256" ]]; then
    printf 'expected quality bridge terminal result is absent\n' >&2
    exit 17
  fi
  terminal_eval "$COFITOK_RUN" 8 4
  terminal_eval "$DENSE_RUN" 1 0
  mkdir -p "$CLASS_FIDELITY_ROOT"
  "$PYTHON" scripts/build_generation_class_fidelity_qualification.py \
    --cofitok-report "$COFITOK_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json" \
    --dense-report "$DENSE_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json" \
    --output "$CLASS_FIDELITY_QUALIFICATION" \
    --stage scaling \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-branch "$EXPECTED_TARGET_BRANCH" \
    --min-top1 0.01 \
    --min-top5 0.05 \
    --min-predicted-class-fraction 0.25 \
    --min-normalized-predicted-entropy 0.50 \
    --max-top1-regression 0.05 \
    --max-top5-regression 0.05 \
    --allow-hold >/dev/null
  "$PYTHON" scripts/build_generation_quality_bridge_result.py \
    "${result_args[@]}" \
    --output "$QUALITY_BRIDGE_RESULT" >/dev/null
  EXPECTED_QUALITY_BRIDGE_RESULT_SHA256="$(sha256sum "$QUALITY_BRIDGE_RESULT" | awk '{print $1}')"
  "$PYTHON" scripts/verify_generation_quality_bridge_result.py \
    "${result_args[@]}" \
    --result "$QUALITY_BRIDGE_RESULT" \
    --expected-result-sha256 "$EXPECTED_QUALITY_BRIDGE_RESULT_SHA256" >/dev/null
  printf 'quality bridge terminal result: %s  %s\n' \
    "$EXPECTED_QUALITY_BRIDGE_RESULT_SHA256" "$QUALITY_BRIDGE_RESULT"
fi

completed=true
write_status completed "quality bridge terminal evidence verified; no larger-training authorization was created"
