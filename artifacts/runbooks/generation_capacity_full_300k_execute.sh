#!/usr/bin/env bash
set -euo pipefail

EXECUTION_PROJECT=${EXECUTION_PROJECT:?set EXECUTION_PROJECT to the exact execution checkout}
TRAINING_PROJECT=${TRAINING_PROJECT:?set TRAINING_PROJECT to the exact training checkout}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}

EXPECTED_EXECUTION_REVISION=${EXPECTED_EXECUTION_REVISION:?set EXPECTED_EXECUTION_REVISION}
EXPECTED_EXECUTION_TREE=${EXPECTED_EXECUTION_TREE:?set EXPECTED_EXECUTION_TREE}
EXPECTED_EXECUTION_BRANCH=${EXPECTED_EXECUTION_BRANCH:?set EXPECTED_EXECUTION_BRANCH}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_READINESS_REVISION=${EXPECTED_READINESS_REVISION:?set EXPECTED_READINESS_REVISION}
EXPECTED_READINESS_TREE=${EXPECTED_READINESS_TREE:?set EXPECTED_READINESS_TREE}
EXPECTED_READINESS_BRANCH=${EXPECTED_READINESS_BRANCH:?set EXPECTED_READINESS_BRANCH}
EXPECTED_READINESS_DEPLOYMENT_SHA256=${EXPECTED_READINESS_DEPLOYMENT_SHA256:?set EXPECTED_READINESS_DEPLOYMENT_SHA256}
EXPECTED_READINESS_CLARIFICATION_SHA256=${EXPECTED_READINESS_CLARIFICATION_SHA256:?set EXPECTED_READINESS_CLARIFICATION_SHA256}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}
EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256=${EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256:?set EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256}

REPORT_ROOT="$FULL_OUTPUT_ROOT/reports"
READINESS="$REPORT_ROOT/capacity_full_300k_readiness.json"
READINESS_SUPERVISOR_STATUS="$SOURCE_OUTPUT_ROOT/reports/capacity_full_300k_readiness_supervisor_status.json"
READINESS_DEPLOYMENT_RECEIPT="$SOURCE_OUTPUT_ROOT/reports/capacity_full_300k_readiness_supervisor_deployment_receipt.json"
READINESS_DEPLOYMENT_CLARIFICATION="$SOURCE_OUTPUT_ROOT/reports/capacity_full_300k_readiness_supervisor_deployment_validation_clarification.json"
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:-/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json}
COFITOK_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
DENSE_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
COFITOK_RUN="$FULL_OUTPUT_ROOT/cofitok"
DENSE_RUN="$FULL_OUTPUT_ROOT/dense_identity"
LAUNCH_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_launch.json"
CURRENT_STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_current.json"
TRAINING_LAUNCH_RECEIPT="$REPORT_ROOT/capacity_full_300k_training_launch_receipt.json"
REFERENCE_COFITOK="$SOURCE_OUTPUT_ROOT/base256_cofitok/checkpoint_step_00100000.pt"
REFERENCE_DENSE="$SOURCE_OUTPUT_ROOT/base256_dense_identity/checkpoint_step_00100000.pt"
MONITOR_REPORT="$FULL_OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$FULL_OUTPUT_ROOT/pair_monitor.log"
MONITOR_PID_FILE="$FULL_OUTPUT_ROOT/pair_monitor.pid"
MONITOR_NAME=generation_capacity_full_300k_v1

[[ -x "$PYTHON" ]]
[[ -d "$EXECUTION_PROJECT/.git" || -f "$EXECUTION_PROJECT/.git" ]]
[[ -d "$TRAINING_PROJECT/.git" || -f "$TRAINING_PROJECT/.git" ]]
[[ "$(git -C "$EXECUTION_PROJECT" rev-parse HEAD)" == "$EXPECTED_EXECUTION_REVISION" ]]
[[ "$(git -C "$EXECUTION_PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_EXECUTION_TREE" ]]
[[ "$(git -C "$EXECUTION_PROJECT" branch --show-current)" == "$EXPECTED_EXECUTION_BRANCH" ]]
[[ -z "$(git -C "$EXECUTION_PROJECT" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "$TRAINING_PROJECT" rev-parse HEAD)" == "$EXPECTED_TRAINING_REVISION" ]]
[[ "$(git -C "$TRAINING_PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_TRAINING_TREE" ]]
[[ "$(git -C "$TRAINING_PROJECT" branch --show-current)" == "$EXPECTED_TRAINING_BRANCH" ]]
[[ -z "$(git -C "$TRAINING_PROJECT" status --porcelain --untracked-files=no)" ]]
[[ -f "$READINESS" ]]
[[ -f "$READINESS_SUPERVISOR_STATUS" ]]
[[ -f "$READINESS_DEPLOYMENT_RECEIPT" ]]
[[ -f "$READINESS_DEPLOYMENT_CLARIFICATION" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ -f "$COFITOK_CONFIG" ]]
[[ -f "$DENSE_CONFIG" ]]
[[ -f "$LAUNCH_STORAGE_CAPACITY" ]]
[[ -f "$TRAINING_LAUNCH_RECEIPT" ]]
[[ -f "$REFERENCE_COFITOK" ]]
[[ -f "$REFERENCE_DENSE" ]]

cd "$EXECUTION_PROJECT"
export PYTHONPATH="$EXECUTION_PROJECT/src:$TRAINING_PROJECT/src"

receipt_args=(
  --execution-project "$EXECUTION_PROJECT"
  --training-project "$TRAINING_PROJECT"
  --formal-project "$FORMAL_PROJECT"
  --readiness "$READINESS"
  --readiness-supervisor-status "$READINESS_SUPERVISOR_STATUS"
  --readiness-deployment-receipt "$READINESS_DEPLOYMENT_RECEIPT"
  --expected-readiness-deployment-sha256 "$EXPECTED_READINESS_DEPLOYMENT_SHA256"
  --readiness-deployment-clarification "$READINESS_DEPLOYMENT_CLARIFICATION"
  --expected-readiness-clarification-sha256 "$EXPECTED_READINESS_CLARIFICATION_SHA256"
  --standing-authorization "$STANDING_AUTHORIZATION"
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256"
  --cofitok-config "$COFITOK_CONFIG"
  --dense-config "$DENSE_CONFIG"
  --launch-storage-capacity "$LAUNCH_STORAGE_CAPACITY"
  --cofitok-run-dir "$COFITOK_RUN"
  --dense-run-dir "$DENSE_RUN"
  --expected-execution-revision "$EXPECTED_EXECUTION_REVISION"
  --expected-execution-tree "$EXPECTED_EXECUTION_TREE"
  --expected-execution-branch "$EXPECTED_EXECUTION_BRANCH"
  --expected-training-revision "$EXPECTED_TRAINING_REVISION"
  --expected-training-tree "$EXPECTED_TRAINING_TREE"
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH"
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION"
  --expected-readiness-tree "$EXPECTED_READINESS_TREE"
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH"
)

"$PYTHON" scripts/verify_generation_capacity_full_300k_training_launch_receipt.py \
  "${receipt_args[@]}" \
  --receipt "$TRAINING_LAUNCH_RECEIPT" \
  --expected-receipt-sha256 "$EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256" >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing capacity-full 300K launch while the GPU is busy\n' >&2
  exit 9
fi

PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
  "$TRAINING_PROJECT/scripts/check_generation_storage_capacity.py" \
  --path "$CHECKPOINT_ROOT" \
  --output "$CURRENT_STORAGE_CAPACITY" \
  --stage full_training \
  --reference-checkpoint "$REFERENCE_COFITOK" \
  --reference-checkpoint "$REFERENCE_DENSE" \
  --checkpoint-count 16 \
  --checkpoint-size-multiplier 1.0 \
  --sample-count 116640 \
  --estimated-sample-kib 256 \
  --additional-gib 16 \
  --safety-margin-gib 64 >/dev/null

read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION < <(
  "$PYTHON" - "$READINESS" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = report["runtime_selection"]
print(runtime["micro_batch_size"], runtime["gradient_accumulation_steps"])
PY
)
if [[ ! "$SELECTED_MICRO_BATCH" =~ ^[0-9]+$ \
  || ! "$SELECTED_ACCUMULATION" =~ ^[0-9]+$ \
  || $((SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION)) -ne 64 ]]; then
  printf 'invalid capacity-full selected runtime: %s x %s\n' \
    "$SELECTED_MICRO_BATCH" "$SELECTED_ACCUMULATION" >&2
  exit 10
fi

monitor_report_passes() {
  "$PYTHON" - "$MONITOR_REPORT" "$MONITOR_NAME" \
    "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" <<'PY'
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

monitor_args=(
  --output-root "$FULL_OUTPUT_ROOT"
  --output "$MONITOR_REPORT"
  --monitor-name "$MONITOR_NAME"
  --cofitok-run "$(basename "$COFITOK_RUN")"
  --dense-run "$(basename "$DENSE_RUN")"
  --expected-steps 300000
  --training-process-pattern '[s]cripts/train_generation.py.*300k'
  --runbook-process-pattern '[g]eneration_capacity_full_300k_execute.sh'
  --checkpoint-interval 5000
  --checkpoint-grace-steps 250
  --checkpoint-integrity-policy required
  --expected-checkpoint-revision "$EXPECTED_TRAINING_REVISION"
  --poll-seconds 300
  --stall-seconds 1800
  --idle-failure-grace-seconds 600
)

start_monitor() {
  if monitor_report_passes; then
    return
  fi
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] \
      && kill -0 "$existing_pid" 2>/dev/null; then
      return
    fi
  fi
  (
    cd "$TRAINING_PROJECT"
    nohup env PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
      scripts/monitor_generation_pair.py "${monitor_args[@]}" \
      >"$MONITOR_LOG" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$MONITOR_PID_FILE.tmp.$$"
    mv "$MONITOR_PID_FILE.tmp.$$" "$MONITOR_PID_FILE"
  )
  sleep 1
  local monitor_pid
  monitor_pid="$(cat "$MONITOR_PID_FILE")"
  if ! kill -0 "$monitor_pid" 2>/dev/null && ! monitor_report_passes; then
    printf 'capacity-full monitor failed during launch\n' >&2
    exit 11
  fi
}

snapshot_monitor() {
  (
    cd "$TRAINING_PROJECT"
    PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
      scripts/monitor_generation_pair.py "${monitor_args[@]}" --once
  )
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
  "$PYTHON" - "$1" "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" \
    "$EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256" <<'PY'
import json
import sys
from pathlib import Path
latest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    latest.get("git_revision") != sys.argv[2]
    or latest.get("git_branch") != sys.argv[3]
    or latest.get("git_dirty") is not False
    or latest.get("authorization_stage") != "capacity_full_experimental"
    or latest.get("authorization_decision")
       != "authorize_fresh_matched_300k_training"
    or latest.get("authorization_gate_sha256") != sys.argv[4]
):
    raise SystemExit("capacity-full resume checkpoint has another identity")
PY
}

train_to_milestone() {
  local config="$1"
  local run_dir="$2"
  local target="$3"
  local current
  current="$(latest_step "$run_dir")"
  if (( current > target )); then
    [[ -f "$run_dir/checkpoint_step_$(printf '%08d' "$target").pt" ]]
    [[ -f "$run_dir/checkpoint_step_$(printf '%08d' "$target").pt.integrity.json" ]]
    return
  fi
  if (( current == target )); then
    return
  fi
  if [[ -d "$run_dir" ]] \
    && find "$run_dir" -mindepth 1 -print -quit | grep -q . \
    && [[ ! -f "$run_dir/latest.json" ]]; then
    printf 'refusing capacity-full state without latest pointer: %s\n' "$run_dir" >&2
    exit 10
  fi
  local delta=$((target - current))
  local resume_args=()
  if (( current > 0 )); then
    validate_resume_identity "$run_dir/latest.json"
    resume_args=(--resume auto)
  fi
  (
    cd "$TRAINING_PROJECT"
    PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
      scripts/run_generation_training_watchdog.py \
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
        --authorization-gate "$TRAINING_LAUNCH_RECEIPT" \
        --micro-batch-size "$SELECTED_MICRO_BATCH" \
        --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
        --stop-after-steps "$delta" \
        "${resume_args[@]}"
  )
  local reached
  reached="$(latest_step "$run_dir")"
  if (( reached != target )); then
    printf 'capacity-full run %s reached step %s instead of %s\n' \
      "$run_dir" "$reached" "$target" >&2
    exit 12
  fi
}

paired_milestone_complete() {
  local step="$1"
  local tag="step_$(printf '%08d' "$step")"
  local report="$REPORT_ROOT/milestones/$tag.json"
  [[ -f "$report" ]] || return 1
  [[ -f "$COFITOK_RUN/checkpoint_${tag}.pt" ]] || return 1
  [[ -f "$COFITOK_RUN/checkpoint_${tag}.pt.integrity.json" ]] || return 1
  [[ -f "$DENSE_RUN/checkpoint_${tag}.pt" ]] || return 1
  [[ -f "$DENSE_RUN/checkpoint_${tag}.pt.integrity.json" ]] || return 1
  PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
    "$TRAINING_PROJECT/scripts/validate_generation_milestone_report.py" \
    --report "$report" --expected-step "$step" >/dev/null
}

evaluate_milestone() {
  local method="$1"
  local run_dir="$2"
  local step="$3"
  local prefix_budget="$4"
  local random_orders="$5"
  local checkpoint="$run_dir/checkpoint_step_$(printf '%08d' "$step").pt"
  PROJECT="$TRAINING_PROJECT" PYTHON="$PYTHON" OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    bash "$TRAINING_PROJECT/artifacts/runbooks/generation_full_milestone_eval.sh" \
      "$method" "$run_dir" "$checkpoint" "$step" \
      "$prefix_budget" "$random_orders"
}

build_paired_milestone() {
  local step="$1"
  local tag="step_$(printf '%08d' "$step")"
  local cofitok_milestone="$COFITOK_RUN/milestones/$tag"
  local dense_milestone="$DENSE_RUN/milestones/$tag"
  mkdir -p "$REPORT_ROOT/milestones"
  PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
    "$TRAINING_PROJECT/scripts/build_generation_milestone_report.py" \
    --cofitok-generation "$cofitok_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --dense-generation "$dense_milestone/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --cofitok-checkpoint-eval "$cofitok_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --dense-checkpoint-eval "$dense_milestone/checkpoint_eval/checkpoint_evaluation_report.json" \
    --milestone-step "$step" \
    --expected-samples 2048 \
    --output "$REPORT_ROOT/milestones/$tag.json"
  PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
    "$TRAINING_PROJECT/scripts/validate_generation_milestone_report.py" \
    --report "$REPORT_ROOT/milestones/$tag.json" --expected-step "$step" >/dev/null
}

require_complete() {
  PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
    "$TRAINING_PROJECT/scripts/validate_generation_training_completion.py" \
    --training-report "$1" \
    --config "$2" \
    --expected-steps 300000 \
    --expected-revision "$EXPECTED_TRAINING_REVISION" \
    --expected-branch "$EXPECTED_TRAINING_BRANCH" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" >/dev/null
}

start_monitor
for milestone in 50000 100000 200000 300000; do
  if paired_milestone_complete "$milestone"; then
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

PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
  "$TRAINING_PROJECT/scripts/validate_generation_training_pair.py" \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 \
  --expected-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-dataset imagenet_256 \
  --expected-recipe-stage stability_full \
  --authorization-gate "$TRAINING_LAUNCH_RECEIPT" \
  >"$REPORT_ROOT/training_pair_validation.json"

PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
  "$TRAINING_PROJECT/scripts/audit_generation_training_progress.py" \
  --run-dir "$COFITOK_RUN" \
  --config "$COFITOK_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/cofitok_training_audit.json"

PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
  "$TRAINING_PROJECT/scripts/audit_generation_training_progress.py" \
  --run-dir "$DENSE_RUN" \
  --config "$DENSE_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/dense_identity_training_audit.json"

printf 'capacity-full experimental matched 300K training completed; no quality promotion or release was issued\n'
