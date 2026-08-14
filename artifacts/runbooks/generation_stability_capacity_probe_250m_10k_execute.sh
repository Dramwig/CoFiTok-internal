#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set exact standing authorization record}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set standing authorization SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set clean capacity-probe execution revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set clean capacity-probe execution branch}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set immutable capacity-probe preparation SHA256}
CAPACITY_PROBE_EXECUTION_ALLOWED=${CAPACITY_PROBE_EXECUTION_ALLOWED:-false}
EXPECTED_EXECUTION_AUTHORIZATION_SHA256=${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:-}
EXPECTED_LAUNCH_RECEIPT_SHA256=${EXPECTED_LAUNCH_RECEIPT_SHA256:-}
EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:-}
PREPARATION_PROJECT=${PREPARATION_PROJECT:?set exact capacity-probe preparation checkout}
EXPECTED_PREPARATION_REVISION=${EXPECTED_PREPARATION_REVISION:?set exact preparation revision}
EXPECTED_PREPARATION_BRANCH=${EXPECTED_PREPARATION_BRANCH:?set exact preparation branch}

QUALITY_BRIDGE_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1"
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
PREPARATION="$REPORT_ROOT/preparation.json"
EXECUTION_AUTHORIZATION="$REPORT_ROOT/execution_authorization.json"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
STORAGE_LAUNCH="$REPORT_ROOT/storage_capacity_launch.json"
STORAGE_CURRENT="$REPORT_ROOT/storage_capacity_current.json"
RUNTIME_ROOT="$OUTPUT_ROOT/runtime_preflight/training"
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
LAUNCH_RECEIPT="$REPORT_ROOT/launch_receipt.json"
EXECUTION_STATUS="$REPORT_ROOT/execution_status.json"
RESULT="$REPORT_ROOT/capacity_probe_result.json"
LOCK="$OUTPUT_ROOT/capacity_probe_execution.lock"
MONITOR_REPORT="$OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/pair_monitor.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/pair_monitor.pid"
MONITOR_NAME=generation_stability_capacity_probe_250m_10k
COFITOK_CONFIG=configs/generation/imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_capacity_probe_rollout_x0_u2_ema_teacher_dense_100k.json
COFITOK_RUN="$OUTPUT_ROOT/base256_cofitok"
DENSE_RUN="$OUTPUT_ROOT/base256_dense_identity"
COFITOK_VALIDATION="$REPORT_ROOT/training/base256_cofitok.json"
DENSE_VALIDATION="$REPORT_ROOT/training/base256_dense_identity.json"
BASE128_REFERENCE_ROOT="$QUALITY_BRIDGE_ROOT/capacity_probe_references/base128_step_00010000"
BASE128_COFITOK_CHECKPOINT="$BASE128_REFERENCE_ROOT/cofitok/checkpoint_step_00010000.pt"
BASE128_DENSE_CHECKPOINT="$BASE128_REFERENCE_ROOT/dense_identity/checkpoint_step_00010000.pt"
REFERENCE_CHECKPOINT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
REAL_DATA=${REAL_DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
EVAL_CACHE="$CHECKPOINT_ROOT/eval_cache/torch_fidelity"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$CAPACITY_PROBE_EXECUTION_ALLOWED" == true ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -f "$PREPARATION" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ -f "$BASE128_COFITOK_CHECKPOINT" ]]
[[ -f "$BASE128_DENSE_CHECKPOINT" ]]
[[ -d "$REAL_DATA" ]]
[[ "$(sha256sum "$PREPARATION" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
mkdir -p "$REPORT_ROOT/training"
exec 7>"$LOCK"
if ! flock -n 7; then
  printf 'refusing concurrent capacity probe execution controller\n' >&2
  exit 75
fi

write_status() {
  local status="$1"
  local detail="$2"
  local exit_code="${3:-}"
  "$PYTHON" - "$EXECUTION_STATUS" "$status" "$detail" "$exit_code" \
    "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" "$$" "$RESULT" <<'PY'
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "role": "stability_full_data_capacity_probe_execution",
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
    "capacity_probe_only": True,
    "intentional_training_stop_step": 10000,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
}
result_path = Path(sys.argv[8])
if sys.argv[2] == "completed":
    if not result_path.is_file():
        raise SystemExit("completed capacity probe lacks a result")
    digest = hashlib.sha256()
    with result_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    payload["result"] = {
        "path": result_path.resolve().as_posix(),
        "bytes": result_path.stat().st_size,
        "sha256": digest.hexdigest(),
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
CURRENT_STAGE=initialization
on_exit() {
  local code=$?
  if [[ "$completed" != true && $code -ne 0 ]]; then
    write_status failed "capacity probe stage failed: $CURRENT_STAGE" "$code" || true
  fi
}
trap on_exit EXIT

set_stage() {
  CURRENT_STAGE="$1"
  write_status running "$2"
}

set_stage authorization "validating exact bounded capacity-probe authorization"

[[ "$(git -C "$PREPARATION_PROJECT" rev-parse HEAD)" == "$EXPECTED_PREPARATION_REVISION" ]]
[[ "$(git -C "$PREPARATION_PROJECT" branch --show-current)" == "$EXPECTED_PREPARATION_BRANCH" ]]
[[ -z "$(git -C "$PREPARATION_PROJECT" status --porcelain --untracked-files=no)" ]]
DECISION="$QUALITY_BRIDGE_ROOT/reports/followup_experiment_decision.json"
DECISION_SHA256="$(sha256sum "$DECISION" | awk '{print $1}')"
PYTHONPATH="$PREPARATION_PROJECT:$PREPARATION_PROJECT/src" \
  "$PYTHON" "$PREPARATION_PROJECT/scripts/verify_generation_capacity_probe_preparation.py" \
    --followup-decision "$DECISION" \
    --expected-followup-decision-sha256 "$DECISION_SHA256" \
    --base-cofitok-config "$PREPARATION_PROJECT/configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json" \
    --base-dense-config "$PREPARATION_PROJECT/configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json" \
    --capacity-cofitok-config "$PREPARATION_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json" \
    --capacity-dense-config "$PREPARATION_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rollout_x0_u2_ema_teacher_dense_100k.json" \
    --cofitok-reference-receipt "$QUALITY_BRIDGE_ROOT/reports/capacity_probe_references/cofitok_step_00010000.json" \
    --dense-reference-receipt "$QUALITY_BRIDGE_ROOT/reports/capacity_probe_references/dense_identity_step_00010000.json" \
    --expected-preparation-revision "$EXPECTED_PREPARATION_REVISION" \
    --expected-preparation-branch "$EXPECTED_PREPARATION_BRANCH" \
    --output-root "$OUTPUT_ROOT" \
    --preparation "$PREPARATION" \
    --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" >/dev/null

authorization_args=(
  --project-root "$PROJECT"
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --standing-authorization "$STANDING_AUTHORIZATION"
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
  --output-root "$OUTPUT_ROOT"
)
if [[ -f "$EXECUTION_AUTHORIZATION" ]]; then
  if [[ -z "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]; then
    printf 'set EXPECTED_EXECUTION_AUTHORIZATION_SHA256 to resume capacity probe\n' >&2
    exit 10
  fi
  "$PYTHON" scripts/verify_generation_capacity_probe_execution_authorization.py \
    "${authorization_args[@]}" \
    --authorization "$EXECUTION_AUTHORIZATION" \
    --expected-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" >/dev/null
else
  if [[ -n "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]; then
    printf 'expected capacity-probe execution authorization is absent\n' >&2
    exit 10
  fi
  "$PYTHON" scripts/build_generation_capacity_probe_execution_authorization.py \
    "${authorization_args[@]}" --output "$EXECUTION_AUTHORIZATION" >/dev/null
  EXPECTED_EXECUTION_AUTHORIZATION_SHA256="$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')"
  printf 'capacity-probe execution authorization: %s  %s\n' \
    "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" "$EXECUTION_AUTHORIZATION"
fi

set_stage config_validation "validating exact matched capacity-probe configs"
"$PYTHON" scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_capacity_probe \
  --output "$CONFIG_VALIDATION" >/dev/null

set_stage launch_guard "checking duplicate processes and GPU idleness"
if pgrep -af '[t]rain_generation.py.*stability_capacity_probe.*100k' >/dev/null; then
  printf 'refusing duplicate capacity-probe trainer\n' >&2
  exit 12
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing capacity-probe launch while GPU is busy\n' >&2
  exit 9
fi

storage_preflight() {
  local output="$1"
  "$PYTHON" scripts/check_generation_storage_capacity.py \
    --path "$CHECKPOINT_ROOT" \
    --output "$output" \
    --stage stability_full_data_capacity_probe_250m_10k_execution \
    --reference-checkpoint "$REFERENCE_CHECKPOINT" \
    --checkpoint-count 4 \
    --checkpoint-size-multiplier 4.0 \
    --sample-count 8192 \
    --estimated-sample-kib 256 \
    --additional-gib 8 \
    --safety-margin-gib 64 >/dev/null
}

receipt_args=(
  --project-root "$PROJECT"
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --execution-authorization "$EXECUTION_AUTHORIZATION"
  --expected-execution-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256"
  --cofitok-config "$COFITOK_CONFIG"
  --dense-config "$DENSE_CONFIG"
  --config-validation "$CONFIG_VALIDATION"
  --storage-capacity "$STORAGE_LAUNCH"
  --runtime-selection "$RUNTIME_SELECTION"
  --training-run-dir "$COFITOK_RUN"
  --training-run-dir "$DENSE_RUN"
  --benchmark-root "$RUNTIME_ROOT"
  --output-root "$OUTPUT_ROOT"
  --storage-path "$CHECKPOINT_ROOT"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
  --require-current-runtime-environment
)

if [[ -f "$LAUNCH_RECEIPT" ]]; then
  set_stage launch_receipt_replay "replaying the immutable capacity-probe launch receipt"
  if [[ -z "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'set EXPECTED_LAUNCH_RECEIPT_SHA256 to resume capacity probe\n' >&2
    exit 10
  fi
  "$PYTHON" scripts/verify_generation_capacity_probe_launch_receipt.py \
    "${receipt_args[@]}" --receipt "$LAUNCH_RECEIPT" \
    --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
  storage_preflight "$STORAGE_CURRENT"
else
  set_stage runtime_selection "selecting a matched capacity-probe runtime"
  if [[ -n "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'expected capacity-probe launch receipt is absent\n' >&2
    exit 10
  fi
  for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
    if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      printf 'refusing unreceipted capacity-probe training state: %s\n' "$run_dir" >&2
      exit 10
    fi
  done
  runtime_selected="$("$PYTHON" scripts/select_generation_training_runtime.py \
    --project-root "$PROJECT" \
    --cofitok-config "$COFITOK_CONFIG" \
    --dense-config "$DENSE_CONFIG" \
    --output-root "$RUNTIME_ROOT" \
    --output "$RUNTIME_SELECTION" \
    --training-run-dir "$COFITOK_RUN" \
    --training-run-dir "$DENSE_RUN" \
    --candidates 1x64,2x32,4x16,8x8,16x4 \
    --baseline-candidate 1x64 \
    --effective-batch-size 64 \
    --benchmark-steps 8 \
    --warmup-steps 2 \
    --max-memory-fraction 0.90)"
  read -r selected_micro selected_accumulation <<<"$runtime_selected"
  if [[ ! "$selected_micro" =~ ^[0-9]+$ \
    || ! "$selected_accumulation" =~ ^[0-9]+$ \
    || $((selected_micro * selected_accumulation)) -ne 64 ]]; then
    printf 'invalid capacity-probe runtime selection: %s\n' "$runtime_selected" >&2
    exit 11
  fi
  set_stage storage_preflight "checking launch storage headroom"
  storage_preflight "$STORAGE_LAUNCH"
  set_stage launch_receipt_build "building the immutable capacity-probe launch receipt"
  "$PYTHON" scripts/build_generation_capacity_probe_launch_receipt.py \
    "${receipt_args[@]}" --output "$LAUNCH_RECEIPT" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256="$(sha256sum "$LAUNCH_RECEIPT" | awk '{print $1}')"
  printf 'capacity-probe launch receipt: %s  %s\n' \
    "$EXPECTED_LAUNCH_RECEIPT_SHA256" "$LAUNCH_RECEIPT"
fi

read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION < <(
  "$PYTHON" - "$LAUNCH_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path
selected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["runtime_selection"]
print(selected["micro_batch_size"], selected["gradient_accumulation_steps"])
PY
)
(( SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION == 64 ))

monitor_passes() {
  "$PYTHON" - "$MONITOR_REPORT" "$MONITOR_NAME" \
    "$EXPECTED_TARGET_REVISION" "$EXPECTED_TARGET_BRANCH" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("monitor") != sys.argv[2]
    or report.get("status") != "pass"
    or report.get("stage") != "complete"
    or report.get("issues") != []
    or report.get("git") != {
        "revision": sys.argv[3],
        "branch": sys.argv[4],
        "tracked_dirty": False,
    }
    or report.get("scope", {}).get("intentional_partial_stop_step") != 10000
    or report.get("scope", {}).get("configured_100k_completion_allowed") is not False
):
    raise SystemExit(1)
PY
}

monitor_args=(
  --output-root "$OUTPUT_ROOT"
  --output "$MONITOR_REPORT"
  --monitor-name "$MONITOR_NAME"
  --cofitok-run "$(basename "$COFITOK_RUN")"
  --dense-run "$(basename "$DENSE_RUN")"
  --cofitok-validation "$COFITOK_VALIDATION"
  --dense-validation "$DENSE_VALIDATION"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
  --expected-stop-step 10000
  --expected-configured-steps 100000
  --checkpoint-interval 5000
  --checkpoint-grace-steps 250
  --poll-seconds 300
  --stall-seconds 1800
  --idle-failure-grace-seconds 600
)

start_monitor() {
  if monitor_passes; then
    return
  fi
  local existing=""
  if [[ -f "$MONITOR_PID_FILE" ]]; then
    existing="$(cat "$MONITOR_PID_FILE")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
      local argv cwd
      argv="$(tr '\0' ' ' <"/proc/$existing/cmdline")"
      cwd="$(readlink -f "/proc/$existing/cwd")"
      if [[ "$argv" != *"monitor_generation_capacity_probe.py"* \
        || "$argv" != *"$OUTPUT_ROOT"* \
        || "$cwd" != "$PROJECT" ]]; then
        printf 'capacity-probe monitor PID file points to another process\n' >&2
        exit 13
      fi
      return
    fi
  fi
  if pgrep -af '[m]onitor_generation_capacity_probe.py.*stability_full_data_100k_capacity_probe_250m_10k_v1' >/dev/null; then
    printf 'refusing duplicate unreceipted capacity-probe monitor\n' >&2
    exit 13
  fi
  nohup "$PYTHON" scripts/monitor_generation_capacity_probe.py \
    "${monitor_args[@]}" >"$MONITOR_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"${MONITOR_PID_FILE}.tmp.$$"
  mv "${MONITOR_PID_FILE}.tmp.$$" "$MONITOR_PID_FILE"
  sleep 1
  kill -0 "$pid" 2>/dev/null || {
    printf 'capacity-probe monitor failed during launch\n' >&2
    exit 13
  }
}

snapshot_monitor() {
  "$PYTHON" scripts/monitor_generation_capacity_probe.py \
    "${monitor_args[@]}" --once
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
latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
if (
    int(integrity.get("step", -1)) != step
    or integrity.get("git_revision") != sys.argv[3]
    or integrity.get("git_branch") != sys.argv[4]
    or integrity.get("git_dirty") is not False
    or latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
    or int(latest.get("step", -1)) != step
):
    raise SystemExit("capacity-probe checkpoint binding differs")
PY
}

train_to_stop() {
  local method="$1"
  local config="$2"
  local run_dir="$3"
  local validation="$4"
  local parameters="$5"
  local current
  current="$(latest_step "$run_dir")"
  if (( current > 10000 )); then
    printf 'capacity-probe run exceeded step 10000: %s\n' "$run_dir" >&2
    exit 14
  fi
  if (( current < 10000 )); then
    local resume_args=()
    if (( current > 0 )); then
      verify_checkpoint "$run_dir" "$current"
      resume_args=(--resume auto)
    elif [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      printf 'refusing fresh capacity-probe launch into non-empty state: %s\n' "$run_dir" >&2
      exit 14
    fi
    local delta=$((10000 - current))
    "$PYTHON" scripts/run_generation_training_watchdog.py \
      --monitor-report "$MONITOR_REPORT" \
      --monitor-pid-file "$MONITOR_PID_FILE" \
      --expected-monitor-name "$MONITOR_NAME" \
      --status-output "$run_dir/training_watchdog_step_00010000.json" \
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
  fi
  [[ "$(latest_step "$run_dir")" == 10000 ]]
  verify_checkpoint "$run_dir" 10000
  set_stage "${method}_training_validation" \
    "validating the source-bound ${method} intentional step-10K stop"
  "$PYTHON" scripts/validate_generation_capacity_probe_training.py \
    --training-report "$run_dir/training_report.json" \
    --config "$config" \
    --expected-stop-step 10000 \
    --expected-configured-steps 100000 \
    --expected-revision "$EXPECTED_TARGET_REVISION" \
    --expected-branch "$EXPECTED_TARGET_BRANCH" \
    --expected-parameter-count "$parameters" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
    --expected-dataset imagenet_256 \
    --output "$validation" >/dev/null
  snapshot_monitor
}

start_monitor
set_stage cofitok_training "training base256 CoFiTok to the intentional step-10K stop"
train_to_stop cofitok "$COFITOK_CONFIG" "$COFITOK_RUN" "$COFITOK_VALIDATION" 250153763
set_stage dense_identity_training "training base256 dense identity to the intentional step-10K stop"
train_to_stop dense_identity "$DENSE_CONFIG" "$DENSE_RUN" "$DENSE_VALIDATION" 250135043
set_stage partial_training_replay "replaying both intentional partial-training reports"
snapshot_monitor
monitor_passes

evaluate_arm() {
  local arm="$1"
  local checkpoint="$2"
  local prefix="$3"
  local random_orders="$4"
  local arm_root="$OUTPUT_ROOT/evaluation/$arm"
  PROJECT="$PROJECT" PYTHON="$PYTHON" OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    bash artifacts/runbooks/generation_full_milestone_eval.sh \
      "$arm" "$arm_root" "$checkpoint" 10000 "$prefix" "$random_orders"
}

if [[ ! -f "$RESULT" ]]; then
  set_stage base128_cofitok_evaluation "evaluating the preserved base128 CoFiTok arm"
  evaluate_arm base128_cofitok "$BASE128_COFITOK_CHECKPOINT" 8 4
  set_stage base128_dense_identity_evaluation "evaluating the preserved base128 dense arm"
  evaluate_arm base128_dense_identity "$BASE128_DENSE_CHECKPOINT" 1 0
  set_stage base256_cofitok_evaluation "evaluating the new base256 CoFiTok arm"
  evaluate_arm base256_cofitok "$COFITOK_RUN/checkpoint_step_00010000.pt" 8 4
  set_stage base256_dense_identity_evaluation "evaluating the new base256 dense arm"
  evaluate_arm base256_dense_identity "$DENSE_RUN/checkpoint_step_00010000.pt" 1 0
fi

result_args=(
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --launch-receipt "$LAUNCH_RECEIPT"
  --expected-launch-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256"
  --base256-cofitok-training-validation "$COFITOK_VALIDATION"
  --base256-dense-training-validation "$DENSE_VALIDATION"
  --expected-revision "$EXPECTED_TARGET_REVISION"
  --expected-branch "$EXPECTED_TARGET_BRANCH"
)
for arm in base128_cofitok base128_dense_identity base256_cofitok base256_dense_identity; do
  flag="${arm//_/-}"
  arm_root="$OUTPUT_ROOT/evaluation/$arm/milestones/step_00010000"
  result_args+=(
    "--${flag}-sampling-preflight" "$arm_root/sampling_preflight.json"
    "--${flag}-generation" "$arm_root/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json"
    "--${flag}-checkpoint-eval" "$arm_root/checkpoint_eval/checkpoint_evaluation_report.json"
  )
done

if [[ -f "$RESULT" ]]; then
  set_stage result_replay "replaying the immutable four-arm capacity-probe result"
  if [[ -z "$EXPECTED_RESULT_SHA256" ]]; then
    printf 'set EXPECTED_RESULT_SHA256 to reuse capacity-probe result\n' >&2
    exit 17
  fi
  "$PYTHON" scripts/verify_generation_capacity_probe_result.py \
    "${result_args[@]}" --result "$RESULT" \
    --expected-result-sha256 "$EXPECTED_RESULT_SHA256" >/dev/null
else
  set_stage result_build "building and replaying the four-arm capacity-probe result"
  if [[ -n "$EXPECTED_RESULT_SHA256" ]]; then
    printf 'expected capacity-probe result is absent\n' >&2
    exit 17
  fi
  "$PYTHON" scripts/build_generation_capacity_probe_result.py \
    "${result_args[@]}" --output "$RESULT"
  EXPECTED_RESULT_SHA256="$(sha256sum "$RESULT" | awk '{print $1}')"
  "$PYTHON" scripts/verify_generation_capacity_probe_result.py \
    "${result_args[@]}" --result "$RESULT" \
    --expected-result-sha256 "$EXPECTED_RESULT_SHA256" >/dev/null
  printf 'capacity-probe result: %s  %s\n' "$EXPECTED_RESULT_SHA256" "$RESULT"
fi

completed=true
CURRENT_STAGE=completed
write_status completed "bounded capacity-probe evidence verified; no 100K/300K authorization created"
