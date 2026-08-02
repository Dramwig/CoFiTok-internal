#!/usr/bin/env bash
set -euo pipefail

CONTROL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINING_PROJECT=${TRAINING_PROJECT:-/tmp/cofitok-stability-50k-preflight-2c2c1f5}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_CONTROL_REVISION=${EXPECTED_CONTROL_REVISION:?set the clean recovery-control revision}
EXPECTED_CONTROL_BRANCH=${EXPECTED_CONTROL_BRANCH:?set the clean recovery-control branch}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:-2c2c1f5166b73d4f28df93b276901671ac1a7836}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:-scale/generation-stability-50k-preflight}
EXPECTED_RUNTIME_SELECTION_SHA256=${EXPECTED_RUNTIME_SELECTION_SHA256:-f7befbcdbc6644fc066a71b85cb1df5cae5b922c526f09fe0000ae252b324663}
EXPECTED_COFITOK_CONFIG_SHA256=${EXPECTED_COFITOK_CONFIG_SHA256:-4b9bf0d89f639983df6f5c046deb36ad6bd65dac8ec8f9b587721dbcef688f02}
EXPECTED_DENSE_CONFIG_SHA256=${EXPECTED_DENSE_CONFIG_SHA256:-fab927c9809d751da0ebb07c39697e6f108273eb60242c2510b316f040c283b3}
EXPECTED_COFITOK_CHECKPOINT_SHA256=${EXPECTED_COFITOK_CHECKPOINT_SHA256:-ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-false}
RECOVERY_EXECUTION_ALLOWED=${RECOVERY_EXECUTION_ALLOWED:-false}

STABILITY_ROOT="$CHECKPOINT_ROOT/stability_probe_2026-07-29"
SOURCE_PAIR="$STABILITY_ROOT/pair5k_rollout_x0_u2_ema_teacher"
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
CONFIG_VALIDATION="$REPORT_ROOT/dense_recovery_config_validation.json"
STORAGE_VALIDATION="$REPORT_ROOT/dense_recovery_storage_capacity.json"
MONITOR_REPORT="$OUTPUT_ROOT/pair_monitor.json"
MONITOR_LOG="$OUTPUT_ROOT/pair_monitor_dense_recovery.log"
MONITOR_PID_FILE="$OUTPUT_ROOT/pair_monitor_dense_recovery.pid"
MONITOR_NAME=generation_stability_ema_teacher_matched_50k
RECOVERY_STATUS="$REPORT_ROOT/dense_recovery_status.json"
RECOVERY_LOCK="$OUTPUT_ROOT/dense_recovery.lock"

write_status() {
  local status="$1"
  local detail="$2"
  local exit_code="${3:-}"
  PYTHONPATH="$CONTROL_PROJECT/src" "$PYTHON" - \
    "$RECOVERY_STATUS" "$status" "$detail" "$exit_code" \
    "$EXPECTED_CONTROL_REVISION" "$EXPECTED_CONTROL_BRANCH" \
    "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" "$$" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "role": "generation_stability_ema_teacher_dense_recovery",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "exit_code": int(sys.argv[4]) if sys.argv[4] else None,
    "pid": int(sys.argv[9]),
    "hostname": socket.gethostname(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "controller": {"revision": sys.argv[5], "branch": sys.argv[6]},
    "training": {"revision": sys.argv[7], "branch": sys.argv[8]},
    "full_training_launch_allowed": False,
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
    write_status failed "dense recovery exited before matched-pair completion" "$exit_code" || true
  fi
}
trap on_exit EXIT

control_python() {
  PYTHONPATH="$CONTROL_PROJECT/src" "$PYTHON" "$@"
}

training_python() {
  (
    cd "$TRAINING_PROJECT"
    PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" "$@"
  )
}

require_complete() {
  control_python "$CONTROL_PROJECT/scripts/validate_generation_training_completion.py" \
    --training-report "$1" \
    --config "$2" \
    --expected-steps 50000 \
    --expected-revision "$EXPECTED_TRAINING_REVISION" \
    --expected-branch "$EXPECTED_TRAINING_BRANCH" \
    --expected-micro-batch-size "$SELECTED_MICRO_BATCH" \
    --expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
    >/dev/null
}

monitor_report_passes() {
  control_python - "$MONITOR_REPORT" "$MONITOR_NAME" <<'PY'
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
):
    raise SystemExit(1)
PY
}

start_monitor() {
  nohup env PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
    "$TRAINING_PROJECT/scripts/monitor_generation_pair.py" \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*ema_teacher.*50k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_dense_recovery_after_transition_failure.sh' \
    --checkpoint-interval 5000 \
    --checkpoint-grace-steps 250 \
    --checkpoint-integrity-policy required \
    --expected-checkpoint-revision "$EXPECTED_TRAINING_REVISION" \
    --poll-seconds 300 \
    --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    >"$MONITOR_LOG" 2>&1 </dev/null &
  local monitor_pid=$!
  local temporary="${MONITOR_PID_FILE}.tmp.$$"
  printf '%s\n' "$monitor_pid" >"$temporary"
  mv "$temporary" "$MONITOR_PID_FILE"
  sleep 2
  if ! kill -0 "$monitor_pid" 2>/dev/null; then
    printf 'dense-recovery pair monitor failed during launch\n' >&2
    exit 10
  fi
}

snapshot_monitor() {
  training_python scripts/monitor_generation_pair.py \
    --output-root "$OUTPUT_ROOT" \
    --output "$MONITOR_REPORT" \
    --monitor-name "$MONITOR_NAME" \
    --cofitok-run "$(basename "$COFITOK_RUN")" \
    --dense-run "$(basename "$DENSE_RUN")" \
    --expected-steps 50000 \
    --training-process-pattern '[s]cripts/train_generation.py.*ema_teacher.*50k' \
    --runbook-process-pattern '[g]eneration_stability_ema_teacher_dense_recovery_after_transition_failure.sh' \
    --checkpoint-interval 5000 \
    --checkpoint-grace-steps 250 \
    --checkpoint-integrity-policy required \
    --expected-checkpoint-revision "$EXPECTED_TRAINING_REVISION" \
    --poll-seconds 300 \
    --stall-seconds 1800 \
    --idle-failure-grace-seconds 600 \
    --once
}

[[ -x "$PYTHON" ]]
[[ "$PREFLIGHT_ONLY" == true || "$PREFLIGHT_ONLY" == false ]]
[[ "$RECOVERY_EXECUTION_ALLOWED" == true || "$RECOVERY_EXECUTION_ALLOWED" == false ]]
[[ -d "$TRAINING_PROJECT/.git" || -f "$TRAINING_PROJECT/.git" ]]
[[ "$(git -C "$CONTROL_PROJECT" rev-parse HEAD)" == "$EXPECTED_CONTROL_REVISION" ]]
[[ "$(git -C "$CONTROL_PROJECT" branch --show-current)" == "$EXPECTED_CONTROL_BRANCH" ]]
[[ -z "$(git -C "$CONTROL_PROJECT" status --porcelain --untracked-files=no)" ]]
[[ "$(git -C "$TRAINING_PROJECT" rev-parse HEAD)" == "$EXPECTED_TRAINING_REVISION" ]]
[[ "$(git -C "$TRAINING_PROJECT" branch --show-current)" == "$EXPECTED_TRAINING_BRANCH" ]]
[[ -z "$(git -C "$TRAINING_PROJECT" status --porcelain --untracked-files=no)" ]]
command -v flock >/dev/null
mkdir -p "$OUTPUT_ROOT"
exec 6>"$RECOVERY_LOCK"
if ! flock -n 6; then
  printf 'refusing concurrent dense recovery controller\n' >&2
  exit 15
fi
mkdir -p "$REPORT_ROOT"
write_status running "validating completed CoFiTok and frozen matched runtime"

read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION < <(
  control_python - \
    "$RUNTIME_SELECTION" \
    "$EXPECTED_RUNTIME_SELECTION_SHA256" \
    "$EXPECTED_TRAINING_REVISION" \
    "$EXPECTED_TRAINING_BRANCH" \
    "$EXPECTED_COFITOK_CONFIG_SHA256" \
    "$EXPECTED_DENSE_CONFIG_SHA256" \
    "$TRAINING_PROJECT/$COFITOK_CONFIG" \
    "$TRAINING_PROJECT/$DENSE_CONFIG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report_path = Path(sys.argv[1])
if sha256(report_path) != sys.argv[2]:
    raise SystemExit("frozen runtime-selection SHA256 mismatch")
report = json.loads(report_path.read_text(encoding="utf-8"))
lock = report.get("selection_lock", {})
git = lock.get("git", {})
if (
    report.get("status") != "selected"
    or report.get("git_revision") != sys.argv[3]
    or git.get("revision") != sys.argv[3]
    or git.get("branch") != sys.argv[4]
    or git.get("tracked_dirty") is not False
):
    raise SystemExit("frozen runtime-selection Git identity mismatch")
expected_configs = {"cofitok": sys.argv[5], "dense_identity": sys.argv[6]}
if report.get("config_sha256") != expected_configs:
    raise SystemExit("frozen runtime-selection config identity mismatch")
if sha256(Path(sys.argv[7])) != sys.argv[5] or sha256(Path(sys.argv[8])) != sys.argv[6]:
    raise SystemExit("training-checkout config SHA256 mismatch")
selected = report.get("selected", {})
micro = int(selected.get("micro_batch_size", -1))
accumulation = int(selected.get("gradient_accumulation_steps", -1))
if micro < 1 or accumulation < 1 or micro * accumulation != 64:
    raise SystemExit("frozen runtime selection does not preserve effective batch 64")
print(micro, accumulation)
PY
)

require_complete \
  "$COFITOK_RUN/training_report.json" \
  "$TRAINING_PROJECT/$COFITOK_CONFIG"
control_python - \
  "$COFITOK_RUN" "$EXPECTED_COFITOK_CHECKPOINT_SHA256" \
  "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.training.checkpointing import verify_training_checkpoint

run = Path(sys.argv[1])
latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
checkpoint = run / latest["checkpoint"]
integrity = verify_training_checkpoint(checkpoint)
if (
    integrity.get("checkpoint_sha256") != sys.argv[2]
    or latest.get("checkpoint_sha256") != sys.argv[2]
    or latest.get("git_revision") != sys.argv[3]
    or latest.get("git_branch") != sys.argv[4]
    or latest.get("git_dirty") is not False
    or int(latest.get("step", -1)) != 50000
):
    raise SystemExit("completed CoFiTok checkpoint trust boundary mismatch")
PY

training_python scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_scaling \
  --output "$CONFIG_VALIDATION" >/dev/null
training_python scripts/check_generation_storage_capacity.py \
  --path "$CHECKPOINT_ROOT" \
  --output "$STORAGE_VALIDATION" \
  --stage stability_scaling_matched_50k \
  --reference-checkpoint "$SOURCE_PAIR/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00005000.pt" \
  --reference-checkpoint "$SOURCE_PAIR/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00005000.pt" \
  --checkpoint-count 10 \
  --sample-count 20000 \
  --estimated-sample-kib 256 \
  --additional-gib 32 \
  --safety-margin-gib 64 >/dev/null

if [[ "$PREFLIGHT_ONLY" == true ]]; then
  completed=true
  write_status prepared "dense recovery preflight passed; no process was launched"
  exit 0
fi
if [[ "$RECOVERY_EXECUTION_ALLOWED" != true ]]; then
  printf 'dense recovery execution requires RECOVERY_EXECUTION_ALLOWED=true\n' >&2
  exit 14
fi

if pgrep -af '[t]rain_generation.py.*ema_teacher.*50k|[m]onitor_generation_pair.py.*stability_scaling_50k_ema_teacher|[g]eneration_stability_ema_teacher_matched_50k_after_gate.sh' >/dev/null; then
  printf 'refusing dense recovery while another stability 50K queue process exists\n' >&2
  exit 12
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing dense recovery while the GPU is busy\n' >&2
  exit 9
fi

write_status running "launching dense-only member at the immutable training revision"
start_monitor

resume_args=()
if [[ -f "$DENSE_RUN/training_report.json" ]] \
  && require_complete \
    "$DENSE_RUN/training_report.json" \
    "$TRAINING_PROJECT/$DENSE_CONFIG"; then
  printf 'dense stability member is already complete\n'
elif [[ -f "$DENSE_RUN/latest.json" ]]; then
  control_python - "$DENSE_RUN/latest.json" \
    "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

latest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    latest.get("git_revision") != sys.argv[2]
    or latest.get("git_branch") != sys.argv[3]
    or latest.get("git_dirty") is not False
):
    raise SystemExit("dense resume checkpoint has another Git identity")
PY
  resume_args=(--resume auto)
else
  if [[ -d "$DENSE_RUN" ]] && find "$DENSE_RUN" -mindepth 1 -print -quit | grep -q .; then
    printf 'refusing fresh dense launch into non-empty state without latest.json\n' >&2
    exit 13
  fi
fi

if ! require_complete \
  "$DENSE_RUN/training_report.json" \
  "$TRAINING_PROJECT/$DENSE_CONFIG" 2>/dev/null; then
  (
    cd "$TRAINING_PROJECT"
    PYTHONPATH="$TRAINING_PROJECT/src" "$PYTHON" \
      scripts/run_generation_training_watchdog.py \
      --monitor-report "$MONITOR_REPORT" \
      --monitor-pid-file "$MONITOR_PID_FILE" \
      --expected-monitor-name "$MONITOR_NAME" \
      --status-output "$DENSE_RUN/training_watchdog.json" \
      --poll-seconds 30 \
      --startup-grace-seconds 600 \
      --monitor-silence-seconds 900 \
      --monitor-process-grace-seconds 120 \
      --termination-grace-seconds 60 \
      -- \
      "$PYTHON" scripts/train_generation.py \
        --config "$DENSE_CONFIG" \
        --output-dir "$DENSE_RUN" \
        --micro-batch-size "$SELECTED_MICRO_BATCH" \
        --gradient-accumulation-steps "$SELECTED_ACCUMULATION" \
        "${resume_args[@]}"
  )
fi

require_complete \
  "$DENSE_RUN/training_report.json" \
  "$TRAINING_PROJECT/$DENSE_CONFIG"
snapshot_monitor
monitor_report_passes

completed=true
write_status pass "matched CoFiTok/dense 50K training pair completed; source-bound summary and post-eval remain separate"
