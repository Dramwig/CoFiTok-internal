#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-sampling-recovery-execution-11d8f954/CoFiTok-internal
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
REVISION=11d8f954030915f1d8848594683ac70518ce39bc
TREE=95a5e656b1e0a8b772ce5d1fe1badc487ef98d16
BRANCH=scale/generation-large-capacity
RECOVERY_SUMMARY=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_recovery_v1/summary.json
RECOVERY_SUMMARY_SHA256=79bc585c8fe9e0f6fef0fa0803c64e4a7e0f99af9da34d0884670ed0b8b03de5
APPROVAL=/tmp/cofitok-sampling-recovery-execution-11d8f954/confirmation_approval.json
APPROVAL_SHA256_FILE=/tmp/cofitok-sampling-recovery-execution-11d8f954/confirmation_approval.sha256
STANDING_AUTHORIZATION=/tmp/cofitok-sampling-recovery-execution-11d8f954/standing_authorization.json
STANDING_AUTHORIZATION_SHA256=5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1
STATUS=/tmp/cofitok-sampling-recovery-execution-11d8f954/confirmation_waiter_status.json
LOG=/tmp/cofitok-sampling-recovery-execution-11d8f954/confirmation_controller.log
WAITER_LOCK=/tmp/cofitok-sampling-confirmation-10k-v1.waiter.lock
STAGE_LOCK=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1.lock
RUNBOOK=artifacts/runbooks/generation_stability_frozen_50k_sampling_confirmation_10k.sh
POLL_SECONDS=60
REQUIRED_IDLE_POLLS=5

exec 9>"$WAITER_LOCK"
if ! flock -n 9; then
  printf 'confirmation waiter already owns its lock\n' >&2
  exit 75
fi

write_status() {
  local status_value="$1"
  local detail_value="$2"
  local idle_polls_value="$3"
  local controller_pid_value="${4:-}"
  STATUS_VALUE="$status_value" \
  DETAIL_VALUE="$detail_value" \
  IDLE_POLLS_VALUE="$idle_polls_value" \
  CONTROLLER_PID_VALUE="$controller_pid_value" \
  "$PYTHON" - "$STATUS" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
controller_pid = os.environ["CONTROLLER_PID_VALUE"]
payload = {
    "schema_version": 1,
    "role": "cofitok_sampling_confirmation_10k_idle_waiter",
    "status": os.environ["STATUS_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "idle_polls": int(os.environ["IDLE_POLLS_VALUE"]),
    "required_idle_polls": 5,
    "poll_seconds": 60,
    "controller_pid": int(controller_pid) if controller_pid else None,
    "project": "/tmp/cofitok-sampling-recovery-execution-11d8f954/CoFiTok-internal",
    "revision": "11d8f954030915f1d8848594683ac70518ce39bc",
    "tree": "95a5e656b1e0a8b772ce5d1fe1badc487ef98d16",
    "branch": "scale/generation-large-capacity",
    "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1",
    "confirmation_non_authorizing": True,
    "replaces_frozen_promotion_gate": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "hostname": socket.gethostname(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

approval_sha256="$(cut -d' ' -f1 "$APPROVAL_SHA256_FILE")"
[[ "$approval_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | cut -d' ' -f1)" == "$STANDING_AUTHORIZATION_SHA256" ]]
[[ -f "$APPROVAL" ]]
[[ "$(sha256sum "$APPROVAL" | cut -d' ' -f1)" == "$approval_sha256" ]]
[[ "$(sha256sum "$RECOVERY_SUMMARY" | cut -d' ' -f1)" == "$RECOVERY_SUMMARY_SHA256" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
[[ ! -f "$OUTPUT_ROOT/confirmation_report.json" ]]

"$PYTHON" - "$STANDING_AUTHORIZATION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("schema_version") != 1
    or payload.get("role") != "cofitok_standing_experiment_authorization_record"
    or payload.get("status") != "active"
    or payload.get("instruction", {}).get("exact_text")
    != "之后不要我授权你直接运行需要的实验"
    or payload.get("instruction", {}).get("language") != "zh-CN"
    or payload.get("preserved_safety_boundaries")
    != {
        "unrelated_project_processes_must_not_be_modified": True,
        "formal_remote_checkout_must_not_be_modified": True,
        "locked_evidence_must_not_be_overwritten": True,
        "independent_clean_checkout_required": True,
        "exact_revision_stage_and_output_binding_required": True,
        "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing": True,
    }
):
    raise SystemExit("standing experiment authorization record differs")
PY

PYTHONPATH="$PROJECT/src" "$PYTHON" "$PROJECT/scripts/validate_generation_stability_sampling_execution_approval.py" \
  --approval "$APPROVAL" \
  --expected-approval-sha256 "$approval_sha256" \
  --evidence "$RECOVERY_SUMMARY" \
  --expected-scope stability_50k_sampling_confirmation_10k_v1_execution_only \
  --expected-revision "$REVISION" \
  --expected-branch "$BRANCH" \
  --expected-output-root "$OUTPUT_ROOT" >/dev/null

if flock -n "$STAGE_LOCK" true 2>/dev/null; then
  :
else
  printf 'confirmation stage lock is already held\n' >&2
  exit 75
fi

idle_polls=0
write_status waiting waiting_for_gpu_idle "$idle_polls"
while true; do
  compute_rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [[ -n "${compute_rows//[[:space:]]/}" ]]; then
    idle_polls=0
    write_status waiting waiting_for_gpu_idle "$idle_polls"
    sleep "$POLL_SECONDS"
    continue
  fi
  idle_polls=$((idle_polls + 1))
  write_status waiting confirming_gpu_idle "$idle_polls"
  if (( idle_polls < REQUIRED_IDLE_POLLS )); then
    sleep "$POLL_SECONDS"
    continue
  fi
  break
done

[[ "$(sha256sum "$APPROVAL" | cut -d' ' -f1)" == "$approval_sha256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | cut -d' ' -f1)" == "$STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$RECOVERY_SUMMARY" | cut -d' ' -f1)" == "$RECOVERY_SUMMARY_SHA256" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)" ]]

cd "$PROJECT"
export PYTHON
export EXPECTED_SAMPLING_PIPELINE_REVISION="$REVISION"
export EXPECTED_SAMPLING_PIPELINE_BRANCH="$BRANCH"
export SAMPLING_CONFIRMATION_EXECUTION_APPROVAL="$APPROVAL"
export EXPECTED_SAMPLING_CONFIRMATION_EXECUTION_APPROVAL_SHA256="$approval_sha256"
export SAMPLING_CONFIRMATION_EXECUTION_ALLOWED=true

write_status launched launching_exact_confirmation_controller "$idle_polls" "$$"
set +e
bash "$RUNBOOK" >>"$LOG" 2>&1
exit_code="$?"
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status completed confirmation_controller_completed "$idle_polls" "$$"
else
  write_status failed confirmation_controller_failed "$idle_polls" "$$"
fi
exit "$exit_code"
