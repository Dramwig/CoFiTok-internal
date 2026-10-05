#!/usr/bin/env bash
set -euo pipefail

BASE=/tmp/cofitok-quality-bridge-execution-cf0e5fa
PROJECT="$BASE/CoFiTok-internal"
FORMAL_PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
BUNDLE=/tmp/cofitok-quality-bridge-cf0e5fa.bundle
BUNDLE_BYTES=40130783
BUNDLE_SHA256=3392e2a66c48237b8bfa0bde519245d4d17d055bb21b4e86321537fe5c3a871a
REVISION=cf0e5faa94bf4ab38d947b921935b3b765b5537a
TREE=6cef27723196fd363379bca2e7b85b1678ebd777
BRANCH=scale/generation-stability-quality-bridge-100k
FORMAL_REVISION=1ebcc15210e63a776a2ba448481cbd8bb94a4066
FORMAL_BRANCH=scale/generative-system
FORMAL_PORCELAIN_COUNT=87
FORMAL_PORCELAIN_SHA256=a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497
FROZEN_GATE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/promotion_gate.json
FROZEN_GATE_SHA256=2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
REPORT_ROOT="$OUTPUT_ROOT/reports"
PREPARATION="$REPORT_ROOT/preparation.json"
PREPARATION_BYTES=23398
PREPARATION_SHA256=7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea
APPROVAL="$BASE/execution_approval.json"
APPROVAL_BYTES=1754
APPROVAL_SHA256=e9da52a4e7ff1b4700b70aadaa8703ee40fb1a9862e847dcfb6e75933d295a4b
STANDING_AUTHORIZATION="$BASE/standing_authorization.json"
STANDING_AUTHORIZATION_BYTES=865
STANDING_AUTHORIZATION_SHA256=5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
RUNBOOK=artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
STATUS="$BASE/idle_waiter_status.json"
LOG="$BASE/controller.log"
WAITER_LOCK=/tmp/cofitok-quality-bridge-100k-cf0e5fa.waiter.lock
EXECUTION_LOCK="$OUTPUT_ROOT/quality_bridge_execution.lock"
LAUNCH_RECEIPT="$REPORT_ROOT/launch_receipt.json"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
POLL_SECONDS=60
REQUIRED_IDLE_POLLS=5

exec 9>"$WAITER_LOCK"
if ! flock -n 9; then
  printf 'quality bridge idle waiter already owns its lock\n' >&2
  exit 75
fi

write_status() {
  local status_value="$1"
  local detail_value="$2"
  local idle_polls_value="$3"
  local gpu_rows_value="${4:-}"
  local controller_pid_value="${5:-}"
  STATUS_VALUE="$status_value" \
  DETAIL_VALUE="$detail_value" \
  IDLE_POLLS_VALUE="$idle_polls_value" \
  GPU_ROWS_VALUE="$gpu_rows_value" \
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
gpu_rows = [row for row in os.environ["GPU_ROWS_VALUE"].splitlines() if row.strip()]
payload = {
    "schema_version": 1,
    "role": "cofitok_full_data_matched_100k_quality_bridge_idle_waiter",
    "status": os.environ["STATUS_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "idle_polls": int(os.environ["IDLE_POLLS_VALUE"]),
    "required_idle_polls": 5,
    "poll_seconds": 60,
    "gpu_compute_rows": gpu_rows,
    "controller_pid": int(controller_pid) if controller_pid else None,
    "project": "/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal",
    "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
    "tree": "6cef27723196fd363379bca2e7b85b1678ebd777",
    "branch": "scale/generation-stability-quality-bridge-100k",
    "dataset": "imagenet_256",
    "steps_per_method": 100000,
    "milestone_steps": [50000, 100000],
    "effective_batch_size": 64,
    "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1",
    "quality_bridge_execution_allowed": True,
    "scope_limited_to_quality_bridge": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "report_is_promotion_gate": False,
    "hostname": socket.gethostname(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(path)
PY
}

require_exact_file() {
  local path="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  [[ -f "$path" ]]
  [[ "$(stat -c %s "$path")" == "$expected_bytes" ]]
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected_sha256" ]]
}

require_empty_or_absent_run_dir() {
  local path="$1"
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
    printf 'refusing non-empty unreceipted quality bridge run: %s\n' "$path" >&2
    exit 76
  fi
}

require_no_matching_bridge_processes() {
  if pgrep -af '[g]eneration_stability_full_data_quality_bridge_100k_execute.sh' >/dev/null; then
    printf 'quality bridge execution controller already exists\n' >&2
    exit 75
  fi
  if pgrep -af '[t]rain_generation.py.*stability_quality_bridge.*100k' >/dev/null; then
    printf 'quality bridge trainer already exists\n' >&2
    exit 75
  fi
  if pgrep -af '[m]onitor_generation_pair.py.*stability_full_data_100k_base128_quality_bridge_v1' >/dev/null; then
    printf 'quality bridge monitor already exists\n' >&2
    exit 75
  fi
}

require_static_bindings() {
  local formal_porcelain
  local formal_porcelain_count
  local formal_porcelain_sha256
  require_exact_file "$BUNDLE" "$BUNDLE_BYTES" "$BUNDLE_SHA256"
  require_exact_file "$FROZEN_GATE" 31870 "$FROZEN_GATE_SHA256"
  require_exact_file "$PREPARATION" "$PREPARATION_BYTES" "$PREPARATION_SHA256"
  require_exact_file "$APPROVAL" "$APPROVAL_BYTES" "$APPROVAL_SHA256"
  require_exact_file "$STANDING_AUTHORIZATION" "$STANDING_AUTHORIZATION_BYTES" "$STANDING_AUTHORIZATION_SHA256"
  [[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$REVISION" ]]
  [[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$TREE" ]]
  [[ "$(git -C "$PROJECT" branch --show-current)" == "$BRANCH" ]]
  [[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
  [[ "$(git -C "$FORMAL_PROJECT" rev-parse HEAD)" == "$FORMAL_REVISION" ]]
  [[ "$(git -C "$FORMAL_PROJECT" branch --show-current)" == "$FORMAL_BRANCH" ]]
  formal_porcelain="$(mktemp /tmp/cofitok-quality-bridge-formal-porcelain.XXXXXX)"
  git -C "$FORMAL_PROJECT" status --porcelain=v1 >"$formal_porcelain"
  formal_porcelain_count="$(wc -l <"$formal_porcelain")"
  formal_porcelain_sha256="$(sha256sum "$formal_porcelain" | awk '{print $1}')"
  rm -f -- "$formal_porcelain"
  [[ "$formal_porcelain_count" == "$FORMAL_PORCELAIN_COUNT" ]]
  [[ "$formal_porcelain_sha256" == "$FORMAL_PORCELAIN_SHA256" ]]
  [[ -d /root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/train ]]
  [[ -d /root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val ]]
}

require_standing_authorization() {
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
}

replay_contracts() {
  cd "$PROJECT"
  PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/validate_generation_quality_bridge_preparation.py \
    --preparation "$PREPARATION" \
    --expected-preparation-sha256 "$PREPARATION_SHA256" \
    --promotion-gate "$FROZEN_GATE" \
    --expected-promotion-gate-sha256 "$FROZEN_GATE_SHA256" \
    --cofitok-config configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json \
    --dense-config configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json >/dev/null
  PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/validate_generation_quality_bridge_execution_approval.py \
    --approval "$APPROVAL" \
    --expected-approval-sha256 "$APPROVAL_SHA256" \
    --preparation "$PREPARATION" \
    --expected-preparation-sha256 "$PREPARATION_SHA256" \
    --expected-revision "$REVISION" \
    --expected-branch "$BRANCH" \
    --expected-output-root "$OUTPUT_ROOT" >/dev/null
}

require_static_bindings
require_standing_authorization
replay_contracts
require_no_matching_bridge_processes
[[ ! -f "$LAUNCH_RECEIPT" ]]
require_empty_or_absent_run_dir "$COFITOK_RUN"
require_empty_or_absent_run_dir "$DENSE_RUN"
if ! flock -n "$EXECUTION_LOCK" true 2>/dev/null; then
  printf 'quality bridge execution lock is already held\n' >&2
  exit 75
fi

idle_polls=0
write_status waiting waiting_for_gpu_idle "$idle_polls"
while true; do
  compute_rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [[ -n "${compute_rows//[[:space:]]/}" ]]; then
    idle_polls=0
    write_status waiting waiting_for_gpu_idle "$idle_polls" "$compute_rows"
    sleep "$POLL_SECONDS"
    continue
  fi
  idle_polls=$((idle_polls + 1))
  write_status waiting confirming_gpu_idle "$idle_polls" "$compute_rows"
  if (( idle_polls < REQUIRED_IDLE_POLLS )); then
    sleep "$POLL_SECONDS"
    continue
  fi
  break
done

require_static_bindings
require_standing_authorization
replay_contracts
require_no_matching_bridge_processes
[[ ! -f "$LAUNCH_RECEIPT" ]]
require_empty_or_absent_run_dir "$COFITOK_RUN"
require_empty_or_absent_run_dir "$DENSE_RUN"
[[ -z "$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)" ]]
if ! flock -n "$EXECUTION_LOCK" true 2>/dev/null; then
  printf 'quality bridge execution lock became held before launch\n' >&2
  exit 75
fi

cd "$PROJECT"
export PYTHON
export EXPECTED_TARGET_REVISION="$REVISION"
export EXPECTED_TARGET_BRANCH="$BRANCH"
export EXPECTED_PREPARATION_SHA256="$PREPARATION_SHA256"
export QUALITY_BRIDGE_EXECUTION_APPROVAL="$APPROVAL"
export EXPECTED_EXECUTION_APPROVAL_SHA256="$APPROVAL_SHA256"
export QUALITY_BRIDGE_EXECUTION_ALLOWED=true

write_status launched launching_exact_quality_bridge_controller "$idle_polls" "" "$$"
set +e
bash "$RUNBOOK" >>"$LOG" 2>&1
exit_code="$?"
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status completed quality_bridge_controller_completed "$idle_polls" "" "$$"
else
  write_status failed quality_bridge_controller_failed "$idle_polls" "" "$$"
fi
exit "$exit_code"
