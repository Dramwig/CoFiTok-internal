#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
EXPECTED_SAMPLING_PIPELINE_REVISION=${EXPECTED_SAMPLING_PIPELINE_REVISION:?set the exact recovery/confirmation revision}
EXPECTED_SAMPLING_PIPELINE_BRANCH=${EXPECTED_SAMPLING_PIPELINE_BRANCH:-scale/generation-large-capacity}
SAMPLING_CONFIRMATION_EXECUTION_APPROVAL=${SAMPLING_CONFIRMATION_EXECUTION_APPROVAL:?set the explicit confirmation execution approval sentinel path}
EXPECTED_SAMPLING_CONFIRMATION_EXECUTION_APPROVAL_SHA256=${EXPECTED_SAMPLING_CONFIRMATION_EXECUTION_APPROVAL_SHA256:?set the immutable confirmation execution approval SHA256}
SAMPLING_CONFIRMATION_EXECUTION_ALLOWED=${SAMPLING_CONFIRMATION_EXECUTION_ALLOWED:-false}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
COFITOK_RUN="$SCALING_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$SCALING_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
PROMOTION_GATE="$SCALING_ROOT/reports/promotion_gate.json"
COFITOK_FORMAL_METRICS="$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
DENSE_FORMAL_METRICS="$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"

RECOVERY_PLAN="$PROJECT/configs/generation/diagnostics/stability_50k_sampling_recovery_v1.json"
RECOVERY_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_recovery_v1"
RECOVERY_CASE_ROOT="$RECOVERY_ROOT/cases"
RECOVERY_SUMMARY="$RECOVERY_ROOT/summary.json"

CONFIRMATION_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_confirmation_10k_v1"
CASE_ROOT="$CONFIRMATION_ROOT/cases"
EVAL_CACHE="$CONFIRMATION_ROOT/eval_cache/torch_fidelity"
PREFLIGHT="$CONFIRMATION_ROOT/preflight.json"
REPORT="$CONFIRMATION_ROOT/confirmation_report.json"
STATUS="$CONFIRMATION_ROOT/status.json"
LOCK="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_confirmation_10k_v1.lock"
CURRENT_STAGE=initializing

cd "$PROJECT"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$PYTHON" ]]
[[ "$SAMPLING_CONFIRMATION_EXECUTION_ALLOWED" == true ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SAMPLING_PIPELINE_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_SAMPLING_PIPELINE_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]

for path in \
  "$RECOVERY_PLAN" \
  "$RECOVERY_SUMMARY" \
  "$PROMOTION_GATE" \
  "$COFITOK_FORMAL_METRICS" \
  "$DENSE_FORMAL_METRICS" \
  "$COFITOK_CHECKPOINT" \
  "$COFITOK_CHECKPOINT.integrity.json" \
  "$COFITOK_RUN/latest.json" \
  "$DENSE_CHECKPOINT" \
  "$DENSE_CHECKPOINT.integrity.json" \
  "$DENSE_RUN/latest.json"; do
  [[ -f "$path" ]]
done
[[ -d "$RECOVERY_CASE_ROOT" ]]
[[ -d "$DATA_ROOT" ]]
[[ -f "$SAMPLING_CONFIRMATION_EXECUTION_APPROVAL" ]]

"$PYTHON" scripts/validate_generation_stability_sampling_execution_approval.py \
  --approval "$SAMPLING_CONFIRMATION_EXECUTION_APPROVAL" \
  --expected-approval-sha256 "$EXPECTED_SAMPLING_CONFIRMATION_EXECUTION_APPROVAL_SHA256" \
  --evidence "$RECOVERY_SUMMARY" \
  --expected-scope stability_50k_sampling_confirmation_10k_v1_execution_only \
  --expected-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
  --expected-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
  --expected-output-root "$CONFIRMATION_ROOT" >/dev/null

mkdir -p "$CONFIRMATION_ROOT" "$CASE_ROOT" "$EVAL_CACHE"
command -v flock >/dev/null
exec 6>"$LOCK"
if ! flock -n 6; then
  printf 'matched 10K sampling confirmation already owns its lock\n' >&2
  exit 75
fi

write_status() {
  local status_value="$1"
  local stage_value="$2"
  local detail_value="$3"
  local exit_code_value="${4:-}"
  STATUS_VALUE="$status_value" \
  STAGE_VALUE="$stage_value" \
  DETAIL_VALUE="$detail_value" \
  EXIT_CODE_VALUE="$exit_code_value" \
  "$PYTHON" - "$STATUS" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
exit_code = os.environ["EXIT_CODE_VALUE"]
payload = {
    "schema_version": 1,
    "role": "non_authorizing_matched_10000_sampling_confirmation",
    "status": os.environ["STATUS_VALUE"],
    "stage": os.environ["STAGE_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "exit_code": int(exit_code) if exit_code else None,
    "confirmation_non_authorizing": True,
    "replaces_frozen_promotion_gate": False,
    "confirmation_report_is_promotion_gate": False,
    "new_gate_required": True,
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

on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -eq 75 ]]; then
    write_status deferred "$CURRENT_STAGE" "GPU or confirmation lock was busy; no stage was launched" "$exit_code"
  elif [[ "$exit_code" -ne 0 ]]; then
    write_status failed "$CURRENT_STAGE" "matched 10K confirmation stopped before completion" "$exit_code"
  fi
}
trap on_exit EXIT

require_gpu_idle() {
  local compute_rows
  compute_rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [[ -n "${compute_rows//[[:space:]]/}" ]]; then
    printf 'refusing matched 10K confirmation while GPU compute is busy:\n%s\n' "$compute_rows" >&2
    exit 75
  fi
}

CURRENT_STAGE=source_verification
write_status running "$CURRENT_STAGE" "rehashing frozen checkpoints and recovery evidence"

"$PYTHON" - \
  "$RECOVERY_PLAN" \
  "$COFITOK_RUN" \
  "$DENSE_RUN" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    resolve_latest_checkpoint,
)

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
training = plan["source"]["provenance_contract"]
runs = {
    "cofitok": Path(sys.argv[2]).resolve(),
    "dense_identity": Path(sys.argv[3]).resolve(),
}
for method, run in runs.items():
    expected = plan["methods"][method]
    checkpoint = resolve_latest_checkpoint(run)
    integrity_path = checkpoint_integrity_path(checkpoint)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (
        checkpoint.name != "checkpoint_step_00050000.pt"
        or int(integrity.get("step", -1)) != int(expected["checkpoint_step"])
        or integrity.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or file_sha256(integrity_path) != expected["checkpoint_integrity_sha256"]
        or integrity.get("git_revision") != training["training_revision"]
        or integrity.get("git_branch") != training["training_branch"]
        or integrity.get("git_dirty") is not False
    ):
        raise SystemExit(f"{method} frozen checkpoint identity differs from the recovery plan")
PY

"$PYTHON" scripts/build_generation_stability_sampling_confirmation.py \
  --mode preflight \
  --project "$PROJECT" \
  --recovery-plan "$RECOVERY_PLAN" \
  --recovery-summary "$RECOVERY_SUMMARY" \
  --recovery-case-root "$RECOVERY_CASE_ROOT" \
  --promotion-gate "$PROMOTION_GATE" \
  --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
  --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
  --expected-recovery-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
  --expected-recovery-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
  --expected-confirmation-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
  --expected-confirmation-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
  --output "$PREFLIGHT"

if [[ -f "$REPORT" ]]; then
  CURRENT_STAGE=reverify_complete
  "$PYTHON" scripts/build_generation_stability_sampling_confirmation.py \
    --mode build \
    --project "$PROJECT" \
    --recovery-plan "$RECOVERY_PLAN" \
    --recovery-summary "$RECOVERY_SUMMARY" \
    --recovery-case-root "$RECOVERY_CASE_ROOT" \
    --promotion-gate "$PROMOTION_GATE" \
    --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
    --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
    --case-root "$CASE_ROOT" \
    --expected-recovery-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
    --expected-recovery-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
    --expected-confirmation-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
    --expected-confirmation-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
    --output "$REPORT"
  report_status="$($PYTHON - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
  write_status "$report_status" complete "existing matched 10K confirmation was independently reverified"
  trap - EXIT
  exit 0
fi

read -r selected_case guidance_scale guidance_rescale < <(
  "$PYTHON" - "$PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
protocol = preflight["protocol"]
print(
    preflight["recovery"]["selected_case"],
    protocol["guidance_scale"],
    protocol["guidance_rescale"],
)
PY
)
[[ "$selected_case" != "cfg150_r000" ]]

require_gpu_idle

for method in cofitok dense_identity; do
  if [[ "$method" == cofitok ]]; then
    checkpoint="$COFITOK_CHECKPOINT"
    prefix_budget=8
  else
    checkpoint="$DENSE_CHECKPOINT"
    prefix_budget=1
  fi
  method_root="$CASE_ROOT/$method"
  sampling_report="$method_root/sampling_report.json"

  if [[ ! -f "$sampling_report" ]]; then
    CURRENT_STAGE="sampling_${method}"
    write_status running "$CURRENT_STAGE" "generating the matched selected-protocol 10K sample set"
    require_gpu_idle
    "$PYTHON" scripts/generate_samples.py \
      --checkpoint "$checkpoint" \
      --output-dir "$method_root" \
      --num-samples 10000 \
      --batch-size 32 \
      --sample-steps 100 \
      --prefix-budgets "$prefix_budget" \
      --guidance-scale "$guidance_scale" \
      --guidance-rescale "$guidance_rescale" \
      --cfg-batch-mode batched \
      --eta 0.0 \
      --seed 0 \
      --start-index 0 \
      --weights ema \
      --precision bf16 \
      --resume
  fi

  CURRENT_STAGE="metrics_${method}"
  write_status running "$CURRENT_STAGE" "evaluating formal-size FID/IS/precision/recall"
  require_gpu_idle
  "$PYTHON" scripts/evaluate_generation_metrics.py \
    --real-dir "$DATA_ROOT" \
    --generated-dir "$method_root/prefix_$prefix_budget" \
    --sampling-report "$sampling_report" \
    --output-dir "$method_root/metrics" \
    --batch-size 64 \
    --prc-batch-size 10000 \
    --min-samples 10000 \
    --seed 2027 \
    --cache-root "$EVAL_CACHE" \
    --real-cache-name imagenet256_val_50k_torch_fidelity_v04 \
    --resume
done

CURRENT_STAGE=summarizing
write_status running "$CURRENT_STAGE" "building the non-authorizing matched 10K confirmation report"
"$PYTHON" scripts/build_generation_stability_sampling_confirmation.py \
  --mode build \
  --project "$PROJECT" \
  --recovery-plan "$RECOVERY_PLAN" \
  --recovery-summary "$RECOVERY_SUMMARY" \
  --recovery-case-root "$RECOVERY_CASE_ROOT" \
  --promotion-gate "$PROMOTION_GATE" \
  --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
  --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
  --case-root "$CASE_ROOT" \
  --expected-recovery-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
  --expected-recovery-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
  --expected-confirmation-revision "$EXPECTED_SAMPLING_PIPELINE_REVISION" \
  --expected-confirmation-branch "$EXPECTED_SAMPLING_PIPELINE_BRANCH" \
  --output "$REPORT"

report_status="$($PYTHON - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
CURRENT_STAGE=complete
write_status "$report_status" "$CURRENT_STAGE" "matched 10K confirmation completed; a separate new gate is still required"
trap - EXIT
printf 'matched 10K sampling confirmation: %s  %s\n' "$report_status" "$REPORT"
