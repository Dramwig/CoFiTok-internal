#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
EXPECTED_SAMPLING_RECOVERY_REVISION=${EXPECTED_SAMPLING_RECOVERY_REVISION:?set the clean sampling-recovery revision}
EXPECTED_SAMPLING_RECOVERY_BRANCH=${EXPECTED_SAMPLING_RECOVERY_BRANCH:-scale/generation-large-capacity}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
COFITOK_RUN="$SCALING_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$SCALING_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
PROMOTION_GATE="$SCALING_ROOT/reports/promotion_gate.json"
COFITOK_FORMAL_METRICS="$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
DENSE_FORMAL_METRICS="$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
PLAN="$PROJECT/configs/generation/diagnostics/stability_50k_sampling_recovery_v1.json"

DIAGNOSTIC_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_recovery_v1"
CASE_ROOT="$DIAGNOSTIC_ROOT/cases"
EVAL_CACHE="$DIAGNOSTIC_ROOT/eval_cache/torch_fidelity"
PREFLIGHT="$DIAGNOSTIC_ROOT/preflight.json"
SUMMARY="$DIAGNOSTIC_ROOT/summary.json"
STATUS="$DIAGNOSTIC_ROOT/status.json"
LOCK="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_recovery_v1.lock"
CURRENT_STAGE=initializing

cd "$PROJECT"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SAMPLING_RECOVERY_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_SAMPLING_RECOVERY_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]

for path in \
  "$PLAN" \
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
[[ -d "$DATA_ROOT" ]]

mkdir -p "$DIAGNOSTIC_ROOT" "$CASE_ROOT" "$EVAL_CACHE"
command -v flock >/dev/null
exec 7>"$LOCK"
if ! flock -n 7; then
  printf 'sampling-recovery diagnostic already owns its lock\n' >&2
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
    "role": "non_authorizing_matched_inference_protocol_diagnostic",
    "status": os.environ["STATUS_VALUE"],
    "stage": os.environ["STAGE_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "exit_code": int(exit_code) if exit_code else None,
    "diagnostic_non_authorizing": True,
    "replaces_frozen_promotion_gate": False,
    "matched_10000_confirmation_required": True,
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
    write_status deferred "$CURRENT_STAGE" "GPU or diagnostic lock was busy; no stage was launched" "$exit_code"
  elif [[ "$exit_code" -ne 0 ]]; then
    write_status failed "$CURRENT_STAGE" "sampling-recovery diagnostic stopped before completion" "$exit_code"
  fi
}
trap on_exit EXIT

require_gpu_idle() {
  local compute_rows
  compute_rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [[ -n "${compute_rows//[[:space:]]/}" ]]; then
    printf 'refusing sampling-recovery GPU stage while compute is busy:\n%s\n' "$compute_rows" >&2
    exit 75
  fi
}

CURRENT_STAGE=source_verification
write_status running "$CURRENT_STAGE" "verifying frozen reports and physical checkpoints"

"$PYTHON" - \
  "$PLAN" \
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

"$PYTHON" scripts/build_generation_stability_sampling_recovery.py \
  --mode preflight \
  --project "$PROJECT" \
  --plan "$PLAN" \
  --promotion-gate "$PROMOTION_GATE" \
  --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
  --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
  --expected-diagnostic-revision "$EXPECTED_SAMPLING_RECOVERY_REVISION" \
  --expected-diagnostic-branch "$EXPECTED_SAMPLING_RECOVERY_BRANCH" \
  --output "$PREFLIGHT"

if [[ -f "$SUMMARY" ]]; then
  CURRENT_STAGE=reverify_complete
  "$PYTHON" scripts/build_generation_stability_sampling_recovery.py \
    --mode build \
    --project "$PROJECT" \
    --plan "$PLAN" \
    --promotion-gate "$PROMOTION_GATE" \
    --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
    --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
    --case-root "$CASE_ROOT" \
    --expected-diagnostic-revision "$EXPECTED_SAMPLING_RECOVERY_REVISION" \
    --expected-diagnostic-branch "$EXPECTED_SAMPLING_RECOVERY_BRANCH" \
    --output "$SUMMARY"
  write_status pass complete "existing matched recovery sweep was independently reverified"
  trap - EXIT
  exit 0
fi

require_gpu_idle

mapfile -t CASES < <(
  "$PYTHON" - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in plan["diagnostic"]["cases"]:
    print(f"{case['id']}\t{case['guidance_scale']}\t{case['guidance_rescale']}")
PY
)

for method in cofitok dense_identity; do
  if [[ "$method" == cofitok ]]; then
    checkpoint="$COFITOK_CHECKPOINT"
    prefix_budget=8
  else
    checkpoint="$DENSE_CHECKPOINT"
    prefix_budget=1
  fi

  for case_spec in "${CASES[@]}"; do
    IFS=$'\t' read -r case_id guidance_scale guidance_rescale <<<"$case_spec"
    case_dir="$CASE_ROOT/${method}_${case_id}"
    sampling_report="$case_dir/sampling_report.json"
    metrics_report="$case_dir/metrics/generation_metrics_report.json"

    if [[ ! -f "$sampling_report" ]]; then
      CURRENT_STAGE="sampling_${method}_${case_id}"
      write_status running "$CURRENT_STAGE" "generating 512 frozen-checkpoint diagnostic samples"
      require_gpu_idle
      "$PYTHON" scripts/generate_samples.py \
        --checkpoint "$checkpoint" \
        --output-dir "$case_dir" \
        --num-samples 512 \
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

    CURRENT_STAGE="metrics_${method}_${case_id}"
    write_status running "$CURRENT_STAGE" "evaluating matched FID/IS without precision-recall"
    require_gpu_idle
    "$PYTHON" scripts/evaluate_generation_metrics.py \
      --real-dir "$DATA_ROOT" \
      --generated-dir "$case_dir/prefix_$prefix_budget" \
      --sampling-report "$sampling_report" \
      --output-dir "$case_dir/metrics" \
      --batch-size 64 \
      --min-samples 512 \
      --seed 2027 \
      --cache-root "$EVAL_CACHE" \
      --real-cache-name imagenet256_val_50k_torch_fidelity_v04 \
      --skip-prc \
      --resume
    [[ -f "$metrics_report" ]]
  done
done

CURRENT_STAGE=summarizing
write_status running "$CURRENT_STAGE" "building the bounded matched recovery report"
"$PYTHON" scripts/build_generation_stability_sampling_recovery.py \
  --mode build \
  --project "$PROJECT" \
  --plan "$PLAN" \
  --promotion-gate "$PROMOTION_GATE" \
  --cofitok-formal-metrics "$COFITOK_FORMAL_METRICS" \
  --dense-formal-metrics "$DENSE_FORMAL_METRICS" \
  --case-root "$CASE_ROOT" \
  --expected-diagnostic-revision "$EXPECTED_SAMPLING_RECOVERY_REVISION" \
  --expected-diagnostic-branch "$EXPECTED_SAMPLING_RECOVERY_BRANCH" \
  --output "$SUMMARY"

CURRENT_STAGE=complete
write_status pass "$CURRENT_STAGE" "matched 512-sample recovery sweep completed; no training authorization was issued"
trap - EXIT
printf 'sampling-recovery diagnostic complete: %s\n' "$SUMMARY"
