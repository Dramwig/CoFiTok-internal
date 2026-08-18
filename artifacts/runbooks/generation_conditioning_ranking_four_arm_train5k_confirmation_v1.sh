#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the isolated training-confirmation checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact training-confirmation revision}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact training-confirmation branch}
SAMPLING_VALIDATION=${SAMPLING_VALIDATION:?set the shared-pass 5K sampling validation}
EXPECTED_SAMPLING_VALIDATION_SHA256=${EXPECTED_SAMPLING_VALIDATION_SHA256:?set its SHA256}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set the standing authorization record}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set its SHA256}
PREPARATION_REPORT=${PREPARATION_REPORT:?set the immutable training preparation}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set its SHA256}
IDLE_GPU_EVIDENCE=${IDLE_GPU_EVIDENCE:?set the five-poll idle GPU evidence}
EXPECTED_IDLE_GPU_EVIDENCE_SHA256=${EXPECTED_IDLE_GPU_EVIDENCE_SHA256:?set its SHA256}
EXECUTION_RECEIPT=${EXECUTION_RECEIPT:?set the source-bound execution receipt}
EXPECTED_EXECUTION_RECEIPT_SHA256=${EXPECTED_EXECUTION_RECEIPT_SHA256:?set its SHA256}
EXPECTED_RUNBOOK_SHA256=${EXPECTED_RUNBOOK_SHA256:?set the runbook SHA256}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_train5k_confirmation_v1}
LOCK_DIR=${OUTPUT_ROOT}.lock
RUNBOOK_PATH=$PROJECT/artifacts/runbooks/generation_conditioning_ranking_four_arm_train5k_confirmation_v1.sh

CONTROL_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json
CONTROL_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json
RANKED_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classrank_k8_confirm5k.json
RANKED_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classrank_dense_confirm5k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$RUNBOOK_PATH" ]]
[[ -f "$SAMPLING_VALIDATION" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$IDLE_GPU_EVIDENCE" ]]
[[ -f "$EXECUTION_RECEIPT" ]]
[[ "$(sha256sum "$RUNBOOK_PATH" | awk '{print $1}')" == "$EXPECTED_RUNBOOK_SHA256" ]]
[[ "$(sha256sum "$SAMPLING_VALIDATION" | awk '{print $1}')" == "$EXPECTED_SAMPLING_VALIDATION_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$IDLE_GPU_EVIDENCE" | awk '{print $1}')" == "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_RECEIPT" | awk '{print $1}')" == "$EXPECTED_EXECUTION_RECEIPT_SHA256" ]]

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing four-arm 5K training confirmation while any GPU compute process is active\n' >&2
  exit 9
fi
test ! -e "$OUTPUT_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'training-confirmation lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 10
fi
temporary_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$temporary_dir"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONPATH=.:src
export PYTHONDONTWRITEBYTECODE=1
recomputed_preparation=$temporary_dir/preparation.json
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" scripts/prepare_generation_conditioning_ranking_training_confirmation.py \
    --sampling-validation "$SAMPLING_VALIDATION" \
    --expected-sampling-validation-sha256 "$EXPECTED_SAMPLING_VALIDATION_SHA256" \
    --standing-authorization "$STANDING_AUTHORIZATION" \
    --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
    --control-cofitok "$CONTROL_COFITOK" \
    --control-dense "$CONTROL_DENSE" \
    --ranked-cofitok "$RANKED_COFITOK" \
    --ranked-dense "$RANKED_DENSE" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-branch "$EXPECTED_BRANCH" \
    --output-root "$OUTPUT_ROOT" \
    --output "$recomputed_preparation" >/dev/null
[[ "$(sha256sum "$recomputed_preparation" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
cmp -- "$PREPARATION_REPORT" "$recomputed_preparation"

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" scripts/verify_generation_conditioning_ranking_training_confirmation_execution_receipt.py \
    --receipt "$EXECUTION_RECEIPT" \
    --expected-receipt-sha256 "$EXPECTED_EXECUTION_RECEIPT_SHA256" \
    --preparation "$PREPARATION_REPORT" \
    --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
    --sampling-validation "$SAMPLING_VALIDATION" \
    --expected-sampling-validation-sha256 "$EXPECTED_SAMPLING_VALIDATION_SHA256" \
    --standing-authorization "$STANDING_AUTHORIZATION" \
    --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
    --idle-gpu-evidence "$IDLE_GPU_EVIDENCE" \
    --expected-idle-gpu-evidence-sha256 "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" \
    --control-cofitok "$CONTROL_COFITOK" \
    --control-dense "$CONTROL_DENSE" \
    --ranked-cofitok "$RANKED_COFITOK" \
    --ranked-dense "$RANKED_DENSE" \
    --runbook "$RUNBOOK_PATH" \
    --expected-runbook-sha256 "$EXPECTED_RUNBOOK_SHA256" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-branch "$EXPECTED_BRANCH" \
    --expected-output-root "$OUTPUT_ROOT" >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing training confirmation because the GPU became busy during preflight\n' >&2
  exit 11
fi
[[ -z "$(git status --porcelain)" ]]
[[ "$(sha256sum "$SAMPLING_VALIDATION" | awk '{print $1}')" == "$EXPECTED_SAMPLING_VALIDATION_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$IDLE_GPU_EVIDENCE" | awk '{print $1}')" == "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_RECEIPT" | awk '{print $1}')" == "$EXPECTED_EXECUTION_RECEIPT_SHA256" ]]

mkdir -p "$OUTPUT_ROOT/reports"
install_bound_copy() {
  local source=$1
  local target=$2
  local expected_sha=$3
  if [[ -e "$target" ]]; then
    [[ -f "$target" ]]
    [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]
    cmp -- "$source" "$target"
  else
    cp -- "$source" "$target"
    [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]
  fi
}
install_bound_copy "$SAMPLING_VALIDATION" "$OUTPUT_ROOT/reports/source_sampling_validation.json" "$EXPECTED_SAMPLING_VALIDATION_SHA256"
install_bound_copy "$STANDING_AUTHORIZATION" "$OUTPUT_ROOT/reports/standing_authorization.json" "$EXPECTED_STANDING_AUTHORIZATION_SHA256"
install_bound_copy "$PREPARATION_REPORT" "$OUTPUT_ROOT/reports/preparation.json" "$EXPECTED_PREPARATION_SHA256"
install_bound_copy "$IDLE_GPU_EVIDENCE" "$OUTPUT_ROOT/reports/idle_gpu_evidence.json" "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256"
install_bound_copy "$EXECUTION_RECEIPT" "$OUTPUT_ROOT/reports/execution_receipt.json" "$EXPECTED_EXECUTION_RECEIPT_SHA256"

run_one() {
  local name=$1
  local config=$2
  local run_dir=$OUTPUT_ROOT/$name
  local audit=$OUTPUT_ROOT/reports/${name}_training_audit.json
  test ! -e "$run_dir"
  "$PYTHON" scripts/train_generation.py \
    --config "$config" \
    --output-dir "$run_dir"
  "$PYTHON" scripts/audit_generation_training_progress.py \
    --run-dir "$run_dir" \
    --config "$config" \
    --expected-steps 5000 \
    --checkpoint-interval 1250 \
    --evaluation-interval 1250 \
    --required-checkpoint-steps 1250,2500,5000 \
    --integrity-policy required \
    --output "$audit"
}

run_one control_cofitok "$CONTROL_COFITOK"
run_one ranked_cofitok "$RANKED_COFITOK"
run_one control_dense_identity "$CONTROL_DENSE"
run_one ranked_dense_identity "$RANKED_DENSE"

"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_REVISION" "$EXPECTED_BRANCH" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
revision = sys.argv[2]
branch = sys.argv[3]
runs = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)


def identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


sources = {}
for name in runs:
    run_dir = root / name
    report_path = run_dir / "training_report.json"
    audit_path = root / "reports" / f"{name}_training_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    latest = report.get("latest_checkpoint", {})
    checkpoint = run_dir / str(latest.get("checkpoint", ""))
    integrity = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    if (
        report.get("training_complete") is not True
        or report.get("completed_steps") != 5000
        or report.get("target_steps") != 5000
        or report.get("git", {}).get("revision") != revision
        or report.get("git", {}).get("branch") != branch
        or report.get("git", {}).get("dirty") is not False
        or latest.get("step") != 5000
        or report.get("final_metrics", {}).get("step") != 5000
        or report.get("final_metrics", {}).get("samples_seen") != 320000
        or not checkpoint.is_file()
        or not integrity.is_file()
        or not audit_path.is_file()
    ):
        raise SystemExit(f"{name} did not complete the exact fresh 5K contract")
    checkpoint_identity = identity(checkpoint)
    if (
        checkpoint_identity["bytes"] != latest.get("checkpoint_bytes")
        or checkpoint_identity["sha256"] != latest.get("checkpoint_sha256")
    ):
        raise SystemExit(f"{name} terminal checkpoint identity differs")
    sources[name] = {
        "run_dir": run_dir.as_posix(),
        "training_report": identity(report_path),
        "training_audit": identity(audit_path),
        "checkpoint": checkpoint_identity,
        "integrity_manifest": identity(integrity),
        "final_metrics": report["final_metrics"],
    }

payload = {
    "schema_version": 1,
    "status": "completed",
    "role": "generation_conditioning_ranking_four_arm_train5k_confirmation_status",
    "stage": "conditioning_ranking_four_arm_train5k_confirmation_v1",
    "revision": revision,
    "branch": branch,
    "output_root": root.as_posix(),
    "runs": sources,
    "execution_boundary": {
        "fresh_four_arm_training_completed": True,
        "steps_per_run": 5000,
        "effective_batch_size": 64,
        "automatic_relaunch_allowed": False,
        "sampling_allowed": False,
        "checkpoint_promotion_allowed": False,
        "followup_training_allowed": False,
        "full_100k_or_300k_launch_allowed": False,
        "release_authorization_allowed": False,
    },
    "claim_boundary": {
        "diagnostic_only": True,
        "training_quality_claim_allowed": False,
        "sample_quality_claim_allowed": False,
        "cofitok_specific_advantage_claim_allowed": False,
        "broad_generation_superiority_claim_allowed": False,
        "authorizes_followup_evaluation": False,
        "authorizes_followup_training": False,
        "authorizes_full_100k_or_300k": False,
        "authorizes_release": False,
        "required_next_evidence": "separately_source_bound_matched_5k_heldout_evaluation",
    },
}
target = root / "reports" / "training_status.json"
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
print(target)
PY

chmod 0444 \
  "$OUTPUT_ROOT/reports/source_sampling_validation.json" \
  "$OUTPUT_ROOT/reports/standing_authorization.json" \
  "$OUTPUT_ROOT/reports/preparation.json" \
  "$OUTPUT_ROOT/reports/idle_gpu_evidence.json" \
  "$OUTPUT_ROOT/reports/execution_receipt.json" \
  "$OUTPUT_ROOT/reports/training_status.json"

printf 'completed fresh four-arm conditioning-ranking 5K training confirmation: %s\n' \
  "$OUTPUT_ROOT/reports/training_status.json"
