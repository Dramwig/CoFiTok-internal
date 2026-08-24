#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the isolated residual-alignment checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact residual-alignment revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact residual-alignment tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact residual-alignment branch}
PREPARATION_REPORT=${PREPARATION_REPORT:?set the immutable preparation report}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set the preparation SHA256}
EXECUTION_AUTHORIZATION=${EXECUTION_AUTHORIZATION:?set the execution authorization}
EXPECTED_EXECUTION_AUTHORIZATION_SHA256=${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:?set the authorization SHA256}
EXPECTED_RUNBOOK_SHA256=${EXPECTED_RUNBOOK_SHA256:?set this runbook SHA256}
OUTPUT_ROOT=${OUTPUT_ROOT:?set the versioned output root}
EXPECTED_OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/semantic_residual_alignment_four_arm_probe1k_v1
LOCK_DIR=${OUTPUT_ROOT}.lock

CONTROL_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json
RESIDUAL_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classresidualalign_k8_probe1k.json
CONTROL_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json
RESIDUAL_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classresidualalign_dense_probe1k.json

cd "$PROJECT"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git rev-parse 'HEAD^{tree}')" == "$EXPECTED_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$EXECUTION_AUTHORIZATION" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$EXPECTED_RUNBOOK_SHA256" ]]

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing residual-alignment probe while any GPU compute process is active\n' >&2
  exit 9
fi
test ! -e "$OUTPUT_ROOT"
test ! -e "$LOCK_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'residual-alignment execution lock already exists: %s\n' "$LOCK_DIR" >&2
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
recomputed_preparation="$temporary_dir/preparation.json"
"$PYTHON" scripts/prepare_generation_semantic_residual_alignment_probe.py \
  --control-cofitok "$CONTROL_COFITOK" \
  --control-dense "$CONTROL_DENSE" \
  --residual-cofitok "$RESIDUAL_COFITOK" \
  --residual-dense "$RESIDUAL_DENSE" \
  --output-root "$OUTPUT_ROOT" \
  --output "$recomputed_preparation"
[[ "$(sha256sum "$recomputed_preparation" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
cmp -- "$PREPARATION_REPORT" "$recomputed_preparation"

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" scripts/verify_generation_semantic_residual_alignment_execution_authorization.py \
    --authorization "$EXECUTION_AUTHORIZATION" \
    --expected-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-tree "$EXPECTED_TREE" \
    --expected-branch "$EXPECTED_BRANCH" \
    --expected-output-root "$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT/reports"
cp -- "$PREPARATION_REPORT" "$OUTPUT_ROOT/reports/preparation.json"
cp -- "$EXECUTION_AUTHORIZATION" "$OUTPUT_ROOT/reports/execution_authorization.json"

run_one() {
  local name=$1
  local config=$2
  local run_dir="$OUTPUT_ROOT/$name"
  local audit="$OUTPUT_ROOT/reports/${name}_training_audit.json"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    printf 'refusing to start %s while another GPU compute process is active\n' "$name" >&2
    exit 11
  fi
  "$PYTHON" scripts/train_generation.py \
    --config "$config" \
    --output-dir "$run_dir"
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nice -n 10 ionice -c 3 "$PYTHON" scripts/audit_generation_training_progress.py \
      --run-dir "$run_dir" \
      --config "$config" \
      --expected-steps 1000 \
      --checkpoint-interval 250 \
      --evaluation-interval 250 \
      --required-checkpoint-steps 500,750,1000 \
      --integrity-policy required \
      --output "$audit"
}

run_one control_cofitok "$CONTROL_COFITOK"
run_one residual_cofitok "$RESIDUAL_COFITOK"
run_one control_dense_identity "$CONTROL_DENSE"
run_one residual_dense_identity "$RESIDUAL_DENSE"

"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_REVISION" "$EXPECTED_TREE" "$EXPECTED_BRANCH" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision, tree, branch = sys.argv[2:5]
runs = [
    "control_cofitok",
    "residual_cofitok",
    "control_dense_identity",
    "residual_dense_identity",
]
sources = {}
for name in runs:
    report_path = root / name / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    latest = report.get("latest_checkpoint", {})
    if (
        report.get("training_complete") is not True
        or report.get("completed_steps") != 1000
        or report.get("target_steps") != 1000
        or report.get("git")
        != {"revision": revision, "branch": branch, "dirty": False}
        or latest.get("checkpoint") != "checkpoint_step_00001000.pt"
        or latest.get("integrity_manifest")
        != "checkpoint_step_00001000.pt.integrity.json"
    ):
        raise SystemExit(f"{name} did not complete the exact clean 1K contract")
    sources[name] = {
        "training_report": report_path.resolve().as_posix(),
        "checkpoint_sha256": latest["checkpoint_sha256"],
        "final_metrics": report["final_metrics"],
    }
payload = {
    "schema_version": 1,
    "role": "generation_semantic_residual_alignment_four_arm_training_status",
    "status": "completed",
    "stage": "semantic_residual_alignment_four_arm_probe1k_v1",
    "revision": revision,
    "tree": tree,
    "branch": branch,
    "runs": sources,
    "execution_boundary": {
        "training_allowed": True,
        "training_runs": runs,
        "steps_per_run": 1000,
        "dataset": "imagenet_256_10pct",
        "required_checkpoint_steps": [500, 750, 1000],
        "held_out_cpu_evaluation_allowed": True,
        "generation_sampling_allowed": False,
        "checkpoint_promotion_allowed": False,
        "followup_training_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "inference_export_allowed": False,
        "release_allowed": False,
        "unrelated_process_signaling_allowed": False,
    },
    "claim_boundary": {
        "diagnostic_non_authorizing": True,
        "training_quality_claim_allowed": False,
        "sample_quality_claim_allowed": False,
        "cofitok_specific_advantage_claim_allowed": False,
        "generation_advantage_proven": False,
        "checkpoint_promotion_allowed": False,
        "followup_training_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "inference_export_allowed": False,
        "release_allowed": False,
        "process_signal_allowed": False,
    },
    "generation_advantage_proven": False,
}
target = root / "reports" / "training_status.json"
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

training_status="$OUTPUT_ROOT/reports/training_status.json"
postevaluation_root="$OUTPUT_ROOT/reports/semantic_residual_alignment_posteval_v1"
mkdir -p "$postevaluation_root"

run_sensitivity() {
  local run=$1
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nice -n 19 ionice -c 3 "$PYTHON" \
      scripts/evaluate_generation_conditioning_sensitivity.py \
      --checkpoint "$OUTPUT_ROOT/$run/checkpoint_step_00001000.pt" \
      --output-dir "$postevaluation_root/$run" \
      --weights ema \
      --num-samples 8 \
      --start-label 128 \
      --wrong-label-offset 250 \
      --timesteps 100 500 700 900 \
      --noise-seed 314159 \
      --threads 2 \
      --resume
}

run_sensitivity control_cofitok
run_sensitivity residual_cofitok
run_sensitivity control_dense_identity
run_sensitivity residual_dense_identity

[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 19 ionice -c 3 "$PYTHON" \
    scripts/build_generation_semantic_residual_alignment_posteval.py \
    --preparation-report "$PREPARATION_REPORT" \
    --execution-authorization "$EXECUTION_AUTHORIZATION" \
    --training-status "$training_status" \
    --control-cofitok-audit "$OUTPUT_ROOT/reports/control_cofitok_training_audit.json" \
    --control-cofitok-report "$postevaluation_root/control_cofitok/conditioning_sensitivity_report.json" \
    --residual-cofitok-audit "$OUTPUT_ROOT/reports/residual_cofitok_training_audit.json" \
    --residual-cofitok-report "$postevaluation_root/residual_cofitok/conditioning_sensitivity_report.json" \
    --control-dense-identity-audit "$OUTPUT_ROOT/reports/control_dense_identity_training_audit.json" \
    --control-dense-identity-report "$postevaluation_root/control_dense_identity/conditioning_sensitivity_report.json" \
    --residual-dense-identity-audit "$OUTPUT_ROOT/reports/residual_dense_identity_training_audit.json" \
    --residual-dense-identity-report "$postevaluation_root/residual_dense_identity/conditioning_sensitivity_report.json" \
    --output "$postevaluation_root/postevaluation.json" \
    --resume

chmod 0444 \
  "$postevaluation_root"/*/conditioning_sensitivity_manifest.json \
  "$postevaluation_root"/*/conditioning_sensitivity_report.json \
  "$postevaluation_root/postevaluation.json"
