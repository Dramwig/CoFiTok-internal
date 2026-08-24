#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the isolated terminal-rebind checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact terminal-rebind revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact terminal-rebind tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact terminal-rebind branch}
PREPARATION_REPORT=${PREPARATION_REPORT:?set the immutable rebind preparation}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set the preparation SHA256}
LEGACY_PREPARATION=${LEGACY_PREPARATION:?set the immutable legacy preparation}
EXPECTED_LEGACY_PREPARATION_SHA256=${EXPECTED_LEGACY_PREPARATION_SHA256:?set the legacy preparation SHA256}
EXECUTION_AUTHORIZATION=${EXECUTION_AUTHORIZATION:?set the source-bound terminal-rebind authorization}
EXPECTED_EXECUTION_AUTHORIZATION_SHA256=${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:?set the authorization SHA256}
EXPECTED_RUNBOOK_SHA256=${EXPECTED_RUNBOOK_SHA256:?set this runbook SHA256}
OUTPUT_ROOT=${OUTPUT_ROOT:?set the new versioned terminal-rebind output root}
EXPECTED_OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_probe1k_terminal_rebind_v2
LOCK_DIR=${OUTPUT_ROOT}.lock

CONTROL_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json
CONTROL_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json
RANKED_COFITOK=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classrank_k8_probe1k.json
RANKED_DENSE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classrank_dense_probe1k.json

cd "$PROJECT"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git rev-parse 'HEAD^{tree}')" == "$EXPECTED_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$LEGACY_PREPARATION" ]]
[[ -f "$EXECUTION_AUTHORIZATION" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$LEGACY_PREPARATION" | awk '{print $1}')" == "$EXPECTED_LEGACY_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$EXPECTED_RUNBOOK_SHA256" ]]

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing terminal-rebind probe while any GPU compute process is active\n' >&2
  exit 9
fi
test ! -e "$OUTPUT_ROOT"
test ! -e "$LOCK_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'terminal-rebind probe lock already exists: %s\n' "$LOCK_DIR" >&2
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
"$PYTHON" scripts/prepare_generation_conditioning_ranking_terminal_rebind.py \
  --control-cofitok "$CONTROL_COFITOK" \
  --control-dense "$CONTROL_DENSE" \
  --ranked-cofitok "$RANKED_COFITOK" \
  --ranked-dense "$RANKED_DENSE" \
  --legacy-preparation "$LEGACY_PREPARATION" \
  --expected-legacy-preparation-sha256 "$EXPECTED_LEGACY_PREPARATION_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --output "$recomputed_preparation"
[[ "$(sha256sum "$recomputed_preparation" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
cmp -- "$PREPARATION_REPORT" "$recomputed_preparation"

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" scripts/verify_generation_conditioning_ranking_terminal_rebind_authorization.py \
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
  "$PYTHON" scripts/train_generation.py \
    --config "$config" \
    --output-dir "$run_dir"
  "$PYTHON" scripts/audit_generation_training_progress.py \
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
run_one ranked_cofitok "$RANKED_COFITOK"
run_one control_dense_identity "$CONTROL_DENSE"
run_one ranked_dense_identity "$RANKED_DENSE"

"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_REVISION" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
runs = [
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
]
sources = {}
for name in runs:
    report_path = root / name / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("training_complete") is not True
        or report.get("completed_steps") != 1000
        or report.get("target_steps") != 1000
        or report.get("git", {}).get("revision") != revision
        or report.get("git", {}).get("dirty") is not False
    ):
        raise SystemExit(f"{name} did not complete the exact clean 1K contract")
    sources[name] = {
        "training_report": report_path.resolve().as_posix(),
        "checkpoint_sha256": report["latest_checkpoint"]["checkpoint_sha256"],
        "final_metrics": report["final_metrics"],
    }
payload = {
    "schema_version": 1,
    "status": "completed",
    "role": "generation_conditioning_ranking_four_arm_probe_training_status",
    "stage": "conditioning_ranking_four_arm_probe1k_terminal_rebind_v2",
    "revision": revision,
    "runs": sources,
    "generation_advantage_proven": False,
    "claim_boundary": {
        "semantic_alignment_screening_only": True,
        "absolute_fid_or_recall_recovery_tested": False,
        "training_quality_claim_allowed": False,
        "sample_quality_claim_allowed": False,
        "promotion_authorization_allowed": False,
        "followup_training_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "export_allowed": False,
        "release_allowed": False,
        "required_next_evidence": "held-out matched conditioning sensitivity evaluation",
    },
}
target = root / "reports" / "training_status.json"
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(target)
PY

training_status=$OUTPUT_ROOT/reports/training_status.json
training_status_sha256=$(sha256sum "$training_status" | awk '{print $1}')
PROJECT="$PROJECT" \
PYTHON="$PYTHON" \
EXPECTED_REVISION="$EXPECTED_REVISION" \
EXPECTED_BRANCH="$EXPECTED_BRANCH" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
PREPARATION_REPORT="$PREPARATION_REPORT" \
EXPECTED_PREPARATION_SHA256="$EXPECTED_PREPARATION_SHA256" \
TRAINING_STATUS="$training_status" \
EXPECTED_TRAINING_STATUS_SHA256="$training_status_sha256" \
bash artifacts/runbooks/generation_conditioning_ranking_four_arm_posteval_v1.sh
