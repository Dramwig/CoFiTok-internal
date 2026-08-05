#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
DATASET_ROOT=${DATASET_ROOT:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256}
EXPECTED_AUDIT_REVISION=${EXPECTED_AUDIT_REVISION:?set the exact frozen-support audit revision}
EXPECTED_AUDIT_BRANCH=${EXPECTED_AUDIT_BRANCH:-scale/generation-stability-frozen-sample-support-audit-v1}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_SAMPLES="$COFITOK_RUN/samples_gate10k_ddim100_cfg15"
DENSE_SAMPLES="$DENSE_RUN/samples_gate10k_ddim100_cfg15"
PROMOTION_GATE="$REPORT_ROOT/promotion_gate.json"
REAL_DIR="$DATASET_ROOT/extracted/val"
DATASET_MANIFEST="$DATASET_ROOT/metadata/image_manifest.jsonl"
AUDIT_ROOT="$REPORT_ROOT/frozen_existing_sample_support_audit"
AUDIT_REPORT="$AUDIT_ROOT/support_audit.json"
AUDIT_LOCK="$OUTPUT_ROOT/frozen_existing_sample_support_audit.lock"

cd "$PROJECT"
export PYTHONPATH=.:src
export CUDA_VISIBLE_DEVICES=""
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_AUDIT_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_AUDIT_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
for path in \
  "$PROMOTION_GATE" \
  "$DATASET_MANIFEST" \
  "$COFITOK_SAMPLES/sampling_manifest.json" \
  "$COFITOK_SAMPLES/sampling_progress.json" \
  "$COFITOK_SAMPLES/sampling_report.json" \
  "$COFITOK_SAMPLES/metrics/generation_metrics_report.json" \
  "$DENSE_SAMPLES/sampling_manifest.json" \
  "$DENSE_SAMPLES/sampling_progress.json" \
  "$DENSE_SAMPLES/sampling_report.json" \
  "$DENSE_SAMPLES/metrics/generation_metrics_report.json"; do
  [[ -f "$path" ]]
done
[[ -d "$REAL_DIR" ]]
[[ -d "$COFITOK_SAMPLES/prefix_8" ]]
[[ -d "$DENSE_SAMPLES/prefix_1" ]]
[[ "$(stat -c '%s' "$DATASET_MANIFEST")" == "405484553" ]]
[[ "$(sha256sum "$DATASET_MANIFEST" | awk '{print $1}')" == "9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0" ]]
[[ "$(sha256sum "$PROMOTION_GATE" | awk '{print $1}')" == "2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90" ]]

mkdir -p "$AUDIT_ROOT"
command -v flock >/dev/null
command -v nice >/dev/null
command -v ionice >/dev/null
exec 6>"$AUDIT_LOCK"
if ! flock -n 6; then
  printf 'refusing concurrent frozen existing-sample support audit\n' >&2
  exit 15
fi
[[ ! -e "$AUDIT_REPORT" ]]

nice -n 15 ionice -c3 "$PYTHON" scripts/build_generation_frozen_sample_support_audit.py \
  --cofitok-sampling-report "$COFITOK_SAMPLES/sampling_report.json" \
  --dense-sampling-report "$DENSE_SAMPLES/sampling_report.json" \
  --cofitok-metrics-report "$COFITOK_SAMPLES/metrics/generation_metrics_report.json" \
  --dense-metrics-report "$DENSE_SAMPLES/metrics/generation_metrics_report.json" \
  --promotion-gate "$PROMOTION_GATE" \
  --real-dir "$REAL_DIR" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output "$AUDIT_REPORT" \
  --expected-audit-revision "$EXPECTED_AUDIT_REVISION" \
  --expected-audit-branch "$EXPECTED_AUDIT_BRANCH"

"$PYTHON" - "$AUDIT_REPORT" "$EXPECTED_AUDIT_REVISION" "$EXPECTED_AUDIT_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_git = {
    "revision": sys.argv[2],
    "branch": sys.argv[3],
    "tracked_dirty": False,
    "full_clean": True,
}
boundary = report.get("claim_boundary")
if (
    report.get("schema_version") != 1
    or report.get("role")
    != "generation_frozen_existing_sample_support_audit"
    or report.get("status") != "completed"
    or report.get("audit_git") != expected_git
    or not isinstance(boundary, dict)
    or boundary.get("supplemental_non_authorizing") is not True
    or boundary.get("full_training_launch_allowed") is not False
    or boundary.get("gpu_execution_authorized") is not False
    or boundary.get("new_sampling_performed") is not False
    or boundary.get("new_training_performed") is not False
    or report.get("interpretation_policy", {}).get("thresholded_gate") is not False
):
    raise SystemExit("frozen existing-sample support audit terminal contract differs")
PY

AUDIT_SHA256="$(sha256sum "$AUDIT_REPORT" | awk '{print $1}')"
printf 'frozen existing-sample support audit: %s  %s\n' \
  "$AUDIT_SHA256" "$AUDIT_REPORT"
