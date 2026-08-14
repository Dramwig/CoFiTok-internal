#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact process-snapshot checkout}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1}
STATIC_ARCHIVE_ROOT=${STATIC_ARCHIVE_ROOT:-$CHECKPOINT_ROOT/control_plane_continuity/capacity_generation_pipeline_v3}
STATIC_MANIFEST="$STATIC_ARCHIVE_ROOT/manifest.json"
STATIC_MANIFEST_SHA256=925d7caf63823f0431d2c3dc31967b26a586642d614568aa937f88a42a315d74
LINEAGE_REPORT="$SOURCE_OUTPUT_ROOT/reports/capacity_generation_pipeline_lineage_observer.json"
OUTPUT_ROOT=${OUTPUT_ROOT:-$CHECKPOINT_ROOT/control_plane_continuity/capacity_generation_pipeline_process_relaunch_v1}
MANIFEST="$OUTPUT_ROOT/process_relaunch_manifest.json"
CAPTURE_VERIFICATION="$OUTPUT_ROOT/capture_live_verification.json"
FINAL_VERIFICATION="$OUTPUT_ROOT/final_live_verification.json"
LOCK="$OUTPUT_ROOT.lock"

[[ "$(sha256sum "$STATIC_MANIFEST" | awk '{print $1}')" == "$STATIC_MANIFEST_SHA256" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$EXPECTED_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain=v1 --untracked-files=no)" ]]
[[ ! -e "$OUTPUT_ROOT" ]]

mkdir -p "$(dirname "$OUTPUT_ROOT")"
exec 9>"$LOCK"
flock -n 9

cd "$PROJECT"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/capture_generation_control_plane_process_snapshot.py \
  --project "$PROJECT" \
  --lineage-report "$LINEAGE_REPORT" \
  --static-continuity-manifest "$STATIC_MANIFEST" \
  --expected-static-continuity-sha256 "$STATIC_MANIFEST_SHA256" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --expected-process-count 17 \
  --output "$MANIFEST" \
  --verification-output "$CAPTURE_VERIFICATION"

MANIFEST_SHA256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/verify_generation_control_plane_process_snapshot.py \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-process-count 17 \
  --require-live \
  --output "$FINAL_VERIFICATION"

printf '%s\n' "$MANIFEST_SHA256"
