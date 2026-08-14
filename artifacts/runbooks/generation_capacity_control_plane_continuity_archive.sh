#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact continuity-builder checkout}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_v3}
PLAN="$PROJECT/configs/generation/diagnostics/capacity_generation_control_plane_continuity_v3.json"
PLAN_SHA256=2f0ae8faa79265c8be85128637186482b2db7e2043f860af2fc314b869164985
LOCK="$OUTPUT_ROOT.lock"

[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$EXPECTED_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain=v1 --untracked-files=no)" ]]

mkdir -p "$(dirname "$OUTPUT_ROOT")"
exec 9>"$LOCK"
flock -n 9

cd "$PROJECT"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/build_generation_control_plane_continuity_archive.py \
  --project "$PROJECT" \
  --plan "$PLAN" \
  --output-root "$OUTPUT_ROOT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH"

MANIFEST_SHA256="$(sha256sum "$OUTPUT_ROOT/manifest.json" | awk '{print $1}')"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/verify_generation_control_plane_continuity_archive.py \
  --archive-root "$OUTPUT_ROOT" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --output "$OUTPUT_ROOT/verification_report.json"

printf '%s\n' "$MANIFEST_SHA256"
