#!/usr/bin/env bash
set -euo pipefail

OBSERVER_PROJECT=${OBSERVER_PROJECT:?set OBSERVER_PROJECT}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
PLAN=${PLAN:-$OBSERVER_PROJECT/configs/generation/diagnostics/conditioning_generation_pipeline_lineage_v1.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CHECKPOINT_ROOT/conditioning_pipeline_lineage_observer_v1}
OUTPUT=${OUTPUT:-$OUTPUT_ROOT/lineage_report.json}
LOCK=${LOCK:-$OUTPUT_ROOT/lineage_observer.lock}

EXPECTED_PLAN_SHA256=${EXPECTED_PLAN_SHA256:?set EXPECTED_PLAN_SHA256}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
POLL_SECONDS=${POLL_SECONDS:-10}
STALE_SECONDS=${STALE_SECONDS:-300}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-31536000}

[[ -x "$PYTHON" ]]
[[ -f "$PLAN" ]]
mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate conditioning pipeline lineage observer\n' >&2
  exit 10
fi

cd "$OBSERVER_PROJECT"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$OBSERVER_PROJECT:$OBSERVER_PROJECT/src"
exec nice -n 19 "$PYTHON" scripts/observe_generation_conditioning_pipeline_lineage.py \
  --project "$OBSERVER_PROJECT" \
  --plan "$PLAN" \
  --expected-plan-sha256 "$EXPECTED_PLAN_SHA256" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output "$OUTPUT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --poll-seconds "$POLL_SECONDS" \
  --stale-seconds "$STALE_SECONDS" \
  --timeout-seconds "$TIMEOUT_SECONDS"
