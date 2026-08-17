#!/usr/bin/env bash
set -euo pipefail

OBSERVER_PROJECT=${OBSERVER_PROJECT:?set OBSERVER_PROJECT}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
QUALITY_BRIDGE_ROOT=${QUALITY_BRIDGE_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1}
PLAN=${PLAN:-$OBSERVER_PROJECT/configs/generation/diagnostics/capacity_generation_pipeline_lineage_v1.json}
OUTPUT=${OUTPUT:-$SOURCE_OUTPUT_ROOT/reports/capacity_generation_pipeline_lineage_observer.json}
LOCK=${LOCK:-$SOURCE_OUTPUT_ROOT/capacity_generation_pipeline_lineage_observer.lock}

EXPECTED_PLAN_SHA256=${EXPECTED_PLAN_SHA256:?set EXPECTED_PLAN_SHA256}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
EXPECTED_FORMAL_REVISION=${EXPECTED_FORMAL_REVISION:?set EXPECTED_FORMAL_REVISION}
EXPECTED_FORMAL_BRANCH=${EXPECTED_FORMAL_BRANCH:?set EXPECTED_FORMAL_BRANCH}
EXPECTED_FORMAL_PORCELAIN_COUNT=${EXPECTED_FORMAL_PORCELAIN_COUNT:?set EXPECTED_FORMAL_PORCELAIN_COUNT}
EXPECTED_FORMAL_PORCELAIN_SHA256=${EXPECTED_FORMAL_PORCELAIN_SHA256:?set EXPECTED_FORMAL_PORCELAIN_SHA256}
RECOVERY_SUPERSESSION_CONTRACT=${RECOVERY_SUPERSESSION_CONTRACT:?set RECOVERY_SUPERSESSION_CONTRACT}
EXPECTED_RECOVERY_SUPERSESSION_CONTRACT_SHA256=${EXPECTED_RECOVERY_SUPERSESSION_CONTRACT_SHA256:?set EXPECTED_RECOVERY_SUPERSESSION_CONTRACT_SHA256}
POLL_SECONDS=${POLL_SECONDS:-60}
STALE_SECONDS=${STALE_SECONDS:-600}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-31536000}

[[ -x "$PYTHON" ]]
[[ -f "$PLAN" ]]
[[ -f "$RECOVERY_SUPERSESSION_CONTRACT" ]]
mkdir -p "$(dirname "$OUTPUT")"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate capacity pipeline lineage observer\n' >&2
  exit 10
fi

cd "$OBSERVER_PROJECT"
export PYTHONPATH="$OBSERVER_PROJECT:$OBSERVER_PROJECT/src"
exec nice -n 19 "$PYTHON" scripts/observe_generation_capacity_pipeline_lineage.py \
  --project "$OBSERVER_PROJECT" \
  --plan "$PLAN" \
  --expected-plan-sha256 "$EXPECTED_PLAN_SHA256" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --source-output-root "$SOURCE_OUTPUT_ROOT" \
  --quality-bridge-root "$QUALITY_BRIDGE_ROOT" \
  --formal-project "$FORMAL_PROJECT" \
  --output "$OUTPUT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --expected-formal-revision "$EXPECTED_FORMAL_REVISION" \
  --expected-formal-branch "$EXPECTED_FORMAL_BRANCH" \
  --expected-formal-porcelain-count "$EXPECTED_FORMAL_PORCELAIN_COUNT" \
  --expected-formal-porcelain-sha256 "$EXPECTED_FORMAL_PORCELAIN_SHA256" \
  --recovery-supersession-contract "$RECOVERY_SUPERSESSION_CONTRACT" \
  --expected-recovery-supersession-contract-sha256 "$EXPECTED_RECOVERY_SUPERSESSION_CONTRACT_SHA256" \
  --poll-seconds "$POLL_SECONDS" \
  --stale-seconds "$STALE_SECONDS" \
  --timeout-seconds "$TIMEOUT_SECONDS"
