#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
QUALITY_BRIDGE_ROOT=${QUALITY_BRIDGE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1}
EXPECTED_DECISION_REVISION=${EXPECTED_DECISION_REVISION:?set EXPECTED_DECISION_REVISION}
EXPECTED_DECISION_BRANCH=${EXPECTED_DECISION_BRANCH:?set EXPECTED_DECISION_BRANCH}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:-}

RESULT="$QUALITY_BRIDGE_ROOT/reports/quality_bridge_result.json"
DECISION="$QUALITY_BRIDGE_ROOT/reports/followup_experiment_decision.json"
LOCK="$QUALITY_BRIDGE_ROOT/followup_experiment_decision.lock"

cd "$PROJECT"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing concurrent quality bridge follow-up decision builder\n' >&2
  exit 75
}

[[ -f "$RESULT" ]] || {
  printf 'quality bridge terminal result is not available: %s\n' "$RESULT" >&2
  exit 76
}

result_sha=$(sha256sum "$RESULT" | awk '{print $1}')
common=(
  --quality-bridge-result "$RESULT"
  --expected-quality-bridge-result-sha256 "$result_sha"
  --expected-decision-revision "$EXPECTED_DECISION_REVISION"
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH"
)

if [[ -f "$DECISION" ]]; then
  [[ -n "$EXPECTED_DECISION_SHA256" ]] || {
    printf 'set EXPECTED_DECISION_SHA256 to replay an existing decision\n' >&2
    exit 77
  }
else
  [[ -z "$EXPECTED_DECISION_SHA256" ]] || {
    printf 'expected quality bridge follow-up decision is absent\n' >&2
    exit 77
  }
  "$PYTHON" scripts/build_generation_quality_bridge_followup_decision.py \
    "${common[@]}" \
    --output "$DECISION"
  EXPECTED_DECISION_SHA256=$(sha256sum "$DECISION" | awk '{print $1}')
fi

"$PYTHON" scripts/verify_generation_quality_bridge_followup_decision.py \
  "${common[@]}" \
  --decision "$DECISION" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256"

printf 'quality bridge follow-up decision: %s  %s\n' \
  "$EXPECTED_DECISION_SHA256" "$DECISION"
