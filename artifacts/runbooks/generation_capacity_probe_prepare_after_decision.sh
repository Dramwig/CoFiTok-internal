#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
QUALITY_BRIDGE_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1"
CAPACITY_PROBE_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1"
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set exact preparation revision}
EXPECTED_TARGET_TREE=${EXPECTED_TARGET_TREE:?set exact preparation tree}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set exact preparation branch}

DECISION="$QUALITY_BRIDGE_ROOT/reports/followup_experiment_decision.json"
REFERENCE_RECEIPTS="$QUALITY_BRIDGE_ROOT/reports/capacity_probe_references"
REPORT_ROOT="$CAPACITY_PROBE_ROOT/reports"
PREPARATION="$REPORT_ROOT/preparation.json"
STATUS="$REPORT_ROOT/preparation_waiter_status.json"
LOCK="$CAPACITY_PROBE_ROOT/preparation_waiter.lock"
BASE_COFITOK="$PROJECT/configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
BASE_DENSE="$PROJECT/configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json"
CAPACITY_COFITOK="$PROJECT/configs/generation/imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
CAPACITY_DENSE="$PROJECT/configs/generation/imagenet256_stability_capacity_probe_rollout_x0_u2_ema_teacher_dense_100k.json"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git rev-parse HEAD^{tree})" == "$EXPECTED_TARGET_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
mkdir -p "$REPORT_ROOT"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'capacity probe preparation waiter already owns the lock\n' >&2
  exit 75
}

"$PYTHON" scripts/wait_for_generation_capacity_probe_preparation.py \
  --followup-decision "$DECISION" \
  --preparation "$PREPARATION" \
  --status "$STATUS" \
  --expected-self-revision "$EXPECTED_TARGET_REVISION" \
  --expected-self-tree "$EXPECTED_TARGET_TREE" \
  --expected-self-branch "$EXPECTED_TARGET_BRANCH" \
  --poll-seconds 60 \
  --build-argument "base-cofitok-config=$BASE_COFITOK" \
  --build-argument "base-dense-config=$BASE_DENSE" \
  --build-argument "capacity-cofitok-config=$CAPACITY_COFITOK" \
  --build-argument "capacity-dense-config=$CAPACITY_DENSE" \
  --build-argument "cofitok-reference-receipt=$REFERENCE_RECEIPTS/cofitok_step_00010000.json" \
  --build-argument "dense-reference-receipt=$REFERENCE_RECEIPTS/dense_identity_step_00010000.json" \
  --build-argument "expected-preparation-revision=$EXPECTED_TARGET_REVISION" \
  --build-argument "expected-preparation-branch=$EXPECTED_TARGET_BRANCH" \
  --build-argument "output-root=$CAPACITY_PROBE_ROOT"

waiter_status=$("$PYTHON" - "$STATUS" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)
if [[ "$waiter_status" == completed ]]; then
  decision_sha=$(sha256sum "$DECISION" | awk '{print $1}')
  preparation_sha=$(sha256sum "$PREPARATION" | awk '{print $1}')
  "$PYTHON" scripts/verify_generation_capacity_probe_preparation.py \
    --followup-decision "$DECISION" \
    --expected-followup-decision-sha256 "$decision_sha" \
    --base-cofitok-config "$BASE_COFITOK" \
    --base-dense-config "$BASE_DENSE" \
    --capacity-cofitok-config "$CAPACITY_COFITOK" \
    --capacity-dense-config "$CAPACITY_DENSE" \
    --cofitok-reference-receipt "$REFERENCE_RECEIPTS/cofitok_step_00010000.json" \
    --dense-reference-receipt "$REFERENCE_RECEIPTS/dense_identity_step_00010000.json" \
    --expected-preparation-revision "$EXPECTED_TARGET_REVISION" \
    --expected-preparation-branch "$EXPECTED_TARGET_BRANCH" \
    --output-root "$CAPACITY_PROBE_ROOT" \
    --preparation "$PREPARATION" \
    --expected-preparation-sha256 "$preparation_sha"
elif [[ "$waiter_status" != not_selected ]]; then
  printf 'unexpected capacity preparation waiter status: %s\n' "$waiter_status" >&2
  exit 78
fi

printf 'capacity probe preparation waiter terminal status: %s\n' "$STATUS"
