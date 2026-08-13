#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the exact reference-waiter revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the exact reference-waiter branch}
EXPECTED_BRIDGE_REVISION=${EXPECTED_BRIDGE_REVISION:?set the exact quality-bridge training revision}
EXPECTED_BRIDGE_BRANCH=${EXPECTED_BRIDGE_BRANCH:?set the exact quality-bridge training branch}

BRIDGE_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1"
COFITOK_RUN="$BRIDGE_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$BRIDGE_ROOT/dense_rollout_x0_u2_ema_teacher"
REFERENCE_ROOT="$BRIDGE_ROOT/capacity_probe_references/base128_step_00010000"
RECEIPT_ROOT="$BRIDGE_ROOT/reports/capacity_probe_references"
STATUS="$RECEIPT_ROOT/waiter_status.json"
LOCK="$BRIDGE_ROOT/capacity_probe_reference_waiter.lock"
REASON="matched base128 step-10K reference for a possible 250M capacity qualification probe"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]

mkdir -p "$RECEIPT_ROOT"
exec 9>"$LOCK"
flock -n 9 || {
  printf 'capacity reference waiter already owns the lock\n' >&2
  exit 75
}

"$PYTHON" scripts/wait_for_generation_checkpoint_references.py \
  --source "cofitok=$COFITOK_RUN/checkpoint_step_00010000.pt" \
  --source "dense_identity=$DENSE_RUN/checkpoint_step_00010000.pt" \
  --reference-root "$REFERENCE_ROOT" \
  --receipt-root "$RECEIPT_ROOT" \
  --expected-step 10000 \
  --expected-revision "$EXPECTED_BRIDGE_REVISION" \
  --expected-branch "$EXPECTED_BRIDGE_BRANCH" \
  --reason "$REASON" \
  --status "$STATUS" \
  --poll-seconds 30

printf 'capacity probe base-128 references preserved: %s\n' "$REFERENCE_ROOT"
