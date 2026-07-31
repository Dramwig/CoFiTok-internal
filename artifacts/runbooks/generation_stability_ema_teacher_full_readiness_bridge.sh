#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_READINESS_SHA256=${EXPECTED_READINESS_SHA256:?set the immutable source readiness SHA256}
EXPECTED_SOURCE_DEPLOYMENT_RECEIPT_SHA256=${EXPECTED_SOURCE_DEPLOYMENT_RECEIPT_SHA256:?set the source readiness deployment receipt SHA256}
EXPECTED_TARGET_DEPLOYMENT_RECEIPT_SHA256=${EXPECTED_TARGET_DEPLOYMENT_RECEIPT_SHA256:?set the target training deployment receipt SHA256}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:?set the source readiness revision}
EXPECTED_SOURCE_BRANCH=${EXPECTED_SOURCE_BRANCH:-scale/generation-large-capacity}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the target training revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:-scale/generation-large-capacity}

REPORT_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher/reports"
READINESS="$REPORT_ROOT/full_training_readiness.json"
BRIDGE="$REPORT_ROOT/full_training_readiness_bridge.json"
SOURCE_DEPLOYMENT_RECEIPT="$CHECKPOINT_ROOT/deployment/large_capacity/deployments/$EXPECTED_SOURCE_REVISION/deployment_receipt.json"
TARGET_DEPLOYMENT_RECEIPT="$CHECKPOINT_ROOT/deployment/large_capacity/deployments/$EXPECTED_TARGET_REVISION/deployment_receipt.json"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$READINESS" ]]
[[ -f "$SOURCE_DEPLOYMENT_RECEIPT" ]]
[[ -f "$TARGET_DEPLOYMENT_RECEIPT" ]]
[[ "$(sha256sum "$READINESS" | awk '{print $1}')" == "$EXPECTED_READINESS_SHA256" ]]
[[ "$(sha256sum "$SOURCE_DEPLOYMENT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_SOURCE_DEPLOYMENT_RECEIPT_SHA256" ]]
[[ "$(sha256sum "$TARGET_DEPLOYMENT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_TARGET_DEPLOYMENT_RECEIPT_SHA256" ]]
if [[ -e "$BRIDGE" ]]; then
  printf 'refusing to overwrite stability-full readiness bridge: %s\n' "$BRIDGE" >&2
  exit 8
fi
mkdir -p "$REPORT_ROOT"

"$PYTHON" scripts/build_generation_full_readiness_bridge.py \
  --project-root "$PROJECT" \
  --readiness "$READINESS" \
  --expected-readiness-sha256 "$EXPECTED_READINESS_SHA256" \
  --source-deployment-receipt "$SOURCE_DEPLOYMENT_RECEIPT" \
  --expected-source-deployment-receipt-sha256 "$EXPECTED_SOURCE_DEPLOYMENT_RECEIPT_SHA256" \
  --target-deployment-receipt "$TARGET_DEPLOYMENT_RECEIPT" \
  --expected-target-deployment-receipt-sha256 "$EXPECTED_TARGET_DEPLOYMENT_RECEIPT_SHA256" \
  --expected-source-revision "$EXPECTED_SOURCE_REVISION" \
  --expected-source-branch "$EXPECTED_SOURCE_BRANCH" \
  --expected-target-revision "$EXPECTED_TARGET_REVISION" \
  --expected-target-branch "$EXPECTED_TARGET_BRANCH" \
  --output "$BRIDGE" >/dev/null

sha256sum "$BRIDGE"
