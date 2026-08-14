#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONTROL_PROCESS_RELAUNCH_ALLOWED:-false}" != "true" ]]; then
  echo "CONTROL_PROCESS_RELAUNCH_ALLOWED must equal true" >&2
  exit 2
fi
PROJECT=${PROJECT:?set PROJECT to the exact process-relaunch checkout}
EXPECTED_REVISION=${EXPECTED_REVISION:?set EXPECTED_REVISION}
EXPECTED_TREE=${EXPECTED_TREE:?set EXPECTED_TREE}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set EXPECTED_BRANCH}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
FORMAL_PROJECT=${FORMAL_PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
MANIFEST=${MANIFEST:-$CHECKPOINT_ROOT/control_plane_continuity/capacity_generation_pipeline_process_relaunch_v1/process_relaunch_manifest.json}
EXPECTED_MANIFEST_SHA256=455ddc919802c697afa4a80d43e70d79fc3d72057c8a194460772cb15f1c08c3
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:-/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json}
EXPECTED_STANDING_AUTHORIZATION_SHA256=5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
OUTPUT_ROOT=${OUTPUT_ROOT:-$CHECKPOINT_ROOT/control_plane_continuity/capacity_generation_pipeline_process_relaunch_execution_v1}
READINESS=$OUTPUT_ROOT/readiness.json
APPROVAL=$OUTPUT_ROOT/approval.json
EXECUTION=$OUTPUT_ROOT/execution.json
LOCK="$OUTPUT_ROOT.lock"

[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$EXPECTED_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain=v1 --untracked-files=no)" ]]
[[ ! -e "$OUTPUT_ROOT" ]]

mkdir -p "$(dirname "$OUTPUT_ROOT")"
exec 9>"$LOCK"
flock -n 9

cd "$PROJECT"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/build_generation_control_plane_process_relaunch_readiness.py \
  --project "$PROJECT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-tree "$EXPECTED_TREE" \
  --expected-branch "$EXPECTED_BRANCH" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --formal-project "$FORMAL_PROJECT" \
  --expected-formal-revision 1ebcc15210e63a776a2ba448481cbd8bb94a4066 \
  --expected-formal-tree 659fa94726c4aec0afef49904f82b828bb62872b \
  --expected-formal-branch scale/generative-system \
  --expected-formal-porcelain-count 87 \
  --expected-formal-porcelain-sha256 a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497 \
  --expected-process-count 17 \
  --output "$READINESS" \
  --issue-approval \
  --approval-output "$APPROVAL" \
  --require-ready

APPROVAL_SHA256="$(sha256sum "$APPROVAL" | awk '{print $1}')"
PYTHONPATH="$PROJECT:$PROJECT/src" "$PYTHON" scripts/execute_generation_control_plane_process_relaunch.py \
  --approval "$APPROVAL" \
  --expected-approval-sha256 "$APPROVAL_SHA256" \
  --output "$EXECUTION" \
  --status-timeout-seconds 30 \
  --rollback-grace-seconds 10
