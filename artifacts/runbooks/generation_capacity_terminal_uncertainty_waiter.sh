#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact capacity uncertainty control checkout}
EVALUATOR_PROJECT=${EVALUATOR_PROJECT:?set EVALUATOR_PROJECT to the exact matched uncertainty evaluator checkout}
SOURCE_KIND=${SOURCE_KIND:?set SOURCE_KIND to capacity_completion_100k or capacity_full_300k}
SOURCE_ANCHOR=${SOURCE_ANCHOR:?set SOURCE_ANCHOR to the exact future terminal source path}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to the dedicated uncertainty output root}
REAL_FEATURE_CACHE_SOURCE=${REAL_FEATURE_CACHE_SOURCE:?set REAL_FEATURE_CACHE_SOURCE}
EXPECTED_REAL_FEATURE_CACHE_BYTES=${EXPECTED_REAL_FEATURE_CACHE_BYTES:?set EXPECTED_REAL_FEATURE_CACHE_BYTES}
EXPECTED_REAL_FEATURE_CACHE_SHA256=${EXPECTED_REAL_FEATURE_CACHE_SHA256:?set EXPECTED_REAL_FEATURE_CACHE_SHA256}
EXPECTED_CONTROL_REVISION=${EXPECTED_CONTROL_REVISION:?set EXPECTED_CONTROL_REVISION}
EXPECTED_CONTROL_TREE=${EXPECTED_CONTROL_TREE:?set EXPECTED_CONTROL_TREE}
EXPECTED_CONTROL_BRANCH=${EXPECTED_CONTROL_BRANCH:?set EXPECTED_CONTROL_BRANCH}
EXPECTED_EVALUATOR_REVISION=${EXPECTED_EVALUATOR_REVISION:?set EXPECTED_EVALUATOR_REVISION}
EXPECTED_EVALUATOR_TREE=${EXPECTED_EVALUATOR_TREE:?set EXPECTED_EVALUATOR_TREE}
EXPECTED_EVALUATOR_BRANCH=${EXPECTED_EVALUATOR_BRANCH-}
EXPECTED_EXECUTION_REVISION=${EXPECTED_EXECUTION_REVISION-}
EXPECTED_EXECUTION_TREE=${EXPECTED_EXECUTION_TREE-}
EXPECTED_EXECUTION_BRANCH=${EXPECTED_EXECUTION_BRANCH-}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_RESULT_REVISION=${EXPECTED_RESULT_REVISION-}
EXPECTED_RESULT_TREE=${EXPECTED_RESULT_TREE-}
EXPECTED_RESULT_BRANCH=${EXPECTED_RESULT_BRANCH-}
EXPECTED_SOURCE_EVALUATION_REVISION=${EXPECTED_SOURCE_EVALUATION_REVISION-}
EXPECTED_SOURCE_EVALUATION_TREE=${EXPECTED_SOURCE_EVALUATION_TREE-}
EXPECTED_SOURCE_EVALUATION_BRANCH=${EXPECTED_SOURCE_EVALUATION_BRANCH-}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
GPU_SLOT_LOCK_TARGET=${GPU_SLOT_LOCK_TARGET:-/root/autodl-tmp/CoFiTok/checkpoints/generation/matched_uncertainty_shared_gpu_slot}
STATUS=${STATUS:-$OUTPUT_ROOT/reports/waiter_status.json}
PID_FILE=${PID_FILE:-$OUTPUT_ROOT/reports/waiter.pid}
MANIFEST=${MANIFEST:-$OUTPUT_ROOT/reports/execution_manifest.json}
QUALIFICATION=${QUALIFICATION:-$OUTPUT_ROOT/reports/statistical_claim_qualification.json}
POLL_SECONDS=${POLL_SECONDS:-60}
REQUIRED_IDLE_POLLS=${REQUIRED_IDLE_POLLS:-5}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-31536000}

[[ "$SOURCE_KIND" == "capacity_completion_100k" || "$SOURCE_KIND" == "capacity_full_300k" ]]
[[ -x "$PYTHON" ]]
[[ -d "$PROJECT/.git" || -f "$PROJECT/.git" ]]
[[ -d "$EVALUATOR_PROJECT/.git" || -f "$EVALUATOR_PROJECT/.git" ]]
[[ -f "$REAL_FEATURE_CACHE_SOURCE" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_CONTROL_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_CONTROL_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_CONTROL_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
[[ "$(git -C "$EVALUATOR_PROJECT" rev-parse HEAD)" == "$EXPECTED_EVALUATOR_REVISION" ]]
[[ "$(git -C "$EVALUATOR_PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_EVALUATOR_TREE" ]]
[[ "$(git -C "$EVALUATOR_PROJECT" branch --show-current)" == "$EXPECTED_EVALUATOR_BRANCH" ]]
[[ -z "$(git -C "$EVALUATOR_PROJECT" status --porcelain)" ]]
[[ "$(stat -c %s "$REAL_FEATURE_CACHE_SOURCE")" == "$EXPECTED_REAL_FEATURE_CACHE_BYTES" ]]
[[ "$(sha256sum "$REAL_FEATURE_CACHE_SOURCE" | awk '{print $1}')" == "$EXPECTED_REAL_FEATURE_CACHE_SHA256" ]]

mkdir -p "$OUTPUT_ROOT/reports"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

exec nice -n 10 "$PYTHON" scripts/run_generation_capacity_terminal_uncertainty_waiter.py \
  --project "$PROJECT" \
  --evaluator-project "$EVALUATOR_PROJECT" \
  --source-kind "$SOURCE_KIND" \
  --source-anchor "$SOURCE_ANCHOR" \
  --output-root "$OUTPUT_ROOT" \
  --status-output "$STATUS" \
  --pid-file "$PID_FILE" \
  --manifest-output "$MANIFEST" \
  --qualification-output "$QUALIFICATION" \
  --gpu-slot-lock-target "$GPU_SLOT_LOCK_TARGET" \
  --real-feature-cache-source "$REAL_FEATURE_CACHE_SOURCE" \
  --expected-real-feature-cache-bytes "$EXPECTED_REAL_FEATURE_CACHE_BYTES" \
  --expected-real-feature-cache-sha256 "$EXPECTED_REAL_FEATURE_CACHE_SHA256" \
  --python-executable "$PYTHON" \
  --expected-control-revision "$EXPECTED_CONTROL_REVISION" \
  --expected-control-tree "$EXPECTED_CONTROL_TREE" \
  --expected-control-branch "$EXPECTED_CONTROL_BRANCH" \
  --expected-evaluator-revision "$EXPECTED_EVALUATOR_REVISION" \
  --expected-evaluator-tree "$EXPECTED_EVALUATOR_TREE" \
  --expected-evaluator-branch "$EXPECTED_EVALUATOR_BRANCH" \
  --expected-execution-revision "$EXPECTED_EXECUTION_REVISION" \
  --expected-execution-tree "$EXPECTED_EXECUTION_TREE" \
  --expected-execution-branch "$EXPECTED_EXECUTION_BRANCH" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-tree "$EXPECTED_TRAINING_TREE" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-result-revision "$EXPECTED_RESULT_REVISION" \
  --expected-result-tree "$EXPECTED_RESULT_TREE" \
  --expected-result-branch "$EXPECTED_RESULT_BRANCH" \
  --expected-source-evaluation-revision "$EXPECTED_SOURCE_EVALUATION_REVISION" \
  --expected-source-evaluation-tree "$EXPECTED_SOURCE_EVALUATION_TREE" \
  --expected-source-evaluation-branch "$EXPECTED_SOURCE_EVALUATION_BRANCH" \
  --poll-seconds "$POLL_SECONDS" \
  --required-idle-polls "$REQUIRED_IDLE_POLLS" \
  --timeout-seconds "$TIMEOUT_SECONDS"
