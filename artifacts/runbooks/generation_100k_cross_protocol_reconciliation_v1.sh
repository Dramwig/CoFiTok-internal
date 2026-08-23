#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the locked evaluator checkout}
QUALITY_ROOT=${QUALITY_ROOT:?set QUALITY_ROOT to the completed 100K bridge}
TRAINING_CHECKOUT=${TRAINING_CHECKOUT:?set TRAINING_CHECKOUT to the authoritative training checkout}
DECISION=${DECISION:?set DECISION to the versioned authoritative decision}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:?set EXPECTED_DECISION_SHA256}
QUALITY_RESULT=${QUALITY_RESULT:?set QUALITY_RESULT to quality_bridge_result.json}
EXPECTED_QUALITY_RESULT_SHA256=${EXPECTED_QUALITY_RESULT_SHA256:?set EXPECTED_QUALITY_RESULT_SHA256}
EXPECTED_EVALUATOR_REVISION=${EXPECTED_EVALUATOR_REVISION:?set EXPECTED_EVALUATOR_REVISION}
EXPECTED_EVALUATOR_TREE=${EXPECTED_EVALUATOR_TREE:?set EXPECTED_EVALUATOR_TREE}
EXPECTED_EVALUATOR_BRANCH=${EXPECTED_EVALUATOR_BRANCH:?set EXPECTED_EVALUATOR_BRANCH}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a versioned non-canonical report directory}
CACHE_ROOT=${CACHE_ROOT:?set CACHE_ROOT to the existing torch-fidelity cache}
PYTHON=${PYTHON:?set PYTHON to the pinned evaluation interpreter}

test -d "$PROJECT"
test -d "$QUALITY_ROOT"
test -d "$TRAINING_CHECKOUT"
test -f "$DECISION"
test -f "$QUALITY_RESULT"
test -d "$CACHE_ROOT"
test -x "$PYTHON"
mkdir -p "$(dirname "$OUTPUT_DIR")"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$PROJECT/src:$PROJECT"

cd "$PROJECT"
exec nice -n 10 ionice -c3 "$PYTHON" \
  scripts/reconcile_generation_100k_cross_protocol.py \
  --quality-bridge-root "$QUALITY_ROOT" \
  --training-checkout "$TRAINING_CHECKOUT" \
  --decision "$DECISION" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --quality-result "$QUALITY_RESULT" \
  --expected-quality-result-sha256 "$EXPECTED_QUALITY_RESULT_SHA256" \
  --expected-evaluator-revision "$EXPECTED_EVALUATOR_REVISION" \
  --expected-evaluator-tree "$EXPECTED_EVALUATOR_TREE" \
  --expected-evaluator-branch "$EXPECTED_EVALUATOR_BRANCH" \
  --output-dir "$OUTPUT_DIR" \
  --cache-root "$CACHE_ROOT" \
  --batch-size 64 \
  --torch-num-threads 8 \
  --resume
