#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:?set the passing 5K decision SHA256}
EXPECTED_SCALING_GATE_SHA256=${EXPECTED_SCALING_GATE_SHA256:?set the passing 50K gate SHA256}
EXPECTED_FULL_READINESS_SHA256=${EXPECTED_FULL_READINESS_SHA256:?set the immutable full readiness SHA256}
EXPECTED_FINAL_GATE_SHA256=${EXPECTED_FINAL_GATE_SHA256:?set the passing full gate SHA256}
EXPECTED_FULL_TRAINING_REVISION=${EXPECTED_FULL_TRAINING_REVISION:?set the full training revision}
EXPECTED_FULL_TRAINING_BRANCH=${EXPECTED_FULL_TRAINING_BRANCH:?set the full training branch}
EXPECTED_FULL_EVALUATION_REVISION=${EXPECTED_FULL_EVALUATION_REVISION:?set the full evaluation revision}
EXPECTED_FULL_EVALUATION_BRANCH=${EXPECTED_FULL_EVALUATION_BRANCH:?set the full evaluation branch}
EXPECTED_AUDIT_REVISION=${EXPECTED_AUDIT_REVISION:?set the clean completion-audit revision}
EXPECTED_AUDIT_BRANCH=${EXPECTED_AUDIT_BRANCH:?set the clean completion-audit branch}

SCALING_TRAINING_REVISION=2c2c1f5166b73d4f28df93b276901671ac1a7836
SCALING_TRAINING_BRANCH=scale/generation-stability-50k-preflight
SCALING_EVALUATION_REVISION=caab51348d546e98858d1203f2958d9e396e2d18
SCALING_EVALUATION_BRANCH=scale/generation-stability-50k-posteval
REPORT_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher/reports"
OUTPUT="$REPORT_ROOT/stability_generation_completion_audit.json"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_AUDIT_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_AUDIT_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
mkdir -p "$REPORT_ROOT"

"$PYTHON" scripts/audit_generation_stability_completion.py \
  --project-root "$PROJECT" \
  --output-root "$CHECKPOINT_ROOT" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --expected-scaling-training-revision "$SCALING_TRAINING_REVISION" \
  --expected-scaling-training-branch "$SCALING_TRAINING_BRANCH" \
  --expected-scaling-evaluation-revision "$SCALING_EVALUATION_REVISION" \
  --expected-scaling-evaluation-branch "$SCALING_EVALUATION_BRANCH" \
  --expected-scaling-gate-sha256 "$EXPECTED_SCALING_GATE_SHA256" \
  --expected-full-readiness-sha256 "$EXPECTED_FULL_READINESS_SHA256" \
  --expected-full-training-revision "$EXPECTED_FULL_TRAINING_REVISION" \
  --expected-full-training-branch "$EXPECTED_FULL_TRAINING_BRANCH" \
  --expected-full-evaluation-revision "$EXPECTED_FULL_EVALUATION_REVISION" \
  --expected-full-evaluation-branch "$EXPECTED_FULL_EVALUATION_BRANCH" \
  --expected-final-gate-sha256 "$EXPECTED_FINAL_GATE_SHA256" \
  --expected-export-revision "$EXPECTED_AUDIT_REVISION" \
  --expected-export-branch "$EXPECTED_AUDIT_BRANCH" \
  --output "$OUTPUT"
