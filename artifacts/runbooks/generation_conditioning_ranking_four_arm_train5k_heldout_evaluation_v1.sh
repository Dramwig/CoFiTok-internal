#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1

PROJECT=${PROJECT:?set the isolated held-out evaluator checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact held-out evaluator revision}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-scale/generation-label-ranking-5k-heldout-evaluation-v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_train5k_confirmation_v1}
PREPARATION_REPORT=${PREPARATION_REPORT:-$OUTPUT_ROOT/reports/preparation.json}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set the immutable preparation SHA256}
TRAINING_STATUS=${TRAINING_STATUS:-$OUTPUT_ROOT/reports/training_status.json}
EXPECTED_TRAINING_STATUS_SHA256=${EXPECTED_TRAINING_STATUS_SHA256:?set the completed training-status SHA256}
TRAINING_EXECUTION_REPLAY=${TRAINING_EXECUTION_REPLAY:?set the exact training-receipt replay evidence}
EXPECTED_TRAINING_EXECUTION_REPLAY_SHA256=${EXPECTED_TRAINING_EXECUTION_REPLAY_SHA256:?set the replay evidence SHA256}
POSTEVAL_ROOT=${POSTEVAL_ROOT:-$OUTPUT_ROOT/reports/conditioning_ranking_train5k_heldout_evaluation_v1}
LOCK_DIR=$OUTPUT_ROOT/reports/.conditioning_ranking_train5k_heldout_evaluation_v1.lock

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$TRAINING_STATUS" ]]
[[ -f "$TRAINING_EXECUTION_REPLAY" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$TRAINING_STATUS" | awk '{print $1}')" == "$EXPECTED_TRAINING_STATUS_SHA256" ]]
[[ "$(sha256sum "$TRAINING_EXECUTION_REPLAY" | awk '{print $1}')" == "$EXPECTED_TRAINING_EXECUTION_REPLAY_SHA256" ]]
[[ -d "$OUTPUT_ROOT" ]]
[[ ! -e "$POSTEVAL_ROOT" ]]

if pgrep -af 'scripts/train_generation.py' | grep -F "$OUTPUT_ROOT/" >/dev/null; then
  printf 'refusing held-out evaluation while a four-arm trainer is active\n' >&2
  exit 11
fi

for run in \
  control_cofitok \
  ranked_cofitok \
  control_dense_identity \
  ranked_dense_identity
do
  [[ -f "$OUTPUT_ROOT/$run/checkpoint_step_00005000.pt" ]]
  [[ -f "$OUTPUT_ROOT/$run/checkpoint_step_00005000.pt.integrity.json" ]]
  [[ -f "$OUTPUT_ROOT/$run/training_report.json" ]]
  [[ -f "$OUTPUT_ROOT/reports/${run}_training_audit.json" ]]
done

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'held-out evaluation lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 12
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

mkdir "$POSTEVAL_ROOT"
export PYTHONPATH=.:src

run_sensitivity() {
  local run=$1
  local checkpoint=$OUTPUT_ROOT/$run/checkpoint_step_00005000.pt
  local output=$POSTEVAL_ROOT/$run
  nice -n 19 ionice -c 3 "$PYTHON" \
    scripts/evaluate_generation_conditioning_sensitivity.py \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --weights ema \
    --num-samples 16 \
    --start-label 192 \
    --wrong-label-offset 500 \
    --timesteps 100 500 900 \
    --noise-seed 304060 \
    --threads 2 \
    --resume
}

run_sensitivity control_cofitok
run_sensitivity ranked_cofitok
run_sensitivity control_dense_identity
run_sensitivity ranked_dense_identity

[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$TRAINING_STATUS" | awk '{print $1}')" == "$EXPECTED_TRAINING_STATUS_SHA256" ]]
[[ "$(sha256sum "$TRAINING_EXECUTION_REPLAY" | awk '{print $1}')" == "$EXPECTED_TRAINING_EXECUTION_REPLAY_SHA256" ]]

nice -n 19 ionice -c 3 "$PYTHON" \
  scripts/build_generation_conditioning_ranking_training_confirmation_posteval.py \
  --preparation-report "$PREPARATION_REPORT" \
  --training-status "$TRAINING_STATUS" \
  --control-cofitok-audit "$OUTPUT_ROOT/reports/control_cofitok_training_audit.json" \
  --control-cofitok-report "$POSTEVAL_ROOT/control_cofitok/conditioning_sensitivity_report.json" \
  --ranked-cofitok-audit "$OUTPUT_ROOT/reports/ranked_cofitok_training_audit.json" \
  --ranked-cofitok-report "$POSTEVAL_ROOT/ranked_cofitok/conditioning_sensitivity_report.json" \
  --control-dense-identity-audit "$OUTPUT_ROOT/reports/control_dense_identity_training_audit.json" \
  --control-dense-identity-report "$POSTEVAL_ROOT/control_dense_identity/conditioning_sensitivity_report.json" \
  --ranked-dense-identity-audit "$OUTPUT_ROOT/reports/ranked_dense_identity_training_audit.json" \
  --ranked-dense-identity-report "$POSTEVAL_ROOT/ranked_dense_identity/conditioning_sensitivity_report.json" \
  --output "$POSTEVAL_ROOT/heldout_evaluation.json" \
  --resume

chmod 0444 \
  "$POSTEVAL_ROOT"/*/conditioning_sensitivity_manifest.json \
  "$POSTEVAL_ROOT"/*/conditioning_sensitivity_report.json \
  "$POSTEVAL_ROOT/heldout_evaluation.json"

printf 'completed source-bound CPU-only held-out evaluation: %s\n' \
  "$POSTEVAL_ROOT/heldout_evaluation.json"
