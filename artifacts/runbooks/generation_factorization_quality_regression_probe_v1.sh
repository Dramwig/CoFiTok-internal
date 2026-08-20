#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${PYTHON:?set PYTHON}"
: "${EXPECTED_REVISION:?set EXPECTED_REVISION}"
: "${EXPECTED_TREE:?set EXPECTED_TREE}"
: "${EXPECTED_BRANCH:?set EXPECTED_BRANCH}"
: "${PREPARATION:?set PREPARATION}"
: "${EXPECTED_PREPARATION_SHA256:?set EXPECTED_PREPARATION_SHA256}"
: "${SOURCE_BINDING:?set SOURCE_BINDING}"
: "${EXPECTED_SOURCE_BINDING_SHA256:?set EXPECTED_SOURCE_BINDING_SHA256}"
: "${EXECUTION_AUTHORIZATION:?set EXECUTION_AUTHORIZATION}"
: "${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:?set EXPECTED_EXECUTION_AUTHORIZATION_SHA256}"
: "${STANDING_AUTHORIZATION:?set STANDING_AUTHORIZATION}"
: "${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}"
: "${FOLLOWUP_DECISION:?set FOLLOWUP_DECISION}"
: "${TERMINAL_SYSTEM_GUARD:?set TERMINAL_SYSTEM_GUARD}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

EXPECTED_OUTPUT_ROOT="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_factorization_quality_regression_v1"
QUALITY_ROOT="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1"
COFITOK_RUN="$QUALITY_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$QUALITY_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00100000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00100000.pt"
COFITOK_TRAINING="$COFITOK_RUN/training_report.json"
DENSE_TRAINING="$DENSE_RUN/training_report.json"
LOCK="${OUTPUT_ROOT}.lock"

if [[ "$OUTPUT_ROOT" != "$EXPECTED_OUTPUT_ROOT" ]]; then
  printf 'factorization-regression output root differs\n' >&2
  exit 2
fi
cd "$PROJECT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_REVISION" ]] \
  || [[ "$(git rev-parse HEAD^{tree})" != "$EXPECTED_TREE" ]] \
  || [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]] \
  || [[ -n "$(git status --porcelain)" ]]; then
  printf 'factorization-regression execution checkout differs\n' >&2
  exit 3
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'factorization-regression Python is not executable\n' >&2
  exit 4
fi

export PYTHONPATH="$PROJECT:$PROJECT/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

"$PYTHON" scripts/verify_generation_factorization_quality_regression_execution_authorization.py \
  --authorization "$EXECUTION_AUTHORIZATION" \
  --expected-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --source-binding "$SOURCE_BINDING" \
  --expected-source-binding-sha256 "$EXPECTED_SOURCE_BINDING_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --followup-decision "$FOLLOWUP_DECISION" \
  --terminal-system-guard "$TERMINAL_SYSTEM_GUARD" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-branch "$EXPECTED_BRANCH" >/dev/null

if ! mkdir "$LOCK"; then
  printf 'factorization-regression output lock already exists\n' >&2
  exit 73
fi
mkdir -p "$OUTPUT_ROOT/checkpoint_eval/cofitok" \
  "$OUTPUT_ROOT/checkpoint_eval/dense_identity" \
  "$OUTPUT_ROOT/reports/factorization_quality_regression"

"$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/checkpoint_eval/cofitok" \
  --num-images 256 \
  --timestep 500 \
  --random-orders 4 \
  --weights ema \
  --precision bf16 \
  --resume >/dev/null
"$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/checkpoint_eval/dense_identity" \
  --num-images 256 \
  --timestep 500 \
  --random-orders 0 \
  --weights ema \
  --precision bf16 \
  --resume >/dev/null

COFITOK_EVAL="$OUTPUT_ROOT/checkpoint_eval/cofitok/checkpoint_evaluation_report.json"
DENSE_EVAL="$OUTPUT_ROOT/checkpoint_eval/dense_identity/checkpoint_evaluation_report.json"
REPORT_ARGS=()
for seed in 2029 2039; do
  seed_root="$OUTPUT_ROOT/seed_$seed"
  cofitok_rollout="$seed_root/cofitok/rollout_stability_report.json"
  dense_rollout="$seed_root/dense_identity/rollout_stability_report.json"
  qualification_root="$seed_root/qualification"
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$seed_root/cofitok" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed "$seed" \
    --weights ema \
    --precision bf16 \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0 >/dev/null
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$seed_root/dense_identity" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed "$seed" \
    --weights ema \
    --precision bf16 \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0 >/dev/null
  "$PYTHON" scripts/build_generation_stability_qualification.py \
    --cofitok-training "$COFITOK_TRAINING" \
    --dense-training "$DENSE_TRAINING" \
    --cofitok-checkpoint "$COFITOK_EVAL" \
    --dense-checkpoint "$DENSE_EVAL" \
    --cofitok-rollout "$cofitok_rollout" \
    --dense-rollout "$dense_rollout" \
    --output-dir "$qualification_root" \
    --weights ema \
    --expected-evaluation-revision "$EXPECTED_REVISION" \
    --expected-evaluation-branch "$EXPECTED_BRANCH" >/dev/null
  REPORT_ARGS+=(
    --cofitok-rollout "$seed=$cofitok_rollout"
    --dense-rollout "$seed=$dense_rollout"
    --qualification "$seed=$qualification_root/qualification_report.json"
  )
done

"$PYTHON" scripts/build_generation_factorization_quality_regression_report.py \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --source-binding "$SOURCE_BINDING" \
  --expected-source-binding-sha256 "$EXPECTED_SOURCE_BINDING_SHA256" \
  --execution-authorization "$EXECUTION_AUTHORIZATION" \
  --expected-execution-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" \
  --cofitok-training "$COFITOK_TRAINING" \
  --dense-training "$DENSE_TRAINING" \
  --cofitok-checkpoint-eval "$COFITOK_EVAL" \
  --dense-checkpoint-eval "$DENSE_EVAL" \
  "${REPORT_ARGS[@]}" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-branch "$EXPECTED_BRANCH" \
  --output "$OUTPUT_ROOT/reports/factorization_quality_regression/diagnostic_report.json" >/dev/null
