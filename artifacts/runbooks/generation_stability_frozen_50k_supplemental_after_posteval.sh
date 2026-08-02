#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:-2c2c1f5166b73d4f28df93b276901671ac1a7836}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:-scale/generation-stability-50k-preflight}
EXPECTED_FROZEN_EVALUATION_REVISION=${EXPECTED_FROZEN_EVALUATION_REVISION:-c1efb12c6640f2d2d62ac7e9982c8804d96e7289}
EXPECTED_FROZEN_EVALUATION_BRANCH=${EXPECTED_FROZEN_EVALUATION_BRANCH:-scale/generation-stability-50k-posteval-v4}
EXPECTED_SUPPLEMENTAL_REVISION=${EXPECTED_SUPPLEMENTAL_REVISION:?set the clean supplemental revision}
EXPECTED_SUPPLEMENTAL_BRANCH=${EXPECTED_SUPPLEMENTAL_BRANCH:-scale/generation-large-capacity}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json
DATASET_MANIFEST="$CHECKPOINT_ROOT/../../datasets/imagenet_256/metadata/image_manifest.jsonl"
POSTEVAL_STATUS="$REPORT_ROOT/posteval_waiter.json"
PROMOTION_GATE="$REPORT_ROOT/promotion_gate.json"
SUPPLEMENTAL_ROOT="$REPORT_ROOT/frozen_posteval_supplemental"
POSTEVAL_VERIFICATION="$SUPPLEMENTAL_ROOT/posteval_verification.json"
COFITOK_CHECKPOINT_EVAL="$SUPPLEMENTAL_ROOT/checkpoint_eval/cofitok"
DENSE_CHECKPOINT_EVAL="$SUPPLEMENTAL_ROOT/checkpoint_eval/dense"
COFITOK_ROLLOUT="$SUPPLEMENTAL_ROOT/rollout_stability/cofitok"
DENSE_ROLLOUT="$SUPPLEMENTAL_ROOT/rollout_stability/dense"
ROLLOUT_QUALIFICATION_ROOT="$SUPPLEMENTAL_ROOT/ema_rollout_stability"
ROLLOUT_QUALIFICATION="$ROLLOUT_QUALIFICATION_ROOT/qualification_report.json"
DISTRIBUTION_SUPPORT="$SUPPLEMENTAL_ROOT/distribution_support/qualification_report.json"
SUPPLEMENTAL_QUALIFICATION="$SUPPLEMENTAL_ROOT/supplemental_qualification.json"
STAGE_STATE_ROOT="$SUPPLEMENTAL_ROOT/stage_receipts"
SUPPLEMENTAL_LOCK="$OUTPUT_ROOT/frozen_posteval_supplemental.lock"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SUPPLEMENTAL_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_SUPPLEMENTAL_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
for path in \
  "$POSTEVAL_STATUS" \
  "$PROMOTION_GATE" \
  "$COFITOK_CHECKPOINT" \
  "$COFITOK_CHECKPOINT.integrity.json" \
  "$DENSE_CHECKPOINT" \
  "$DENSE_CHECKPOINT.integrity.json" \
  "$COFITOK_RUN/training_report.json" \
  "$DENSE_RUN/training_report.json" \
  "$COFITOK_CONFIG" \
  "$DENSE_CONFIG" \
  "$DATASET_MANIFEST"; do
  [[ -f "$path" ]]
done
command -v flock >/dev/null
exec 8>"$SUPPLEMENTAL_LOCK"
if ! flock -n 8; then
  printf 'refusing concurrent frozen stability supplemental evaluation\n' >&2
  exit 75
fi
mkdir -p "$SUPPLEMENTAL_ROOT" "$STAGE_STATE_ROOT"

"$PYTHON" scripts/validate_generation_stability_frozen_posteval.py \
  --posteval-status "$POSTEVAL_STATUS" \
  --output "$POSTEVAL_VERIFICATION" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-evaluation-revision "$EXPECTED_FROZEN_EVALUATION_REVISION" \
  --expected-evaluation-branch "$EXPECTED_FROZEN_EVALUATION_BRANCH" \
  --resume >/dev/null

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$PROMOTION_GATE" \
  --stage scaling \
  --sources-only >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing frozen stability supplemental evaluation while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok_checkpoint_eval.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$POSTEVAL_VERIFICATION" \
  --input-file "$PROMOTION_GATE" \
  --input-file "$COFITOK_CHECKPOINT" \
  --input-file "$COFITOK_CHECKPOINT.integrity.json" \
  --input-file "$COFITOK_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$COFITOK_CHECKPOINT_EVAL" \
  -- \
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_CHECKPOINT_EVAL" \
  --num-images 1024 \
  --timestep 500 \
  --random-orders 16 \
  --weights ema \
  --precision bf16

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/dense_checkpoint_eval.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$POSTEVAL_VERIFICATION" \
  --input-file "$PROMOTION_GATE" \
  --input-file "$DENSE_CHECKPOINT" \
  --input-file "$DENSE_CHECKPOINT.integrity.json" \
  --input-file "$DENSE_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$DENSE_CHECKPOINT_EVAL" \
  -- \
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_CHECKPOINT_EVAL" \
  --num-images 1024 \
  --timestep 500 \
  --random-orders 0 \
  --weights ema \
  --precision bf16

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok_rollout_stability.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$POSTEVAL_VERIFICATION" \
  --input-file "$PROMOTION_GATE" \
  --input-file "$COFITOK_CHECKPOINT" \
  --input-file "$COFITOK_CHECKPOINT.integrity.json" \
  --input-file "$COFITOK_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$COFITOK_ROLLOUT" \
  -- \
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_ROLLOUT" \
  --num-images 64 \
  --batch-size 8 \
  --sample-steps 100 \
  --seed 2029 \
  --weights ema \
  --precision bf16 \
  --guidance-scale 1.5 \
  --guidance-rescale 0.0 \
  --teacher-guidance-scale 1.0 \
  --cfg-batch-mode batched \
  --clip-x0

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/dense_rollout_stability.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$POSTEVAL_VERIFICATION" \
  --input-file "$PROMOTION_GATE" \
  --input-file "$DENSE_CHECKPOINT" \
  --input-file "$DENSE_CHECKPOINT.integrity.json" \
  --input-file "$DENSE_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$DENSE_ROLLOUT" \
  -- \
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_ROLLOUT" \
  --num-images 64 \
  --batch-size 8 \
  --sample-steps 100 \
  --seed 2029 \
  --weights ema \
  --precision bf16 \
  --guidance-scale 1.5 \
  --guidance-rescale 0.0 \
  --teacher-guidance-scale 1.0 \
  --cfg-batch-mode batched \
  --clip-x0

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/ema_rollout_stability_qualification.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_RUN/training_report.json" \
  --input-file "$DENSE_RUN/training_report.json" \
  --input-file "$COFITOK_CHECKPOINT_EVAL/checkpoint_evaluation_report.json" \
  --input-file "$DENSE_CHECKPOINT_EVAL/checkpoint_evaluation_report.json" \
  --input-file "$COFITOK_ROLLOUT/rollout_stability_report.json" \
  --input-file "$DENSE_ROLLOUT/rollout_stability_report.json" \
  --output-tree "$ROLLOUT_QUALIFICATION_ROOT" \
  -- \
  "$PYTHON" scripts/build_generation_stability_qualification.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-checkpoint "$COFITOK_CHECKPOINT_EVAL/checkpoint_evaluation_report.json" \
  --dense-checkpoint "$DENSE_CHECKPOINT_EVAL/checkpoint_evaluation_report.json" \
  --cofitok-rollout "$COFITOK_ROLLOUT/rollout_stability_report.json" \
  --dense-rollout "$DENSE_ROLLOUT/rollout_stability_report.json" \
  --weights ema \
  --expected-evaluation-revision "$EXPECTED_SUPPLEMENTAL_REVISION" \
  --expected-evaluation-branch "$EXPECTED_SUPPLEMENTAL_BRANCH" \
  --output-dir "$ROLLOUT_QUALIFICATION_ROOT"

mkdir -p "$(dirname "$DISTRIBUTION_SUPPORT")"
"$PYTHON" scripts/build_generation_stability_distribution_support.py \
  --gate "$PROMOTION_GATE" \
  --output "$DISTRIBUTION_SUPPORT" \
  --resume >/dev/null

"$PYTHON" scripts/build_generation_stability_frozen_supplemental.py \
  --promotion-gate "$PROMOTION_GATE" \
  --posteval-verification "$POSTEVAL_VERIFICATION" \
  --distribution-support "$DISTRIBUTION_SUPPORT" \
  --rollout-stability "$ROLLOUT_QUALIFICATION" \
  --output "$SUPPLEMENTAL_QUALIFICATION" \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-frozen-evaluation-revision "$EXPECTED_FROZEN_EVALUATION_REVISION" \
  --expected-frozen-evaluation-branch "$EXPECTED_FROZEN_EVALUATION_BRANCH" \
  --expected-supplemental-revision "$EXPECTED_SUPPLEMENTAL_REVISION" \
  --expected-supplemental-branch "$EXPECTED_SUPPLEMENTAL_BRANCH" \
  --resume

printf 'frozen stability supplemental: %s  %s\n' \
  "$(sha256sum "$SUPPLEMENTAL_QUALIFICATION" | awk '{print $1}')" \
  "$SUPPLEMENTAL_QUALIFICATION"
