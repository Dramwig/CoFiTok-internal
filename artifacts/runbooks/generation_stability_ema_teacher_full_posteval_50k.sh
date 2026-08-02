#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
DATA=${DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
OFFICIAL_RELATED=${OFFICIAL_RELATED:-"$PROJECT/artifacts/reports/baselines/official_related_methods_2026-07-11_final/official_related_methods_table.json"}
EXPECTED_SCALING_GATE_SHA256=${EXPECTED_SCALING_GATE_SHA256:?set the passing stability scaling gate SHA256}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set the stability-full training revision}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set the stability-full training branch}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean full post-evaluation revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean full post-evaluation branch}

SCALING_GATE="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher/reports/promotion_gate.json"
COFITOK_CONFIG=configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00300000.pt"
EVAL_CACHE="$CHECKPOINT_ROOT/eval_cache/torch_fidelity"
SAMPLING_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/sampling_50k"
SAMPLING_SELECTION="$REPORT_ROOT/sampling_runtime_selection.json"
COFITOK_ROLLOUT_DIR="$COFITOK_RUN/rollout_stability_ema_n64_seed2029"
DENSE_ROLLOUT_DIR="$DENSE_RUN/rollout_stability_ema_n64_seed2029"
EMA_ROLLOUT_QUALIFICATION_ROOT="$REPORT_ROOT/ema_rollout_stability"
EMA_ROLLOUT_QUALIFICATION="$EMA_ROLLOUT_QUALIFICATION_ROOT/qualification_report.json"
TRAINING_CONTENTION="$OUTPUT_ROOT/pair_monitor.json"
FINAL_GATE="$REPORT_ROOT/final_generation_gate.json"
STAGE_STATE_ROOT="$REPORT_ROOT/stage_receipts"
DATASET_MANIFEST="$CHECKPOINT_ROOT/../../datasets/imagenet_256/metadata/image_manifest.jsonl"
POSTEVAL_LOCK="$OUTPUT_ROOT/posteval.lock"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$SCALING_GATE" ]]
[[ "$(sha256sum "$SCALING_GATE" | awk '{print $1}')" == "$EXPECTED_SCALING_GATE_SHA256" ]]
[[ -f "$COFITOK_CHECKPOINT" ]]
[[ -f "$DENSE_CHECKPOINT" ]]
[[ -f "$OFFICIAL_RELATED" ]]
[[ -f "$DATASET_MANIFEST" ]]
[[ -f "$TRAINING_CONTENTION" ]]
command -v flock >/dev/null
exec 8>"$POSTEVAL_LOCK"
if ! flock -n 8; then
  printf 'refusing concurrent stability full post-evaluation\n' >&2
  exit 75
fi
mkdir -p "$REPORT_ROOT" "$STAGE_STATE_ROOT"

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$SCALING_GATE" \
  --stage scaling >/dev/null

"$PYTHON" scripts/validate_generation_training_pair.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 \
  --expected-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-dataset imagenet_256 \
  --expected-recipe-stage stability_full \
  --authorization-gate "$SCALING_GATE" \
  >"$REPORT_ROOT/posteval_training_pair_validation.json"

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" \
  --config "$COFITOK_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/preposteval_cofitok_training_audit.json" >/dev/null

"$PYTHON" scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" \
  --config "$DENSE_CONFIG" \
  --expected-steps 300000 \
  --checkpoint-interval 5000 \
  --evaluation-interval 2000 \
  --required-checkpoint-steps 50000,100000,200000,300000 \
  --integrity-policy required \
  --output "$REPORT_ROOT/preposteval_dense_training_audit.json" >/dev/null

"$PYTHON" scripts/check_generation_storage_capacity.py \
  --path "$CHECKPOINT_ROOT" \
  --output "$REPORT_ROOT/posteval_storage_capacity.json" \
  --stage full_posteval \
  --checkpoint-count 0 \
  --sample-count 100256 \
  --estimated-sample-kib 256 \
  --additional-gib 16 \
  --safety-margin-gib 64 >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing stability full post-evaluation while the GPU is busy\n' >&2
  exit 9
fi

SAMPLING_BATCH="$("$PYTHON" scripts/select_generation_sampling_batch.py \
  --cofitok-checkpoint "$COFITOK_CHECKPOINT" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --cofitok-prefix-budget 8 \
  --dense-prefix-budget 1 \
  --output-root "$SAMPLING_BENCHMARK_ROOT" \
  --output "$SAMPLING_SELECTION" \
  --sampling-output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15" \
  --sampling-output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15" \
  --candidates 16,32,64,128 \
  --baseline-batch-size 32 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --warmup-forwards 2 \
  --measured-forwards 5 \
  --max-memory-fraction 0.90)"
if [[ ! "$SAMPLING_BATCH" =~ ^[0-9]+$ ]]; then
  printf 'invalid stability full sampling batch: %s\n' "$SAMPLING_BATCH" >&2
  exit 10
fi

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok_checkpoint_eval.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_CHECKPOINT" \
  --input-file "$COFITOK_CHECKPOINT.integrity.json" \
  --input-file "$COFITOK_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$COFITOK_RUN/checkpoint_eval_ema_t500_1024" \
  -- \
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/checkpoint_eval_ema_t500_1024" \
  --num-images 1024 \
  --timestep 500 \
  --random-orders 16 \
  --weights ema \
  --precision bf16

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/dense_checkpoint_eval.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$DENSE_CHECKPOINT" \
  --input-file "$DENSE_CHECKPOINT.integrity.json" \
  --input-file "$DENSE_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$DENSE_RUN/checkpoint_eval_ema_t500_1024" \
  -- \
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/checkpoint_eval_ema_t500_1024" \
  --num-images 1024 \
  --timestep 500 \
  --random-orders 0 \
  --weights ema \
  --precision bf16

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok_rollout_stability.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_CHECKPOINT" \
  --input-file "$COFITOK_CHECKPOINT.integrity.json" \
  --input-file "$COFITOK_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$COFITOK_ROLLOUT_DIR" \
  -- \
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_ROLLOUT_DIR" \
  --num-images 64 \
  --batch-size 8 \
  --sample-steps 250 \
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
  --input-file "$DENSE_CHECKPOINT" \
  --input-file "$DENSE_CHECKPOINT.integrity.json" \
  --input-file "$DENSE_CONFIG" \
  --input-file "$DATASET_MANIFEST" \
  --output-tree "$DENSE_ROLLOUT_DIR" \
  -- \
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_ROLLOUT_DIR" \
  --num-images 64 \
  --batch-size 8 \
  --sample-steps 250 \
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
  --input-file "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --input-file "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --input-file "$COFITOK_ROLLOUT_DIR/rollout_stability_report.json" \
  --input-file "$DENSE_ROLLOUT_DIR/rollout_stability_report.json" \
  --output-tree "$EMA_ROLLOUT_QUALIFICATION_ROOT" \
  -- \
  "$PYTHON" scripts/build_generation_stability_qualification.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-checkpoint "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --dense-checkpoint "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --cofitok-rollout "$COFITOK_ROLLOUT_DIR/rollout_stability_report.json" \
  --dense-rollout "$DENSE_ROLLOUT_DIR/rollout_stability_report.json" \
  --weights ema \
  --expected-evaluation-revision "$EXPECTED_TARGET_REVISION" \
  --expected-evaluation-branch "$EXPECTED_TARGET_BRANCH" \
  --output-dir "$EMA_ROLLOUT_QUALIFICATION_ROOT"

"$PYTHON" scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15" \
  --num-samples 50000 \
  --batch-size "$SAMPLING_BATCH" \
  --sample-steps 250 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --resume

"$PYTHON" scripts/generate_samples.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15" \
  --num-samples 50000 \
  --batch-size "$SAMPLING_BATCH" \
  --sample-steps 250 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --resume

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok_generation_metrics.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_manifest.json" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_progress.json" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --input-file "$DATASET_MANIFEST" \
  --input-tree "$DATA" \
  --input-tree "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --output-tree "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics" \
  -- \
  "$PYTHON" scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --sampling-report "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" \
  --min-samples 50000

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/dense_generation_metrics.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_manifest.json" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_progress.json" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --input-file "$DATASET_MANIFEST" \
  --input-tree "$DATA" \
  --input-tree "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --output-tree "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics" \
  -- \
  "$PYTHON" scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --sampling-report "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" \
  --min-samples 50000

"$PYTHON" scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15" \
  --num-samples 64 \
  --batch-size 16 \
  --sample-steps 250 \
  --prefix-budgets 1,2,4,8 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --resume

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/visual_audit.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --input-file "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15/sampling_report.json" \
  --input-tree "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --input-tree "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --input-tree "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15" \
  --output-tree "$REPORT_ROOT/visual_audit" \
  -- \
  "$PYTHON" scripts/build_generation_visual_audit.py \
  --cofitok-sampling-report "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --dense-sampling-report "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --prefix-sampling-report "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15/sampling_report.json" \
  --cofitok-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --dense-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --indices 0,1,2,3,250,251,1000,1001,10000,10001,25000,25001,49998,49999 \
  --prefix-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --prefix-budgets 1,2,4,8 \
  --output-dir "$REPORT_ROOT/visual_audit"


"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/final_gate.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_RUN/training_report.json" \
  --input-file "$DENSE_RUN/training_report.json" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --input-file "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --input-file "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --input-file "$EMA_ROLLOUT_QUALIFICATION" \
  --output-file "$FINAL_GATE" \
  -- \
  "$PYTHON" scripts/build_generation_gate_report.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --cofitok-checkpoint-eval "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --dense-checkpoint-eval "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --rollout-stability-qualification "$EMA_ROLLOUT_QUALIFICATION" \
  --output "$FINAL_GATE" \
  --stage full \
  --source-profile stability_full \
  --expected-training-revision "$EXPECTED_TRAINING_REVISION" \
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH" \
  --expected-evaluation-revision "$EXPECTED_TARGET_REVISION" \
  --expected-evaluation-branch "$EXPECTED_TARGET_BRANCH" \
  --min-samples 50000 \
  --max-absolute-fid 20.0 \
  --min-coarse-token-energy-ratio 0.05 \
  --min-precision 0.30 \
  --min-recall 0.30 \
  --max-precision-regression 0.05 \
  --max-recall-regression 0.05 \
  --allow-fail

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$FINAL_GATE" \
  --stage full \
  --sources-only >/dev/null

if "$PYTHON" - "$FINAL_GATE" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if gate.get("status") == "pass" else 1)
PY
then
  "$PYTHON" scripts/validate_generation_gate_report.py \
    --gate "$FINAL_GATE" \
    --stage full >/dev/null
fi

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/comparison.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_RUN/training_report.json" \
  --input-file "$DENSE_RUN/training_report.json" \
  --input-file "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --input-file "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --input-file "$FINAL_GATE" \
  --input-file "$OFFICIAL_RELATED" \
  --input-file "$TRAINING_CONTENTION" \
  --output-tree "$REPORT_ROOT/comparison" \
  -- \
  "$PYTHON" scripts/build_large_scale_generation_comparison.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --final-gate "$FINAL_GATE" \
  --official-related "$OFFICIAL_RELATED" \
  --training-contention "$TRAINING_CONTENTION" \
  --source-profile stability_full \
  --output-dir "$REPORT_ROOT/comparison"
