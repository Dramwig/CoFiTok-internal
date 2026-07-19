#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
OFFICIAL_RELATED="$PROJECT/artifacts/reports/baselines/official_related_methods_2026-07-11_final/official_related_methods_table.json"
SAMPLING_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/imagenet256_full_50k_sampling"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
eval "$(python scripts/print_generation_workspace_paths.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" --format shell)"
COFITOK_RUN="$FULL_COFITOK_RUN"
DENSE_RUN="$FULL_DENSE_RUN"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00300000.pt"
EVAL_CACHE="$OUTPUT_ROOT/eval_cache/torch_fidelity"
REPORT_ROOT="$FULL_REPORT_ROOT"
SAMPLING_SELECTION="$REPORT_ROOT/sampling_runtime_selection.json"
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'formal 50K post-evaluation requires a clean tracked worktree\n' >&2
  exit 66
fi
mkdir -p "$REPORT_ROOT"

python scripts/validate_generation_gate_report.py \
  --gate "$SCALING_GATE" --stage scaling

python scripts/validate_generation_training_pair.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --expected-steps 300000 --expected-revision "$(git rev-parse HEAD)" \
  --expected-dataset imagenet_256 --expected-recipe-stage full \
  --authorization-gate "$SCALING_GATE" \
  >"$REPORT_ROOT/posteval_training_pair_validation.json"

test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"
test -f "$OFFICIAL_RELATED"

python scripts/check_generation_storage_capacity.py \
  --path "$OUTPUT_ROOT" \
  --output "$REPORT_ROOT/storage_preflight_posteval.json" \
  --stage full_posteval --checkpoint-count 0 --sample-count 100256 \
  --estimated-sample-kib 256 --additional-gib 16 --safety-margin-gib 64

SAMPLING_BATCH="$(python scripts/select_generation_sampling_batch.py \
  --cofitok-checkpoint "$COFITOK_CHECKPOINT" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --cofitok-prefix-budget 8 --dense-prefix-budget 1 \
  --output-root "$SAMPLING_BENCHMARK_ROOT" --output "$SAMPLING_SELECTION" \
  --sampling-output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15" \
  --sampling-output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15" \
  --candidates 16,32,64,128 --baseline-batch-size 32 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 \
  --warmup-forwards 2 --measured-forwards 5 --max-memory-fraction 0.90)"
if [[ ! "$SAMPLING_BATCH" =~ ^[0-9]+$ ]]; then
  printf 'invalid selected sampling batch: %s\n' "$SAMPLING_BATCH" >&2
  exit 1
fi

python scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/checkpoint_eval_ema_t500_1024" \
  --num-images 1024 --timestep 500 --random-orders 16 --weights ema --precision bf16

python scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/checkpoint_eval_ema_t500_1024" \
  --num-images 1024 --timestep 500 --random-orders 0 --weights ema --precision bf16

python scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15" \
  --num-samples 50000 --batch-size "$SAMPLING_BATCH" --sample-steps 250 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 --resume

python scripts/generate_samples.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15" \
  --num-samples 50000 --batch-size "$SAMPLING_BATCH" --sample-steps 250 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 --resume

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" --generated-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --sampling-report "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --output-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 50000

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" --generated-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --sampling-report "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 50000

python scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15" \
  --num-samples 64 --batch-size 16 --sample-steps 250 \
  --prefix-budgets 1,2,4,8 --guidance-scale 1.5 --cfg-batch-mode batched \
  --weights ema --precision bf16 --resume

python scripts/build_generation_visual_audit.py \
  --cofitok-sampling-report "$COFITOK_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --dense-sampling-report "$DENSE_RUN/samples_50k_ddim250_cfg15/sampling_report.json" \
  --prefix-sampling-report "$COFITOK_RUN/prefix_diagnostic_64_ddim250_cfg15/sampling_report.json" \
  --cofitok-dir "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8" \
  --dense-dir "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1" \
  --indices 0,1,2,3,250,251,1000,1001,10000,10001,25000,25001,49998,49999 \
  --prefix-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --prefix-budgets 1,2,4,8 \
  --output-dir "$REPORT_ROOT/visual_audit"

python scripts/build_generation_gate_report.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --cofitok-checkpoint-eval "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --dense-checkpoint-eval "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --output "$REPORT_ROOT/final_generation_gate.json" --stage full \
  --min-samples 50000 --max-absolute-fid 20.0 \
  --min-precision 0.30 --min-recall 0.30 \
  --max-precision-regression 0.05 --max-recall-regression 0.05 \
  --allow-fail

python scripts/build_large_scale_generation_comparison.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --final-gate "$REPORT_ROOT/final_generation_gate.json" \
  --official-related "$OFFICIAL_RELATED" \
  --output-dir "$REPORT_ROOT/comparison"
