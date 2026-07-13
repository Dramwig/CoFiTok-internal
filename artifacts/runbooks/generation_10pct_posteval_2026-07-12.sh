#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_10pct_compressed_cofitok_k8_50k"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_10pct_compressed_dense_50k"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
EVAL_CACHE="$OUTPUT_ROOT/eval_cache/torch_fidelity"
SAMPLING_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/imagenet256_10pct_gate10k_sampling"
SCALING_REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_10pct_compressed_matched_50k"
SAMPLING_SELECTION="$SCALING_REPORT_ROOT/sampling_runtime_selection.json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'formal 10%% post-evaluation requires a clean tracked worktree\n' >&2
  exit 66
fi

validate_training_report() {
  python - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training is incomplete: {path}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"step mismatch: {path}")
if report.get("git", {}).get("dirty") is not False:
    raise SystemExit(f"training used a dirty tracked worktree: {path}")
PY
}

validate_training_report "$COFITOK_RUN/training_report.json"
validate_training_report "$DENSE_RUN/training_report.json"
test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"

python scripts/check_generation_storage_capacity.py \
  --path "$OUTPUT_ROOT" \
  --output "$SCALING_REPORT_ROOT/storage_preflight.json" \
  --stage 10pct_posteval --checkpoint-count 0 --sample-count 20256 \
  --estimated-sample-kib 256 --additional-gib 16 --safety-margin-gib 32

python scripts/audit_generation_training_progress.py \
  --run-dir "$COFITOK_RUN" --expected-steps 50000 \
  --checkpoint-interval 5000 --evaluation-interval 1000 \
  --integrity-policy required \
  --output "$COFITOK_RUN/prepromotion_training_audit.json"

python scripts/audit_generation_training_progress.py \
  --run-dir "$DENSE_RUN" --expected-steps 50000 \
  --checkpoint-interval 5000 --evaluation-interval 1000 \
  --integrity-policy required \
  --output "$DENSE_RUN/prepromotion_training_audit.json"

SAMPLING_BATCH="$(python scripts/select_generation_sampling_batch.py \
  --cofitok-checkpoint "$COFITOK_CHECKPOINT" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --cofitok-prefix-budget 8 --dense-prefix-budget 1 \
  --output-root "$SAMPLING_BENCHMARK_ROOT" --output "$SAMPLING_SELECTION" \
  --sampling-output-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15" \
  --sampling-output-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15" \
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
  --num-images 1024 --timestep 500 --random-orders 16 \
  --weights ema --precision bf16

python scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/checkpoint_eval_ema_t500_1024" \
  --num-images 1024 --timestep 500 --random-orders 0 \
  --weights ema --precision bf16

python scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15" \
  --num-samples 10000 --batch-size "$SAMPLING_BATCH" --sample-steps 100 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 --resume

python scripts/generate_samples.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15" \
  --num-samples 10000 --batch-size "$SAMPLING_BATCH" --sample-steps 100 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 --resume

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/prefix_8" \
  --sampling-report "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/sampling_report.json" \
  --output-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 10000

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15/prefix_1" \
  --sampling-report "$DENSE_RUN/samples_gate10k_ddim100_cfg15/sampling_report.json" \
  --output-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 10000

python scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/prefix_diagnostic_64_ddim100_cfg15" \
  --num-samples 64 --batch-size 16 --sample-steps 100 \
  --prefix-budgets 1,2,4,8 --guidance-scale 1.5 --cfg-batch-mode batched \
  --weights ema --precision bf16 --resume

python scripts/build_generation_visual_audit.py \
  --cofitok-sampling-report "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/sampling_report.json" \
  --dense-sampling-report "$DENSE_RUN/samples_gate10k_ddim100_cfg15/sampling_report.json" \
  --prefix-sampling-report "$COFITOK_RUN/prefix_diagnostic_64_ddim100_cfg15/sampling_report.json" \
  --cofitok-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/prefix_8" \
  --dense-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15/prefix_1" \
  --indices 0,1,2,3,250,251,1000,1001,5000,5001,9998,9999 \
  --prefix-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --prefix-budgets 1,2,4,8 \
  --output-dir "$SCALING_REPORT_ROOT/visual_audit"

python scripts/build_generation_gate_report.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json" \
  --cofitok-checkpoint-eval "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --dense-checkpoint-eval "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --output "$SCALING_REPORT_ROOT/promotion_gate.json" \
  --stage scaling --min-samples 10000 --max-absolute-fid 100.0 --allow-fail
