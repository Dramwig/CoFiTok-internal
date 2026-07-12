#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_full_cofitok_k8_300k"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_full_dense_300k"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00300000.pt"
EVAL_CACHE="$OUTPUT_ROOT/eval_cache/torch_fidelity"
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$REPORT_ROOT"

validate_training_report() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training is incomplete: {sys.argv[1]}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"step mismatch: {sys.argv[1]}")
if report.get("git", {}).get("dirty") is not False:
    raise SystemExit(f"training used a dirty tracked worktree: {sys.argv[1]}")
PY
}

validate_training_report "$COFITOK_RUN/training_report.json"
validate_training_report "$DENSE_RUN/training_report.json"
test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"

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
  --num-samples 50000 --batch-size 32 --sample-steps 250 \
  --guidance-scale 1.5 --cfg-batch-mode batched --weights ema --precision bf16 --resume

python scripts/generate_samples.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/samples_50k_ddim250_cfg15" \
  --num-samples 50000 --batch-size 32 --sample-steps 250 \
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

python scripts/build_generation_gate_report.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-generation "$COFITOK_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --dense-generation "$DENSE_RUN/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json" \
  --cofitok-checkpoint-eval "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --dense-checkpoint-eval "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  --output "$REPORT_ROOT/final_generation_gate.json" --stage full \
  --min-samples 50000 --max-absolute-fid 20.0 --allow-fail
