#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
EVAL_CACHE="$OUTPUT_ROOT/eval_cache/torch_fidelity"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

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
  --num-samples 10000 --batch-size 32 --sample-steps 100 \
  --guidance-scale 1.5 --weights ema --precision bf16

python scripts/generate_samples.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15" \
  --num-samples 10000 --batch-size 32 --sample-steps 100 \
  --guidance-scale 1.5 --weights ema --precision bf16

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/prefix_8" \
  --output-dir "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 10000

python scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" \
  --generated-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15/prefix_1" \
  --output-dir "$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics" \
  --cache-root "$EVAL_CACHE" --min-samples 10000

python scripts/generate_samples.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output-dir "$COFITOK_RUN/prefix_diagnostic_64_ddim100_cfg15" \
  --num-samples 64 --batch-size 16 --sample-steps 100 \
  --prefix-budgets 1,2,4,8 --guidance-scale 1.5 --weights ema --precision bf16
