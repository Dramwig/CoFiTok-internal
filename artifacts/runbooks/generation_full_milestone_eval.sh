#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 METHOD RUN_DIR CHECKPOINT STEP PREFIX_BUDGET RANDOM_ORDERS" >&2
  exit 2
fi

METHOD="$1"
RUN_DIR="$2"
CHECKPOINT="$3"
STEP="$4"
PREFIX_BUDGET="$5"
RANDOM_ORDERS="$6"
PROJECT=${PROJECT:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
PYTHON=${PYTHON:-python}
DATA=${DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EVAL_CACHE="$OUTPUT_ROOT/eval_cache/torch_fidelity"
MILESTONE_DIR="$RUN_DIR/milestones/step_$(printf '%08d' "$STEP")"
SAMPLES="$MILESTONE_DIR/samples_2048_ddim50_cfg15"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

"$PYTHON" scripts/preflight_generation_sampling.py \
  --checkpoint "$CHECKPOINT" --output "$MILESTONE_DIR/sampling_preflight.json" \
  --batch-size 32 --prefix-budget "$PREFIX_BUDGET" --guidance-scale 1.5 \
  --cfg-batch-mode batched --weights ema --precision bf16

"$PYTHON" scripts/evaluate_generation_checkpoint.py \
  --checkpoint "$CHECKPOINT" --output-dir "$MILESTONE_DIR/checkpoint_eval" \
  --num-images 256 --timestep 500 --random-orders "$RANDOM_ORDERS" \
  --weights ema --precision bf16 --resume

"$PYTHON" scripts/generate_samples.py \
  --checkpoint "$CHECKPOINT" --output-dir "$SAMPLES" \
  --num-samples 2048 --batch-size 32 --sample-steps 50 \
  --prefix-budgets "$PREFIX_BUDGET" --guidance-scale 1.5 \
  --cfg-batch-mode batched --weights ema --precision bf16 --resume

"$PYTHON" scripts/evaluate_generation_metrics.py \
  --real-dir "$DATA" --generated-dir "$SAMPLES/prefix_$PREFIX_BUDGET" \
  --sampling-report "$SAMPLES/sampling_report.json" \
  --output-dir "$SAMPLES/metrics" --cache-root "$EVAL_CACHE" \
  --min-samples 2048 --skip-prc

printf '%s\n' "$METHOD milestone step $STEP completed"
