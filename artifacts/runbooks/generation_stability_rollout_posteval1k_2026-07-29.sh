#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-eval-98d9437
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
RUN_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout/cofitok_rgbtail3_rollout
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation1k_rollout
EXPECTED_CODE_REVISION=317277b0a41d18d88f7e39b4a3212a6baaab4fa8
CHECKPOINT="$RUN_ROOT/checkpoint_step_00001000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -x "$PYTHON" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing rollout post-evaluation while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - "$RUN_ROOT/training_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("training_complete") is not True
    or report.get("completed_steps") != 1000
    or report.get("target_steps") != 1000
):
    raise SystemExit(f"incomplete training report: {sys.argv[1]}")
PY

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

for weights in model ema; do
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/cofitok_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 16 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/cofitok_rollout" \
    --num-images 8 \
    --batch-size 2 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0
done
