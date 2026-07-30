#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-scale-68ca820
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation5k_rollout_x0
EXPECTED_CODE_REVISION=68ca820d5d901eff2623b97b86c4b88bf68ef500
COFITOK_CHECKPOINT="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0/checkpoint_step_00005000.pt"
DENSE_CHECKPOINT="$PAIR_ROOT/dense_identity/checkpoint_step_00005000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --short)" ]]
[[ -x "$PYTHON" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing rollout-x0 5K post-evaluation while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - \
  "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0/training_report.json" \
  "$PAIR_ROOT/dense_identity/training_report.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if (
        report.get("training_complete") is not True
        or report.get("completed_steps") != 5000
        or report.get("target_steps") != 5000
        or report.get("git", {}).get("dirty") is not False
    ):
        raise SystemExit(f"invalid 5K training report: {path}")
PY

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

for weights in model ema; do
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/cofitok_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 16 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/dense_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 0 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  for method in cofitok dense; do
    checkpoint="$COFITOK_CHECKPOINT"
    if [[ "$method" == dense ]]; then
      checkpoint="$DENSE_CHECKPOINT"
    fi
    "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
      --checkpoint "$checkpoint" \
      --output-dir "$OUTPUT_ROOT/$weights/${method}_rollout" \
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
done
