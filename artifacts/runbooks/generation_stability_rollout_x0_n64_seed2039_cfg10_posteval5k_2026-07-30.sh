#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-scale-68ca820
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation5k_rollout_x0_n64_seed2039_cfg10
EXPECTED_CODE_REVISION=68ca820d5d901eff2623b97b86c4b88bf68ef500
COFITOK_CHECKPOINT="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0/checkpoint_step_00005000.pt"
DENSE_CHECKPOINT="$PAIR_ROOT/dense_identity/checkpoint_step_00005000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --short)" ]]
[[ -x "$PYTHON" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing rollout-x0 5K n64 CFG-1.0 diagnostic while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

for method in cofitok dense; do
  checkpoint="$COFITOK_CHECKPOINT"
  if [[ "$method" == dense ]]; then
    checkpoint="$DENSE_CHECKPOINT"
  fi
  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$checkpoint" \
    --output-dir "$OUTPUT_ROOT/model/${method}_rollout" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed 2039 \
    --weights model \
    --precision bf16 \
    --guidance-scale 1.0 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0
done
