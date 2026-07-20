#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k_v3"
CHECKPOINT="$RUN/checkpoint_step_00005000.pt"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'timestep diagnostic requires a clean tracked worktree\n' >&2
  exit 66
fi
if pgrep -f '[g]eneration_rank_recovery_equal_progress_probe_2026-07-20.sh' >/dev/null; then
  printf 'equal-progress probe runbook is still active\n' >&2
  exit 67
fi
if pgrep -f '[t]rain_generation.py|[g]enerate_samples.py|[e]valuate_generation_checkpoint.py' >/dev/null; then
  printf 'another generation GPU process is still active\n' >&2
  exit 68
fi

test -f "$CHECKPOINT"
test -f "$CHECKPOINT.integrity.json"
test -f "$RUN/training_report.json"

for timestep in 50 250 500 750 950; do
  output_dir="$RUN/checkpoint_eval_ema_t${timestep}_256_energy_scope"
  if [[ ! -f "$output_dir/checkpoint_evaluation_report.json" ]]; then
    python scripts/evaluate_generation_checkpoint.py \
      --checkpoint "$CHECKPOINT" --output-dir "$output_dir" \
      --num-images 256 --timestep "$timestep" --random-orders 16 \
      --weights ema --precision bf16
  fi
done
