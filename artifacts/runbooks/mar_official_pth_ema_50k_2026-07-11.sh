#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/CoFiTok
INTERNAL_ROOT="$PROJECT_ROOT/CoFiTok-internal"
OUTPUT_ROOT="$PROJECT_ROOT/checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_official_pth"
SAMPLE_STEM=MAR-B-official-pth-ema-size-256-ariter-256-diffsteps-100-cfg-2.9-linear-temp-1.0-seed-0
SAMPLE_DIR="$OUTPUT_ROOT/$SAMPLE_STEM"
SAMPLE_NPZ="$OUTPUT_ROOT/$SAMPLE_STEM.npz"
REFERENCE_NPZ="$PROJECT_ROOT/checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR_ROOT="$PROJECT_ROOT/checkpoints/baselines/official_refs"
EVALUATOR_SCRIPT="$PROJECT_ROOT/baselines/repos/d_ar/evaluations/c2i/evaluator.py"
LOG_PATH="$OUTPUT_ROOT/mar_official_pth_ema_50k_2026-07-11.log"
STATUS_PATH="$INTERNAL_ROOT/artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.status"

mkdir -p "$OUTPUT_ROOT" "$SAMPLE_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

finish() {
  local code=$?
  if [[ "$code" == 0 ]]; then
    printf 'completed\n' > "$STATUS_PATH"
  else
    printf 'failed:%s\n' "$code" > "$STATUS_PATH"
  fi
  date --iso-8601=seconds
}
trap finish EXIT

printf 'running\n' > "$STATUS_PATH"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm

cd "$INTERNAL_ROOT"
PYTHONUNBUFFERED=1 python scripts/baselines/sample_mar_official_pth.py \
  --output-dir "$SAMPLE_DIR" \
  --num-images 50000 \
  --class-num 1000 \
  --batch-size 768 \
  --decode-batch-size 16 \
  --num-iter 256 \
  --num-sampling-steps 100 \
  --cfg 2.9 \
  --cfg-schedule linear \
  --temperature 1.0 \
  --precision fp16 \
  --seed 0 \
  --pack-npz "$SAMPLE_NPZ"

cd "$EVALUATOR_ROOT"
python "$EVALUATOR_SCRIPT" "$REFERENCE_NPZ" "$SAMPLE_NPZ"

cd "$INTERNAL_ROOT"
python scripts/baselines/build_official_related_methods_table.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir artifacts/reports/baselines/official_related_methods_2026-07-11_mar_ema
