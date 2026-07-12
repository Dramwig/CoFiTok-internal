#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_10pct_matched_50k_2026-07-12"
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_dense_50k.json \
  --output "$REPORT_ROOT/config_pair.json"

if [[ -f "$COFITOK_RUN/latest.json" ]]; then
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
    --output-dir "$COFITOK_RUN" \
    --resume auto
else
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
    --output-dir "$COFITOK_RUN"
fi

if [[ -f "$DENSE_RUN/latest.json" ]]; then
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_10pct_dense_50k.json \
    --output-dir "$DENSE_RUN" \
    --resume auto
else
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_10pct_dense_50k.json \
    --output-dir "$DENSE_RUN"
fi

