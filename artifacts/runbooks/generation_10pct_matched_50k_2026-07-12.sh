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

require_complete() {
  python - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training did not reach its target: {path}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"training step mismatch: {path}")
PY
}

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
require_complete "$COFITOK_RUN/training_report.json"

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
require_complete "$DENSE_RUN/training_report.json"
