#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
GATE="$PROJECT/artifacts/reports/generation/imagenet256_10pct_matched_50k_2026-07-12/promotion_gate.json"
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_full_cofitok_k8_300k"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_full_dense_300k"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

python - "$GATE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    gate = json.load(handle)
if gate.get("status") != "pass" or gate.get("decision") != "promote_to_full_imagenet256":
    raise SystemExit("10% generation gate did not authorize full ImageNet-256 training")
PY

require_complete() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("training_complete") is not True:
    raise SystemExit(f"training did not reach its target: {sys.argv[1]}")
if report.get("completed_steps") != report.get("target_steps"):
    raise SystemExit(f"training step mismatch: {sys.argv[1]}")
PY
}

if [[ -f "$COFITOK_RUN/latest.json" ]]; then
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_cofitok_k8_300k.json \
    --output-dir "$COFITOK_RUN" --resume auto
else
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_cofitok_k8_300k.json \
    --output-dir "$COFITOK_RUN"
fi
require_complete "$COFITOK_RUN/training_report.json"

if [[ -f "$DENSE_RUN/latest.json" ]]; then
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_dense_300k.json \
    --output-dir "$DENSE_RUN" --resume auto
else
  python scripts/train_generation.py \
    --config configs/generation/imagenet256_dense_300k.json \
    --output-dir "$DENSE_RUN"
fi
require_complete "$DENSE_RUN/training_report.json"

