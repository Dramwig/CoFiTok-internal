#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
REPO="${REPO:-$PROJECT_ROOT/baselines/repos/improved_diffusion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/baselines/adapters/improved_diffusion:$REPO:$CODE_DIR/src"
mkdir -p "artifacts/reports/baselines/improved_diffusion"

"$PYTHON" scripts/baselines/probe_improved_diffusion_adapter.py \
  --repo "$REPO" \
  --adapter "$CODE_DIR/baselines/adapters/improved_diffusion" \
  --output "artifacts/reports/baselines/improved_diffusion/adapter_probe_${DATE_TAG}.json"

echo "improved_diffusion_adapter_probe_done"
