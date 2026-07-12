#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
REPO="${REPO:-$PROJECT_ROOT/baselines/repos/edm}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"

cd "$CODE_DIR"
mkdir -p artifacts/reports/baselines/edm
PYTHONPATH="$REPO:$CODE_DIR/src:${PYTHONPATH:-}" "$PYTHON" scripts/baselines/probe_edm_adapter.py \
  --repo "$REPO" \
  --output "artifacts/reports/baselines/edm/adapter_probe_${DATE_TAG}.json"

echo "edm_adapter_probe_done"
