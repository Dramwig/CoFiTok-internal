#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/CoFiTok/CoFiTok-internal

PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
REPORT=artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json

"${PYTHON}" scripts/baselines/probe_dar_adapter.py \
  --repo /root/autodl-tmp/CoFiTok/baselines/repos/d_ar \
  --output "${REPORT}"

echo "d_ar_feasibility_report=${REPORT}"
