#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-27edb63}
GENERATION_ROOT=${GENERATION_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
PROBE_ROOT=${PROBE_ROOT:-$GENERATION_ROOT/stability_probe_2026-07-29}
REPORT_ROOT=${REPORT_ROOT:-$GENERATION_ROOT/retention_audit_2026-08-01}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
REQUIRED_FREE_BYTES=${REQUIRED_FREE_BYTES:-180880415360}

mkdir -p "$REPORT_ROOT"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_generation_checkpoint_retention_inventory.py" \
  --inventory-root "$PROBE_ROOT" \
  --reference-root "tracked_probe=$PROJECT_ROOT/artifacts/reports/generation/stability_probe_2026-07-29=authoritative" \
  --reference-root "tracked_records=$PROJECT_ROOT/docs/records=authoritative" \
  --physical-hash \
  --output "$REPORT_ROOT/checkpoint_retention_inventory.json"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_generation_checkpoint_retention_inventory.py" \
  --inventory "$REPORT_ROOT/checkpoint_retention_inventory.json"
INVENTORY_SHA256="$(sha256sum "$REPORT_ROOT/checkpoint_retention_inventory.json" | awk '{print $1}')"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_generation_retention_runway.py" \
  --retention-inventory "$REPORT_ROOT/checkpoint_retention_inventory.json" \
  --expected-retention-inventory-sha256 "$INVENTORY_SHA256" \
  --path "$GENERATION_ROOT" \
  --required-free-bytes "$REQUIRED_FREE_BYTES" \
  --output "$REPORT_ROOT/retention_runway.json"
