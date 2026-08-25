#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
BRIDGE_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1
REPORT_ROOT="$BRIDGE_ROOT/reports/min_snr_matched_50k_pilot_v1_20260825"
PREPARATION="$REPORT_ROOT/preparation.json"
EXECUTION_GATE="$REPORT_ROOT/execution_gate.json"
EXECUTION_LOCK=/tmp/cofitok-min-snr-matched-50k-pilot-v1.lock
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set immutable preparation SHA256}
USER_INSTRUCTION=${USER_INSTRUCTION:?set the direct user execution instruction}

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ -x "$PYTHON" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "$PREPARATION" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ ! -e "$EXECUTION_GATE" ]]

"$PYTHON" scripts/build_generation_min_snr_pilot_execution_gate.py \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --dataset-root /root/autodl-tmp/CoFiTok/datasets \
  --storage-path /root/autodl-tmp/CoFiTok/checkpoints/generation \
  --output-root "$OUTPUT_ROOT" \
  --execution-lock "$EXECUTION_LOCK" \
  --user-instruction "$USER_INSTRUCTION" \
  --output "$EXECUTION_GATE"

GATE_SHA256="$(sha256sum "$EXECUTION_GATE" | awk '{print $1}')"
"$PYTHON" scripts/validate_generation_min_snr_pilot_execution_gate.py \
  --gate "$EXECUTION_GATE" \
  --expected-gate-sha256 "$GATE_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" >/dev/null
printf '%s  %s\n' "$GATE_SHA256" "$EXECUTION_GATE"
