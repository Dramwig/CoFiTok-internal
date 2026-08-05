#!/usr/bin/env bash
set -euo pipefail

EXPECTED_AUDIT_REVISION=${EXPECTED_AUDIT_REVISION:?set the exact convergence-audit revision}
EXPECTED_AUDIT_BRANCH=${EXPECTED_AUDIT_BRANCH:-scale/generation-stability-50k-convergence-audit-v1}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
SCALING_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher
FULL_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_300k_ema_teacher
AUDIT_DIR="$SCALING_ROOT/reports/frozen_training_convergence_audit"
AUDIT_REPORT="$AUDIT_DIR/convergence_audit.json"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH=.:src
export PYTHONDONTWRITEBYTECODE=1

[[ "$(git rev-parse HEAD)" == "$EXPECTED_AUDIT_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_AUDIT_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ ! -e "$AUDIT_DIR" ]]

nice -n 15 ionice -c3 "$PYTHON" scripts/build_generation_stability_50k_convergence_audit.py \
  --scaling-root "$SCALING_ROOT" \
  --full-root "$FULL_ROOT" \
  --expected-audit-revision "$EXPECTED_AUDIT_REVISION" \
  --expected-audit-branch "$EXPECTED_AUDIT_BRANCH" \
  --output "$AUDIT_REPORT"

REPLAY_REPORT="$AUDIT_DIR/.convergence_audit.replay.json"
nice -n 15 ionice -c3 "$PYTHON" scripts/build_generation_stability_50k_convergence_audit.py \
  --scaling-root "$SCALING_ROOT" \
  --full-root "$FULL_ROOT" \
  --expected-audit-revision "$EXPECTED_AUDIT_REVISION" \
  --expected-audit-branch "$EXPECTED_AUDIT_BRANCH" \
  --output "$REPLAY_REPORT"
cmp "$AUDIT_REPORT" "$REPLAY_REPORT"
rm -f -- "$REPLAY_REPORT"

"$PYTHON" - "$AUDIT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "pass"
assert report["evidence_synthesis"]["decision"] == (
    "sampling_recovery_first_then_full_data_scale_if_not_recovered"
)
boundary = report["claim_boundary"]
assert boundary["gpu_work_performed"] is False
assert boundary["new_sampling_performed"] is False
assert boundary["new_training_performed"] is False
assert boundary["sampling_execution_allowed"] is False
assert boundary["quality_bridge_execution_allowed"] is False
assert boundary["full_training_launch_allowed"] is False
assert boundary["full_300k_launch_allowed"] is False
PY

sha256sum "$AUDIT_REPORT"
