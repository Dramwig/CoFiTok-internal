#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
BRIDGE_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1
REPORT_ROOT="$BRIDGE_ROOT/reports/min_snr_matched_50k_pilot_v1_20260825"
PREPARATION="$REPORT_ROOT/preparation.json"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
[[ -x "$PYTHON" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
mkdir -p "$REPORT_ROOT"

"$PYTHON" scripts/build_generation_min_snr_pilot_preparation.py \
  --post-diagnostic-decision "$BRIDGE_ROOT/reports/epsilon_stability_post_diagnostic_decision_v1_20260825/post_diagnostic_decision.json" \
  --expected-post-diagnostic-decision-sha256 b9ce02d19a01e48652277eecc82c12fa193b036b54e7d57edc52225105a34aff \
  --pair-monitor "$BRIDGE_ROOT/pair_monitor.json" \
  --legacy-cofitok-config configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json \
  --legacy-dense-config configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json \
  --pilot-cofitok-config configs/generation/imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json \
  --pilot-dense-config configs/generation/imagenet256_min_snr_gamma5_quality_repair_rollout_x0_u2_ema_teacher_dense_100k_horizon_50k_pilot.json \
  --legacy-cofitok-training-report "$BRIDGE_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/training_report.json" \
  --legacy-dense-training-report "$BRIDGE_ROOT/dense_rollout_x0_u2_ema_teacher/training_report.json" \
  --legacy-cofitok-checkpoint-audit-50k "$BRIDGE_ROOT/reports/checkpoint_audits/cofitok_checkpoint_step_00050000_physical_integrity_audit.json" \
  --legacy-dense-checkpoint-audit-50k "$BRIDGE_ROOT/reports/checkpoint_audits/dense_checkpoint_step_00050000_physical_integrity_audit.json" \
  --output-root "$OUTPUT_ROOT" \
  --output "$PREPARATION"

PREPARATION_SHA256="$(sha256sum "$PREPARATION" | awk '{print $1}')"
"$PYTHON" scripts/validate_generation_min_snr_pilot_preparation.py \
  --preparation "$PREPARATION" \
  --expected-sha256 "$PREPARATION_SHA256" >/dev/null
printf '%s  %s\n' "$PREPARATION_SHA256" "$PREPARATION"
