#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean quality-bridge preparation revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean quality-bridge preparation branch}
EXPECTED_PROMOTION_GATE_SHA256=${EXPECTED_PROMOTION_GATE_SHA256:?set the frozen failed stability gate SHA256}

SOURCE_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
PROMOTION_GATE="$SOURCE_ROOT/reports/promotion_gate.json"
REFERENCE_COFITOK="$SOURCE_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
REFERENCE_DENSE="$SOURCE_ROOT/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
COFITOK_CONFIG=configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_base128_quality_bridge_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
PREPARATION="$REPORT_ROOT/preparation.json"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
GATE_SOURCE_VALIDATION="$REPORT_ROOT/source_gate_validation.json"
STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity_preparation.json"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$PROMOTION_GATE" ]]
[[ -f "$REFERENCE_COFITOK" ]]
[[ -f "$REFERENCE_DENSE" ]]
[[ "$(sha256sum "$PROMOTION_GATE" | awk '{print $1}')" == "$EXPECTED_PROMOTION_GATE_SHA256" ]]
mkdir -p "$REPORT_ROOT"

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$PROMOTION_GATE" \
  --stage scaling \
  --sources-only >"$GATE_SOURCE_VALIDATION"

"$PYTHON" scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_quality_bridge \
  --output "$CONFIG_VALIDATION" >/dev/null

"$PYTHON" scripts/build_generation_quality_bridge_preparation.py \
  --promotion-gate "$PROMOTION_GATE" \
  --expected-promotion-gate-sha256 "$EXPECTED_PROMOTION_GATE_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --output "$PREPARATION" >/dev/null

PREPARATION_SHA256="$(sha256sum "$PREPARATION" | awk '{print $1}')"
"$PYTHON" scripts/validate_generation_quality_bridge_preparation.py \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$PREPARATION_SHA256" \
  --promotion-gate "$PROMOTION_GATE" \
  --expected-promotion-gate-sha256 "$EXPECTED_PROMOTION_GATE_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" >/dev/null

"$PYTHON" scripts/check_generation_storage_capacity.py \
  --path "$CHECKPOINT_ROOT" \
  --output "$STORAGE_CAPACITY" \
  --stage stability_full_data_quality_bridge_100k \
  --reference-checkpoint "$REFERENCE_COFITOK" \
  --reference-checkpoint "$REFERENCE_DENSE" \
  --checkpoint-count 10 \
  --sample-count 30000 \
  --estimated-sample-kib 256 \
  --additional-gib 16 \
  --safety-margin-gib 64 >/dev/null

printf 'quality bridge remains non-authorizing; explicit execution approval is required\n'
printf '%s  %s\n' "$PREPARATION_SHA256" "$PREPARATION"
