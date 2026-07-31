#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_FINAL_GATE_SHA256=${EXPECTED_FINAL_GATE_SHA256:?set the passing stability-full final gate SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean inference-export revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean inference-export branch}

FULL_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
COFITOK_CHECKPOINT="$FULL_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00300000.pt"
DENSE_CHECKPOINT="$FULL_ROOT/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00300000.pt"
FINAL_GATE="$FULL_ROOT/reports/final_generation_gate.json"
EXPORT_ROOT="$CHECKPOINT_ROOT/exports/stability_full_300k_ema_teacher"
REPORT_ROOT="$FULL_ROOT/reports/exports"
COFITOK_ARTIFACT="$EXPORT_ROOT/cofitok_k8_ema_inference.pt"
DENSE_ARTIFACT="$EXPORT_ROOT/dense_identity_ema_inference.pt"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$COFITOK_CHECKPOINT" ]]
[[ -f "$DENSE_CHECKPOINT" ]]
[[ -f "$FINAL_GATE" ]]
[[ "$(sha256sum "$FINAL_GATE" | awk '{print $1}')" == "$EXPECTED_FINAL_GATE_SHA256" ]]
mkdir -p "$EXPORT_ROOT" "$REPORT_ROOT"

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$FINAL_GATE" \
  --stage full >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing stability inference export while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" scripts/export_generation_inference_artifact.py \
  --checkpoint "$COFITOK_CHECKPOINT" \
  --output "$COFITOK_ARTIFACT" \
  --release-gate "$FINAL_GATE" \
  --report "$REPORT_ROOT/cofitok_export_report.json"

"$PYTHON" scripts/export_generation_inference_artifact.py \
  --checkpoint "$DENSE_CHECKPOINT" \
  --output "$DENSE_ARTIFACT" \
  --release-gate "$FINAL_GATE" \
  --report "$REPORT_ROOT/dense_export_report.json"

"$PYTHON" scripts/preflight_generation_sampling.py \
  --checkpoint "$COFITOK_ARTIFACT" \
  --output "$REPORT_ROOT/cofitok_export_preflight.json" \
  --batch-size 2 \
  --prefix-budget 8 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --require-release-authorization \
  --warmup-forwards 1 \
  --measured-forwards 2

"$PYTHON" scripts/preflight_generation_sampling.py \
  --checkpoint "$DENSE_ARTIFACT" \
  --output "$REPORT_ROOT/dense_export_preflight.json" \
  --batch-size 2 \
  --prefix-budget 1 \
  --guidance-scale 1.5 \
  --cfg-batch-mode batched \
  --weights ema \
  --precision bf16 \
  --require-release-authorization \
  --warmup-forwards 1 \
  --measured-forwards 2

"$PYTHON" scripts/infer_generation.py \
  --checkpoint "$COFITOK_ARTIFACT" \
  --output-dir "$EXPORT_ROOT/smoke/cofitok" \
  --report "$REPORT_ROOT/cofitok_export_inference_smoke.json" \
  --class-ids 0 \
  --seeds 0,1 \
  --prefix-budgets 1,8 \
  --batch-size 2 \
  --sample-steps 10 \
  --guidance-scale 1.5 \
  --weights ema \
  --precision bf16 \
  --require-release-authorization \
  --overwrite

"$PYTHON" scripts/infer_generation.py \
  --checkpoint "$DENSE_ARTIFACT" \
  --output-dir "$EXPORT_ROOT/smoke/dense_identity" \
  --report "$REPORT_ROOT/dense_export_inference_smoke.json" \
  --class-ids 0 \
  --seeds 0,1 \
  --prefix-budgets 1 \
  --batch-size 2 \
  --sample-steps 10 \
  --guidance-scale 1.5 \
  --weights ema \
  --precision bf16 \
  --require-release-authorization \
  --overwrite
