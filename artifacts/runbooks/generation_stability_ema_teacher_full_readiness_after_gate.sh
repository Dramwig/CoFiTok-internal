#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EXPECTED_SCALING_GATE_SHA256=${EXPECTED_SCALING_GATE_SHA256:?set the passing stability scaling gate SHA256}
EXPECTED_DEPLOYMENT_RECEIPT_SHA256=${EXPECTED_DEPLOYMENT_RECEIPT_SHA256:?set the isolated deployment receipt SHA256}
EXPECTED_TARGET_REVISION=${EXPECTED_TARGET_REVISION:?set the clean stability-full training revision}
EXPECTED_TARGET_BRANCH=${EXPECTED_TARGET_BRANCH:?set the clean stability-full training branch}

SCALING_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
GATE="$SCALING_ROOT/reports/promotion_gate.json"
REFERENCE_COFITOK="$SCALING_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
REFERENCE_DENSE="$SCALING_ROOT/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt"
COFITOK_CONFIG=configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json
DENSE_CONFIG=configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json
OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_300k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
RUNTIME_BENCHMARK_ROOT="$OUTPUT_ROOT/runtime_preflight/training"
CONFIG_VALIDATION="$REPORT_ROOT/config_validation.json"
STORAGE_CAPACITY="$REPORT_ROOT/storage_capacity.json"
RUNTIME_SELECTION="$REPORT_ROOT/runtime_selection.json"
READINESS="$REPORT_ROOT/full_training_readiness.json"
DEPLOYMENT_RECEIPT="$CHECKPOINT_ROOT/deployment/large_capacity/deployments/$EXPECTED_TARGET_REVISION/deployment_receipt.json"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_TARGET_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_TARGET_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ -f "$GATE" ]]
[[ -f "$DEPLOYMENT_RECEIPT" ]]
[[ -f "$REFERENCE_COFITOK" ]]
[[ -f "$REFERENCE_DENSE" ]]
[[ "$(sha256sum "$GATE" | awk '{print $1}')" == "$EXPECTED_SCALING_GATE_SHA256" ]]
[[ "$(sha256sum "$DEPLOYMENT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" ]]
for path in \
  "$CONFIG_VALIDATION" \
  "$STORAGE_CAPACITY" \
  "$RUNTIME_SELECTION" \
  "$READINESS" \
  "$RUNTIME_BENCHMARK_ROOT"; do
  if [[ -e "$path" ]]; then
    printf 'refusing to overwrite stability-full readiness evidence: %s\n' "$path" >&2
    exit 8
  fi
done
for run_dir in "$COFITOK_RUN" "$DENSE_RUN"; do
  if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
    printf 'stability-full training path is not a directory: %s\n' "$run_dir" >&2
    exit 8
  fi
  if [[ -d "$run_dir" ]] \
    && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'refusing readiness after stability-full training state exists: %s\n' \
      "$run_dir" >&2
    exit 8
  fi
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing stability-full readiness while the GPU is busy\n' >&2
  exit 9
fi
mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT"

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$GATE" \
  --stage scaling >/dev/null

"$PYTHON" scripts/validate_generation_large_capacity_deployment.py \
  --receipt "$DEPLOYMENT_RECEIPT" \
  --expected-receipt-sha256 "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" >/dev/null

"$PYTHON" scripts/validate_generation_configs.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --stage stability_full \
  --output "$CONFIG_VALIDATION" >/dev/null

"$PYTHON" scripts/check_generation_storage_capacity.py \
  --path "$CHECKPOINT_ROOT" \
  --output "$STORAGE_CAPACITY" \
  --stage full_training \
  --reference-checkpoint "$REFERENCE_COFITOK" \
  --reference-checkpoint "$REFERENCE_DENSE" \
  --checkpoint-count 16 \
  --checkpoint-size-multiplier 4.0 \
  --sample-count 16384 \
  --estimated-sample-kib 256 \
  --additional-gib 16 \
  --safety-margin-gib 64 >/dev/null

"$PYTHON" scripts/select_generation_training_runtime.py \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --output-root "$RUNTIME_BENCHMARK_ROOT" \
  --output "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN" \
  --training-run-dir "$DENSE_RUN" \
  --candidates 1x64,2x32,4x16,8x8,16x4 \
  --baseline-candidate 1x64 \
  --effective-batch-size 64 \
  --benchmark-steps 8 \
  --warmup-steps 2 \
  --max-memory-fraction 0.90 >/dev/null

"$PYTHON" scripts/build_generation_full_readiness.py \
  --project-root "$PROJECT" \
  --deployment-receipt "$DEPLOYMENT_RECEIPT" \
  --expected-deployment-receipt-sha256 "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" \
  --promotion-gate "$GATE" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --config-validation "$CONFIG_VALIDATION" \
  --storage-capacity "$STORAGE_CAPACITY" \
  --runtime-selection "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN" \
  --training-run-dir "$DENSE_RUN" \
  --benchmark-root "$RUNTIME_BENCHMARK_ROOT" \
  --storage-path "$CHECKPOINT_ROOT" \
  --expected-revision "$EXPECTED_TARGET_REVISION" \
  --expected-branch "$EXPECTED_TARGET_BRANCH" \
  --expected-gate-sha256 "$EXPECTED_SCALING_GATE_SHA256" \
  --output "$READINESS" >/dev/null

printf 'stability-full readiness: %s  %s\n' \
  "$(sha256sum "$READINESS" | awk '{print $1}')" "$READINESS"
