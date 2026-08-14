#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact capacity-full readiness checkout}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:?set SOURCE_OUTPUT_ROOT}
DECISION=${DECISION:?set DECISION to the immutable capacity-full readiness decision}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:?set EXPECTED_DECISION_SHA256}
EXPECTED_READINESS_REVISION=${EXPECTED_READINESS_REVISION:?set EXPECTED_READINESS_REVISION}
EXPECTED_READINESS_TREE=${EXPECTED_READINESS_TREE:?set EXPECTED_READINESS_TREE}
EXPECTED_READINESS_BRANCH=${EXPECTED_READINESS_BRANCH:?set EXPECTED_READINESS_BRANCH}
EXPECTED_DECISION_REVISION=${EXPECTED_DECISION_REVISION:?set EXPECTED_DECISION_REVISION}
EXPECTED_DECISION_TREE=${EXPECTED_DECISION_TREE:?set EXPECTED_DECISION_TREE}
EXPECTED_DECISION_BRANCH=${EXPECTED_DECISION_BRANCH:?set EXPECTED_DECISION_BRANCH}
EXPECTED_RESULT_REVISION=${EXPECTED_RESULT_REVISION:?set EXPECTED_RESULT_REVISION}
EXPECTED_RESULT_TREE=${EXPECTED_RESULT_TREE:?set EXPECTED_RESULT_TREE}
EXPECTED_RESULT_BRANCH=${EXPECTED_RESULT_BRANCH:?set EXPECTED_RESULT_BRANCH}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}

COFITOK_CONFIG=${COFITOK_CONFIG:-$PROJECT/configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json}
DENSE_CONFIG=${DENSE_CONFIG:-$PROJECT/configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}
REPORT_ROOT=${REPORT_ROOT:-$FULL_OUTPUT_ROOT/reports}
COFITOK_RUN_DIR=${COFITOK_RUN_DIR:-$FULL_OUTPUT_ROOT/cofitok}
DENSE_RUN_DIR=${DENSE_RUN_DIR:-$FULL_OUTPUT_ROOT/dense_identity}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$REPORT_ROOT/runtime_benchmark}
CONFIG_VALIDATION=${CONFIG_VALIDATION:-$REPORT_ROOT/config_validation.json}
STORAGE_CAPACITY=${STORAGE_CAPACITY:-$REPORT_ROOT/storage_capacity.json}
RUNTIME_SELECTION=${RUNTIME_SELECTION:-$REPORT_ROOT/runtime_selection.json}
READINESS=${READINESS:-$REPORT_ROOT/capacity_full_300k_readiness.json}
REFERENCE_COFITOK=${REFERENCE_COFITOK:-$SOURCE_OUTPUT_ROOT/base256_cofitok/checkpoint_step_00100000.pt}
REFERENCE_DENSE=${REFERENCE_DENSE:-$SOURCE_OUTPUT_ROOT/base256_dense_identity/checkpoint_step_00100000.pt}
LOCK=${LOCK:-$FULL_OUTPUT_ROOT/readiness.lock}

[[ -x "$PYTHON" ]]
[[ -d "$PROJECT/.git" || -f "$PROJECT/.git" ]]
[[ -d "$CHECKPOINT_ROOT" ]]
[[ -f "$DECISION" ]]
[[ -f "$COFITOK_CONFIG" ]]
[[ -f "$DENSE_CONFIG" ]]
[[ -f "$REFERENCE_COFITOK" ]]
[[ -f "$REFERENCE_DENSE" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED_READINESS_REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_READINESS_TREE" ]]
[[ "$(git -C "$PROJECT" branch --show-current)" == "$EXPECTED_READINESS_BRANCH" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
[[ "$(sha256sum "$DECISION" | awk '{print $1}')" == "$EXPECTED_DECISION_SHA256" ]]

mkdir -p "$FULL_OUTPUT_ROOT"
exec 7>"$LOCK"
flock -n 7 || {
  printf 'refusing duplicate capacity-full readiness execution\n' >&2
  exit 75
}

for path in \
  "$CONFIG_VALIDATION" \
  "$STORAGE_CAPACITY" \
  "$RUNTIME_SELECTION" \
  "$READINESS" \
  "$BENCHMARK_ROOT"; do
  if [[ -e "$path" ]]; then
    printf 'refusing to overwrite capacity-full readiness evidence: %s\n' "$path" >&2
    exit 8
  fi
done
for run_dir in "$COFITOK_RUN_DIR" "$DENSE_RUN_DIR"; do
  if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
    printf 'capacity-full training path is not a directory: %s\n' "$run_dir" >&2
    exit 8
  fi
  if [[ -d "$run_dir" ]] \
    && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'refusing readiness after capacity-full training state exists: %s\n' \
      "$run_dir" >&2
    exit 8
  fi
done

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/validate_generation_capacity_full_300k_readiness_decision.py \
  --decision "$DECISION" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --expected-decision-revision "$EXPECTED_DECISION_REVISION" \
  --expected-decision-tree "$EXPECTED_DECISION_TREE" \
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH" \
  --expected-result-revision "$EXPECTED_RESULT_REVISION" \
  --expected-result-tree "$EXPECTED_RESULT_TREE" \
  --expected-result-branch "$EXPECTED_RESULT_BRANCH" >/dev/null

# Recheck immediately before the first command that can allocate CUDA memory.
# Exit 9 is a bounded launch race, not permission to signal or share the GPU.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; then
  printf 'refusing capacity-full readiness benchmark while the GPU is busy\n' >&2
  exit 9
fi

mkdir -p "$REPORT_ROOT"

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
  --checkpoint-size-multiplier 1.0 \
  --sample-count 116640 \
  --estimated-sample-kib 256 \
  --additional-gib 16 \
  --safety-margin-gib 64 >/dev/null

"$PYTHON" scripts/select_generation_training_runtime.py \
  --project-root "$PROJECT" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --output-root "$BENCHMARK_ROOT" \
  --output "$RUNTIME_SELECTION" \
  --training-run-dir "$COFITOK_RUN_DIR" \
  --training-run-dir "$DENSE_RUN_DIR" \
  --candidates 1x64,2x32,4x16,8x8,16x4 \
  --baseline-candidate 1x64 \
  --effective-batch-size 64 \
  --benchmark-steps 8 \
  --warmup-steps 2 \
  --max-memory-fraction 0.90 >/dev/null

"$PYTHON" scripts/build_generation_capacity_full_300k_readiness.py \
  --decision "$DECISION" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --config-validation "$CONFIG_VALIDATION" \
  --storage-capacity "$STORAGE_CAPACITY" \
  --runtime-selection "$RUNTIME_SELECTION" \
  --cofitok-run-dir "$COFITOK_RUN_DIR" \
  --dense-run-dir "$DENSE_RUN_DIR" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --storage-path "$CHECKPOINT_ROOT" \
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION" \
  --expected-readiness-tree "$EXPECTED_READINESS_TREE" \
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH" \
  --expected-decision-revision "$EXPECTED_DECISION_REVISION" \
  --expected-decision-tree "$EXPECTED_DECISION_TREE" \
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH" \
  --expected-result-revision "$EXPECTED_RESULT_REVISION" \
  --expected-result-tree "$EXPECTED_RESULT_TREE" \
  --expected-result-branch "$EXPECTED_RESULT_BRANCH" \
  --output "$READINESS" >/dev/null

"$PYTHON" scripts/verify_generation_capacity_full_300k_readiness.py \
  --readiness "$READINESS" \
  --decision "$DECISION" \
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" \
  --cofitok-config "$COFITOK_CONFIG" \
  --dense-config "$DENSE_CONFIG" \
  --config-validation "$CONFIG_VALIDATION" \
  --storage-capacity "$STORAGE_CAPACITY" \
  --runtime-selection "$RUNTIME_SELECTION" \
  --cofitok-run-dir "$COFITOK_RUN_DIR" \
  --dense-run-dir "$DENSE_RUN_DIR" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --storage-path "$CHECKPOINT_ROOT" \
  --expected-readiness-revision "$EXPECTED_READINESS_REVISION" \
  --expected-readiness-tree "$EXPECTED_READINESS_TREE" \
  --expected-readiness-branch "$EXPECTED_READINESS_BRANCH" \
  --expected-decision-revision "$EXPECTED_DECISION_REVISION" \
  --expected-decision-tree "$EXPECTED_DECISION_TREE" \
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH" \
  --expected-result-revision "$EXPECTED_RESULT_REVISION" \
  --expected-result-tree "$EXPECTED_RESULT_TREE" \
  --expected-result-branch "$EXPECTED_RESULT_BRANCH" >/dev/null

printf 'capacity-full readiness: %s  %s\n' \
  "$(sha256sum "$READINESS" | awk '{print $1}')" "$READINESS"
