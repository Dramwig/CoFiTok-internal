#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the exact exposure-qualification checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact preparation revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact preparation tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact preparation branch}
EXPECTED_BUILDER_SHA256=${EXPECTED_BUILDER_SHA256:?set the builder SHA256}
EXPECTED_VERIFIER_SHA256=${EXPECTED_VERIFIER_SHA256:?set the verifier SHA256}
EXPECTED_RUNBOOK_SHA256=${EXPECTED_RUNBOOK_SHA256:?set this runbook SHA256}

QUALITY_BRIDGE_ROOT=${QUALITY_BRIDGE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1}
CAPACITY_PROBE_ROOT=${CAPACITY_PROBE_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_300k_base128_exposure_qualification_v1}
FOLLOWUP_DECISION=${FOLLOWUP_DECISION:-$QUALITY_BRIDGE_ROOT/reports/followup_experiment_decision_exposure_aware_v2.json}
CAPACITY_PROBE_RESULT=${CAPACITY_PROBE_RESULT:-$CAPACITY_PROBE_ROOT/reports/capacity_probe_result.json}
EXPECTED_FOLLOWUP_DECISION_SHA256=${EXPECTED_FOLLOWUP_DECISION_SHA256:?set the v2 decision SHA256}
EXPECTED_CAPACITY_PROBE_RESULT_SHA256=${EXPECTED_CAPACITY_PROBE_RESULT_SHA256:?set the capacity result SHA256}
EXPECTED_FOLLOWUP_REVISION=${EXPECTED_FOLLOWUP_REVISION:?set the v2 decision revision}
EXPECTED_FOLLOWUP_BRANCH=${EXPECTED_FOLLOWUP_BRANCH:?set the v2 decision branch}
EXPECTED_CAPACITY_REVISION=${EXPECTED_CAPACITY_REVISION:?set the capacity result revision}
EXPECTED_CAPACITY_BRANCH=${EXPECTED_CAPACITY_BRANCH:?set the capacity result branch}

BUILDER="$PROJECT/scripts/build_generation_training_exposure_qualification.py"
VERIFIER="$PROJECT/scripts/verify_generation_training_exposure_qualification.py"
RUNBOOK=$(readlink -f "${BASH_SOURCE[0]}")
SOURCE_COFITOK_CONFIG="$PROJECT/configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
SOURCE_DENSE_CONFIG="$PROJECT/configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json"
TARGET_COFITOK_CONFIG="$PROJECT/configs/generation/imagenet256_stability_exposure_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
TARGET_DENSE_CONFIG="$PROJECT/configs/generation/imagenet256_stability_exposure_rollout_x0_u2_ema_teacher_dense_300k.json"
PREPARATION="$OUTPUT_ROOT/reports/training_exposure_qualification_preparation.json"
TEMP_PREPARATION="$PREPARATION.tmp.$$"
LOCK=/tmp/cofitok-training-exposure-qualification-prepare.lock

cd "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_REVISION"
test "$(git rev-parse 'HEAD^{tree}')" = "$EXPECTED_TREE"
test "$(git branch --show-current)" = "$EXPECTED_BRANCH"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$EXPECTED_BUILDER_SHA256"
test "$(sha256sum "$VERIFIER" | awk '{print $1}')" = "$EXPECTED_VERIFIER_SHA256"
test "$(sha256sum "$RUNBOOK" | awk '{print $1}')" = "$EXPECTED_RUNBOOK_SHA256"
test ! -e "$PREPARATION"

exec 9>"$LOCK"
flock -n 9 || {
  printf 'refusing a concurrent exposure-qualification preparation\n' >&2
  exit 75
}
trap 'rm -f -- "$TEMP_PREPARATION"' EXIT

env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/build_generation_training_exposure_qualification.py \
    --followup-decision "$FOLLOWUP_DECISION" \
    --expected-followup-decision-sha256 "$EXPECTED_FOLLOWUP_DECISION_SHA256" \
    --capacity-probe-result "$CAPACITY_PROBE_RESULT" \
    --expected-capacity-probe-result-sha256 "$EXPECTED_CAPACITY_PROBE_RESULT_SHA256" \
    --source-cofitok-config "$SOURCE_COFITOK_CONFIG" \
    --source-dense-config "$SOURCE_DENSE_CONFIG" \
    --target-cofitok-config "$TARGET_COFITOK_CONFIG" \
    --target-dense-config "$TARGET_DENSE_CONFIG" \
    --expected-followup-revision "$EXPECTED_FOLLOWUP_REVISION" \
    --expected-followup-branch "$EXPECTED_FOLLOWUP_BRANCH" \
    --expected-capacity-revision "$EXPECTED_CAPACITY_REVISION" \
    --expected-capacity-branch "$EXPECTED_CAPACITY_BRANCH" \
    --expected-preparation-revision "$EXPECTED_REVISION" \
    --expected-preparation-branch "$EXPECTED_BRANCH" \
    --output-root "$OUTPUT_ROOT" \
    --output "$TEMP_PREPARATION"

PREPARATION_SHA256=$(sha256sum "$TEMP_PREPARATION" | awk '{print $1}')
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/verify_generation_training_exposure_qualification.py \
    --followup-decision "$FOLLOWUP_DECISION" \
    --expected-followup-decision-sha256 "$EXPECTED_FOLLOWUP_DECISION_SHA256" \
    --capacity-probe-result "$CAPACITY_PROBE_RESULT" \
    --expected-capacity-probe-result-sha256 "$EXPECTED_CAPACITY_PROBE_RESULT_SHA256" \
    --source-cofitok-config "$SOURCE_COFITOK_CONFIG" \
    --source-dense-config "$SOURCE_DENSE_CONFIG" \
    --target-cofitok-config "$TARGET_COFITOK_CONFIG" \
    --target-dense-config "$TARGET_DENSE_CONFIG" \
    --expected-followup-revision "$EXPECTED_FOLLOWUP_REVISION" \
    --expected-followup-branch "$EXPECTED_FOLLOWUP_BRANCH" \
    --expected-capacity-revision "$EXPECTED_CAPACITY_REVISION" \
    --expected-capacity-branch "$EXPECTED_CAPACITY_BRANCH" \
    --expected-preparation-revision "$EXPECTED_REVISION" \
    --expected-preparation-branch "$EXPECTED_BRANCH" \
    --output-root "$OUTPUT_ROOT" \
    --preparation "$TEMP_PREPARATION" \
    --expected-preparation-sha256 "$PREPARATION_SHA256"

mv -T "$TEMP_PREPARATION" "$PREPARATION"
trap - EXIT
test "$(sha256sum "$PREPARATION" | awk '{print $1}')" = "$PREPARATION_SHA256"

env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PROJECT:$PROJECT/src" \
  "$PYTHON" scripts/verify_generation_training_exposure_qualification.py \
    --followup-decision "$FOLLOWUP_DECISION" \
    --expected-followup-decision-sha256 "$EXPECTED_FOLLOWUP_DECISION_SHA256" \
    --capacity-probe-result "$CAPACITY_PROBE_RESULT" \
    --expected-capacity-probe-result-sha256 "$EXPECTED_CAPACITY_PROBE_RESULT_SHA256" \
    --source-cofitok-config "$SOURCE_COFITOK_CONFIG" \
    --source-dense-config "$SOURCE_DENSE_CONFIG" \
    --target-cofitok-config "$TARGET_COFITOK_CONFIG" \
    --target-dense-config "$TARGET_DENSE_CONFIG" \
    --expected-followup-revision "$EXPECTED_FOLLOWUP_REVISION" \
    --expected-followup-branch "$EXPECTED_FOLLOWUP_BRANCH" \
    --expected-capacity-revision "$EXPECTED_CAPACITY_REVISION" \
    --expected-capacity-branch "$EXPECTED_CAPACITY_BRANCH" \
    --expected-preparation-revision "$EXPECTED_REVISION" \
    --expected-preparation-branch "$EXPECTED_BRANCH" \
    --output-root "$OUTPUT_ROOT" \
    --preparation "$PREPARATION" \
    --expected-preparation-sha256 "$PREPARATION_SHA256"

printf '%s  %s\n' "$PREPARATION_SHA256" "$PREPARATION"
