#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the isolated exposure-trajectory checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact trajectory revision}
EXPECTED_TREE=${EXPECTED_TREE:?set the exact trajectory tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set the exact trajectory branch}
PREPARATION_REPORT=${PREPARATION_REPORT:?set the immutable preparation report}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set the preparation SHA256}
EXECUTION_AUTHORIZATION=${EXECUTION_AUTHORIZATION:?set the execution authorization}
EXPECTED_EXECUTION_AUTHORIZATION_SHA256=${EXPECTED_EXECUTION_AUTHORIZATION_SHA256:?set the authorization SHA256}
EXPECTED_RUNBOOK_SHA256=${EXPECTED_RUNBOOK_SHA256:?set this runbook SHA256}
OUTPUT_ROOT=${OUTPUT_ROOT:?set the versioned output root}

EXPECTED_OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure_semantic_trajectory_pair5k_v1
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher
STANDING_AUTHORIZATION=/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json
REJECTED_POSTEVALUATION=/root/autodl-tmp/CoFiTok/checkpoints/generation/semantic_residual_alignment_four_arm_probe1k_v1/reports/semantic_residual_alignment_posteval_v1/postevaluation.json
LOCK_DIR=${OUTPUT_ROOT}.lock

cd "$PROJECT"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git rev-parse 'HEAD^{tree}')" == "$EXPECTED_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$EXECUTION_AUTHORIZATION" ]]
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$EXPECTED_RUNBOOK_SHA256" ]]

test ! -e "$OUTPUT_ROOT"
test ! -e "$LOCK_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'exposure-trajectory execution lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 10
fi
temporary_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$temporary_dir"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=.:src
export PYTHONDONTWRITEBYTECODE=1

recomputed_preparation="$temporary_dir/preparation.json"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 ionice -c 3 "$PYTHON" \
    scripts/prepare_generation_exposure_semantic_trajectory.py \
      --standing-authorization "$STANDING_AUTHORIZATION" \
      --rejected-postevaluation "$REJECTED_POSTEVALUATION" \
      --pair-root "$PAIR_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --runbook "${BASH_SOURCE[0]}" \
      --output "$recomputed_preparation"
[[ "$(sha256sum "$recomputed_preparation" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
cmp -- "$PREPARATION_REPORT" "$recomputed_preparation"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 ionice -c 3 "$PYTHON" \
    scripts/verify_generation_exposure_semantic_trajectory_execution_authorization.py \
      --authorization "$EXECUTION_AUTHORIZATION" \
      --expected-authorization-sha256 "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" \
      --expected-revision "$EXPECTED_REVISION" \
      --expected-tree "$EXPECTED_TREE" \
      --expected-branch "$EXPECTED_BRANCH" \
      --expected-output-root "$OUTPUT_ROOT"

REPORT_ROOT="$OUTPUT_ROOT/reports/exposure_semantic_trajectory_v1"
mkdir -p "$REPORT_ROOT"
cp -- "$PREPARATION_REPORT" "$OUTPUT_ROOT/preparation.json"
cp -- "$EXECUTION_AUTHORIZATION" "$OUTPUT_ROOT/execution_authorization.json"

run_sensitivity() {
  local method=$1
  local run_dir=$2
  local step=$3
  local padded
  padded=$(printf '%08d' "$step")
  local checkpoint="$PAIR_ROOT/$run_dir/checkpoint_step_${padded}.pt"
  local output="$REPORT_ROOT/$method/step_${padded}"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nice -n 19 ionice -c 3 "$PYTHON" \
      scripts/evaluate_generation_conditioning_sensitivity.py \
        --checkpoint "$checkpoint" \
        --output-dir "$output" \
        --weights ema \
        --num-samples 16 \
        --start-label 128 \
        --wrong-label-offset 250 \
        --timesteps 100 500 700 900 \
        --noise-seed 314159 \
        --threads 2 \
        --resume
}

for step in 1250 2500 5000; do
  run_sensitivity cofitok cofitok_rgbtail3_rollout_x0_u2_ema_teacher "$step"
done
for step in 1250 2500 5000; do
  run_sensitivity dense_identity dense_rollout_x0_u2_ema_teacher "$step"
done

builder_args=(
  --preparation "$PREPARATION_REPORT"
  --execution-authorization "$EXECUTION_AUTHORIZATION"
  --output "$REPORT_ROOT/trajectory_report.json"
  --resume
)
for method in cofitok dense_identity; do
  for step in 1250 2500 5000; do
    padded=$(printf '%08d' "$step")
    option=${method//_/-}
    builder_args+=(
      "--${option}-step-${step}-report"
      "$REPORT_ROOT/$method/step_${padded}/conditioning_sensitivity_report.json"
    )
  done
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 ionice -c 3 "$PYTHON" \
    scripts/build_generation_exposure_semantic_trajectory_report.py \
      "${builder_args[@]}"

[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_AUTHORIZATION_SHA256" ]]
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]

chmod 0444 "$OUTPUT_ROOT/preparation.json" "$OUTPUT_ROOT/execution_authorization.json"
find "$REPORT_ROOT" -type f \( \
  -name 'conditioning_sensitivity_manifest.json' -o \
  -name 'conditioning_sensitivity_report.json' -o \
  -name 'trajectory_report.json' \
\) -exec chmod 0444 {} +
