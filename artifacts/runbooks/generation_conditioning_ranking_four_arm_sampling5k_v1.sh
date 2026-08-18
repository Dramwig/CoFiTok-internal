#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set the isolated sampling checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set the exact sampling revision}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-scale/generation-label-ranking-standing-authorization-v1}
POSTEVALUATION=${POSTEVALUATION:?set the completed four-arm 1K postevaluation}
EXPECTED_POSTEVALUATION_SHA256=${EXPECTED_POSTEVALUATION_SHA256:?set its SHA256}
PREPARATION_REPORT=${PREPARATION_REPORT:?set the immutable sampling preparation}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set its SHA256}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set the standing authorization record}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set its SHA256}
IDLE_GPU_EVIDENCE=${IDLE_GPU_EVIDENCE:?set the five-poll idle GPU evidence}
EXPECTED_IDLE_GPU_EVIDENCE_SHA256=${EXPECTED_IDLE_GPU_EVIDENCE_SHA256:?set its SHA256}
CLASSIFIER_CHECKPOINT=${CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/evaluators/torchvision/resnet50-11ad3fa6.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_sampling5k_v1}
DATA=${DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
EVAL_CACHE=${EVAL_CACHE:-$CHECKPOINT_ROOT/eval_cache/torch_fidelity}
RUNBOOK_PATH=$PROJECT/artifacts/runbooks/generation_conditioning_ranking_four_arm_sampling5k_v1.sh
LOCK_DIR=${OUTPUT_ROOT}.lock
SELECTION=$OUTPUT_ROOT/reports/sampling_batch_selection.json
EXECUTION_RECEIPT=$OUTPUT_ROOT/reports/execution_receipt.json
FINAL_REPORT=$OUTPUT_ROOT/reports/sampling_validation.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$POSTEVALUATION" ]]
[[ -f "$PREPARATION_REPORT" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ -f "$IDLE_GPU_EVIDENCE" ]]
[[ -f "$CLASSIFIER_CHECKPOINT" ]]
[[ -f "$RUNBOOK_PATH" ]]
[[ "$(sha256sum "$POSTEVALUATION" | awk '{print $1}')" == "$EXPECTED_POSTEVALUATION_SHA256" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$IDLE_GPU_EVIDENCE" | awk '{print $1}')" == "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" ]]

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing four-arm 5K sampling while any GPU compute process is active\n' >&2
  exit 9
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'conditioning-ranking sampling lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 10
fi
temporary_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$temporary_dir"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONPATH=.:src
export PYTHONDONTWRITEBYTECODE=1

recomputed_preparation=$temporary_dir/preparation.json
"$PYTHON" scripts/prepare_generation_conditioning_ranking_sampling_validation.py \
  --postevaluation "$POSTEVALUATION" \
  --expected-postevaluation-sha256 "$EXPECTED_POSTEVALUATION_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --runbook "$RUNBOOK_PATH" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-branch "$EXPECTED_BRANCH" \
  --output-root "$OUTPUT_ROOT" \
  --output "$recomputed_preparation"
[[ "$(sha256sum "$recomputed_preparation" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
cmp -- "$PREPARATION_REPORT" "$recomputed_preparation"

mkdir -p "$OUTPUT_ROOT/reports"
install_bound_copy() {
  local source=$1
  local target=$2
  local expected_sha=$3
  if [[ -e "$target" ]]; then
    [[ -f "$target" ]]
    [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]
    cmp -- "$source" "$target"
  else
    cp -- "$source" "$target"
    [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]
  fi
}
install_bound_copy "$PREPARATION_REPORT" "$OUTPUT_ROOT/reports/preparation.json" "$EXPECTED_PREPARATION_SHA256"
install_bound_copy "$IDLE_GPU_EVIDENCE" "$OUTPUT_ROOT/reports/idle_gpu_evidence.json" "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256"

selection_args=()
if [[ -f "$SELECTION" ]]; then
  selection_args+=(--resume)
fi
SAMPLING_BATCH=$("$PYTHON" scripts/select_generation_conditioning_ranking_sampling_batch.py \
  --preparation "$PREPARATION_REPORT" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --output "$SELECTION" \
  "${selection_args[@]}")
if [[ ! "$SAMPLING_BATCH" =~ ^(16|32|64)$ ]]; then
  printf 'invalid selected four-arm sampling batch: %s\n' "$SAMPLING_BATCH" >&2
  exit 11
fi
EXPECTED_SELECTION_SHA256=$(sha256sum "$SELECTION" | awk '{print $1}')

receipt_args=()
if [[ -f "$EXECUTION_RECEIPT" ]]; then
  receipt_args+=(--resume)
fi
EXPECTED_EXECUTION_RECEIPT_SHA256=$("$PYTHON" scripts/build_generation_conditioning_ranking_sampling_execution_receipt.py \
  --preparation "$PREPARATION_REPORT" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --standing-authorization "$STANDING_AUTHORIZATION" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --idle-gpu-evidence "$IDLE_GPU_EVIDENCE" \
  --expected-idle-gpu-evidence-sha256 "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" \
  --batch-selection "$SELECTION" \
  --expected-batch-selection-sha256 "$EXPECTED_SELECTION_SHA256" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-branch "$EXPECTED_BRANCH" \
  --expected-output-root "$OUTPUT_ROOT" \
  --output "$EXECUTION_RECEIPT" \
  "${receipt_args[@]}")

"$PYTHON" scripts/verify_generation_conditioning_ranking_sampling_execution_receipt.py \
  --receipt "$EXECUTION_RECEIPT" \
  --expected-receipt-sha256 "$EXPECTED_EXECUTION_RECEIPT_SHA256" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" \
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
  --expected-idle-gpu-evidence-sha256 "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" \
  --expected-batch-selection-sha256 "$EXPECTED_SELECTION_SHA256" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-branch "$EXPECTED_BRANCH" \
  --expected-output-root "$OUTPUT_ROOT" >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  printf 'refusing four-arm sampling after preflight because the GPU became busy\n' >&2
  exit 12
fi
[[ -z "$(git status --porcelain)" ]]
[[ "$(sha256sum "$POSTEVALUATION" | awk '{print $1}')" == "$EXPECTED_POSTEVALUATION_SHA256" ]]
[[ "$(sha256sum "$PREPARATION_REPORT" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$IDLE_GPU_EVIDENCE" | awk '{print $1}')" == "$EXPECTED_IDLE_GPU_EVIDENCE_SHA256" ]]

checkpoint_for() {
  local run=$1
  "$PYTHON" - "$EXECUTION_RECEIPT" "$run" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(receipt["training_checkpoints"][sys.argv[2]]["checkpoint"]["path"])
PY
}

run_sampling() {
  local run=$1
  local budget=$2
  local checkpoint
  checkpoint=$(checkpoint_for "$run")
  "$PYTHON" scripts/generate_samples.py \
    --checkpoint "$checkpoint" \
    --output-dir "$OUTPUT_ROOT/$run/samples_5000_ddim50_cfg15" \
    --num-samples 5000 \
    --batch-size "$SAMPLING_BATCH" \
    --sample-steps 50 \
    --prefix-budgets "$budget" \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --cfg-batch-mode batched \
    --eta 0.0 \
    --seed 406020 \
    --start-index 0 \
    --weights ema \
    --precision bf16 \
    --resume
}

run_sampling control_cofitok 8
run_sampling ranked_cofitok 8
run_sampling control_dense_identity 1
run_sampling ranked_dense_identity 1

run_metrics() {
  local run=$1
  local budget=$2
  local sample_root=$OUTPUT_ROOT/$run/samples_5000_ddim50_cfg15
  "$PYTHON" scripts/evaluate_generation_metrics.py \
    --real-dir "$DATA" \
    --generated-dir "$sample_root/prefix_$budget" \
    --sampling-report "$sample_root/sampling_report.json" \
    --output-dir "$sample_root/metrics" \
    --cache-root "$EVAL_CACHE" \
    --min-samples 5000 \
    --skip-prc \
    --resume
}

run_metrics control_cofitok 8
run_metrics ranked_cofitok 8
run_metrics control_dense_identity 1
run_metrics ranked_dense_identity 1

run_paired_class_fidelity() {
  local method=$1
  local control_run=$2
  local ranked_run=$3
  local budget=$4
  local control_root=$OUTPUT_ROOT/$control_run/samples_5000_ddim50_cfg15
  local ranked_root=$OUTPUT_ROOT/$ranked_run/samples_5000_ddim50_cfg15
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 ionice -c 3 \
    "$PYTHON" scripts/evaluate_generation_conditioning_ranking_samples.py \
      --method "$method" \
      --control-generated-dir "$control_root/prefix_$budget" \
      --control-sampling-report "$control_root/sampling_report.json" \
      --ranked-generated-dir "$ranked_root/prefix_$budget" \
      --ranked-sampling-report "$ranked_root/sampling_report.json" \
      --output-dir "$OUTPUT_ROOT/reports/paired_class_fidelity/$method" \
      --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
      --batch-size 32 \
      --num-workers 8 \
      --min-samples 5000 \
      --cpu \
      --resume
}

run_paired_class_fidelity cofitok control_cofitok ranked_cofitok 8
run_paired_class_fidelity dense_identity control_dense_identity ranked_dense_identity 1

CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" scripts/build_generation_conditioning_ranking_sampling_validation.py \
    --postevaluation "$POSTEVALUATION" \
    --output-root "$OUTPUT_ROOT" \
    --output "$FINAL_REPORT" \
    --resume
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" scripts/build_generation_conditioning_ranking_sampling_validation.py \
    --postevaluation "$POSTEVALUATION" \
    --output-root "$OUTPUT_ROOT" \
    --output "$FINAL_REPORT" \
    --resume >/dev/null

[[ "$(sha256sum "$EXECUTION_RECEIPT" | awk '{print $1}')" == "$EXPECTED_EXECUTION_RECEIPT_SHA256" ]]
[[ "$(sha256sum "$SELECTION" | awk '{print $1}')" == "$EXPECTED_SELECTION_SHA256" ]]
[[ -f "$FINAL_REPORT" ]]
printf 'completed four-arm conditioning-ranking 5K sampling validation: %s\n' "$FINAL_REPORT"
