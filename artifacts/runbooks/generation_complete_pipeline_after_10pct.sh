#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
STATUS="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.status.json"
LOCK="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.lock"
SCALING_GATE="$PROJECT/artifacts/reports/generation/imagenet256_10pct_matched_50k_2026-07-12/promotion_gate.json"
FINAL_GATE="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/final_generation_gate.json"
FINAL_COMPARISON="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/comparison/large_scale_generation_comparison.json"
FINAL_VISUAL_AUDIT="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/visual_audit/visual_audit_report.json"
COFITOK_10PCT="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12/training_report.json"
DENSE_10PCT="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12/training_report.json"
PINNED_10PCT_REVISION=781a01444fddbf0d48a427ba58bdeed50167b5be

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
FULL_REVISION="$(git rev-parse HEAD)"
mkdir -p "$OUTPUT_ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'generation completion pipeline already holds %s\n' "$LOCK" >&2
  exit 75
fi

STAGE=preconditions
FINISHED=0

write_status() {
  local status="$1"
  local detail="$2"
  shift 2
  python scripts/write_generation_pipeline_status.py \
    --output "$STATUS" --status "$status" --stage "$STAGE" \
    --detail "$detail" "$@"
}

record_failure() {
  local exit_code=$?
  if (( FINISHED == 0 )); then
    write_status failed "pipeline stopped during ${STAGE}" --exit-code "$exit_code" || true
  fi
  exit "$exit_code"
}
trap record_failure EXIT

validate_completed_training_pair() {
  python scripts/validate_generation_training_pair.py \
    --cofitok-training "$COFITOK_10PCT" \
    --dense-training "$DENSE_10PCT" \
    --expected-steps 50000 --expected-revision "$PINNED_10PCT_REVISION" \
    --expected-recipe-stage scaling \
    --allow-legacy-missing-dataset-provenance
}

validate_gate() {
  local gate_path="$1"
  local stage="$2"
  python scripts/validate_generation_gate_report.py \
    --gate "$gate_path" --stage "$stage"
}

write_status running "validating the completed 10% matched training pair"
validate_completed_training_pair

if [[ ! -f "$SCALING_GATE" ]]; then
  STAGE=posteval_10pct
  write_status running "running matched 10K sampling, metrics, and scaling gate"
  bash artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh
fi

STAGE=promotion_gate
write_status running "checking authorization for full ImageNet-256 training"
validate_gate "$SCALING_GATE" scaling

STAGE=full_training
write_status running "running alternating matched full ImageNet-256 300K training"
bash artifacts/runbooks/generation_full_matched_300k_after_gate.sh

if [[ ! -f "$FINAL_GATE" || ! -f "$FINAL_COMPARISON" || ! -f "$FINAL_VISUAL_AUDIT" ]]; then
  STAGE=full_posteval
  write_status running "running matched 50K-sample formal evaluation"
  bash artifacts/runbooks/generation_full_posteval_50k.sh
fi

STAGE=final_gate
write_status running "checking the large-scale generation readiness gate"
validate_gate "$FINAL_GATE" full

STAGE=inference_export
write_status running "exporting and smoke-testing deployable EMA artifacts"
bash artifacts/runbooks/generation_export_inference_artifacts.sh

STAGE=completion_audit
write_status running "auditing all required large-scale generation evidence"
python scripts/audit_large_scale_generation_completion.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" \
  --expected-10pct-revision "$PINNED_10PCT_REVISION" \
  --expected-full-revision "$FULL_REVISION" \
  --output "$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/completion_audit.json"

STAGE=complete
write_status pass "large-scale generation training and formal evaluation passed"
FINISHED=1
trap - EXIT
