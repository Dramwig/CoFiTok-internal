#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
STATUS="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.status.json"
LOCK="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.lock"
SCALING_GATE="$PROJECT/artifacts/reports/generation/imagenet256_10pct_matched_50k_2026-07-12/promotion_gate.json"
FINAL_GATE="$PROJECT/artifacts/reports/generation/imagenet256_full_matched_300k/final_generation_gate.json"
COFITOK_10PCT="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12/training_report.json"
DENSE_10PCT="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12/training_report.json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
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
  python - "$COFITOK_10PCT" "$DENSE_10PCT" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("training_complete") is not True:
        raise SystemExit(f"training is incomplete: {path}")
    if report.get("completed_steps") != 50_000 or report.get("target_steps") != 50_000:
        raise SystemExit(f"training did not finish exactly 50K steps: {path}")
    if report.get("git", {}).get("dirty") is not False:
        raise SystemExit(f"training used a dirty tracked worktree: {path}")
PY
}

validate_gate() {
  local gate_path="$1"
  local expected_decision="$2"
  python - "$gate_path" "$expected_decision" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    gate = json.load(handle)
if gate.get("status") != "pass" or gate.get("decision") != sys.argv[2]:
    raise SystemExit(
        f"gate did not pass: status={gate.get('status')!r}, "
        f"decision={gate.get('decision')!r}, expected={sys.argv[2]!r}"
    )
PY
}

write_status running "validating the completed 10% matched training pair"
validate_completed_training_pair

STAGE=posteval_10pct
write_status running "running matched 10K sampling, metrics, and scaling gate"
bash artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh

STAGE=promotion_gate
write_status running "checking authorization for full ImageNet-256 training"
validate_gate "$SCALING_GATE" promote_to_full_imagenet256

STAGE=full_training
write_status running "running alternating matched full ImageNet-256 300K training"
bash artifacts/runbooks/generation_full_matched_300k_after_gate.sh

STAGE=full_posteval
write_status running "running matched 50K-sample formal evaluation"
bash artifacts/runbooks/generation_full_posteval_50k.sh

STAGE=final_gate
write_status running "checking the large-scale generation readiness gate"
validate_gate "$FINAL_GATE" large_scale_generation_ready

STAGE=complete
write_status pass "large-scale generation training and formal evaluation passed"
FINISHED=1
trap - EXIT
