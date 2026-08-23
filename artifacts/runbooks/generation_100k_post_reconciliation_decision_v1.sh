#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_POST_RECONCILIATION_REVISION=${EXPECTED_POST_RECONCILIATION_REVISION:?set exact decision revision}
EXPECTED_POST_RECONCILIATION_TREE=${EXPECTED_POST_RECONCILIATION_TREE:?set exact decision tree}
EXPECTED_POST_RECONCILIATION_BRANCH=${EXPECTED_POST_RECONCILIATION_BRANCH:-analysis/generation-100k-post-reconciliation-decision-v1}

QUALITY_ROOT=${QUALITY_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1}
REPORT_ROOT="$QUALITY_ROOT/reports"
AUTHORITATIVE_DECISION="$REPORT_ROOT/followup_decision_authoritative_verifier_v3_20260824/followup_experiment_decision.json"
RECONCILIATION="$REPORT_ROOT/100k_cross_protocol_reconciliation_v1_20260824/cross_protocol_reconciliation.json"
QUALITY_RESULT="$REPORT_ROOT/quality_bridge_result.json"
TRAINING_EXPOSURE="$REPORT_ROOT/training_exposure_terminal_100k_authoritative_verifier_v2/training_exposure_report.json"
OUTPUT_DIR="$REPORT_ROOT/100k_post_reconciliation_decision_v1_20260824"
DECISION="$OUTPUT_DIR/post_reconciliation_decision.json"
VERIFICATION="$OUTPUT_DIR/verification.json"

cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_POST_RECONCILIATION_REVISION" ]]
[[ "$(git rev-parse 'HEAD^{tree}')" == "$EXPECTED_POST_RECONCILIATION_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_POST_RECONCILIATION_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]

for path in \
  "$AUTHORITATIVE_DECISION" \
  "$RECONCILIATION" \
  "$QUALITY_RESULT" \
  "$TRAINING_EXPOSURE"; do
  [[ -f "$path" ]]
done

mkdir -p "$OUTPUT_DIR"

common_args=(
  --authoritative-decision "$AUTHORITATIVE_DECISION"
  --expected-authoritative-decision-sha256 ae989b2e0b0f06a2b28346aae629f5368079a45a9a5b53d90c2d6a20c137175a
  --reconciliation "$RECONCILIATION"
  --expected-reconciliation-sha256 6fc639caec0320225c2e8d26316490cde2d765d49c6062a18e0e671f79384d19
  --quality-result "$QUALITY_RESULT"
  --expected-quality-result-sha256 15752e05611fa888e15352934c1627ccb95418df0339f9ad120e195bc088d165
  --training-exposure "$TRAINING_EXPOSURE"
  --expected-training-exposure-sha256 d95c4d9c41327be36cb462af3076064123803b53d9f2f9e927ac89351513300b
  --expected-builder-revision "$EXPECTED_POST_RECONCILIATION_REVISION"
  --expected-builder-tree "$EXPECTED_POST_RECONCILIATION_TREE"
  --expected-builder-branch "$EXPECTED_POST_RECONCILIATION_BRANCH"
)

build_args=("${common_args[@]}" --output "$DECISION")
if [[ -f "$DECISION" ]]; then
  build_args+=(--resume)
fi
"$PYTHON" scripts/build_generation_100k_post_reconciliation_decision.py "${build_args[@]}"

decision_sha256="$(sha256sum "$DECISION" | awk '{print $1}')"
verify_args=(
  "${common_args[@]}"
  --decision "$DECISION"
  --expected-decision-sha256 "$decision_sha256"
  --output "$VERIFICATION"
)
if [[ -f "$VERIFICATION" ]]; then
  verify_args+=(--resume)
fi
"$PYTHON" scripts/verify_generation_100k_post_reconciliation_decision.py "${verify_args[@]}"

printf 'post-reconciliation decision: %s\n' "$DECISION"
printf 'decision SHA256: %s\n' "$decision_sha256"
printf 'verification: %s\n' "$VERIFICATION"
