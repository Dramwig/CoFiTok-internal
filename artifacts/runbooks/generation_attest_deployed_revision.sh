#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  printf 'usage: %s SOURCE_REVISION TARGET_REVISION\n' "$0" >&2
  exit 64
fi

SOURCE_REVISION="$1"
TARGET_REVISION="$2"
PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
PAIR_VALIDATION="$OUTPUT_ROOT/generation_10pct_pair_validation.json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
eval "$(python scripts/print_generation_workspace_paths.py \
  --project-root "$PROJECT" --output-root "$OUTPUT_ROOT" --format shell \
  --target-revision "$TARGET_REVISION")"

if [[ "$(git rev-parse HEAD)" != "$TARGET_REVISION" ]]; then
  printf 'deployed HEAD does not equal target revision %s\n' "$TARGET_REVISION" >&2
  exit 65
fi
if [[ "$(git branch --show-current)" != "scale/generative-system" ]]; then
  printf 'deployment attestation requires scale/generative-system\n' >&2
  exit 66
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'deployment attestation requires a clean tracked worktree\n' >&2
  exit 67
fi
if ! git merge-base --is-ancestor "$SOURCE_REVISION" "$TARGET_REVISION"; then
  printf 'source revision is not an ancestor of the target\n' >&2
  exit 68
fi
if [[ ! -f "$PAIR_VALIDATION" ]]; then
  printf 'source training-pair validation is missing: %s\n' "$PAIR_VALIDATION" >&2
  exit 69
fi
if [[ -e "$DEPLOYMENT_RECEIPT" ]]; then
  printf 'refusing to overwrite deployment receipt: %s\n' \
    "$DEPLOYMENT_RECEIPT" >&2
  exit 70
fi
if pgrep -af '[s]cripts/train_generation.py' >/dev/null; then
  printf 'generation training is active; refusing GPU validation\n' >&2
  exit 71
fi

mkdir -p "$DEPLOYMENT_EVIDENCE_ROOT"
bundle_temporary="${DEPLOYMENT_BUNDLE}.tmp.$$"
pytest_temporary="${DEPLOYMENT_PYTEST}.tmp.$$"
cleanup() {
  rm -f -- "$bundle_temporary" "$pytest_temporary"
}
trap cleanup EXIT

git bundle create "$bundle_temporary" HEAD "^$SOURCE_REVISION"
git bundle verify "$bundle_temporary"
mapfile -t bundle_heads < <(git bundle list-heads "$bundle_temporary")
if (( ${#bundle_heads[@]} != 1 )); then
  printf 'attestation bundle must advertise exactly one head\n' >&2
  exit 72
fi
read -r bundle_head bundle_ref <<<"${bundle_heads[0]}"
if [[ "$bundle_head" != "$TARGET_REVISION" || \
      ( "$bundle_ref" != "HEAD" && \
        "$bundle_ref" != "refs/heads/scale/generative-system" ) ]]; then
  printf 'attestation bundle head differs from the target revision\n' >&2
  exit 73
fi
mv "$bundle_temporary" "$DEPLOYMENT_BUNDLE"

python scripts/check_generation_deployment_conflicts.py \
  --repository "$PROJECT" \
  --current-commit "$SOURCE_REVISION" \
  --target-commit "$TARGET_REVISION" \
  --output "$DEPLOYMENT_CONFLICT_SCAN" >/dev/null

if python -m pytest -q --junitxml "$pytest_temporary"; then
  mv "$pytest_temporary" "$DEPLOYMENT_PYTEST"
else
  pytest_exit_code=$?
  exit "$pytest_exit_code"
fi
python scripts/check_generation_runbook_syntax.py \
  --project-root "$PROJECT" --output "$DEPLOYMENT_RUNBOOK_SYNTAX"

python scripts/write_generation_deployment_receipt.py \
  --output "$DEPLOYMENT_RECEIPT" \
  --bundle "$DEPLOYMENT_BUNDLE" \
  --pair-validation "$PAIR_VALIDATION" \
  --conflict-scan "$DEPLOYMENT_CONFLICT_SCAN" \
  --runbook-syntax "$DEPLOYMENT_RUNBOOK_SYNTAX" \
  --pytest-report "$DEPLOYMENT_PYTEST" \
  --expected-training-revision "$SOURCE_REVISION" \
  --target-revision "$TARGET_REVISION"

printf 'wrote immutable deployment attestation %s\n' "$DEPLOYMENT_RECEIPT"
