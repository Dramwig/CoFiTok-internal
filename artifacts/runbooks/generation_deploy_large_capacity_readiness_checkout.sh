#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
FORMAL_REPOSITORY=${FORMAL_REPOSITORY:-/root/autodl-tmp/CoFiTok/CoFiTok-internal}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
TARGET_REVISION=${TARGET_REVISION:?set the full large-capacity target revision}
TARGET_BRANCH=${TARGET_BRANCH:-scale/generation-large-capacity}
EXPECTED_FORMAL_REVISION=${EXPECTED_FORMAL_REVISION:?set the pinned formal repository revision}
EXPECTED_FORMAL_BRANCH=${EXPECTED_FORMAL_BRANCH:-scale/generative-system}
EXPECTED_BUNDLE_SHA256=${EXPECTED_BUNDLE_SHA256:?set the verified deployment bundle SHA256}
EXPECTED_BUNDLE_BYTES=${EXPECTED_BUNDLE_BYTES:?set the verified deployment bundle bytes}
EXPECTED_PREREQUISITE_ONE=${EXPECTED_PREREQUISITE_ONE:?set the first bundle prerequisite}
EXPECTED_PREREQUISITE_TWO=${EXPECTED_PREREQUISITE_TWO:?set the second bundle prerequisite}
PAPER_ROOT=${PAPER_ROOT:-/root/autodl-tmp/CoFiTok/paper}

SHORT_TARGET="${TARGET_REVISION:0:7}"
BUNDLE="$CHECKPOINT_ROOT/deployment_bundles/cofitok-generation-large-capacity-${SHORT_TARGET}-from-${EXPECTED_FORMAL_REVISION:0:7}.bundle"
DEPLOYMENT_ROOT="$CHECKPOINT_ROOT/deployment/large_capacity"
EVIDENCE_ROOT="$DEPLOYMENT_ROOT/deployments/$TARGET_REVISION"
CHECKOUT="$DEPLOYMENT_ROOT/checkout-$SHORT_TARGET"
RECEIPT="$EVIDENCE_ROOT/deployment_receipt.json"
RUNBOOK_SYNTAX="$EVIDENCE_ROOT/runbook_syntax.json"
PYTEST_REPORT="$EVIDENCE_ROOT/pytest.xml"
TEMP_CHECKOUT="${CHECKOUT}.tmp.$$"
TEMP_RUNBOOK_SYNTAX="${RUNBOOK_SYNTAX}.tmp.$$"
TEMP_PYTEST_REPORT="${PYTEST_REPORT}.tmp.$$"
PAPER_LINK="$DEPLOYMENT_ROOT/paper"

[[ -x "$PYTHON" ]]
[[ "$TARGET_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_FORMAL_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_PREREQUISITE_ONE" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_PREREQUISITE_TWO" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_BUNDLE_BYTES" =~ ^[0-9]+$ ]]
[[ -d "$FORMAL_REPOSITORY/.git" ]]
[[ -d "$PAPER_ROOT" ]]
[[ "$(git -C "$FORMAL_REPOSITORY" rev-parse HEAD)" == "$EXPECTED_FORMAL_REVISION" ]]
[[ "$(git -C "$FORMAL_REPOSITORY" branch --show-current)" == "$EXPECTED_FORMAL_BRANCH" ]]
[[ -z "$(git -C "$FORMAL_REPOSITORY" status --porcelain --untracked-files=no)" ]]
[[ -f "$BUNDLE" ]]
[[ "$(stat -c %s "$BUNDLE")" == "$EXPECTED_BUNDLE_BYTES" ]]
[[ "$(sha256sum "$BUNDLE" | awk '{print $1}')" == "$EXPECTED_BUNDLE_SHA256" ]]

if [[ -e "$EVIDENCE_ROOT" || -e "$CHECKOUT" \
  || -e "$RUNBOOK_SYNTAX" || -e "$PYTEST_REPORT" \
  || -e "$TEMP_CHECKOUT" || -e "$TEMP_RUNBOOK_SYNTAX" \
  || -e "$TEMP_PYTEST_REPORT" ]]; then
  printf 'refusing to overwrite large-capacity deployment state under %s\n' \
    "$DEPLOYMENT_ROOT" >&2
  exit 8
fi
mkdir -p "$DEPLOYMENT_ROOT" "$EVIDENCE_ROOT"
cleanup() {
  resolved_root="$(realpath -m "$DEPLOYMENT_ROOT")"
  for candidate in "$TEMP_CHECKOUT" "$CHECKOUT"; do
    [[ -d "$candidate" ]] || continue
    resolved_candidate="$(realpath -m "$candidate")"
    case "$resolved_candidate" in
      "$resolved_root"/checkout-"$SHORT_TARGET"|\
      "$resolved_root"/checkout-"$SHORT_TARGET".tmp.*)
        if [[ ! -e "$RECEIPT" ]]; then
          rm -rf -- "$resolved_candidate"
        fi
        ;;
      *)
        printf 'refusing unsafe deployment checkout removal: %s\n' \
          "$resolved_candidate" >&2
        ;;
    esac
  done
  if [[ ! -e "$RECEIPT" ]]; then
    rm -f -- "$TEMP_RUNBOOK_SYNTAX" "$TEMP_PYTEST_REPORT" \
      "$RUNBOOK_SYNTAX" "$PYTEST_REPORT"
  fi
  if [[ ! -e "$RECEIPT" ]]; then
    rmdir -- "$EVIDENCE_ROOT" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ -L "$PAPER_LINK" ]]; then
  [[ "$(readlink -f "$PAPER_LINK")" == "$(readlink -f "$PAPER_ROOT")" ]]
elif [[ -e "$PAPER_LINK" ]]; then
  printf 'deployment paper path exists but is not the expected symlink: %s\n' \
    "$PAPER_LINK" >&2
  exit 8
else
  ln -s "$PAPER_ROOT" "$PAPER_LINK"
fi

git -C "$FORMAL_REPOSITORY" bundle verify "$BUNDLE" >/dev/null
heads="$(git bundle list-heads "$BUNDLE")"
if [[ "$heads" != "$TARGET_REVISION refs/heads/$TARGET_BRANCH" ]]; then
  printf 'large-capacity bundle advertises another target: %s\n' "$heads" >&2
  exit 9
fi

git clone --no-local "$FORMAL_REPOSITORY" "$TEMP_CHECKOUT" >/dev/null
git -C "$TEMP_CHECKOUT" fetch "$BUNDLE" \
  "refs/heads/$TARGET_BRANCH:refs/heads/$TARGET_BRANCH"
git -C "$TEMP_CHECKOUT" checkout "$TARGET_BRANCH" >/dev/null
[[ "$(git -C "$TEMP_CHECKOUT" rev-parse HEAD)" == "$TARGET_REVISION" ]]
[[ -z "$(git -C "$TEMP_CHECKOUT" status --porcelain --untracked-files=no)" ]]

cd "$TEMP_CHECKOUT"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=-1
"$PYTHON" scripts/check_generation_runbook_syntax.py \
  --project-root "$TEMP_CHECKOUT" \
  --output "$TEMP_RUNBOOK_SYNTAX" >/dev/null
"$PYTHON" -m pytest -q --junitxml="$TEMP_PYTEST_REPORT"

cd "$DEPLOYMENT_ROOT"
mv "$TEMP_CHECKOUT" "$CHECKOUT"
cd "$CHECKOUT"
"$PYTHON" scripts/check_generation_runbook_syntax.py \
  --project-root "$CHECKOUT" \
  --output "$RUNBOOK_SYNTAX" >/dev/null
mv "$TEMP_PYTEST_REPORT" "$PYTEST_REPORT"
rm -f -- "$TEMP_RUNBOOK_SYNTAX"

"$PYTHON" scripts/build_generation_large_capacity_deployment_receipt.py \
  --formal-repository "$FORMAL_REPOSITORY" \
  --checkout "$CHECKOUT" \
  --bundle "$BUNDLE" \
  --runbook-syntax "$RUNBOOK_SYNTAX" \
  --pytest-report "$PYTEST_REPORT" \
  --expected-bundle-sha256 "$EXPECTED_BUNDLE_SHA256" \
  --expected-bundle-bytes "$EXPECTED_BUNDLE_BYTES" \
  --expected-formal-revision "$EXPECTED_FORMAL_REVISION" \
  --expected-formal-branch "$EXPECTED_FORMAL_BRANCH" \
  --expected-target-revision "$TARGET_REVISION" \
  --expected-target-branch "$TARGET_BRANCH" \
  --expected-prerequisite "$EXPECTED_PREREQUISITE_ONE" \
  --expected-prerequisite "$EXPECTED_PREREQUISITE_TWO" \
  --minimum-pytest-passed 800 \
  --maximum-pytest-skipped 5 \
  --output "$RECEIPT" >/dev/null
trap - EXIT

printf 'large-capacity deployment receipt: %s  %s\n' \
  "$(sha256sum "$RECEIPT" | awk '{print $1}')" "$RECEIPT"
