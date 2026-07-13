#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  printf 'usage: %s BUNDLE EXPECTED_COMMIT TARGET_COMMIT\n' "$0" >&2
  exit 64
fi

BUNDLE="$1"
EXPECTED_COMMIT="$2"
TARGET_COMMIT="$3"
PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
PIPELINE=artifacts/runbooks/generation_complete_pipeline_after_10pct.sh
SUPERVISOR=artifacts/runbooks/generation_completion_supervisor.sh
LOG="$OUTPUT_ROOT/generation_completion_supervisor.log"
PID_FILE="$OUTPUT_ROOT/generation_completion_supervisor.pid"
COFITOK_REPORT="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12/training_report.json"
DENSE_REPORT="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12/training_report.json"
PAIR_VALIDATION="$OUTPUT_ROOT/generation_10pct_pair_validation.json"
DEPLOYMENT_RECEIPT="$OUTPUT_ROOT/generation_upgrade_deployment_receipt.json"
CONFLICT_REPORT="$OUTPUT_ROOT/generation_upgrade_conflict_scan.json"
RUNBOOK_SYNTAX_REPORT="$OUTPUT_ROOT/generation_upgrade_runbook_syntax.json"
PYTEST_REPORT="$OUTPUT_ROOT/generation_upgrade_pytest.xml"
DEPLOYMENT_EVIDENCE_ROOT="$OUTPUT_ROOT/deployment"
ARCHIVED_BUNDLE="$DEPLOYMENT_EVIDENCE_ROOT/cofitok-generation-upgrade-${TARGET_COMMIT}.bundle"

cd "$PROJECT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm

current_commit="$(git rev-parse HEAD)"
if [[ "$current_commit" != "$EXPECTED_COMMIT" && "$current_commit" != "$TARGET_COMMIT" ]]; then
  printf 'remote HEAD %s is neither expected pinned commit %s nor target %s\n' \
    "$current_commit" "$EXPECTED_COMMIT" "$TARGET_COMMIT" >&2
  exit 65
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'remote tracked worktree has local changes\n' >&2
  exit 66
fi
if [[ "$current_commit" == "$EXPECTED_COMMIT" && ! -f "$BUNDLE" ]]; then
  printf 'upgrade bundle is missing: %s\n' "$BUNDLE" >&2
  exit 67
fi

prevalidation_root=""
cleanup_prevalidation() {
  if [[ -n "$prevalidation_root" && -d "$prevalidation_root" ]]; then
    rm -rf -- "$prevalidation_root"
  fi
}
trap cleanup_prevalidation EXIT

if [[ "$current_commit" == "$EXPECTED_COMMIT" ]]; then
  bundle_prerequisites=()
  while IFS= read -r bundle_header_line; do
    [[ -z "$bundle_header_line" ]] && break
    if [[ "$bundle_header_line" == -* ]]; then
      bundle_prerequisite="${bundle_header_line#-}"
      bundle_prerequisite="${bundle_prerequisite%% *}"
      bundle_prerequisites+=("$bundle_prerequisite")
    fi
  done < "$BUNDLE"
  if (( ${#bundle_prerequisites[@]} != 1 )) || \
      [[ "${bundle_prerequisites[0]:-}" != "$EXPECTED_COMMIT" ]]; then
    printf 'bundle prerequisites must equal only pinned commit %s; found: %s\n' \
      "$EXPECTED_COMMIT" "${bundle_prerequisites[*]:-none}" >&2
    exit 78
  fi
  git bundle verify "$BUNDLE"
  git fetch "$BUNDLE" HEAD
  fetched_commit="$(git rev-parse FETCH_HEAD)"
  if [[ "$fetched_commit" != "$TARGET_COMMIT" ]]; then
    printf 'bundle resolved to %s instead of target %s\n' \
      "$fetched_commit" "$TARGET_COMMIT" >&2
    exit 70
  fi
  prevalidation_root="$(mktemp -d /tmp/cofitok-generation-prevalidation.XXXXXX)"
  git archive "$TARGET_COMMIT" -- \
    scripts/validate_generation_training_pair.py \
    scripts/check_generation_deployment_conflicts.py \
    src/cofitok | tar -x -C "$prevalidation_root"
  validator="$prevalidation_root/scripts/validate_generation_training_pair.py"
  conflict_checker="$prevalidation_root/scripts/check_generation_deployment_conflicts.py"
  validator_pythonpath="$prevalidation_root/src"
else
  fetched_commit="$TARGET_COMMIT"
  validator="scripts/validate_generation_training_pair.py"
  conflict_checker="scripts/check_generation_deployment_conflicts.py"
  validator_pythonpath="src"
fi

mkdir -p "$OUTPUT_ROOT" "$DEPLOYMENT_EVIDENCE_ROOT"
pair_validation_temporary="${PAIR_VALIDATION}.tmp.$$"
if PYTHONPATH="$validator_pythonpath" python "$validator" \
    --cofitok-training "$COFITOK_REPORT" \
    --dense-training "$DENSE_REPORT" \
    --expected-steps 50000 --expected-revision "$EXPECTED_COMMIT" \
    --expected-recipe-stage legacy_scaling \
    --allow-legacy-missing-dataset-provenance \
    >"$pair_validation_temporary"; then
  mv "$pair_validation_temporary" "$PAIR_VALIDATION"
else
  validation_exit_code=$?
  rm -f "$pair_validation_temporary"
  exit "$validation_exit_code"
fi

if pgrep -af '[s]cripts/train_generation.py.*configs/generation/imagenet256_10pct_' >/dev/null; then
  printf 'a 10%% generation training process is still active\n' >&2
  exit 68
fi
if pgrep -af '[g]eneration_10pct_matched_50k_2026-07-12.sh' >/dev/null; then
  printf 'the 10%% matched training runbook is still active\n' >&2
  exit 69
fi

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(cat "$PID_FILE")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    if [[ "$current_commit" != "$TARGET_COMMIT" ]]; then
      printf 'live supervisor PID %s cannot authorize deployment from HEAD %s\n' \
        "$previous_pid" "$current_commit" >&2
      exit 79
    fi
    previous_command="$(ps -p "$previous_pid" -o args=)"
    if [[ "$previous_command" != *"generation_completion_supervisor.sh"* ]]; then
      printf 'PID %s is live but is not the generation completion supervisor\n' \
        "$previous_pid" >&2
      exit 80
    fi
    if [[ ! -f "$DEPLOYMENT_RECEIPT" ]]; then
      printf 'live supervisor lacks deployment receipt: %s\n' \
        "$DEPLOYMENT_RECEIPT" >&2
      exit 81
    fi
    python - "$DEPLOYMENT_RECEIPT" "$EXPECTED_COMMIT" "$TARGET_COMMIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt.get("status") != "pass":
    raise SystemExit("existing deployment receipt did not pass")
if receipt.get("expected_training_revision") != sys.argv[2]:
    raise SystemExit("existing deployment receipt has the wrong training revision")
if receipt.get("target_revision") != sys.argv[3]:
    raise SystemExit("existing deployment receipt has the wrong target revision")
PY
    printf 'generation completion supervisor is already active as PID %s\n' \
      "$previous_pid"
    exit 0
  fi
fi

if [[ "$current_commit" == "$EXPECTED_COMMIT" ]]; then
  python "$conflict_checker" \
    --repository "$PROJECT" \
    --current-commit "$current_commit" \
    --target-commit "$fetched_commit" \
    --output "$CONFLICT_REPORT"
  git merge --ff-only FETCH_HEAD
  if [[ "$(git rev-parse HEAD)" != "$TARGET_COMMIT" ]]; then
    printf 'remote fast-forward did not reach target commit\n' >&2
    exit 71
  fi
elif [[ ! -f "$CONFLICT_REPORT" ]]; then
  printf 'target revision is active but the original conflict report is missing: %s\n' \
    "$CONFLICT_REPORT" >&2
  exit 77
fi

export PYTHONPATH=src
pytest_temporary="${PYTEST_REPORT}.tmp.$$"
if python -m pytest -q --junitxml "$pytest_temporary"; then
  mv "$pytest_temporary" "$PYTEST_REPORT"
else
  pytest_exit_code=$?
  rm -f "$pytest_temporary"
  exit "$pytest_exit_code"
fi
python scripts/check_generation_runbook_syntax.py \
  --project-root "$PROJECT" --output "$RUNBOOK_SYNTAX_REPORT"

bundle_temporary="${ARCHIVED_BUNDLE}.tmp.$$"
cp -- "$BUNDLE" "$bundle_temporary"
mv "$bundle_temporary" "$ARCHIVED_BUNDLE"
git bundle verify "$ARCHIVED_BUNDLE"

python scripts/write_generation_deployment_receipt.py \
  --output "$DEPLOYMENT_RECEIPT" --bundle "$ARCHIVED_BUNDLE" \
  --pair-validation "$PAIR_VALIDATION" \
  --conflict-scan "$CONFLICT_REPORT" \
  --runbook-syntax "$RUNBOOK_SYNTAX_REPORT" \
  --pytest-report "$PYTEST_REPORT" \
  --expected-training-revision "$EXPECTED_COMMIT" \
  --target-revision "$TARGET_COMMIT"

nohup bash "$SUPERVISOR" >"$LOG" 2>&1 </dev/null &
supervisor_pid=$!
pid_temporary="${PID_FILE}.tmp.$$"
printf '%s\n' "$supervisor_pid" >"$pid_temporary"
mv "$pid_temporary" "$PID_FILE"
sleep 2
if ! kill -0 "$supervisor_pid" 2>/dev/null; then
  printf 'generation completion supervisor exited during launch; inspect %s\n' "$LOG" >&2
  exit 73
fi
printf 'launched generation completion supervisor PID %s at commit %s\n' \
  "$supervisor_pid" "$TARGET_COMMIT"
