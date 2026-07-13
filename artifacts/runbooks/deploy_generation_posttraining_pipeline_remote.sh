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
    src/cofitok | tar -x -C "$prevalidation_root"
  validator="$prevalidation_root/scripts/validate_generation_training_pair.py"
  validator_pythonpath="$prevalidation_root/src"
else
  fetched_commit="$TARGET_COMMIT"
  validator="scripts/validate_generation_training_pair.py"
  validator_pythonpath="src"
fi

mkdir -p "$OUTPUT_ROOT"
pair_validation_temporary="${PAIR_VALIDATION}.tmp.$$"
if PYTHONPATH="$validator_pythonpath" python "$validator" \
    --cofitok-training "$COFITOK_REPORT" \
    --dense-training "$DENSE_REPORT" \
    --expected-steps 50000 --expected-revision "$EXPECTED_COMMIT" \
    --expected-recipe-stage scaling \
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
    printf 'generation completion supervisor is already active as PID %s\n' \
      "$previous_pid"
    exit 0
  fi
fi

if [[ "$current_commit" == "$EXPECTED_COMMIT" ]]; then
  mapfile -t untracked_conflicts < <(
    comm -12 \
      <(git ls-files --others --exclude-standard | LC_ALL=C sort) \
      <(git ls-tree -r --name-only "$fetched_commit" | LC_ALL=C sort)
  )
  if (( ${#untracked_conflicts[@]} > 0 )); then
    printf 'untracked files would conflict with target revision:\n' >&2
    printf '  %s\n' "${untracked_conflicts[@]}" >&2
    exit 76
  fi
  git merge --ff-only FETCH_HEAD
  if [[ "$(git rev-parse HEAD)" != "$TARGET_COMMIT" ]]; then
    printf 'remote fast-forward did not reach target commit\n' >&2
    exit 71
  fi
fi

export PYTHONPATH=src
python -m pytest -q
bash -n artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh
bash -n artifacts/runbooks/generation_full_milestone_eval.sh
bash -n artifacts/runbooks/generation_full_matched_300k_after_gate.sh
bash -n artifacts/runbooks/generation_full_posteval_50k.sh
bash -n artifacts/runbooks/generation_export_inference_artifacts.sh
bash -n "$PIPELINE"
bash -n "$SUPERVISOR"

python scripts/write_generation_deployment_receipt.py \
  --output "$DEPLOYMENT_RECEIPT" --bundle "$BUNDLE" \
  --pair-validation "$PAIR_VALIDATION" \
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
