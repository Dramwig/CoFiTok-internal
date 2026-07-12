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
LOG="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.log"
PID_FILE="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.pid"
COFITOK_REPORT="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12/training_report.json"
DENSE_REPORT="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12/training_report.json"

cd "$PROJECT"

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

python - "$COFITOK_REPORT" "$DENSE_REPORT" "$EXPECTED_COMMIT" <<'PY'
import json
import sys

for path in sys.argv[1:3]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("training_complete") is not True:
        raise SystemExit(f"training is incomplete: {path}")
    if report.get("completed_steps") != 50_000 or report.get("target_steps") != 50_000:
        raise SystemExit(f"training did not finish exactly 50K steps: {path}")
    if report.get("git", {}).get("commit") != sys.argv[3]:
        raise SystemExit(f"training report commit does not match pinned revision: {path}")
    if report.get("git", {}).get("dirty") is not False:
        raise SystemExit(f"training used a dirty tracked worktree: {path}")
PY

if pgrep -af '[s]cripts/train_generation.py.*configs/generation/imagenet256_10pct_' >/dev/null; then
  printf 'a 10%% generation training process is still active\n' >&2
  exit 68
fi
if pgrep -af '[g]eneration_10pct_matched_50k_2026-07-12.sh' >/dev/null; then
  printf 'the 10%% matched training runbook is still active\n' >&2
  exit 69
fi

mkdir -p "$OUTPUT_ROOT"
if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(cat "$PID_FILE")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    printf 'generation completion pipeline is already active as PID %s\n' \
      "$previous_pid"
    exit 0
  fi
fi

if [[ "$current_commit" == "$EXPECTED_COMMIT" ]]; then
  git bundle verify "$BUNDLE"
  git fetch "$BUNDLE" HEAD
  fetched_commit="$(git rev-parse FETCH_HEAD)"
  if [[ "$fetched_commit" != "$TARGET_COMMIT" ]]; then
    printf 'bundle resolved to %s instead of target %s\n' \
      "$fetched_commit" "$TARGET_COMMIT" >&2
    exit 70
  fi
  git merge --ff-only FETCH_HEAD
  if [[ "$(git rev-parse HEAD)" != "$TARGET_COMMIT" ]]; then
    printf 'remote fast-forward did not reach target commit\n' >&2
    exit 71
  fi
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
export PYTHONPATH=src
python -m pytest -q
bash -n artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh
bash -n artifacts/runbooks/generation_full_milestone_eval.sh
bash -n artifacts/runbooks/generation_full_matched_300k_after_gate.sh
bash -n artifacts/runbooks/generation_full_posteval_50k.sh
bash -n "$PIPELINE"

nohup bash "$PIPELINE" >"$LOG" 2>&1 </dev/null &
pipeline_pid=$!
pid_temporary="${PID_FILE}.tmp.$$"
printf '%s\n' "$pipeline_pid" >"$pid_temporary"
mv "$pid_temporary" "$PID_FILE"
sleep 2
if ! kill -0 "$pipeline_pid" 2>/dev/null; then
  printf 'generation completion pipeline exited during launch; inspect %s\n' "$LOG" >&2
  exit 73
fi
printf 'launched generation completion pipeline PID %s at commit %s\n' \
  "$pipeline_pid" "$TARGET_COMMIT"
