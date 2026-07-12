#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
PIPELINE="$PROJECT/artifacts/runbooks/generation_complete_pipeline_after_10pct.sh"
PIPELINE_STATUS="$OUTPUT_ROOT/generation_complete_pipeline_after_10pct.status.json"
STATUS="$OUTPUT_ROOT/generation_completion_supervisor.status.json"
LOCK="$OUTPUT_ROOT/generation_completion_supervisor.lock"
MAX_ATTEMPTS="${COFITOK_SUPERVISOR_MAX_ATTEMPTS:-4}"
BASE_DELAY_SECONDS="${COFITOK_SUPERVISOR_BASE_DELAY_SECONDS:-60}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src
mkdir -p "$OUTPUT_ROOT"

if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'invalid supervisor max attempts: %s\n' "$MAX_ATTEMPTS" >&2
  exit 64
fi
if [[ ! "$BASE_DELAY_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'invalid supervisor base delay: %s\n' "$BASE_DELAY_SECONDS" >&2
  exit 64
fi

exec 8>"$LOCK"
if ! flock -n 8; then
  printf 'generation completion supervisor already holds %s\n' "$LOCK" >&2
  exit 75
fi

write_status() {
  local status="$1"
  local detail="$2"
  local attempt="$3"
  local pipeline_stage="$4"
  local pipeline_status="$5"
  shift 5
  python scripts/write_generation_supervisor_status.py \
    --output "$STATUS" --status "$status" --detail "$detail" \
    --attempt "$attempt" --max-attempts "$MAX_ATTEMPTS" \
    --pipeline-stage "$pipeline_stage" --pipeline-status "$pipeline_status" \
    "$@"
}

for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
  write_status running "launching completion pipeline" "$attempt" unknown unknown
  printf 'supervisor attempt %s/%s\n' "$attempt" "$MAX_ATTEMPTS"
  set +e
  bash "$PIPELINE"
  child_exit_code=$?
  set -e

  classification="$(python scripts/classify_generation_pipeline_exit.py \
    --status "$PIPELINE_STATUS" --exit-code "$child_exit_code")"
  read -r decision pipeline_stage pipeline_status <<<"$classification"
  failure_exit_code="$child_exit_code"
  if (( failure_exit_code == 0 )); then
    failure_exit_code=1
  fi
  if [[ "$decision" == "pass" ]]; then
    write_status pass "completion pipeline passed" "$attempt" \
      "$pipeline_stage" "$pipeline_status"
    exit 0
  fi
  if [[ "$decision" != "retry" ]]; then
    write_status failed "completion pipeline stopped at a nonretryable stage" \
      "$attempt" "$pipeline_stage" "$pipeline_status" \
      --child-exit-code "$child_exit_code"
    exit "$failure_exit_code"
  fi
  if (( attempt == MAX_ATTEMPTS )); then
    write_status failed "completion pipeline exhausted bounded retries" \
      "$attempt" "$pipeline_stage" "$pipeline_status" \
      --child-exit-code "$child_exit_code"
    exit "$failure_exit_code"
  fi

  delay=$(( BASE_DELAY_SECONDS * (2 ** (attempt - 1)) ))
  if (( delay > 900 )); then
    delay=900
  fi
  write_status retrying "retrying recoverable pipeline stage" \
    "$attempt" "$pipeline_stage" "$pipeline_status" \
    --child-exit-code "$child_exit_code" --next-retry-seconds "$delay"
  sleep "$delay"
done

exit 1
