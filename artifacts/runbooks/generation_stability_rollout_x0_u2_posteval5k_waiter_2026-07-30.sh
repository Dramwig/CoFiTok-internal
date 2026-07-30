#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2
MONITOR="${PAIR_ROOT}_monitor.json"
PAIR_SUMMARY="$PAIR_ROOT/pair_summary.json"
POSTEVAL=/tmp/generation_stability_rollout_x0_u2_posteval5k_2026-07-30.sh
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_POSTEVAL_SHA256=f6358383cabd61c2215dea748014b8d93e189d57c5690e307c4f34078f0a64cd
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-57600}
POLL_SECONDS=${POLL_SECONDS:-60}

if (( TIMEOUT_SECONDS <= 0 || POLL_SECONDS <= 0 )); then
  printf 'waiter timeout and poll interval must be positive\n' >&2
  exit 64
fi

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ -f "$MONITOR" ]]
[[ -f "$POSTEVAL" ]]
[[ "$(sha256sum "$POSTEVAL" | awk '{print $1}')" == "$EXPECTED_POSTEVAL_SHA256" ]]

deadline=$((SECONDS + TIMEOUT_SECONDS))
while true; do
  state="$("$PYTHON" - "$MONITOR" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

monitor = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_revision = sys.argv[2]
if monitor["git"]["revision"] != expected_revision:
    raise SystemExit("monitor revision mismatch")
if monitor["git"]["tracked_dirty"]:
    raise SystemExit("monitor reports a dirty tracked checkout")
status = monitor["status"]
if status not in {"running", "waiting", "pass", "failed", "stalled"}:
    raise SystemExit(f"unknown monitor status: {status}")
print(f"{status}\t{monitor['stage']}\t{monitor['updated_at']}")
PY
)"
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$state"

  status="${state%%$'\t'*}"
  if [[ "$status" == failed || "$status" == stalled ]]; then
    printf 'matched 5K monitor reached terminal status %s\n' "$status" >&2
    exit 65
  fi
  if [[ "$status" == pass ]] && [[ -f "$PAIR_SUMMARY" ]] && \
      ! pgrep -af '[g]eneration_stability_rollout_x0_u2_probe5k_2026-07-30.sh' >/dev/null && \
      ! pgrep -af '[s]cripts/train_generation.py' >/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    printf 'timed out waiting for the matched two-step 5K pair\n' >&2
    exit 66
  fi
  sleep "$POLL_SECONDS"
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing post-evaluation because the GPU is still busy\n' >&2
  exit 67
fi

pair_summary_sha256="$(sha256sum "$PAIR_SUMMARY" | awk '{print $1}')"
printf 'launching bound two-step 5K post-evaluation with pair summary %s\n' \
  "$pair_summary_sha256"
EXPECTED_PAIR_SUMMARY_SHA256="$pair_summary_sha256" bash "$POSTEVAL"
