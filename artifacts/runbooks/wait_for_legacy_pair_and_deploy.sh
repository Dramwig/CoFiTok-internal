#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  printf 'usage: %s BUNDLE EXPECTED_COMMIT TARGET_COMMIT DEPLOY_HELPER\n' "$0" >&2
  exit 64
fi

BUNDLE="$1"
EXPECTED_COMMIT="$2"
TARGET_COMMIT="$3"
DEPLOY_HELPER="$4"
PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_REPORT="$OUTPUT_ROOT/imagenet256_10pct_cofitok_k8_50k_2026-07-12/training_report.json"
DENSE_REPORT="$OUTPUT_ROOT/imagenet256_10pct_dense_50k_2026-07-12/training_report.json"
MONITOR="$OUTPUT_ROOT/generation_10pct_pair_monitor.json"
STATUS="$OUTPUT_ROOT/generation_upgrade_deployment_waiter.json"
LOCK="$OUTPUT_ROOT/generation_upgrade_deployment_waiter.lock"
POLL_SECONDS="${COFITOK_DEPLOY_WAITER_POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${COFITOK_DEPLOY_WAITER_MAX_SECONDS:-172800}"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$MAX_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'waiter intervals must be positive integers\n' >&2
  exit 64
fi
if [[ ! -f "$BUNDLE" || ! -f "$DEPLOY_HELPER" ]]; then
  printf 'deployment bundle or helper is missing\n' >&2
  exit 66
fi

cd "$PROJECT"
current_commit="$(git rev-parse HEAD)"
if [[ "$current_commit" != "$EXPECTED_COMMIT" && "$current_commit" != "$TARGET_COMMIT" ]]; then
  printf 'unexpected formal HEAD: %s\n' "$current_commit" >&2
  exit 65
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'formal tracked worktree is dirty\n' >&2
  exit 66
fi
git bundle verify "$BUNDLE" >/dev/null
bundle_head="$(git bundle list-heads "$BUNDLE" | awk 'NR == 1 {print $1}')"
if [[ "$bundle_head" != "$TARGET_COMMIT" ]]; then
  printf 'bundle head %s does not match target %s\n' "$bundle_head" "$TARGET_COMMIT" >&2
  exit 67
fi

mkdir -p "$OUTPUT_ROOT"
exec 7>"$LOCK"
if ! flock -n 7; then
  printf 'deployment waiter already holds %s\n' "$LOCK" >&2
  exit 75
fi

write_status() {
  local state="$1"
  local detail="$2"
  local elapsed="$3"
  /root/miniconda3/bin/python - "$STATUS" "$state" "$detail" "$elapsed" \
    "$EXPECTED_COMMIT" "$TARGET_COMMIT" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "waiter": "legacy_pair_to_compressed_generation_deployment",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "elapsed_seconds": int(sys.argv[4]),
    "expected_commit": sys.argv[5],
    "target_commit": sys.argv[6],
    "hostname": socket.gethostname(),
    "pid": os.getppid(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

pair_complete() {
  /root/miniconda3/bin/python - "$COFITOK_REPORT" "$DENSE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(1)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("training_complete") is not True:
        raise SystemExit(1)
    if report.get("completed_steps") != 50_000 or report.get("target_steps") != 50_000:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

legacy_process_active() {
  pgrep -af '[s]cripts/train_generation.py.*configs/generation/imagenet256_10pct_' >/dev/null || \
    pgrep -af '[g]eneration_10pct_matched_50k_2026-07-12.sh' >/dev/null
}

monitor_failed() {
  /root/miniconda3/bin/python - "$MONITOR" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(0)
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
raise SystemExit(0 if report.get("status") == "failed" else 1)
PY
}

started="$(date +%s)"
while true; do
  now="$(date +%s)"
  elapsed=$(( now - started ))
  if (( elapsed > MAX_WAIT_SECONDS )); then
    write_status failed "timed out waiting for the legacy matched pair" "$elapsed"
    exit 78
  fi
  if monitor_failed; then
    write_status failed "legacy matched-pair monitor reported failure" "$elapsed"
    exit 1
  fi
  if pair_complete; then
    if legacy_process_active; then
      write_status waiting "reports complete; waiting for legacy processes to exit" "$elapsed"
    else
      write_status deploying "legacy pair complete; invoking verified deployment" "$elapsed"
      bash "$DEPLOY_HELPER" "$BUNDLE" "$EXPECTED_COMMIT" "$TARGET_COMMIT"
      now="$(date +%s)"
      write_status pass "deployment completed and supervisor launched" "$(( now - started ))"
      exit 0
    fi
  else
    write_status waiting "waiting for both legacy 50K training reports" "$elapsed"
  fi
  sleep "$POLL_SECONDS"
done
