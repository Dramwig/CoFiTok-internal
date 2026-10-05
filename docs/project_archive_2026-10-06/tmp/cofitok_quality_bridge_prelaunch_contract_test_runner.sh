#!/usr/bin/env bash
set -euo pipefail

BASE=/tmp/cofitok-quality-bridge-execution-cf0e5fa
PROJECT="$BASE/CoFiTok-internal"
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
LOG="$BASE/prelaunch_contract_tests_cpu.log"
STATUS="$BASE/prelaunch_contract_tests_cpu_status.json"
REVISION=cf0e5faa94bf4ab38d947b921935b3b765b5537a
TREE=6cef27723196fd363379bca2e7b85b1678ebd777

TESTS=(
  tests/test_generation_quality_bridge.py
  tests/test_generation_milestone_report.py
  tests/test_generation_exact_resume.py
  tests/test_generation_metrics_resume.py
  tests/test_generation_checkpoint_retention.py
  tests/test_generation_training_watchdog.py
  tests/test_monitor_generation_10pct_pair.py
  tests/test_generation_training_completion.py
  tests/test_audit_generation_training_progress.py
  tests/test_generation_runtime_selection.py
)

write_status() {
  local state="$1"
  local exit_code="${2:-}"
  local log_path="${3:-}"
  STATE="$state" EXIT_CODE="$exit_code" LOG_PATH="$log_path" \
    REVISION="$REVISION" TREE="$TREE" RUNNER_PID="$$" \
    "$PYTHON" - "$STATUS" "${TESTS[@]}" <<'PY'
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
tests = sys.argv[2:]
log_path = Path(os.environ["LOG_PATH"]) if os.environ["LOG_PATH"] else None
log_identity = None
if log_path is not None and log_path.is_file():
    payload = log_path.read_bytes()
    log_identity = {
        "path": log_path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
report = {
    "schema_version": 1,
    "role": "cofitok_quality_bridge_prelaunch_contract_test",
    "status": os.environ["STATE"],
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ["EXIT_CODE"] else None,
    "runner_pid": int(os.environ["RUNNER_PID"]),
    "hostname": socket.gethostname(),
    "git": {
        "revision": os.environ["REVISION"],
        "tree": os.environ["TREE"],
        "tracked_dirty": False,
    },
    "cuda_visible_devices": "-1",
    "pytest_cache_disabled": True,
    "tests": tests,
    "log": log_identity,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

[[ -x "$PYTHON" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$REVISION" ]]
[[ "$(git -C "$PROJECT" rev-parse HEAD^{tree})" == "$TREE" ]]
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
if [[ -f "$STATUS" ]] && "$PYTHON" - "$STATUS" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text())["status"] == "passed" else 1)
PY
then
  printf 'prelaunch contract tests already passed\n'
  exit 0
fi

write_status running
temporary_log="${LOG}.tmp.$$"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="$PROJECT:$PROJECT/src"
export PYTHONDONTWRITEBYTECODE=1
set +e
"$PYTHON" -m pytest -q -p no:cacheprovider "${TESTS[@]}" >"$temporary_log" 2>&1
exit_code=$?
set -e
mv "$temporary_log" "$LOG"
if [[ "$exit_code" -eq 0 ]]; then
  write_status passed "$exit_code" "$LOG"
else
  write_status failed "$exit_code" "$LOG"
fi
[[ -z "$(git -C "$PROJECT" status --porcelain)" ]]
exit "$exit_code"
