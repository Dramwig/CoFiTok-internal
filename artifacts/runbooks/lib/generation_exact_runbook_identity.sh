#!/usr/bin/env bash

cofitok_capture_runbook_monitor_identity() {
  if (( $# != 3 )); then
    printf 'usage: cofitok_capture_runbook_monitor_identity PYTHON PROJECT PID\n' >&2
    return 64
  fi
  local python_executable="$1"
  local project_root="$2"
  local controller_pid="$3"
  local -a identity_args=()
  mapfile -d '' -t identity_args < <(
    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$python_executable" \
      "$project_root/scripts/print_generation_process_identity.py" \
      --pid "$controller_pid" \
      --format monitor-args0
  )
  local -a expected_options=(
    --runbook-process-pid
    --runbook-process-start-ticks
    --runbook-process-executable
    --runbook-process-cwd
    --runbook-process-cmdline-sha256
  )
  if (( ${#identity_args[@]} != 10 )); then
    printf 'runbook controller identity did not produce ten monitor tokens\n' >&2
    return 65
  fi
  local index
  for index in 0 1 2 3 4; do
    if [[ "${identity_args[index * 2]}" != "${expected_options[index]}" ]]; then
      printf 'runbook controller identity option order is invalid\n' >&2
      return 65
    fi
  done
  COFITOK_RUNBOOK_PROCESS_PID="${identity_args[1]}"
  COFITOK_RUNBOOK_PROCESS_START_TICKS="${identity_args[3]}"
  COFITOK_RUNBOOK_PROCESS_EXECUTABLE="${identity_args[5]}"
  COFITOK_RUNBOOK_PROCESS_CWD="${identity_args[7]}"
  COFITOK_RUNBOOK_PROCESS_CMDLINE_SHA256="${identity_args[9]}"
  if [[ ! "$COFITOK_RUNBOOK_PROCESS_PID" =~ ^[1-9][0-9]*$ \
    || ! "$COFITOK_RUNBOOK_PROCESS_START_TICKS" =~ ^[1-9][0-9]*$ \
    || "$COFITOK_RUNBOOK_PROCESS_EXECUTABLE" != /* \
    || "$COFITOK_RUNBOOK_PROCESS_CWD" != /* \
    || ! "$COFITOK_RUNBOOK_PROCESS_CMDLINE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'runbook controller identity values are invalid\n' >&2
    return 65
  fi
}

cofitok_monitor_report_matches_bound_controller() {
  if (( $# != 3 )); then
    printf 'usage: cofitok_monitor_report_matches_bound_controller PYTHON REPORT MONITOR\n' >&2
    return 64
  fi
  local python_executable="$1"
  local report_path="$2"
  local monitor_name="$3"
  "$python_executable" - \
    "$report_path" \
    "$monitor_name" \
    "$COFITOK_RUNBOOK_PROCESS_PID" \
    "$COFITOK_RUNBOOK_PROCESS_START_TICKS" \
    "$COFITOK_RUNBOOK_PROCESS_EXECUTABLE" \
    "$COFITOK_RUNBOOK_PROCESS_CWD" \
    "$COFITOK_RUNBOOK_PROCESS_CMDLINE_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
identity = report.get("runbook_identity", {})
expected = {
    "pid": int(sys.argv[3]),
    "start_ticks": int(sys.argv[4]),
    "executable": sys.argv[5],
    "cwd": sys.argv[6],
    "cmdline_sha256": sys.argv[7],
}
if (
    report.get("monitor") != sys.argv[2]
    or identity.get("role") != "generation_exact_runbook_process_identity"
    or identity.get("mode") != "exact_process_identity"
    or identity.get("status") != "active"
    or identity.get("mismatches") != []
):
    raise SystemExit(1)
for field, value in expected.items():
    if identity.get("expected", {}).get(field) != value:
        raise SystemExit(1)
    if identity.get("observed", {}).get(field) != value:
        raise SystemExit(1)
PY
}
