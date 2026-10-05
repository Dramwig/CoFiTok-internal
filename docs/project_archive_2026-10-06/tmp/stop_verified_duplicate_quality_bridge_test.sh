#!/usr/bin/env bash
set -euo pipefail

TARGET_PGID=931172
TARGET_PARENT=931172
TARGET_PYTEST=931173
EXPECTED_CWD=/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
FIELDSCOPE_PID=910099

if ! kill -0 "$TARGET_PARENT" 2>/dev/null && ! kill -0 "$TARGET_PYTEST" 2>/dev/null; then
  printf 'verified duplicate test already exited\n'
  exit 0
fi

for pid in "$TARGET_PARENT" "$TARGET_PYTEST"; do
  [[ -r "/proc/$pid/stat" ]]
  actual_pgid="$(ps -o pgid= -p "$pid" | xargs)"
  actual_cwd="$(readlink -f "/proc/$pid/cwd")"
  actual_cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$actual_pgid" == "$TARGET_PGID" ]]
  [[ "$actual_cwd" == "$EXPECTED_CWD" ]]
  if [[ "$pid" == "$TARGET_PARENT" ]]; then
    [[ "$actual_cmd" == *prelaunch_recovery_contract_tests.log* ]]
  else
    [[ "$actual_cmd" == *"python -m pytest -q tests/test_generation_quality_bridge.py"* ]]
  fi
done

fieldscope_pgid="$(ps -o pgid= -p "$FIELDSCOPE_PID" | xargs)"
[[ -n "$fieldscope_pgid" ]]
[[ "$fieldscope_pgid" != "$TARGET_PGID" ]]

mapfile -t members < <(ps -eo pid=,pgid= | awk -v pgid="$TARGET_PGID" '$2 == pgid {print $1}')
(( ${#members[@]} >= 2 ))
for pid in "${members[@]}"; do
  cwd="$(readlink -f "/proc/$pid/cwd")"
  [[ "$cwd" == "$EXPECTED_CWD" || "$cwd" == /tmp/pytest-of-root/* ]]
done

printf 'terminating verified duplicate test group %s with members:' "$TARGET_PGID"
printf ' %s' "${members[@]}"
printf '\n'
kill -TERM -- "-$TARGET_PGID"

for _ in $(seq 1 15); do
  if ! ps -eo pgid= | awk -v pgid="$TARGET_PGID" '$1 == pgid {found=1} END {exit found ? 0 : 1}'; then
    printf 'duplicate test group stopped\n'
    exit 0
  fi
  sleep 1
done

printf 'duplicate test group did not exit after TERM\n' >&2
ps -eo pid=,ppid=,pgid=,stat=,cmd= | awk -v pgid="$TARGET_PGID" '$3 == pgid'
exit 91
