#!/usr/bin/env bash
set -euo pipefail

root=/tmp/cofitok-quality-bridge-followup-decision-9b02fa8
waiter="$root/wait_for_quality_bridge_result.py"
status="$root/wait_for_quality_bridge_result_status.json"
log="$root/wait_for_quality_bridge_result.log"
pid_file="$root/wait_for_quality_bridge_result.pid"
launcher_log="$root/wait_for_quality_bridge_result_launcher.log"

if [[ -f "$pid_file" ]]; then
  existing_pid=$(cat "$pid_file")
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'decision waiter is already active: %s\n' "$existing_pid" >&2
    exit 75
  fi
fi

if pgrep -af -- "$waiter" | grep -v -F "pgrep -af" >/dev/null; then
  printf 'unreceipted decision waiter process already exists\n' >&2
  exit 75
fi

nohup /root/autodl-tmp/conda/envs/pf-vlm/bin/python "$waiter" \
  --project "$root" \
  --quality-bridge-root /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1 \
  --expected-revision 9b02fa83d20b1459a2706d6d82371caf5c023f54 \
  --expected-tree 474520aa848f3da143379a7b7f72d59a085fbe38 \
  --expected-branch scale/generation-quality-bridge-followup-decision-v1 \
  --expected-self-sha256 77859b4bdcff20e79449654a1b20a42d6ad21d1c96bec35e189b01b22f01edd2 \
  --python /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  --status "$status" \
  --log "$log" \
  --poll-seconds 60 \
  >"$launcher_log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
printf 'launched decision waiter pid=%s\n' "$pid"
