#!/usr/bin/env bash
# Recover a dynamic archive only if the initial two-stream transfer exits early.
set -euo pipefail

readonly parent_pid="$1"
readonly state_dir="/tmp/cofitok_generation_archive_parallel_20260817"
readonly source_root="/root/autodl-tmp/CoFiTok/checkpoints/generation"
readonly target_root="/home/yubohuang/zixi/CoFiTok/checkpoints/generation"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179"

mkdir -p "$state_dir"
printf 'started %s for parent %s\n' "$(date --iso-8601=seconds)" "$parent_pid" \
  >> "$state_dir/completion_guard.log"

while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 300
done

# Do not overlap a surviving initial rsync or the scaling-stream helper.
while pgrep -f '/tmp/cofitok_generation_scaling_second_stream_20260817.sh' >/dev/null \
  || pgrep -f 'rsync.*cofitok_ybforever_push_20260817' >/dev/null; do
  sleep 120
done

if [[ -f "$state_dir/completed" ]]; then
  printf 'initial transfer completed without recovery %s\n' "$(date --iso-8601=seconds)" \
    >> "$state_dir/completion_guard.log"
  rm -f -- "$0"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  printf 'recovery attempt %s started %s\n' "$attempt" "$(date --iso-8601=seconds)" \
    >> "$state_dir/completion_guard.log"

  if ! rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
    -e "$ssh_command" \
    "$source_root/" \
    "yubohuang@127.0.0.1:$target_root/" \
    > "$state_dir/recovery_rsync.log" 2>&1; then
    printf 'recovery rsync attempt %s failed %s\n' "$attempt" "$(date --iso-8601=seconds)" \
      >> "$state_dir/completion_guard.log"
    sleep 300
    continue
  fi

  if rsync -rLt --dry-run --itemize-changes \
    -e "$ssh_command" \
    "$source_root/" \
    "yubohuang@127.0.0.1:$target_root/" \
    > "$state_dir/final_dry_run.log" 2>&1 \
    && [[ ! -s "$state_dir/final_dry_run.log" ]]; then
    printf 'completed %s via recovery attempt %s\n' "$(date --iso-8601=seconds)" "$attempt" \
      > "$state_dir/completed"
    printf 'recovery succeeded %s\n' "$(date --iso-8601=seconds)" \
      >> "$state_dir/completion_guard.log"
    rm -f -- "$0"
    exit 0
  fi

  printf 'recovery attempt %s found source changes; retrying %s\n' "$attempt" "$(date --iso-8601=seconds)" \
    >> "$state_dir/completion_guard.log"
  sleep 300
done
