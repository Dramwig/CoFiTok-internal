#!/usr/bin/env bash
# Resume the full generation archive after an interrupted transfer session.
set -euo pipefail

readonly state_dir="/tmp/cofitok_generation_archive_parallel_20260817"
readonly source_root="/root/autodl-tmp/CoFiTok/checkpoints/generation"
readonly target_root="/home/yubohuang/zixi/CoFiTok/checkpoints/generation"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p 22179"

mkdir -p "$state_dir"
attempt=0
while [[ ! -f "$state_dir/completed" ]]; do
  attempt=$((attempt + 1))
  printf 'recovery attempt %s started %s\n' "$attempt" "$(date --iso-8601=seconds)" \
    >> "$state_dir/recovery_supervisor.log"

  if rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
    -e "$ssh_command" \
    "$source_root/" \
    "yubohuang@127.0.0.1:$target_root/" \
    > "$state_dir/recovery_rsync.log" 2>&1; then
    if rsync -rLt --dry-run --itemize-changes \
      -e "$ssh_command" \
      "$source_root/" \
      "yubohuang@127.0.0.1:$target_root/" \
      > "$state_dir/final_dry_run.log" 2>&1 \
      && [[ ! -s "$state_dir/final_dry_run.log" ]]; then
      printf 'completed %s via recovery attempt %s\n' "$(date --iso-8601=seconds)" "$attempt" \
        > "$state_dir/completed"
      exit 0
    fi
  fi

  printf 'recovery attempt %s did not converge; retrying in 300 seconds\n' "$attempt" \
    >> "$state_dir/recovery_supervisor.log"
  sleep 300
done
