#!/usr/bin/env bash
# Maintain two transfer streams after the non-stability generation stream ends.
set -euo pipefail

readonly stream_b_pid="$1"
readonly source_dir="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher"
readonly target_dir="/home/yubohuang/zixi/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179"
readonly state_dir="/tmp/cofitok_generation_archive_parallel_20260817"

while kill -0 "$stream_b_pid" 2>/dev/null; do
  sleep 180
done

rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
  -e "$ssh_command" \
  "$source_dir/" \
  "yubohuang@127.0.0.1:$target_dir" \
  > "$state_dir/stream_scaling.log" 2>&1

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > "$state_dir/stream_scaling.completed"
