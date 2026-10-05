#!/usr/bin/env bash
# Resume the CoFiTok generation archive over two measured SSH streams.
set -euo pipefail

readonly source_root="/root/autodl-tmp/CoFiTok/checkpoints/generation"
readonly target_root="/home/yubohuang/zixi/CoFiTok/checkpoints/generation"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179"
readonly common_options=(
  -rLt
  --partial
  --append-verify
  --human-readable
  --info=progress2,stats2
  -e "$ssh_command"
)

mkdir -p /tmp/cofitok_generation_archive_parallel_20260817

# The two streams are balanced by source allocation: 51 GiB + 8.8 GiB versus
# the remaining 57 GiB. -L materializes the one project symlink as a real file.
rsync "${common_options[@]}" \
  "$source_root/stability_probe_2026-07-29" \
  "$source_root/stability_scaling_50k_ema_teacher" \
  "yubohuang@127.0.0.1:$target_root/" \
  > /tmp/cofitok_generation_archive_parallel_20260817/stream_a.log 2>&1 &
stream_a_pid=$!

rsync "${common_options[@]}" \
  --exclude='stability_probe_2026-07-29/***' \
  --exclude='stability_scaling_50k_ema_teacher/***' \
  "$source_root/" \
  "yubohuang@127.0.0.1:$target_root/" \
  > /tmp/cofitok_generation_archive_parallel_20260817/stream_b.log 2>&1 &
stream_b_pid=$!

wait "$stream_a_pid"
wait "$stream_b_pid"

# Metadata-only reconciliation detects any remaining source paths without
# modifying the target. A clean report has no itemized change lines.
rsync -rLt --dry-run --itemize-changes \
  -e "$ssh_command" \
  "$source_root/" \
  "yubohuang@127.0.0.1:$target_root/" \
  > /tmp/cofitok_generation_archive_parallel_20260817/final_dry_run.log 2>&1

if [[ -s /tmp/cofitok_generation_archive_parallel_20260817/final_dry_run.log ]]; then
  printf 'generation archive dry-run reported outstanding changes\n' >&2
  exit 1
fi

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > /tmp/cofitok_generation_archive_parallel_20260817/completed
