#!/usr/bin/env bash
# Sync public-source fallback datasets after the generation archive is verified.
set -euo pipefail

readonly state_dir="/tmp/cofitok_generation_archive_parallel_20260817"
readonly source_root="/root/autodl-tmp/CoFiTok/datasets"
readonly target_root="/home/yubohuang/zixi/CoFiTok/datasets"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p 22179"

while [[ ! -f "$state_dir/completed" ]]; do
  if [[ -f "$state_dir/failed" ]]; then
    printf 'generation archive failed; refusing legacy dataset fallback\n' >&2
    exit 1
  fi
  sleep 300
done

until rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
  -e "$ssh_command" \
  "$source_root/eurosat" \
  "$source_root/resisc45" \
  "$source_root/gtsrb" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$state_dir/legacy_dataset_fallback.log" 2>&1; do
  sleep 300
done

rsync -rLt --dry-run --itemize-changes \
  -e "$ssh_command" \
  "$source_root/eurosat" \
  "$source_root/resisc45" \
  "$source_root/gtsrb" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$state_dir/legacy_dataset_fallback_dry_run.log" 2>&1

if [[ -s "$state_dir/legacy_dataset_fallback_dry_run.log" ]]; then
  printf 'legacy dataset dry-run reported outstanding changes\n' >&2
  exit 1
fi

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > "$state_dir/legacy_dataset_fallback.completed"
