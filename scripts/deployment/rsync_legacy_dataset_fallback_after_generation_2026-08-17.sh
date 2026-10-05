#!/usr/bin/env bash
# Synchronize legacy optional datasets only after the generation archive passes.
set -euo pipefail

readonly generation_state_dir="/tmp/cofitok_generation_archive_parallel_20260817"
readonly source_root="/root/autodl-tmp/CoFiTok/datasets"
readonly target_root="/home/yubohuang/zixi/CoFiTok/datasets"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179"
readonly parent_pid="$1"

while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 300
done

test -f "$generation_state_dir/completed"

rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
  -e "$ssh_command" \
  "$source_root/eurosat" \
  "$source_root/resisc45" \
  "$source_root/gtsrb" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$generation_state_dir/legacy_dataset_fallback.log" 2>&1

rsync -rLt --dry-run --itemize-changes \
  -e "$ssh_command" \
  "$source_root/eurosat" \
  "$source_root/resisc45" \
  "$source_root/gtsrb" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$generation_state_dir/legacy_dataset_fallback_dry_run.log" 2>&1

if [[ -s "$generation_state_dir/legacy_dataset_fallback_dry_run.log" ]]; then
  printf 'legacy dataset dry-run reported outstanding changes\n' >&2
  exit 1
fi

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > "$generation_state_dir/legacy_dataset_fallback.completed"
