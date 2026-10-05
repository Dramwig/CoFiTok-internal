#!/usr/bin/env bash
# Preserve the small provenance files for optional datasets rebuilt from public hosts.
set -euo pipefail

readonly source_root="/root/autodl-tmp/CoFiTok/datasets"
readonly target_root="/home/yubohuang/zixi/CoFiTok/datasets"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179"
readonly aliases=(
  cifar100
  flowers102
  pets
  dtd
  caltech101
  cub_200_2011
  svhn
  stanford_dogs
  fgvc_aircraft
)

for alias in "${aliases[@]}"; do
  rsync -rLt -e "$ssh_command" \
    "$source_root/$alias/metadata/" \
    "yubohuang@127.0.0.1:$target_root/$alias/metadata/"
  rsync -Lt -e "$ssh_command" \
    "$source_root/$alias/raw/checksums.sha256" \
    "yubohuang@127.0.0.1:$target_root/$alias/raw/checksums.sha256"
done

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > /tmp/cofitok_generation_archive_parallel_20260817/optional_metadata.completed
