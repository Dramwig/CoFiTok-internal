#!/usr/bin/env bash
# Preserve first-party historical CoFiTok runs after the large archive settles.
set -euo pipefail

readonly state_dir="/tmp/cofitok_generation_archive_parallel_20260817"
readonly source_root="/root/autodl-tmp/CoFiTok/checkpoints"
readonly target_root="/home/yubohuang/zixi/CoFiTok/checkpoints"
readonly source_code_root="/root/autodl-tmp/CoFiTok/CoFiTok-internal"
readonly target_code_root="/home/yubohuang/zixi/CoFiTok/CoFiTok-internal"
readonly ssh_command="ssh -i /root/.ssh/cofitok_ybforever_push_20260817 -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p 22179"
readonly exclusions=(
  --exclude='generation/***'
  --exclude='baselines/***'
  --exclude='hf_cache/***'
  --exclude='torch_cache/***'
  --exclude='evaluators/***'
)

while [[ ! -f "$state_dir/completed" || ! -f "$state_dir/legacy_dataset_fallback.completed" ]]; do
  if [[ -f "$state_dir/failed" ]]; then
    printf 'generation archive failed; refusing legacy checkpoint transfer\n' >&2
    exit 1
  fi
  sleep 300
done

rsync -rLt --partial --append-verify --human-readable --info=progress2,stats2 \
  "${exclusions[@]}" \
  -e "$ssh_command" \
  "$source_root/" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$state_dir/legacy_checkpoint_runs.log" 2>&1

rsync -rLt --dry-run --itemize-changes \
  "${exclusions[@]}" \
  -e "$ssh_command" \
  "$source_root/" \
  "yubohuang@127.0.0.1:$target_root/" \
  > "$state_dir/legacy_checkpoint_runs_dry_run.log" 2>&1

if [[ -s "$state_dir/legacy_checkpoint_runs_dry_run.log" ]]; then
  printf 'legacy checkpoint dry-run reported outstanding changes\n' >&2
  exit 1
fi

for relative_path in \
  'scripts/deployment/rsync_legacy_checkpoint_runs_after_generation_2026-08-17.sh' \
  'scripts/deployment/finalize_ybforever_transfer_2026-08-17.ps1' \
  'docs/experiment_conditions/ybforever_deployment_2026-08-17.md'; do
  rsync -Lt --partial --append-verify \
    -e "$ssh_command" \
    "$source_code_root/$relative_path" \
    "yubohuang@127.0.0.1:$target_code_root/$relative_path"

  dry_run_output="$(rsync -Lt --dry-run --itemize-changes \
    -e "$ssh_command" \
    "$source_code_root/$relative_path" \
    "yubohuang@127.0.0.1:$target_code_root/$relative_path")"
  if [[ -n "$dry_run_output" ]]; then
    printf 'core provenance dry-run reported outstanding changes for %s\n' "$relative_path" >&2
    exit 1
  fi
done

printf 'completed %s\n' "$(date --iso-8601=seconds)" \
  > "$state_dir/legacy_checkpoint_runs.completed"
