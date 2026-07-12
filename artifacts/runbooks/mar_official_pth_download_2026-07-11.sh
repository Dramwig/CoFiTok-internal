#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CoFiTok
ASSET_DIR="$ROOT/checkpoints/baselines/mar/official_imagenet256_eval_only/official_pth"
STATUS="$ROOT/CoFiTok-internal/artifacts/runbooks/mar_official_pth_download_2026-07-11.status"
LOG="$ASSET_DIR/download.log"
REVISION=773c9bdcd98740c7b876ce0d66f684a0e2468990
MODEL_SHA=7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c
VAE_SHA=34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f

mkdir -p "$ASSET_DIR"
printf 'running\n' > "$STATUS"

download_and_verify() {
  local filename=$1
  local expected_sha=$2
  local destination="$ASSET_DIR/$filename"
  local partial="$destination.part"
  local url="https://huggingface.co/jadechoghari/mar/resolve/$REVISION/$filename"

  if [[ -f "$destination" ]] && printf '%s  %s\n' "$expected_sha" "$destination" | sha256sum --check --status; then
    printf '[verified-existing] %s\n' "$destination"
    return
  fi

  curl --fail --location --retry 8 --retry-all-errors --continue-at - \
    --output "$partial" "$url"
  printf '%s  %s\n' "$expected_sha" "$partial" | sha256sum --check
  mv "$partial" "$destination"
  printf '[downloaded] %s\n' "$destination"
}

{
  source /etc/network_turbo
  download_and_verify checkpoint-last.pth "$MODEL_SHA"
  download_and_verify kl16.ckpt "$VAE_SHA"
  sha256sum "$ASSET_DIR/checkpoint-last.pth" "$ASSET_DIR/kl16.ckpt"
} 2>&1 | tee -a "$LOG"

printf 'completed\n' > "$STATUS"
