#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
source /etc/network_turbo || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm

export PYTHONPATH=src
export HF_HOME=/root/autodl-tmp/CoFiTok/checkpoints/hf_cache
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MAX_IMAGES="${MAX_IMAGES:-256}"
TAG="${TAG:-eval${MAX_IMAGES}_2026-07-10}"

datasets=(
  "cifar10:32"
  "tiny_imagenet_200:64"
  "imagenet_1k_64x64_hf:64"
  "downsampled_imagenet_64:64"
  "ffhq_64:64"
  "afhqv2_64:64"
  "imagenet_256_10pct:256"
  "imagenet_256:256"
)

for item in "${datasets[@]}"; do
  dataset="${item%%:*}"
  source_resolution="${item##*:}"

  python scripts/baselines/evaluate_tokenizer_reconstruction.py \
    --baseline titok_1d_tokenizer \
    --dataset "${dataset}" \
    --repo-dir /root/autodl-tmp/CoFiTok/baselines/repos/titok_1d_tokenizer \
    --output-dir "artifacts/reports/baselines/titok_1d_tokenizer/${dataset}_recon256_${TAG}" \
    --split val \
    --image-size 256 \
    --source-resolution "${source_resolution}" \
    --batch-size 4 \
    --max-images "${MAX_IMAGES}" \
    --device cuda

  python scripts/baselines/evaluate_tokenizer_reconstruction.py \
    --baseline ml_flextok \
    --dataset "${dataset}" \
    --repo-dir /root/autodl-tmp/CoFiTok/baselines/repos/ml_flextok \
    --output-dir "artifacts/reports/baselines/ml_flextok/${dataset}_recon256_${TAG}" \
    --split val \
    --image-size 256 \
    --source-resolution "${source_resolution}" \
    --batch-size 1 \
    --max-images "${MAX_IMAGES}" \
    --flextok-timesteps 4 \
    --device cuda
done
