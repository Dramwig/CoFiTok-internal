#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/CoFiTok/CoFiTok-internal

source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/CoFiTok/checkpoints/hf_cache

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python \
  scripts/prepare_hf_image_dataset.py \
  --dataset-id benjamin-paine/imagenet-1k-64x64 \
  --config default \
  --splits train validation \
  --output-root /root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted \
  --format png \
  --shard-size 10000
