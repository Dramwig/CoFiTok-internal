#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-6688652
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k
EXPECTED_CODE_REVISION=a608b5e484707ed4a9fe6890c7cfc0277add7643
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_k8_probe1k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_dense_probe1k.json

cd "$PROJECT"
git merge-base --is-ancestor "$EXPECTED_CODE_REVISION" HEAD
git diff --quiet "$EXPECTED_CODE_REVISION" HEAD -- configs scripts src tests
[[ -x "$PYTHON" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing to start while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - "$COFITOK_CONFIG" "$DENSE_CONFIG" <<'PY'
import json
import sys

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

cofitok = config_to_dict(load_config(sys.argv[1]))
dense = config_to_dict(load_config(sys.argv[2]))
contract = generation_pair_contract(cofitok, dense)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
print(json.dumps(contract, indent=2))
PY

mkdir -p "$OUTPUT_ROOT"
for output in cofitok_rgbtail3 dense_identity; do
  if [[ -e "$OUTPUT_ROOT/$output/training_report.json" ]]; then
    printf 'refusing to overwrite completed output: %s\n' "$OUTPUT_ROOT/$output" >&2
    exit 10
  fi
done

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$OUTPUT_ROOT/cofitok_rgbtail3"

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$OUTPUT_ROOT/dense_identity"
