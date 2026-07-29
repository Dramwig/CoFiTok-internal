#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-6688652
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0
EXPECTED_CODE_REVISION=9e7ed713cd5bc46d3722e177cae343f0c0be483b
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_k8_probe1k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_dense_probe1k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
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

"$PYTHON" - "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0/training_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("training_complete") is not True
    or report.get("completed_steps") != 1000
    or report.get("target_steps") != 1000
):
    raise SystemExit(f"incomplete CoFiTok training report: {sys.argv[1]}")
PY

test ! -e "$PAIR_ROOT/dense_identity"

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$PAIR_ROOT/dense_identity"
