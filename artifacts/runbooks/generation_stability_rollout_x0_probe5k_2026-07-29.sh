#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-scale-68ca820
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0
QUALIFICATION=/tmp/cofitok_stability_qualification1k_rollout_x0.json
EXPECTED_QUALIFICATION_SHA256=225839d69a239483852b40413fce17e0ff56467c6be3bf3d2c5a4b26228715ef
EXPECTED_CODE_REVISION=68ca820d5d901eff2623b97b86c4b88bf68ef500
EXPECTED_1K_REVISION=9e7ed713cd5bc46d3722e177cae343f0c0be483b
EXPECTED_COFITOK_1K_SHA256=6867beb7f698e5ad5eea790fef8e4c8d14ace9879bcaccefc30d8a2f0d046d63
EXPECTED_DENSE_1K_SHA256=f913d3e740b03b75790294cabc950dda70e5dc270fcfde148d30919a24e9290e
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_k8_probe5k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_dense_probe5k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --short)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_QUALIFICATION_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing rollout-x0 5K qualification while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - "$QUALIFICATION" <<PY
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("status") != "pass":
    raise SystemExit("the matched 1K qualification did not pass")
identity = report.get("identity", {})
expected = {
    "git_revision": "$EXPECTED_1K_REVISION",
    "cofitok_checkpoint_sha256": "$EXPECTED_COFITOK_1K_SHA256",
    "dense_checkpoint_sha256": "$EXPECTED_DENSE_1K_SHA256",
}
if identity != expected:
    raise SystemExit(f"unexpected 1K qualification identity: {identity!r}")
PY

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

test ! -e "$PAIR_ROOT"
mkdir -p "$PAIR_ROOT"

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0"

"$PYTHON" - "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0/training_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("training_complete") is not True
    or report.get("completed_steps") != 5000
    or report.get("target_steps") != 5000
):
    raise SystemExit(f"incomplete CoFiTok training report: {sys.argv[1]}")
PY

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$PAIR_ROOT/dense_identity"
