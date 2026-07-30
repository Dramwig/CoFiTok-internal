#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_5k_2026-07-30
SCALING_DECISION=/tmp/scaling_decision1k_rollout_x0_u2_to_5k.json
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_SCALING_DECISION_SHA256=d47c2518e3c7fff18e8c4a9d6a2c605e1c875b9100a4641d9b8250285daac53b
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_k8_probe5k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_dense_probe5k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$SCALING_DECISION" | awk '{print $1}')" == "$EXPECTED_SCALING_DECISION_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing to benchmark while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - "$COFITOK_CONFIG" "$DENSE_CONFIG" "$SCALING_DECISION" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

cofitok = config_to_dict(load_config(sys.argv[1]))
dense = config_to_dict(load_config(sys.argv[2]))
contract = generation_pair_contract(cofitok, dense)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
decision = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if decision["status"] != "pass":
    raise SystemExit("1K stability scaling decision did not pass")
if decision["decision"] != "authorize_fresh_matched_5k":
    raise SystemExit("1K decision does not authorize matched 5K")
if decision["authorized_next_stage"] != "matched_5k":
    raise SystemExit("1K decision stage mismatch")
print(json.dumps(contract, indent=2))
PY

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

for method in cofitok dense; do
  if [[ "$method" == cofitok ]]; then
    config="$COFITOK_CONFIG"
  else
    config="$DENSE_CONFIG"
  fi
  "$PYTHON" scripts/train_generation.py \
    --config "$config" \
    --output-dir "$OUTPUT_ROOT/$method" \
    --benchmark-steps 8 \
    --benchmark-warmup-steps 2 \
    --benchmark-output "$OUTPUT_ROOT/$method/benchmark_report.json"
done

"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_revision = sys.argv[2]
summary = {"schema_version": 1, "status": "completed", "methods": {}}
for method in ("cofitok", "dense"):
    path = root / method / "benchmark_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status": report["status"] == "completed",
        "revision": report["git"]["revision"] == expected_revision,
        "benchmark_steps": report["benchmark_steps"] == 8,
        "warmup_steps": report["warmup_steps"] == 2,
        "measured_steps": report["measured_steps"] == 6,
        "checkpoint_free": report["checkpoint_written"] is False,
        "finite_throughput": math.isfinite(report["images_per_second"])
        and report["images_per_second"] > 0,
        "memory_below_90pct": report["peak_vram_bytes"]
        < 0.90 * report["device_total_memory_bytes"],
    }
    if not all(checks.values()):
        raise SystemExit(f"{method} benchmark checks failed: {checks}")
    summary["methods"][method] = {
        "checks": checks,
        "parameter_count": report["parameter_count"],
        "mean_optimizer_step_seconds": report["mean_optimizer_step_seconds"],
        "images_per_second": report["images_per_second"],
        "peak_vram_bytes": report["peak_vram_bytes"],
        "device_total_memory_bytes": report["device_total_memory_bytes"],
    }
(root / "benchmark_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
