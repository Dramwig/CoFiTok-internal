#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-6b77ef7
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2
BENCHMARK_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_2026-07-30
HOLD_DECISION=/tmp/scaling_decision5k_rollout_x0_v1.json
EXPECTED_CODE_REVISION=6b77ef7254356d551b2e392aba07df1c96ed067c
EXPECTED_HOLD_SHA256=5c6f602325e99d8209047692ea06208991d869e8b995b4a98c2f46320fb36214
EXPECTED_COFITOK_BENCHMARK_SHA256=ab6da62c303dc8722aa9ef0abf4f8f65bc8205ad6752363f08748419b6f2579f
EXPECTED_DENSE_BENCHMARK_SHA256=a6b40c3f23d44bb91553bc6419a613c6a3ecd53daf511d08e9b186407524a734
EXPECTED_BENCHMARK_SUMMARY_SHA256=9473afaaa944e70e12211ada2d4cf872f34afcadd478d5e90c8e4a0d41786e69
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_k8_probe1k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_dense_probe1k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$HOLD_DECISION" | awk '{print $1}')" == "$EXPECTED_HOLD_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/cofitok/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_COFITOK_BENCHMARK_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/dense/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_DENSE_BENCHMARK_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/benchmark_summary.json" | awk '{print $1}')" == "$EXPECTED_BENCHMARK_SUMMARY_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing to start while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - \
  "$COFITOK_CONFIG" \
  "$DENSE_CONFIG" \
  "$HOLD_DECISION" \
  "$BENCHMARK_ROOT" \
  "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

cofitok_path, dense_path, hold_path, benchmark_root, expected_revision = sys.argv[1:]
cofitok = config_to_dict(load_config(cofitok_path))
dense = config_to_dict(load_config(dense_path))
contract = generation_pair_contract(cofitok, dense)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
if cofitok["loss"]["rollout_consistency_unroll_steps"] != 2:
    raise SystemExit("CoFiTok config is not the two-step correction")
if dense["loss"]["rollout_consistency_unroll_steps"] != 2:
    raise SystemExit("dense config is not the matched two-step correction")
hold = json.loads(Path(hold_path).read_text(encoding="utf-8"))
if hold["status"] != "fail" or hold["decision"] != "hold_for_stability_correction":
    raise SystemExit("the prior 5K decision does not require stability correction")
if hold["identity"]["git_revision"] != "68ca820d5d901eff2623b97b86c4b88bf68ef500":
    raise SystemExit("unexpected prior 5K training identity")
benchmark_root = Path(benchmark_root)
summary = json.loads(
    (benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8")
)
if summary["status"] != "completed":
    raise SystemExit("two-step CUDA benchmark did not complete")
for method in ("cofitok", "dense"):
    report = json.loads(
        (benchmark_root / method / "benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )
    if report["status"] != "completed":
        raise SystemExit(f"{method} benchmark did not complete")
    if report["git"]["revision"] != expected_revision:
        raise SystemExit(f"{method} benchmark revision mismatch")
    if report["checkpoint_written"] is not False:
        raise SystemExit(f"{method} benchmark unexpectedly wrote a checkpoint")
print(json.dumps(contract, indent=2))
PY

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2"

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$OUTPUT_ROOT/dense_rollout_x0_u2"

"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_CODE_REVISION" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from cofitok.generation_pair import generation_pair_contract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(sys.argv[1])
expected_revision = sys.argv[2]
reports = {}
for method, run_name in (
    ("cofitok", "cofitok_rgbtail3_rollout_x0_u2"),
    ("dense", "dense_rollout_x0_u2"),
):
    run_dir = root / run_name
    report = json.loads(
        (run_dir / "training_report.json").read_text(encoding="utf-8")
    )
    if not report["training_complete"]:
        raise SystemExit(f"{method} training did not complete")
    if report["completed_steps"] != 1000 or report["target_steps"] != 1000:
        raise SystemExit(f"{method} did not complete the exact 1K budget")
    if report["final_metrics"]["samples_seen"] != 64000:
        raise SystemExit(f"{method} images-seen mismatch")
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    latest = report["latest_checkpoint"]
    if latest["step"] != 1000 or latest["git_revision"] != expected_revision:
        raise SystemExit(f"{method} latest checkpoint provenance mismatch")
    checkpoint = run_dir / latest["checkpoint"]
    if checkpoint.stat().st_size != latest["checkpoint_bytes"]:
        raise SystemExit(f"{method} checkpoint byte count mismatch")
    if sha256(checkpoint) != latest["checkpoint_sha256"]:
        raise SystemExit(f"{method} checkpoint SHA256 mismatch")
    if not (run_dir / latest["integrity_manifest"]).is_file():
        raise SystemExit(f"{method} checkpoint integrity sidecar is missing")
    validation_mse = report["final_metrics"]["validation_epsilon_mse"]
    if not math.isfinite(validation_mse):
        raise SystemExit(f"{method} validation MSE is non-finite")
    reports[method] = report

contract = generation_pair_contract(
    reports["cofitok"]["config"],
    reports["dense"]["config"],
)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
validation_ratio = (
    reports["cofitok"]["final_metrics"]["validation_epsilon_mse"]
    / reports["dense"]["final_metrics"]["validation_epsilon_mse"]
)
parameter_ratio_delta = abs(
    reports["cofitok"]["parameter_count"] / reports["dense"]["parameter_count"] - 1.0
)
if parameter_ratio_delta > 0.02:
    raise SystemExit("matched pair parameter delta exceeds 2%")
summary = {
    "schema_version": 1,
    "status": "completed",
    "git_revision": expected_revision,
    "completed_steps": 1000,
    "images_seen": 64000,
    "validation_ratio": validation_ratio,
    "parameter_ratio_delta": parameter_ratio_delta,
    "cofitok": {
        "training_report": str(root / "cofitok_rgbtail3_rollout_x0_u2" / "training_report.json"),
        "checkpoint_sha256": reports["cofitok"]["latest_checkpoint"]["checkpoint_sha256"],
        "validation_epsilon_mse": reports["cofitok"]["final_metrics"]["validation_epsilon_mse"],
        "elapsed_seconds": reports["cofitok"]["elapsed_seconds"],
        "peak_vram_bytes": reports["cofitok"]["peak_vram_bytes"],
    },
    "dense": {
        "training_report": str(root / "dense_rollout_x0_u2" / "training_report.json"),
        "checkpoint_sha256": reports["dense"]["latest_checkpoint"]["checkpoint_sha256"],
        "validation_epsilon_mse": reports["dense"]["final_metrics"]["validation_epsilon_mse"],
        "elapsed_seconds": reports["dense"]["elapsed_seconds"],
        "peak_vram_bytes": reports["dense"]["peak_vram_bytes"],
    },
}
(root / "pair_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
