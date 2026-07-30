#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2
BENCHMARK_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_5k_2026-07-30
SCALING_DECISION=/tmp/scaling_decision1k_rollout_x0_u2_to_5k.json
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_1K_REVISION=6b77ef7254356d551b2e392aba07df1c96ed067c
EXPECTED_SCALING_DECISION_SHA256=d47c2518e3c7fff18e8c4a9d6a2c605e1c875b9100a4641d9b8250285daac53b
EXPECTED_COFITOK_1K_SHA256=9885273a3bd3773274d140db433af047c7cefeb87e460b9868406b45b7e1bb05
EXPECTED_DENSE_1K_SHA256=e27a03434ec3bef7f7528d6d7a0d61867845c5f95165cd1ccb51a07adafe0046
EXPECTED_COFITOK_BENCHMARK_SHA256=05887d3920fc3e7851d38e32cf1ece4a7fe45ac49b541a980904a9b5e569a473
EXPECTED_DENSE_BENCHMARK_SHA256=e5a861372a9c5f9b295deae77909a0c6f49f3b02d5b0c4a8d6ce169eb3913054
EXPECTED_BENCHMARK_SUMMARY_SHA256=c8bf26c0c75cba386373ca9adbd17601aa679581eed9ffa87be6c31059e3f5f7
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_k8_probe5k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_dense_probe5k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$SCALING_DECISION" | awk '{print $1}')" == "$EXPECTED_SCALING_DECISION_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/cofitok/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_COFITOK_BENCHMARK_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/dense/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_DENSE_BENCHMARK_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/benchmark_summary.json" | awk '{print $1}')" == "$EXPECTED_BENCHMARK_SUMMARY_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing two-step matched 5K while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - \
  "$COFITOK_CONFIG" \
  "$DENSE_CONFIG" \
  "$SCALING_DECISION" \
  "$BENCHMARK_ROOT" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_1K_REVISION" \
  "$EXPECTED_COFITOK_1K_SHA256" \
  "$EXPECTED_DENSE_1K_SHA256" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

(
    cofitok_path,
    dense_path,
    decision_path,
    benchmark_root,
    expected_revision,
    expected_1k_revision,
    expected_cofitok_1k_sha,
    expected_dense_1k_sha,
) = sys.argv[1:]
cofitok = config_to_dict(load_config(cofitok_path))
dense = config_to_dict(load_config(dense_path))
contract = generation_pair_contract(cofitok, dense)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
if cofitok["runtime"]["steps"] != 5000 or dense["runtime"]["steps"] != 5000:
    raise SystemExit("matched pair does not use the exact 5K budget")
for method, config in (("cofitok", cofitok), ("dense", dense)):
    if config["loss"]["rollout_consistency_unroll_steps"] != 2:
        raise SystemExit(f"{method} is not the two-step correction")
    if config["loss"]["rollout_consistency_batch_fraction"] != 0.125:
        raise SystemExit(f"{method} rollout subset mismatch")
decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
if decision["status"] != "pass":
    raise SystemExit("1K stability decision did not pass")
if decision["decision"] != "authorize_fresh_matched_5k":
    raise SystemExit("1K stability decision does not authorize matched 5K")
if decision["authorized_next_stage"] != "matched_5k":
    raise SystemExit("1K stability decision stage mismatch")
expected_identity = {
    "git_revision": expected_1k_revision,
    "cofitok_checkpoint_sha256": expected_cofitok_1k_sha,
    "dense_checkpoint_sha256": expected_dense_1k_sha,
}
if decision["identity"] != expected_identity:
    raise SystemExit("1K stability decision identity mismatch")
benchmark_root = Path(benchmark_root)
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

test ! -e "$PAIR_ROOT"
mkdir -p "$PAIR_ROOT"

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2"

"$PYTHON" - "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2/training_report.json" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["training_complete"]:
    raise SystemExit("CoFiTok 5K training is incomplete")
if report["completed_steps"] != 5000 or report["target_steps"] != 5000:
    raise SystemExit("CoFiTok exact 5K budget mismatch")
if report["final_metrics"]["samples_seen"] != 320000:
    raise SystemExit("CoFiTok images-seen mismatch")
if report["git"]["revision"] != sys.argv[2] or report["git"]["dirty"]:
    raise SystemExit("CoFiTok Git provenance mismatch")
PY

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$PAIR_ROOT/dense_rollout_x0_u2"

"$PYTHON" - "$PAIR_ROOT" "$EXPECTED_CODE_REVISION" <<'PY'
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
    if report["completed_steps"] != 5000 or report["target_steps"] != 5000:
        raise SystemExit(f"{method} exact 5K budget mismatch")
    if report["final_metrics"]["samples_seen"] != 320000:
        raise SystemExit(f"{method} images-seen mismatch")
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    latest = report["latest_checkpoint"]
    if latest["step"] != 5000 or latest["git_revision"] != expected_revision:
        raise SystemExit(f"{method} latest checkpoint provenance mismatch")
    checkpoint = run_dir / latest["checkpoint"]
    if checkpoint.stat().st_size != latest["checkpoint_bytes"]:
        raise SystemExit(f"{method} final checkpoint byte count mismatch")
    if sha256(checkpoint) != latest["checkpoint_sha256"]:
        raise SystemExit(f"{method} final checkpoint SHA256 mismatch")
    for step in (1250, 2500, 3750, 5000):
        milestone = run_dir / f"checkpoint_step_{step:08d}.pt"
        sidecar = run_dir / f"checkpoint_step_{step:08d}.pt.integrity.json"
        if not milestone.is_file() or not sidecar.is_file():
            raise SystemExit(f"{method} milestone {step} is missing")
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
    "completed_steps": 5000,
    "images_seen": 320000,
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
