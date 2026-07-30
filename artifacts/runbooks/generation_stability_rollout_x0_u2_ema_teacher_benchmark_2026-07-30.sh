#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_ema_teacher_v2_2026-07-30
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
COFITOK_SOURCE=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json
DENSE_SOURCE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing to benchmark while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" - "$COFITOK_SOURCE" "$DENSE_SOURCE" "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

output_root = Path(sys.argv[3])
configs = {}
for method, source in zip(("cofitok", "dense"), sys.argv[1:3]):
    config = config_to_dict(load_config(source))
    config["name"] += "_active_benchmark"
    config["loss"]["ema_teacher_consistency_start_step"] = 0
    config["loss"]["ema_teacher_consistency_warmup_steps"] = 0
    path = output_root / f"{method}_benchmark_config.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    configs[method] = config
contract = generation_pair_contract(configs["cofitok"], configs["dense"])
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
for method, config in configs.items():
    loss = config["loss"]
    if (
        loss["ema_teacher_consistency_weight"] != 0.25
        or loss["ema_teacher_consistency_start_step"] != 0
        or loss["ema_teacher_consistency_warmup_steps"] != 0
        or loss["ema_teacher_consistency_batch_fraction"] != 0.0625
    ):
        raise SystemExit(f"{method} active EMA-teacher benchmark config is invalid")
print(json.dumps(contract, indent=2))
PY

for method in cofitok dense; do
  "$PYTHON" scripts/train_generation.py \
    --config "$OUTPUT_ROOT/${method}_benchmark_config.json" \
    --output-dir "$OUTPUT_ROOT/$method" \
    --benchmark-steps 8 \
    --benchmark-warmup-steps 2 \
    --benchmark-output "$OUTPUT_ROOT/$method/benchmark_report.json"
done

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(sys.argv[1])
expected_revision = sys.argv[2]
reports = {}
configs = {}
summary = {
    "schema_version": 1,
    "status": "completed",
    "role": "training_runtime_selection_only",
    "methods": {},
    "sources": {},
}
for method in ("cofitok", "dense"):
    report_path = root / method / "benchmark_report.json"
    config_path = root / f"{method}_benchmark_config.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = report["last_metrics"]
    checks = {
        "status": report["status"] == "completed",
        "revision": report["git"]["revision"] == expected_revision,
        "clean_git": report["git"]["dirty"] is False,
        "benchmark_steps": report["benchmark_steps"] == 8,
        "warmup_steps": report["warmup_steps"] == 2,
        "measured_steps": report["measured_steps"] == 6,
        "checkpoint_free": report["checkpoint_written"] is False,
        "finite_throughput": (
            math.isfinite(report["images_per_second"])
            and report["images_per_second"] > 0
        ),
        "memory_below_90pct": (
            report["peak_vram_bytes"]
            < 0.90 * report["device_total_memory_bytes"]
        ),
        "teacher_scale_active": metrics["ema_teacher_consistency_scale"] == 1.0,
        "teacher_loss_finite_positive": (
            math.isfinite(metrics["ema_teacher_consistency"])
            and metrics["ema_teacher_consistency"] > 0.0
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"{method} benchmark checks failed: {checks}")
    reports[method] = report
    configs[method] = config
    summary["methods"][method] = {
        "checks": checks,
        "parameter_count": report["parameter_count"],
        "mean_optimizer_step_seconds": report["mean_optimizer_step_seconds"],
        "images_per_second": report["images_per_second"],
        "peak_vram_bytes": report["peak_vram_bytes"],
        "device_total_memory_bytes": report["device_total_memory_bytes"],
        "ema_teacher_consistency": metrics["ema_teacher_consistency"],
        "ema_teacher_consistency_scale": metrics[
            "ema_teacher_consistency_scale"
        ],
        "grad_norm": metrics["grad_norm"],
    }
    summary["sources"][method] = {
        "config": {
            "path": config_path.resolve().as_posix(),
            "bytes": config_path.stat().st_size,
            "sha256": sha256(config_path),
        },
        "report": {
            "path": report_path.resolve().as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256(report_path),
        },
    }
contract = generation_pair_contract(configs["cofitok"], configs["dense"])
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
summary["generation_pair_contract"] = contract
(root / "benchmark_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
