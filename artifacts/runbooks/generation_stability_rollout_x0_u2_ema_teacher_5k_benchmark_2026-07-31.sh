#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-5k-59db142
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
OUTPUT_ROOT="$ROOT/runtime_benchmark_rollout_x0_u2_ema_teacher_5k_2026-07-31"
SCALING_DECISION="$ROOT/scaling_decision1k_rollout_x0_u2_ema_teacher/scaling_decision.json"
EXPECTED_CODE_REVISION=59db142fc45d69dc92bb0333be5ac2d0162d9dc4
EXPECTED_SCALING_DECISION_SHA256=7f2e6e691e26ef24f18d42e0f337229a35f2eec1a774e82f12141bb15dc48c9d
EXPECTED_1K_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_COFITOK_1K_SHA256=85c61f83a333f330dde42ab1df1f3eb462788164caf05950327f590ab732695b
EXPECTED_DENSE_1K_SHA256=a0494f10d5f36cef707654b9bd5406d464ee1b894f32f302cfa7e4a9a2614aa2
COFITOK_SOURCE=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json
DENSE_SOURCE=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$SCALING_DECISION" | awk '{print $1}')" == "$EXPECTED_SCALING_DECISION_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher 5K benchmark while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" - \
  "$COFITOK_SOURCE" \
  "$DENSE_SOURCE" \
  "$SCALING_DECISION" \
  "$OUTPUT_ROOT" \
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
    output_root,
    expected_1k_revision,
    expected_cofitok_sha,
    expected_dense_sha,
) = sys.argv[1:]
configs = {}
for method, source in zip(("cofitok", "dense"), (cofitok_path, dense_path)):
    config = config_to_dict(load_config(source))
    loss = config["loss"]
    if (
        config["runtime"]["steps"] != 5000
        or config["runtime"]["checkpoint_interval"] != 1250
        or config["runtime"]["evaluation_interval"] != 1250
        or loss["rollout_consistency_unroll_steps"] != 2
        or loss["ema_teacher_consistency_weight"] != 0.25
        or loss["ema_teacher_consistency_start_step"] != 3000
        or loss["ema_teacher_consistency_warmup_steps"] != 1000
        or loss["ema_teacher_consistency_batch_fraction"] != 0.0625
    ):
        raise SystemExit(f"{method} EMA-teacher 5K source config is invalid")
    configs[method] = config

contract = generation_pair_contract(configs["cofitok"], configs["dense"])
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))

decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
expected_identity = {
    "git_revision": expected_1k_revision,
    "cofitok_checkpoint_sha256": expected_cofitok_sha,
    "dense_checkpoint_sha256": expected_dense_sha,
}
if (
    decision["status"] != "pass"
    or decision["decision"] != "authorize_fresh_matched_5k"
    or decision["authorized_next_stage"] != "matched_5k"
    or decision["identity"] != expected_identity
    or decision["issues"]
):
    raise SystemExit("robust 1K decision does not authorize this matched 5K")

output = Path(output_root)
benchmark_configs = {}
for method, config in configs.items():
    benchmark = json.loads(json.dumps(config))
    benchmark["name"] += "_active_benchmark"
    benchmark["loss"]["ema_teacher_consistency_start_step"] = 0
    benchmark["loss"]["ema_teacher_consistency_warmup_steps"] = 0
    path = output / f"{method}_benchmark_config.json"
    path.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benchmark_configs[method] = benchmark
benchmark_contract = generation_pair_contract(
    benchmark_configs["cofitok"],
    benchmark_configs["dense"],
)
if not benchmark_contract["valid"]:
    raise SystemExit(json.dumps(benchmark_contract, indent=2))
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

"$PYTHON" - \
  "$OUTPUT_ROOT" \
  "$SCALING_DECISION" \
  "$EXPECTED_CODE_REVISION" <<'PY'
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


def source(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


root = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
expected_revision = sys.argv[3]
reports = {}
configs = {}
summary = {
    "schema_version": 1,
    "status": "completed",
    "role": "matched_5k_runtime_rehearsal_only",
    "scaling_authorization_allowed": False,
    "methods": {},
    "sources": {
        "scaling_decision": source(decision_path),
    },
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
        "teacher_scale_active": (
            metrics["ema_teacher_consistency_scale"] == 1.0
        ),
        "teacher_loss_finite_positive": (
            math.isfinite(metrics["ema_teacher_consistency"])
            and metrics["ema_teacher_consistency"] > 0.0
        ),
        "gradient_finite_positive": (
            math.isfinite(metrics["grad_norm"])
            and metrics["grad_norm"] > 0.0
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"{method} benchmark checks failed: {checks}")
    reports[method] = report
    configs[method] = config
    summary["methods"][method] = {
        "checks": checks,
        "parameter_count": report["parameter_count"],
        "mean_optimizer_step_seconds": report[
            "mean_optimizer_step_seconds"
        ],
        "images_per_second": report["images_per_second"],
        "peak_vram_bytes": report["peak_vram_bytes"],
        "device_total_memory_bytes": report["device_total_memory_bytes"],
        "ema_teacher_consistency": metrics[
            "ema_teacher_consistency"
        ],
        "ema_teacher_consistency_scale": metrics[
            "ema_teacher_consistency_scale"
        ],
        "grad_norm": metrics["grad_norm"],
    }
    summary["sources"][method] = {
        "config": source(config_path),
        "report": source(report_path),
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
