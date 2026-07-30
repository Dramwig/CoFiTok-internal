#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
OUTPUT_ROOT="$ROOT/pair1k_rollout_x0_u2_ema_teacher"
BENCHMARK_ROOT="$ROOT/runtime_benchmark_rollout_x0_u2_ema_teacher_v2_2026-07-30"
RAW_QUALIFICATION="$ROOT/qualification5k_rollout_x0_u2_n8/qualification_report.json"
MILESTONE_DIAGNOSTIC="$ROOT/diagnostic5k_rollout_x0_u2_raw_milestones/diagnostic_report.json"
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_BENCHMARK_SUMMARY_SHA256=e5a88a1e7d18bef30de56ee43b46e94447c76302a84c0aa09d0928495fbaab49
EXPECTED_RAW_QUALIFICATION_SHA256=c4603cbcdfad7537e8a53d9cb5493cd0896eb17444a9cf04d7ef1362bf42f6a1
EXPECTED_MILESTONE_DIAGNOSTIC_SHA256=e741e84b23c090e219b10b36ba0d403d067a9a30d567ef2aa0b80d1f3dbcf9ec
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe1k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe1k.json

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/benchmark_summary.json" | awk '{print $1}')" == "$EXPECTED_BENCHMARK_SUMMARY_SHA256" ]]
[[ "$(sha256sum "$RAW_QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_RAW_QUALIFICATION_SHA256" ]]
[[ "$(sha256sum "$MILESTONE_DIAGNOSTIC" | awk '{print $1}')" == "$EXPECTED_MILESTONE_DIAGNOSTIC_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher 1K probe while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - \
  "$COFITOK_CONFIG" \
  "$DENSE_CONFIG" \
  "$BENCHMARK_ROOT/benchmark_summary.json" \
  "$RAW_QUALIFICATION" \
  "$MILESTONE_DIAGNOSTIC" \
  "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation_pair import generation_pair_contract

(
    cofitok_path,
    dense_path,
    benchmark_path,
    qualification_path,
    milestone_path,
    expected_revision,
) = sys.argv[1:]
cofitok = config_to_dict(load_config(cofitok_path))
dense = config_to_dict(load_config(dense_path))
contract = generation_pair_contract(cofitok, dense)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
for method, config in (("cofitok", cofitok), ("dense", dense)):
    loss = config["loss"]
    if (
        config["runtime"]["steps"] != 1000
        or loss["rollout_consistency_unroll_steps"] != 2
        or loss["ema_teacher_consistency_weight"] != 0.25
        or loss["ema_teacher_consistency_start_step"] != 600
        or loss["ema_teacher_consistency_warmup_steps"] != 300
        or loss["ema_teacher_consistency_batch_fraction"] != 0.0625
    ):
        raise SystemExit(f"{method} EMA-teacher 1K config is invalid")
benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
if (
    benchmark["status"] != "completed"
    or benchmark["generation_pair_contract"]["valid"] is not True
):
    raise SystemExit("active EMA-teacher benchmark did not pass")
for method in ("cofitok", "dense"):
    if not all(benchmark["methods"][method]["checks"].values()):
        raise SystemExit(f"{method} active benchmark checks did not all pass")
qualification = json.loads(Path(qualification_path).read_text(encoding="utf-8"))
failed = sorted(
    name
    for name, gate in qualification["gates"].items()
    if gate["passed"] is not True
)
if qualification["status"] != "fail" or failed != ["predicted_x0_high_frequency"]:
    raise SystemExit("source raw qualification failure identity changed")
milestone = json.loads(Path(milestone_path).read_text(encoding="utf-8"))
if (
    milestone["status"] != "completed"
    or milestone["role"] != "diagnostic_only"
    or milestone["scaling_authorization_allowed"] is not False
    or milestone["first_milestone_above_raw_high_frequency_gate"] != 5000
):
    raise SystemExit("source raw milestone diagnosis identity changed")
if expected_revision != "10f2f6bd9977fb1a63de4b2939ca641107a0ccaa":
    raise SystemExit("unexpected target revision")
print(json.dumps(contract, indent=2))
PY

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"

"$PYTHON" - \
  "$OUTPUT_ROOT" \
  "$EXPECTED_CODE_REVISION" \
  "$BENCHMARK_ROOT/benchmark_summary.json" \
  "$RAW_QUALIFICATION" \
  "$MILESTONE_DIAGNOSTIC" <<'PY'
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


def source(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


root = Path(sys.argv[1])
expected_revision = sys.argv[2]
benchmark_path = Path(sys.argv[3])
qualification_path = Path(sys.argv[4])
milestone_path = Path(sys.argv[5])
reports = {}
for method, run_name in (
    ("cofitok", "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"),
    ("dense", "dense_rollout_x0_u2_ema_teacher"),
):
    run_dir = root / run_name
    report_path = run_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 1000
        or report["target_steps"] != 1000
        or report["final_metrics"]["samples_seen"] != 64000
    ):
        raise SystemExit(f"{method} did not complete the exact matched 1K budget")
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    latest = report["latest_checkpoint"]
    if latest["step"] != 1000 or latest["git_revision"] != expected_revision:
        raise SystemExit(f"{method} latest checkpoint provenance mismatch")
    checkpoint = run_dir / latest["checkpoint"]
    if (
        checkpoint.stat().st_size != latest["checkpoint_bytes"]
        or sha256(checkpoint) != latest["checkpoint_sha256"]
    ):
        raise SystemExit(f"{method} checkpoint identity mismatch")
    integrity_path = run_dir / latest["integrity_manifest"]
    if not integrity_path.is_file():
        raise SystemExit(f"{method} checkpoint integrity sidecar is missing")
    metrics = report["final_metrics"]
    for field in (
        "validation_epsilon_mse",
        "ema_teacher_consistency",
        "ema_teacher_consistency_scale",
    ):
        if not math.isfinite(metrics[field]):
            raise SystemExit(f"{method} final {field} is non-finite")
    if metrics["ema_teacher_consistency_scale"] != 1.0:
        raise SystemExit(f"{method} final EMA-teacher scale is not active")
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
    reports["cofitok"]["parameter_count"]
    / reports["dense"]["parameter_count"]
    - 1.0
)
if parameter_ratio_delta > 0.02:
    raise SystemExit("matched pair parameter delta exceeds 2%")
summary = {
    "schema_version": 1,
    "status": "completed",
    "stage": "matched_1k_qualification_probe",
    "git_revision": expected_revision,
    "completed_steps": 1000,
    "images_seen": 64000,
    "validation_ratio": validation_ratio,
    "parameter_ratio_delta": parameter_ratio_delta,
    "ema_teacher_contract": {
        "weight": 0.25,
        "start_step": 600,
        "warmup_steps": 300,
        "batch_fraction": 0.0625,
    },
    "generation_pair_contract": contract,
    "sources": {
        "active_benchmark": source(benchmark_path),
        "failed_raw_5k_qualification": source(qualification_path),
        "raw_milestone_diagnostic": source(milestone_path),
    },
}
for method, run_name in (
    ("cofitok", "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"),
    ("dense", "dense_rollout_x0_u2_ema_teacher"),
):
    report = reports[method]
    run_dir = root / run_name
    summary[method] = {
        "training_report": source(run_dir / "training_report.json"),
        "checkpoint_sha256": report["latest_checkpoint"]["checkpoint_sha256"],
        "validation_epsilon_mse": report["final_metrics"][
            "validation_epsilon_mse"
        ],
        "ema_teacher_consistency": report["final_metrics"][
            "ema_teacher_consistency"
        ],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_vram_bytes": report["peak_vram_bytes"],
    }
(root / "pair_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
