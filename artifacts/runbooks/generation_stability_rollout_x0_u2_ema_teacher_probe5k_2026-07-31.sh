#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-5k-59db142
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
OUTPUT_ROOT="$ROOT/pair5k_rollout_x0_u2_ema_teacher"
BENCHMARK_ROOT="$ROOT/runtime_benchmark_rollout_x0_u2_ema_teacher_5k_2026-07-31"
SCALING_DECISION="$ROOT/scaling_decision1k_rollout_x0_u2_ema_teacher/scaling_decision.json"
MONITOR_REPORT="${OUTPUT_ROOT}_monitor.json"
MONITOR_LOG="${OUTPUT_ROOT}_monitor.log"
MONITOR_PID_FILE="${OUTPUT_ROOT}_monitor.pid"
EXPECTED_CODE_REVISION=59db142fc45d69dc92bb0333be5ac2d0162d9dc4
EXPECTED_1K_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_SCALING_DECISION_SHA256=7f2e6e691e26ef24f18d42e0f337229a35f2eec1a774e82f12141bb15dc48c9d
EXPECTED_COFITOK_1K_SHA256=85c61f83a333f330dde42ab1df1f3eb462788164caf05950327f590ab732695b
EXPECTED_DENSE_1K_SHA256=a0494f10d5f36cef707654b9bd5406d464ee1b894f32f302cfa7e4a9a2614aa2
EXPECTED_BENCHMARK_SUMMARY_SHA256=066a03a0bf9f6a7d40cd98de41468974d5d4e0d629a259c4a550026e6e399305
EXPECTED_COFITOK_BENCHMARK_SHA256=a26e550e4eb85a1f280f7f51004ad621d6c5b3df4ef1b7f642e2744701011faa
EXPECTED_DENSE_BENCHMARK_SHA256=eb6f4beb6e8a099b437b80ddf423c14cfec9ba363b3576ac723c172fae5fd04e
COFITOK_CONFIG=configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json
DENSE_CONFIG=configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$SCALING_DECISION" | awk '{print $1}')" == "$EXPECTED_SCALING_DECISION_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/benchmark_summary.json" | awk '{print $1}')" == "$EXPECTED_BENCHMARK_SUMMARY_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/cofitok/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_COFITOK_BENCHMARK_SHA256" ]]
[[ "$(sha256sum "$BENCHMARK_ROOT/dense/benchmark_report.json" | awk '{print $1}')" == "$EXPECTED_DENSE_BENCHMARK_SHA256" ]]
export PYTHONPATH=src

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher matched 5K while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" - \
  "$COFITOK_CONFIG" \
  "$DENSE_CONFIG" \
  "$SCALING_DECISION" \
  "$BENCHMARK_ROOT/benchmark_summary.json" \
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
    benchmark_path,
    expected_revision,
    expected_1k_revision,
    expected_cofitok_1k_sha,
    expected_dense_1k_sha,
) = sys.argv[1:]
configs = {
    "cofitok": config_to_dict(load_config(cofitok_path)),
    "dense": config_to_dict(load_config(dense_path)),
}
contract = generation_pair_contract(configs["cofitok"], configs["dense"])
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
for method, config in configs.items():
    runtime = config["runtime"]
    loss = config["loss"]
    if (
        runtime["steps"] != 5000
        or runtime["checkpoint_interval"] != 1250
        or runtime["evaluation_interval"] != 1250
        or runtime["keep_last_checkpoints"] != 3
        or runtime["protected_checkpoint_steps"] != [1250, 2500, 5000]
        or loss["rollout_consistency_weight"] != 0.1
        or loss["rollout_consistency_start_step"] != 0
        or loss["rollout_consistency_warmup_steps"] != 1000
        or loss["rollout_consistency_unroll_steps"] != 2
        or loss["rollout_consistency_batch_fraction"] != 0.125
        or loss["rollout_consistency_clip_x0"] is not True
        or loss["rollout_consistency_mode"] != "clipped_x0"
        or loss["ema_teacher_consistency_weight"] != 0.25
        or loss["ema_teacher_consistency_start_step"] != 3000
        or loss["ema_teacher_consistency_warmup_steps"] != 1000
        or loss["ema_teacher_consistency_batch_fraction"] != 0.0625
    ):
        raise SystemExit(f"{method} EMA-teacher 5K config is invalid")

decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
if (
    decision["status"] != "pass"
    or decision["decision"] != "authorize_fresh_matched_5k"
    or decision["authorized_next_stage"] != "matched_5k"
):
    raise SystemExit("robust 1K decision does not authorize fresh matched 5K")
expected_identity = {
    "git_revision": expected_1k_revision,
    "cofitok_checkpoint_sha256": expected_cofitok_1k_sha,
    "dense_checkpoint_sha256": expected_dense_1k_sha,
}
if decision["identity"] != expected_identity:
    raise SystemExit("robust 1K decision identity mismatch")

benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
if (
    benchmark["status"] != "completed"
    or benchmark["role"] != "matched_5k_runtime_rehearsal_only"
    or benchmark["scaling_authorization_allowed"] is not False
    or benchmark["generation_pair_contract"]["valid"] is not True
):
    raise SystemExit("matched 5K CUDA benchmark identity mismatch")
if benchmark["sources"]["scaling_decision"]["sha256"] != (
    "7f2e6e691e26ef24f18d42e0f337229a35f2eec1a774e82f12141bb15dc48c9d"
):
    raise SystemExit("benchmark does not bind the robust 1K decision")
for method in ("cofitok", "dense"):
    row = benchmark["methods"][method]
    if not all(row["checks"].values()):
        raise SystemExit(f"{method} benchmark checks did not all pass")
    if row["ema_teacher_consistency_scale"] != 1.0:
        raise SystemExit(f"{method} benchmark teacher is not fully active")
    report_path = Path(benchmark["sources"][method]["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} benchmark Git provenance mismatch")
print(json.dumps(contract, indent=2))
PY

test ! -e "$OUTPUT_ROOT"
test ! -e "$MONITOR_REPORT"
test ! -e "$MONITOR_LOG"
test ! -e "$MONITOR_PID_FILE"
mkdir -p "$OUTPUT_ROOT"

monitor_args=(
  --output-root "$ROOT"
  --output "$MONITOR_REPORT"
  --monitor-name generation_stability_rollout_x0_u2_ema_teacher_5k
  --cofitok-run "pair5k_rollout_x0_u2_ema_teacher/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
  --dense-run "pair5k_rollout_x0_u2_ema_teacher/dense_rollout_x0_u2_ema_teacher"
  --expected-steps 5000
  --training-process-pattern '[s]cripts/train_generation.py.*ema_teacher.*probe5k'
  --runbook-process-pattern '[g]eneration_stability_rollout_x0_u2_ema_teacher_probe5k_2026-07-31.sh'
  --checkpoint-interval 1250
  --checkpoint-grace-steps 100
  --poll-seconds 120
  --stall-seconds 1800
  --idle-failure-grace-seconds 600
)

nohup "$PYTHON" scripts/monitor_generation_pair.py "${monitor_args[@]}" \
  >"$MONITOR_LOG" 2>&1 </dev/null &
monitor_pid=$!
temporary_pid="${MONITOR_PID_FILE}.tmp.$$"
printf '%s\n' "$monitor_pid" >"$temporary_pid"
mv "$temporary_pid" "$MONITOR_PID_FILE"
sleep 2
if ! kill -0 "$monitor_pid" 2>/dev/null; then
  printf 'matched 5K monitor exited during launch\n' >&2
  exit 67
fi

"$PYTHON" scripts/train_generation.py \
  --config "$COFITOK_CONFIG" \
  --output-dir "$COFITOK_RUN"

"$PYTHON" scripts/train_generation.py \
  --config "$DENSE_CONFIG" \
  --output-dir "$DENSE_RUN"

"$PYTHON" - \
  "$OUTPUT_ROOT" \
  "$EXPECTED_CODE_REVISION" \
  "$SCALING_DECISION" \
  "$BENCHMARK_ROOT/benchmark_summary.json" <<'PY'
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
decision_path = Path(sys.argv[3])
benchmark_path = Path(sys.argv[4])
reports = {}
run_names = {
    "cofitok": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
    "dense": "dense_rollout_x0_u2_ema_teacher",
}
for method, run_name in run_names.items():
    run_dir = root / run_name
    report_path = run_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 5000
        or report["target_steps"] != 5000
        or report["final_metrics"]["samples_seen"] != 320000
    ):
        raise SystemExit(f"{method} did not complete the exact matched 5K budget")
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    latest = report["latest_checkpoint"]
    if latest["step"] != 5000 or latest["git_revision"] != expected_revision:
        raise SystemExit(f"{method} latest checkpoint provenance mismatch")
    for field in (
        "validation_epsilon_mse",
        "ema_teacher_consistency",
        "ema_teacher_consistency_scale",
    ):
        if not math.isfinite(report["final_metrics"][field]):
            raise SystemExit(f"{method} final {field} is non-finite")
    if report["final_metrics"]["ema_teacher_consistency"] <= 0.0:
        raise SystemExit(f"{method} final EMA-teacher loss is not positive")
    if report["final_metrics"]["ema_teacher_consistency_scale"] != 1.0:
        raise SystemExit(f"{method} final EMA-teacher scale is not one")

    milestones = {}
    for step in (1250, 2500, 3750, 5000):
        checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
        integrity_path = checkpoint.with_name(checkpoint.name + ".integrity.json")
        if not checkpoint.is_file() or not integrity_path.is_file():
            raise SystemExit(f"{method} milestone {step} is missing")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        checkpoint_sha = sha256(checkpoint)
        if (
            integrity["checkpoint"] != checkpoint.name
            or integrity["checkpoint_bytes"] != checkpoint.stat().st_size
            or integrity["checkpoint_sha256"] != checkpoint_sha
            or integrity["step"] != step
            or integrity["git_revision"] != expected_revision
            or integrity["git_dirty"] is not False
        ):
            raise SystemExit(f"{method} milestone {step} integrity mismatch")
        milestones[str(step)] = {
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "integrity_manifest": source(integrity_path),
        }
    if latest["checkpoint_sha256"] != milestones["5000"]["checkpoint_sha256"]:
        raise SystemExit(f"{method} latest checkpoint identity mismatch")
    reports[method] = {
        "report": report,
        "report_source": source(report_path),
        "milestones": milestones,
    }

contract = generation_pair_contract(
    reports["cofitok"]["report"]["config"],
    reports["dense"]["report"]["config"],
)
if not contract["valid"]:
    raise SystemExit(json.dumps(contract, indent=2))
cofitok_report = reports["cofitok"]["report"]
dense_report = reports["dense"]["report"]
validation_ratio = (
    cofitok_report["final_metrics"]["validation_epsilon_mse"]
    / dense_report["final_metrics"]["validation_epsilon_mse"]
)
parameter_ratio_delta = abs(
    cofitok_report["parameter_count"] / dense_report["parameter_count"] - 1.0
)
if parameter_ratio_delta > 0.02:
    raise SystemExit("matched pair parameter delta exceeds 2%")

summary = {
    "schema_version": 2,
    "status": "completed",
    "stage": "matched_5k_stability_gate",
    "formal_scaling_authorization_allowed": False,
    "git_revision": expected_revision,
    "completed_steps": 5000,
    "images_seen": 320000,
    "validation_ratio": validation_ratio,
    "parameter_ratio_delta": parameter_ratio_delta,
    "ema_teacher_contract": {
        "weight": 0.25,
        "start_step": 3000,
        "warmup_steps": 1000,
        "batch_fraction": 0.0625,
    },
    "generation_pair_contract": contract,
    "sources": {
        "scaling_decision": source(decision_path),
        "active_5k_benchmark": source(benchmark_path),
    },
}
for method in ("cofitok", "dense"):
    report = reports[method]["report"]
    summary[method] = {
        "training_report": reports[method]["report_source"],
        "checkpoint_sha256": report["latest_checkpoint"]["checkpoint_sha256"],
        "validation_epsilon_mse": report["final_metrics"][
            "validation_epsilon_mse"
        ],
        "ema_teacher_consistency": report["final_metrics"][
            "ema_teacher_consistency"
        ],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_vram_bytes": report["peak_vram_bytes"],
        "parameter_count": report["parameter_count"],
        "milestones": reports[method]["milestones"],
    }
(root / "pair_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

"$PYTHON" scripts/monitor_generation_pair.py "${monitor_args[@]}" --once
"$PYTHON" - "$MONITOR_REPORT" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report["status"] != "pass"
    or report["stage"] != "complete"
    or report["git"]["revision"] != sys.argv[2]
    or report["git"]["tracked_dirty"]
    or report["issues"]
):
    raise SystemExit(json.dumps(report, indent=2))
PY
