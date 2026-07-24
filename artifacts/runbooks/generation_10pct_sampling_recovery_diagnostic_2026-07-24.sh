#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
COFITOK_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2"
DENSE_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_dense_50k_v2"
DIAGNOSTIC_ROOT="$OUTPUT_ROOT/diagnostics/imagenet256_10pct_rankcomplete_v2_sampling_recovery_2026-07-24"
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/sampling_recovery_diagnostic"
STATUS="$OUTPUT_ROOT/generation_10pct_rankcomplete_v2_sampling_recovery_diagnostic.status.json"
EXPECTED_REVISION=a8f43f4b8695ce1ac92ca9bd774309c62947396e
SAMPLE_COUNT=128
SAMPLE_STEPS=100
BATCH_SIZE=64
CURRENT_STAGE=initializing

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

mkdir -p "$DIAGNOSTIC_ROOT" "$REPORT_ROOT/source_reports" "$REPORT_ROOT/previews"
exec 9>"$OUTPUT_ROOT/generation_10pct_rankcomplete_v2_sampling_recovery_diagnostic.lock"
if ! flock -n 9; then
  printf 'sampling recovery diagnostic is already running\n' >&2
  exit 75
fi

write_status() {
  local status_value="$1"
  local stage_value="$2"
  local detail_value="$3"
  local exit_code_value="${4:-}"
  STATUS_VALUE="$status_value" \
  STAGE_VALUE="$stage_value" \
  DETAIL_VALUE="$detail_value" \
  EXIT_CODE_VALUE="$exit_code_value" \
  python - "$STATUS" <<'PY'
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
exit_code = os.environ["EXIT_CODE_VALUE"]
payload = {
    "schema_version": 1,
    "diagnostic": "imagenet256_10pct_rankcomplete_v2_sampling_recovery",
    "status": os.environ["STATUS_VALUE"],
    "stage": os.environ["STAGE_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "exit_code": int(exit_code) if exit_code else None,
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "hostname": socket.gethostname(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -ne 0 ]]; then
    write_status failed "$CURRENT_STAGE" "diagnostic stopped before completion" "$exit_code"
  fi
}
trap on_exit EXIT

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_REVISION" ]]; then
  printf 'diagnostic requires revision %s\n' "$EXPECTED_REVISION" >&2
  exit 66
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'diagnostic requires a clean tracked worktree\n' >&2
  exit 66
fi

COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00050000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00050000.pt"
test -f "$COFITOK_CHECKPOINT"
test -f "$DENSE_CHECKPOINT"
test -f "$COFITOK_CHECKPOINT.integrity.json"
test -f "$DENSE_CHECKPOINT.integrity.json"

write_status running "$CURRENT_STAGE" "validated revision and checkpoint inputs"

CURRENT_STAGE=raw_model_endpoint
write_status running "$CURRENT_STAGE" "comparing raw model endpoint against formal EMA endpoint"
if [[ ! -f "$DIAGNOSTIC_ROOT/cofitok_checkpoint_eval_model_t500_1024/checkpoint_evaluation_report.json" ]]; then
  python scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$DIAGNOSTIC_ROOT/cofitok_checkpoint_eval_model_t500_1024" \
    --num-images 1024 --timestep 500 --random-orders 16 \
    --weights model --precision bf16
fi
if [[ ! -f "$DIAGNOSTIC_ROOT/dense_checkpoint_eval_model_t500_1024/checkpoint_evaluation_report.json" ]]; then
  python scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$DIAGNOSTIC_ROOT/dense_checkpoint_eval_model_t500_1024" \
    --num-images 1024 --timestep 500 --random-orders 0 \
    --weights model --precision bf16
fi
cp \
  "$DIAGNOSTIC_ROOT/cofitok_checkpoint_eval_model_t500_1024/checkpoint_evaluation_report.json" \
  "$REPORT_ROOT/source_reports/cofitok_checkpoint_eval_model_t500_1024.json"
cp \
  "$DIAGNOSTIC_ROOT/dense_checkpoint_eval_model_t500_1024/checkpoint_evaluation_report.json" \
  "$REPORT_ROOT/source_reports/dense_checkpoint_eval_model_t500_1024.json"

CASES=(
  "cofitok ema 0.75 0.0"
  "cofitok ema 1.0 0.0"
  "cofitok ema 1.25 0.0"
  "cofitok ema 1.5 0.0"
  "cofitok ema 1.5 0.5"
  "cofitok ema 1.5 1.0"
  "dense ema 0.75 0.0"
  "dense ema 1.0 0.0"
  "dense ema 1.25 0.0"
  "dense ema 1.5 0.0"
  "dense ema 1.5 0.5"
  "dense ema 1.5 1.0"
)

CURRENT_STAGE=sampling_sweep
for case_spec in "${CASES[@]}"; do
  read -r method weights guidance_scale guidance_rescale <<<"$case_spec"
  if [[ "$method" == cofitok ]]; then
    checkpoint="$COFITOK_CHECKPOINT"
    budget=8
  else
    checkpoint="$DENSE_CHECKPOINT"
    budget=1
  fi
  scale_slug="${guidance_scale/./}"
  rescale_slug="${guidance_rescale/./}"
  case_name="${method}_${weights}_cfg${scale_slug}_rescale${rescale_slug}"
  case_root="$DIAGNOSTIC_ROOT/$case_name"
  write_status running "$CURRENT_STAGE" "running $case_name"
  if [[ ! -f "$case_root/sampling_report.json" ]]; then
    python scripts/generate_samples.py \
      --checkpoint "$checkpoint" \
      --output-dir "$case_root" \
      --num-samples "$SAMPLE_COUNT" --batch-size "$BATCH_SIZE" \
      --sample-steps "$SAMPLE_STEPS" --prefix-budgets "$budget" \
      --guidance-scale "$guidance_scale" \
      --guidance-rescale "$guidance_rescale" \
      --cfg-batch-mode batched --weights "$weights" --precision bf16 --resume
  fi
  if [[ ! -f "$case_root/metrics/generation_metrics_report.json" ]]; then
    python scripts/evaluate_generation_metrics.py \
      --real-dir "$DATA" \
      --generated-dir "$case_root/prefix_$budget" \
      --sampling-report "$case_root/sampling_report.json" \
      --output-dir "$case_root/metrics" \
      --cache-root "$OUTPUT_ROOT/eval_cache/torch_fidelity" \
      --min-samples "$SAMPLE_COUNT" --skip-prc
  fi
  cp "$case_root/sampling_report.json" \
    "$REPORT_ROOT/source_reports/${case_name}_sampling_report.json"
  cp "$case_root/metrics/generation_metrics_report.json" \
    "$REPORT_ROOT/source_reports/${case_name}_generation_metrics_report.json"
  python - "$case_root/prefix_$budget" "$REPORT_ROOT/previews/${case_name}.png" <<'PY'
import sys
from pathlib import Path
from PIL import Image

source = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = sorted(source.glob("*.png"))[:16]
if len(paths) != 16:
    raise SystemExit(f"expected 16 preview images in {source}, found {len(paths)}")
tile_size = 128
canvas = Image.new("RGB", (tile_size * 4, tile_size * 4))
for index, path in enumerate(paths):
    with Image.open(path) as image:
        tile = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas.paste(tile, ((index % 4) * tile_size, (index // 4) * tile_size))
output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(output)
PY
done

CURRENT_STAGE=summarizing
write_status running "$CURRENT_STAGE" "building bounded diagnostic summary"
python - \
  "$REPORT_ROOT" \
  "$COFITOK_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  "$DENSE_RUN/checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json" \
  "$COFITOK_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json" \
  "$DENSE_RUN/samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json" <<'PY'
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

report_root = Path(sys.argv[1])
cofitok_ema_eval = Path(sys.argv[2])
dense_ema_eval = Path(sys.argv[3])
cofitok_formal_metrics = Path(sys.argv[4])
dense_formal_metrics = Path(sys.argv[5])

def load(path):
    return json.loads(path.read_text())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def endpoint(path):
    payload = load(path)
    return float(payload["metrics"]["orders"]["ordered"]["endpoint_clean_mse"])

rows = []
for metrics_path in sorted(
    (report_root / "source_reports").glob("*_generation_metrics_report.json")
):
    case_name = metrics_path.name.removesuffix("_generation_metrics_report.json")
    metrics = load(metrics_path)
    sampling_path = (
        report_root / "source_reports" / f"{case_name}_sampling_report.json"
    )
    sampling = load(sampling_path)
    metric_values = metrics["metrics"]
    rows.append(
        {
            "case": case_name,
            "method": case_name.split("_", 1)[0],
            "weights": sampling["weights"],
            "guidance_scale": float(sampling["sampling"]["guidance_scale"]),
            "guidance_rescale": float(sampling["sampling"]["guidance_rescale"]),
            "sample_count": int(sampling["sampling"]["num_samples"]),
            "sample_steps": int(sampling["sampling"]["sample_steps"]),
            "fid": float(metric_values["frechet_inception_distance"]),
            "inception_score_mean": float(metric_values["inception_score_mean"]),
            "inception_score_std": float(metric_values["inception_score_std"]),
            "sampling_elapsed_seconds": float(sampling["elapsed_seconds"]),
            "metrics_report": metrics_path.relative_to(report_root).as_posix(),
            "metrics_report_sha256": sha256(metrics_path),
            "sampling_report": sampling_path.relative_to(report_root).as_posix(),
            "sampling_report_sha256": sha256(sampling_path),
            "preview": f"previews/{case_name}.png",
        }
    )

if len(rows) != 12:
    raise SystemExit(f"expected 12 completed sweep cases, found {len(rows)}")
for row in rows:
    if (
        row["sample_count"] != 128
        or row["sample_steps"] != 100
        or not math.isfinite(row["fid"])
        or not math.isfinite(row["inception_score_mean"])
    ):
        raise SystemExit(f"invalid diagnostic row: {row['case']}")

rankings = {
    method: [
        row["case"]
        for row in sorted(
            (candidate for candidate in rows if candidate["method"] == method),
            key=lambda candidate: candidate["fid"],
        )
    ]
    for method in ("cofitok", "dense")
}
raw_paths = {
    "cofitok": report_root
    / "source_reports/cofitok_checkpoint_eval_model_t500_1024.json",
    "dense": report_root
    / "source_reports/dense_checkpoint_eval_model_t500_1024.json",
}
ema_paths = {
    "cofitok": cofitok_ema_eval,
    "dense": dense_ema_eval,
}
weight_comparison = {}
for method in ("cofitok", "dense"):
    raw_value = endpoint(raw_paths[method])
    ema_value = endpoint(ema_paths[method])
    weight_comparison[method] = {
        "raw_model_endpoint_mse": raw_value,
        "ema_endpoint_mse": ema_value,
        "raw_relative_to_ema": raw_value / ema_value - 1.0,
        "raw_report": raw_paths[method].relative_to(report_root).as_posix(),
        "raw_report_sha256": sha256(raw_paths[method]),
        "ema_report": ema_paths[method].as_posix(),
        "ema_report_sha256": sha256(ema_paths[method]),
    }

formal_reference = {}
for method, path in (
    ("cofitok", cofitok_formal_metrics),
    ("dense", dense_formal_metrics),
):
    payload = load(path)
    formal_reference[method] = {
        "sample_count": int(payload["counts"]["generated_image_count"]),
        "fid": float(payload["metrics"]["frechet_inception_distance"]),
        "inception_score_mean": float(payload["metrics"]["inception_score_mean"]),
        "precision": float(payload["metrics"]["precision"]),
        "recall": float(payload["metrics"]["recall"]),
        "report": path.as_posix(),
        "report_sha256": sha256(path),
    }

summary = {
    "schema_version": 1,
    "status": "complete",
    "role": "non_promotion_sampling_recovery_diagnostic",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "limitations": [
        "The 128-sample FID and Inception scores are noisy configuration-selection diagnostics.",
        "No row in this report is valid promotion or formal comparison evidence.",
        "Any selected protocol must be confirmed with a fresh matched 10,000-sample evaluation.",
    ],
    "sweep": {
        "sample_count_per_case": 128,
        "sample_steps": 100,
        "shared_seed": 0,
        "class_schedule": "balanced_modulo",
        "rows": rows,
        "rankings_by_diagnostic_fid": rankings,
    },
    "raw_model_vs_ema_endpoint": weight_comparison,
    "formal_cfg15_reference": formal_reference,
}
summary_path = report_root / "summary.json"
temporary = summary_path.with_name(f".{summary_path.name}.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(summary_path)
print(summary_path)
PY

CURRENT_STAGE=complete
write_status pass "$CURRENT_STAGE" "128-sample matched guidance sweep and raw-model endpoint comparison completed"
trap - EXIT
