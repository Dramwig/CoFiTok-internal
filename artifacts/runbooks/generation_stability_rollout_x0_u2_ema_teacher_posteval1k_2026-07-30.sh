#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair1k_rollout_x0_u2_ema_teacher"
OUTPUT_ROOT="$ROOT/evaluation1k_rollout_x0_u2_ema_teacher"
QUALIFICATION_ROOT="$ROOT/qualification1k_rollout_x0_u2_ema_teacher"
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_PAIR_SUMMARY_SHA256=${EXPECTED_PAIR_SUMMARY_SHA256:?}
EXPECTED_COFITOK_CHECKPOINT_SHA256=${EXPECTED_COFITOK_CHECKPOINT_SHA256:?}
EXPECTED_DENSE_CHECKPOINT_SHA256=${EXPECTED_DENSE_CHECKPOINT_SHA256:?}
COFITOK_RUN="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$PAIR_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00001000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00001000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$PAIR_ROOT/pair_summary.json" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]
export PYTHONPATH=src

"$PYTHON" - \
  "$PAIR_ROOT/pair_summary.json" \
  "$COFITOK_RUN/training_report.json" \
  "$DENSE_RUN/training_report.json" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_COFITOK_CHECKPOINT_SHA256" \
  "$EXPECTED_DENSE_CHECKPOINT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

summary_path, cofitok_path, dense_path, revision, cofitok_sha, dense_sha = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
reports = {
    "cofitok": json.loads(Path(cofitok_path).read_text(encoding="utf-8")),
    "dense": json.loads(Path(dense_path).read_text(encoding="utf-8")),
}
if (
    summary["status"] != "completed"
    or summary["stage"] != "matched_1k_qualification_probe"
    or summary["git_revision"] != revision
    or summary["completed_steps"] != 1000
    or summary["images_seen"] != 64000
):
    raise SystemExit("invalid matched EMA-teacher 1K summary")
for method, expected_sha in (
    ("cofitok", cofitok_sha),
    ("dense", dense_sha),
):
    report = reports[method]
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 1000
        or report["target_steps"] != 1000
        or report["git"]["revision"] != revision
        or report["git"]["dirty"]
        or report["latest_checkpoint"]["checkpoint_sha256"] != expected_sha
        or report["final_metrics"]["ema_teacher_consistency_scale"] != 1.0
    ):
        raise SystemExit(f"{method} training identity is invalid")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher 1K post-evaluation while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$OUTPUT_ROOT"
test ! -e "$QUALIFICATION_ROOT"
mkdir -p "$OUTPUT_ROOT" "$QUALIFICATION_ROOT"

for weights in model ema; do
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/cofitok_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 16 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT/$weights/dense_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 0 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  for method in cofitok dense; do
    checkpoint="$COFITOK_CHECKPOINT"
    if [[ "$method" == dense ]]; then
      checkpoint="$DENSE_CHECKPOINT"
    fi
    "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
      --checkpoint "$checkpoint" \
      --output-dir "$OUTPUT_ROOT/$weights/${method}_rollout" \
      --num-images 8 \
      --batch-size 2 \
      --sample-steps 100 \
      --teacher-timesteps 999,900,750,500,250,100,10 \
      --seed 2029 \
      --weights "$weights" \
      --precision bf16 \
      --guidance-scale 1.5 \
      --teacher-guidance-scale 1.0 \
      --cfg-batch-mode batched \
      --clip-x0
  done
done

"$PYTHON" scripts/build_generation_stability_qualification.py \
  --cofitok-training "$COFITOK_RUN/training_report.json" \
  --dense-training "$DENSE_RUN/training_report.json" \
  --cofitok-checkpoint "$OUTPUT_ROOT/model/cofitok_checkpoint/checkpoint_evaluation_report.json" \
  --dense-checkpoint "$OUTPUT_ROOT/model/dense_checkpoint/checkpoint_evaluation_report.json" \
  --cofitok-rollout "$OUTPUT_ROOT/model/cofitok_rollout/rollout_stability_report.json" \
  --dense-rollout "$OUTPUT_ROOT/model/dense_rollout/rollout_stability_report.json" \
  --output-dir "$QUALIFICATION_ROOT/model"

"$PYTHON" - \
  "$QUALIFICATION_ROOT" \
  "$OUTPUT_ROOT" \
  "$PAIR_ROOT/pair_summary.json" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_COFITOK_CHECKPOINT_SHA256" \
  "$EXPECTED_DENSE_CHECKPOINT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def free_step(report: dict, timestep: int) -> dict:
    for step in report["free_sampling_rollout"]["steps"]:
        if int(step["timestep"]) == timestep:
            return step
    raise SystemExit(f"missing free-rollout timestep {timestep}")


qualification_root = Path(sys.argv[1])
evaluation_root = Path(sys.argv[2])
pair_summary_path = Path(sys.argv[3])
expected_revision = sys.argv[4]
expected_shas = {"cofitok": sys.argv[5], "dense": sys.argv[6]}
raw_path = qualification_root / "model" / "qualification_report.json"
raw = read(raw_path)
if (
    raw["protocol"]["weights"] != "model"
    or raw["identity"]["git_revision"] != expected_revision
):
    raise SystemExit("authoritative qualification identity is invalid")

ema = {}
ema_sources = {}
for method in ("cofitok", "dense"):
    checkpoint_path = (
        evaluation_root
        / "ema"
        / f"{method}_checkpoint"
        / "checkpoint_evaluation_report.json"
    )
    rollout_path = (
        evaluation_root
        / "ema"
        / f"{method}_rollout"
        / "rollout_stability_report.json"
    )
    checkpoint = read(checkpoint_path)
    rollout = read(rollout_path)
    if (
        checkpoint["status"] != "completed"
        or rollout["status"] != "completed"
        or checkpoint["weights"] != "ema"
        or rollout["weights"] != "ema"
        or checkpoint["checkpoint_sha256"] != expected_shas[method]
        or rollout["checkpoint_sha256"] != expected_shas[method]
        or checkpoint["checkpoint_step"] != 1000
        or rollout["checkpoint_step"] != 1000
        or checkpoint["git"]["revision"] != expected_revision
        or rollout["git"]["revision"] != expected_revision
    ):
        raise SystemExit(f"{method} EMA evaluation identity is invalid")
    ema[method] = {"checkpoint": checkpoint, "rollout": rollout}
    ema_sources[method] = {
        "checkpoint": source(checkpoint_path),
        "rollout": source(rollout_path),
    }
if ema["cofitok"]["rollout"]["protocol"] != ema["dense"]["rollout"]["protocol"]:
    raise SystemExit("EMA rollout protocols differ")

cofitok_checkpoint = ema["cofitok"]["checkpoint"]["metrics"]
dense_checkpoint = ema["dense"]["checkpoint"]["metrics"]
cofitok_endpoint = float(
    cofitok_checkpoint["orders"]["ordered"]["endpoint_clean_mse"]
)
dense_endpoint = float(
    dense_checkpoint["orders"]["ordered"]["endpoint_clean_mse"]
)
high_frequency = []
for timestep in (595, 394, 192, 91):
    cofitok_value = float(
        free_step(ema["cofitok"]["rollout"], timestep)[
            "predicted_x0_high_frequency_ratio"
        ]
    )
    dense_value = float(
        free_step(ema["dense"]["rollout"], timestep)[
            "predicted_x0_high_frequency_ratio"
        ]
    )
    high_frequency.append(
        {
            "timestep": timestep,
            "cofitok": cofitok_value,
            "dense": dense_value,
            "ratio": cofitok_value / dense_value,
        }
    )
cofitok_reconstruction = ema["cofitok"]["rollout"][
    "reconstruction_rollout"
]["summary"]
dense_reconstruction = ema["dense"]["rollout"][
    "reconstruction_rollout"
]["summary"]
summary = {
    "schema_version": 1,
    "status": "completed",
    "stage": "matched_1k_n8_screening",
    "scaling_authorization_allowed": False,
    "authoritative_raw_model_qualification": {
        "status": raw["status"],
        "failed_gates": [
            name for name, gate in raw["gates"].items() if not gate["passed"]
        ],
        "metrics": raw["metrics"],
        "source": source(raw_path),
    },
    "ema_diagnostic": {
        "role": "diagnostic_only",
        "ordered_rank_by_path_auc": int(
            cofitok_checkpoint["ordered_rank_by_path_auc"]
        ),
        "endpoint_ratio": cofitok_endpoint / dense_endpoint,
        "peak_predicted_x0_high_frequency_ratio": max(
            row["ratio"] for row in high_frequency
        ),
        "predicted_x0_high_frequency": high_frequency,
        "reconstruction_ratio": (
            float(cofitok_reconstruction["final_clipped_x0_mse"])
            / float(dense_reconstruction["final_clipped_x0_mse"])
        ),
        "cofitok_reconstruction_amplification": float(
            cofitok_reconstruction["final_to_best_x0_mse_amplification"]
        ),
        "dense_reconstruction_amplification": float(
            dense_reconstruction["final_to_best_x0_mse_amplification"]
        ),
        "sources": ema_sources,
    },
    "sources": {
        "pair_summary": source(pair_summary_path),
    },
}
(qualification_root / "screening_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
