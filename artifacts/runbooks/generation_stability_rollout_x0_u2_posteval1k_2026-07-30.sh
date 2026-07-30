#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-6b77ef7
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation1k_rollout_x0_u2
QUALIFICATION_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2
EXPECTED_CODE_REVISION=6b77ef7254356d551b2e392aba07df1c96ed067c
EXPECTED_PAIR_SUMMARY_SHA256=bd6f1575f6250e58d710e7fa58c8cdcf3f0eadc729d867535a9bf46aa764e5d7
EXPECTED_COFITOK_CHECKPOINT_SHA256=9885273a3bd3773274d140db433af047c7cefeb87e460b9868406b45b7e1bb05
EXPECTED_DENSE_CHECKPOINT_SHA256=e27a03434ec3bef7f7528d6d7a0d61867845c5f95165cd1ccb51a07adafe0046
COFITOK_RUN="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2"
DENSE_RUN="$PAIR_ROOT/dense_rollout_x0_u2"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00001000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00001000.pt"
FINALIZE_ONLY=${FINALIZE_ONLY:-0}

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$FINALIZE_ONLY" == 0 || "$FINALIZE_ONLY" == 1 ]]
export PYTHONPATH=src

[[ "$(sha256sum "$PAIR_ROOT/pair_summary.json" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]

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
cofitok = json.loads(Path(cofitok_path).read_text(encoding="utf-8"))
dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))
if summary["status"] != "completed" or summary["git_revision"] != revision:
    raise SystemExit("invalid matched 1K summary")
for method, report, expected_sha in (
    ("cofitok", cofitok, cofitok_sha),
    ("dense", dense, dense_sha),
):
    if not report["training_complete"]:
        raise SystemExit(f"{method} training is incomplete")
    if report["completed_steps"] != 1000 or report["target_steps"] != 1000:
        raise SystemExit(f"{method} exact training budget mismatch")
    if report["git"]["revision"] != revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    if report["latest_checkpoint"]["checkpoint_sha256"] != expected_sha:
        raise SystemExit(f"{method} checkpoint identity mismatch")
PY

if [[ "$FINALIZE_ONLY" == 0 ]]; then
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    printf 'refusing two-step 1K post-evaluation while the GPU is busy\n' >&2
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
else
  test -d "$OUTPUT_ROOT"
  test -f "$QUALIFICATION_ROOT/model/qualification_report.json"
fi

"$PYTHON" - \
  "$QUALIFICATION_ROOT" \
  "$OUTPUT_ROOT" \
  "$COFITOK_RUN/training_report.json" \
  "$DENSE_RUN/training_report.json" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_COFITOK_CHECKPOINT_SHA256" \
  "$EXPECTED_DENSE_CHECKPOINT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path



def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def free_step(report: dict, timestep: int) -> dict:
    for step in report["free_sampling_rollout"]["steps"]:
        if int(step["timestep"]) == timestep:
            return step
    raise ValueError(f"missing EMA free-rollout timestep {timestep}")


qualification_root = Path(sys.argv[1])
evaluation_root = Path(sys.argv[2])
cofitok_training_path = Path(sys.argv[3])
dense_training_path = Path(sys.argv[4])
expected_revision = sys.argv[5]
expected_shas = {"cofitok": sys.argv[6], "dense": sys.argv[7]}
training = {
    "cofitok": read(cofitok_training_path),
    "dense": read(dense_training_path),
}
raw_path = qualification_root / "model" / "qualification_report.json"
raw = read(raw_path)
if raw["protocol"]["weights"] != "model":
    raise SystemExit("authoritative qualification must use raw model weights")

ema = {}
ema_paths = {}
for method in ("cofitok", "dense"):
    checkpoint_path = (
        evaluation_root / "ema" / f"{method}_checkpoint" / "checkpoint_evaluation_report.json"
    )
    rollout_path = (
        evaluation_root / "ema" / f"{method}_rollout" / "rollout_stability_report.json"
    )
    checkpoint = read(checkpoint_path)
    rollout = read(rollout_path)
    if checkpoint["status"] != "completed" or rollout["status"] != "completed":
        raise SystemExit(f"{method} EMA evaluation is incomplete")
    if checkpoint["weights"] != "ema" or rollout["weights"] != "ema":
        raise SystemExit(f"{method} EMA evaluation uses wrong weights")
    if checkpoint["checkpoint_sha256"] != expected_shas[method]:
        raise SystemExit(f"{method} EMA checkpoint identity mismatch")
    if rollout["checkpoint_sha256"] != expected_shas[method]:
        raise SystemExit(f"{method} EMA rollout checkpoint identity mismatch")
    if checkpoint["checkpoint_step"] != 1000 or rollout["checkpoint_step"] != 1000:
        raise SystemExit(f"{method} EMA evaluation step mismatch")
    if checkpoint["git"]["revision"] != expected_revision:
        raise SystemExit(f"{method} EMA checkpoint revision mismatch")
    if rollout["git"]["revision"] != expected_revision:
        raise SystemExit(f"{method} EMA rollout revision mismatch")
    if checkpoint["config"] != training[method]["config"]:
        raise SystemExit(f"{method} EMA checkpoint config mismatch")
    if rollout["config"] != training[method]["config"]:
        raise SystemExit(f"{method} EMA rollout config mismatch")
    ema[method] = {"checkpoint": checkpoint, "rollout": rollout}
    ema_paths[method] = {
        "checkpoint": source(checkpoint_path),
        "rollout": source(rollout_path),
    }
if ema["cofitok"]["rollout"]["protocol"] != ema["dense"]["rollout"]["protocol"]:
    raise SystemExit("EMA rollout protocols do not match")

cofitok_checkpoint = ema["cofitok"]["checkpoint"]["metrics"]
dense_checkpoint = ema["dense"]["checkpoint"]["metrics"]
cofitok_endpoint = float(
    cofitok_checkpoint["orders"]["ordered"]["endpoint_clean_mse"]
)
dense_endpoint = float(dense_checkpoint["orders"]["ordered"]["endpoint_clean_mse"])
high_frequency = []
for timestep in (595, 394, 192, 91):
    cofitok_value = float(
        free_step(
            ema["cofitok"]["rollout"],
            timestep,
        )["predicted_x0_high_frequency_ratio"]
    )
    dense_value = float(
        free_step(
            ema["dense"]["rollout"],
            timestep,
        )["predicted_x0_high_frequency_ratio"]
    )
    high_frequency.append(
        {
            "timestep": timestep,
            "cofitok": cofitok_value,
            "dense": dense_value,
            "ratio": cofitok_value / dense_value,
        }
    )
cofitok_reconstruction = ema["cofitok"]["rollout"]["reconstruction_rollout"][
    "summary"
]
dense_reconstruction = ema["dense"]["rollout"]["reconstruction_rollout"]["summary"]
cofitok_final = float(cofitok_reconstruction["final_clipped_x0_mse"])
dense_final = float(dense_reconstruction["final_clipped_x0_mse"])

summary = {
    "schema_version": 2,
    "authoritative_raw_model_qualification": {
        "status": raw["status"],
        "failed_gates": [
            name for name, gate in raw["gates"].items() if not gate["passed"]
        ],
        "metrics": raw["metrics"],
        "source": source(raw_path),
    },
    "ema_diagnostic": {
        "status": "diagnostic_only",
        "authoritative_for_scaling": False,
        "ordered_rank_by_path_auc": int(
            cofitok_checkpoint["ordered_rank_by_path_auc"]
        ),
        "endpoint_ratio": cofitok_endpoint / dense_endpoint,
        "peak_predicted_x0_high_frequency_ratio": max(
            row["ratio"] for row in high_frequency
        ),
        "predicted_x0_high_frequency": high_frequency,
        "reconstruction_ratio": cofitok_final / dense_final,
        "cofitok_reconstruction_amplification": float(
            cofitok_reconstruction["final_to_best_x0_mse_amplification"]
        ),
        "dense_reconstruction_amplification": float(
            dense_reconstruction["final_to_best_x0_mse_amplification"]
        ),
        "sources": ema_paths,
    },
}
(qualification_root / "qualification_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
