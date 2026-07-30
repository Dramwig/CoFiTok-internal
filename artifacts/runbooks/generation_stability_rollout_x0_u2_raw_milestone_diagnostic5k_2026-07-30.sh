#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair5k_rollout_x0_u2"
PAIR_SUMMARY="$PAIR_ROOT/pair_summary.json"
RAW_QUALIFICATION="$ROOT/qualification5k_rollout_x0_u2_n8/qualification_report.json"
FINAL_EVALUATION="$ROOT/evaluation5k_rollout_x0_u2/model"
OUTPUT_ROOT="$ROOT/evaluation5k_rollout_x0_u2_raw_milestones"
SUMMARY="$ROOT/diagnostic5k_rollout_x0_u2_raw_milestones/diagnostic_report.json"
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_PAIR_SUMMARY_SHA256=ce604c2e9d2a1864bb6ba6fad31cdcee1cc92219256d361333abc00feaded28c
EXPECTED_RAW_QUALIFICATION_SHA256=c4603cbcdfad7537e8a53d9cb5493cd0896eb17444a9cf04d7ef1362bf42f6a1

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$PAIR_SUMMARY" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]
[[ "$(sha256sum "$RAW_QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_RAW_QUALIFICATION_SHA256" ]]
test ! -e "$OUTPUT_ROOT"
test ! -e "$(dirname "$SUMMARY")"
export PYTHONPATH=src

"$PYTHON" - "$PAIR_SUMMARY" "$RAW_QUALIFICATION" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

pair = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
qualification = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_revision = sys.argv[3]
if (
    pair["status"] != "completed"
    or pair["completed_steps"] != 5000
    or pair["images_seen"] != 320000
    or pair["git_revision"] != expected_revision
):
    raise SystemExit("matched 5K pair identity is invalid")
failed = sorted(
    name
    for name, gate in qualification["gates"].items()
    if gate["passed"] is not True
)
if qualification["status"] != "fail" or failed != ["predicted_x0_high_frequency"]:
    raise SystemExit("raw n=8 failure identity changed")
if qualification["protocol"]["weights"] != "model":
    raise SystemExit("raw qualification no longer represents model weights")
if qualification["identity"]["git_revision"] != expected_revision:
    raise SystemExit("raw qualification revision mismatch")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing raw milestone diagnostic while the GPU is busy\n' >&2
  exit 9
fi

mkdir -p "$OUTPUT_ROOT"
for step in 1250 2500 3750; do
  printf -v padded_step '%08d' "$step"
  step_root="$OUTPUT_ROOT/step_$padded_step"
  mkdir -p "$step_root"

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2/checkpoint_step_${padded_step}.pt" \
    --output-dir "$step_root/cofitok_rollout" \
    --num-images 8 \
    --batch-size 2 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed 2029 \
    --weights model \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$PAIR_ROOT/dense_rollout_x0_u2/checkpoint_step_${padded_step}.pt" \
    --output-dir "$step_root/dense_rollout" \
    --num-images 8 \
    --batch-size 2 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed 2029 \
    --weights model \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0
done

mkdir -p "$(dirname "$SUMMARY")"
"$PYTHON" - \
  "$OUTPUT_ROOT" \
  "$FINAL_EVALUATION" \
  "$SUMMARY" \
  "$PAIR_SUMMARY" \
  "$RAW_QUALIFICATION" \
  "$PAIR_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


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


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric(path: Path, step: int) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row["step"]) == step]
    if len(matches) != 1:
        raise SystemExit(f"expected one metrics row for step {step}: {path}")
    return matches[0]


def rollout_step(report: dict[str, object], timestep: int) -> dict[str, object]:
    for row in report["free_sampling_rollout"]["steps"]:
        if int(row["timestep"]) == timestep:
            return row
    raise SystemExit(f"missing free-rollout timestep {timestep}")


output_root = Path(sys.argv[1])
final_evaluation = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
pair_path = Path(sys.argv[4])
qualification_path = Path(sys.argv[5])
pair_root = Path(sys.argv[6])
timesteps = (595, 394, 192, 91)
milestones = (1250, 2500, 3750, 5000)
metrics_paths = {
    "cofitok": pair_root / "cofitok_rgbtail3_rollout_x0_u2" / "train_metrics.jsonl",
    "dense": pair_root / "dense_rollout_x0_u2" / "train_metrics.jsonl",
}
sources = {
    "pair_summary": source(pair_path),
    "raw_qualification": source(qualification_path),
    "training_metrics": {
        method: source(path) for method, path in metrics_paths.items()
    },
    "rollouts": [],
}
rows = []
for milestone in milestones:
    if milestone == 5000:
        report_root = final_evaluation
    else:
        report_root = output_root / f"step_{milestone:08d}"
    reports = {}
    for method in ("cofitok", "dense"):
        path = report_root / f"{method}_rollout" / "rollout_stability_report.json"
        report = load_json(path)
        if (
            report["status"] != "completed"
            or report["weights"] != "model"
            or int(report["checkpoint_step"]) != milestone
        ):
            raise SystemExit(f"invalid raw {method} rollout at step {milestone}")
        reports[method] = report
        sources["rollouts"].append(source(path))
    if reports["cofitok"]["protocol"] != reports["dense"]["protocol"]:
        raise SystemExit(f"step {milestone} rollout protocols differ")
    high_frequency = []
    for timestep in timesteps:
        cofitok_value = float(
            rollout_step(reports["cofitok"], timestep)[
                "predicted_x0_high_frequency_ratio"
            ]
        )
        dense_value = float(
            rollout_step(reports["dense"], timestep)[
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
    reconstruction = {
        method: reports[method]["reconstruction_rollout"]["summary"]
        for method in ("cofitok", "dense")
    }
    training = {
        method: load_metric(metrics_paths[method], milestone)
        for method in ("cofitok", "dense")
    }
    rows.append(
        {
            "step": milestone,
            "checkpoint_sha256": {
                method: reports[method]["checkpoint_sha256"]
                for method in ("cofitok", "dense")
            },
            "training": {
                method: {
                    key: training[method][key]
                    for key in (
                        "epsilon",
                        "grad_norm",
                        "learning_rate",
                        "low_snr_high_frequency",
                        "rollout_consistency",
                        "total",
                        "validation_epsilon_mse",
                    )
                }
                for method in ("cofitok", "dense")
            },
            "peak_predicted_x0_high_frequency_ratio": max(
                row["ratio"] for row in high_frequency
            ),
            "predicted_x0_high_frequency": high_frequency,
            "reconstruction_ratio": (
                float(reconstruction["cofitok"]["final_clipped_x0_mse"])
                / float(reconstruction["dense"]["final_clipped_x0_mse"])
            ),
            "reconstruction": {
                method: {
                    "final_clipped_x0_mse": float(
                        reconstruction[method]["final_clipped_x0_mse"]
                    ),
                    "final_to_best_x0_mse_amplification": float(
                        reconstruction[method][
                            "final_to_best_x0_mse_amplification"
                        ]
                    ),
                }
                for method in ("cofitok", "dense")
            },
        }
    )
crossings = [
    row["step"]
    for row in rows
    if row["peak_predicted_x0_high_frequency_ratio"] > 1.5
]
summary = {
    "schema_version": 1,
    "status": "completed",
    "role": "diagnostic_only",
    "weights": "model",
    "scaling_authorization_allowed": False,
    "reason": (
        "The authoritative raw n=8 qualification failed predicted_x0_high_frequency; "
        "this report locates when that failure emerges along the immutable 5K trajectory."
    ),
    "protocol": {
        "num_images": 8,
        "seed": 2029,
        "sample_steps": 100,
        "selected_timesteps": list(timesteps),
        "milestones": list(milestones),
    },
    "first_milestone_above_raw_high_frequency_gate": (
        min(crossings) if crossings else None
    ),
    "rows": rows,
    "sources": sources,
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
