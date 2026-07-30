#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair5k_rollout_x0_u2"
PAIR_SUMMARY="$PAIR_ROOT/pair_summary.json"
RAW_QUALIFICATION="$ROOT/qualification5k_rollout_x0_u2_n8/qualification_report.json"
OUTPUT_BASE="$ROOT/evaluation5k_rollout_x0_u2_ema_n64"
SUMMARY="$ROOT/diagnostic5k_rollout_x0_u2_ema_n64/diagnostic_report.json"
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_PAIR_SUMMARY_SHA256=ce604c2e9d2a1864bb6ba6fad31cdcee1cc92219256d361333abc00feaded28c
EXPECTED_RAW_QUALIFICATION_SHA256=c4603cbcdfad7537e8a53d9cb5493cd0896eb17444a9cf04d7ef1362bf42f6a1
COFITOK_CHECKPOINT="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2/checkpoint_step_00005000.pt"
DENSE_CHECKPOINT="$PAIR_ROOT/dense_rollout_x0_u2/checkpoint_step_00005000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$PAIR_SUMMARY" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]
[[ "$(sha256sum "$RAW_QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_RAW_QUALIFICATION_SHA256" ]]
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
  printf 'refusing EMA diagnostic while the GPU is busy\n' >&2
  exit 9
fi

for seed in 2029 2039; do
  output_root="${OUTPUT_BASE}_seed${seed}"
  test ! -e "$output_root"
  mkdir -p "$output_root"

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$output_root/cofitok_rollout" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed "$seed" \
    --weights ema \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$output_root/dense_rollout" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed "$seed" \
    --weights ema \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0
done

test ! -e "$(dirname "$SUMMARY")"
mkdir -p "$(dirname "$SUMMARY")"
"$PYTHON" - "$OUTPUT_BASE" "$SUMMARY" "$PAIR_SUMMARY" "$RAW_QUALIFICATION" <<'PY'
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


def step(report: dict[str, object], timestep: int) -> dict[str, object]:
    for row in report["free_sampling_rollout"]["steps"]:
        if int(row["timestep"]) == timestep:
            return row
    raise SystemExit(f"missing free-rollout timestep {timestep}")


base = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
pair_path = Path(sys.argv[3])
qualification_path = Path(sys.argv[4])
timesteps = (595, 394, 192, 91)
rows = []
sources = {
    "pair_summary": source(pair_path),
    "raw_qualification": source(qualification_path),
    "rollouts": [],
}
for seed in (2029, 2039):
    reports = {}
    for method in ("cofitok", "dense"):
        path = Path(f"{base}_seed{seed}") / f"{method}_rollout" / "rollout_stability_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "completed" or report["weights"] != "ema":
            raise SystemExit(f"{method} seed {seed} is not a completed EMA diagnostic")
        reports[method] = report
        sources["rollouts"].append(source(path))
    if reports["cofitok"]["protocol"] != reports["dense"]["protocol"]:
        raise SystemExit(f"seed {seed} rollout protocols differ")
    high_frequency = []
    for timestep in timesteps:
        cofitok_value = float(
            step(reports["cofitok"], timestep)["predicted_x0_high_frequency_ratio"]
        )
        dense_value = float(
            step(reports["dense"], timestep)["predicted_x0_high_frequency_ratio"]
        )
        high_frequency.append(
            {
                "timestep": timestep,
                "cofitok": cofitok_value,
                "dense": dense_value,
                "ratio": cofitok_value / dense_value,
            }
        )
    cofitok_reconstruction = reports["cofitok"]["reconstruction_rollout"]["summary"]
    dense_reconstruction = reports["dense"]["reconstruction_rollout"]["summary"]
    reconstruction_ratio = (
        float(cofitok_reconstruction["final_clipped_x0_mse"])
        / float(dense_reconstruction["final_clipped_x0_mse"])
    )
    peak_high_frequency_ratio = max(row["ratio"] for row in high_frequency)
    rows.append(
        {
            "seed": seed,
            "num_images": int(reports["cofitok"]["protocol"]["num_images"]),
            "peak_predicted_x0_high_frequency_ratio": peak_high_frequency_ratio,
            "predicted_x0_high_frequency": high_frequency,
            "reconstruction_ratio": reconstruction_ratio,
            "cofitok_reconstruction_amplification": float(
                cofitok_reconstruction["final_to_best_x0_mse_amplification"]
            ),
            "dense_reconstruction_amplification": float(
                dense_reconstruction["final_to_best_x0_mse_amplification"]
            ),
            "diagnostic_threshold_checks": {
                "high_frequency_ratio_at_most_1_5": peak_high_frequency_ratio <= 1.5,
                "reconstruction_ratio_at_most_1_05": reconstruction_ratio <= 1.05,
            },
        }
    )
summary = {
    "schema_version": 1,
    "status": "completed",
    "role": "diagnostic_only",
    "weights": "ema",
    "scaling_authorization_allowed": False,
    "reason": (
        "The authoritative raw n=8 qualification failed predicted_x0_high_frequency; "
        "this report only tests whether the production EMA path is stable across seeds."
    ),
    "rows": rows,
    "all_diagnostic_threshold_checks_pass": all(
        all(row["diagnostic_threshold_checks"].values()) for row in rows
    ),
    "sources": sources,
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
