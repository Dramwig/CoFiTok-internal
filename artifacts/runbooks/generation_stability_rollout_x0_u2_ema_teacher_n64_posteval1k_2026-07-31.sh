#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair1k_rollout_x0_u2_ema_teacher"
SCREENING_ROOT="$ROOT/evaluation1k_rollout_x0_u2_ema_teacher"
SCREENING_QUALIFICATION="$ROOT/qualification1k_rollout_x0_u2_ema_teacher/model/qualification_report.json"
OUTPUT_BASE="$ROOT/evaluation1k_rollout_x0_u2_ema_teacher_n64"
QUALIFICATION_BASE="$ROOT/qualification1k_rollout_x0_u2_ema_teacher_n64"
DECISION_ROOT="$ROOT/scaling_decision1k_rollout_x0_u2_ema_teacher"
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_PAIR_SUMMARY_SHA256=${EXPECTED_PAIR_SUMMARY_SHA256:?}
EXPECTED_SCREENING_QUALIFICATION_SHA256=${EXPECTED_SCREENING_QUALIFICATION_SHA256:?}
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
[[ "$(sha256sum "$SCREENING_QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_SCREENING_QUALIFICATION_SHA256" ]]
export PYTHONPATH=src

"$PYTHON" - \
  "$SCREENING_QUALIFICATION" \
  "$PAIR_ROOT/pair_summary.json" \
  "$COFITOK_RUN/training_report.json" \
  "$DENSE_RUN/training_report.json" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_COFITOK_CHECKPOINT_SHA256" \
  "$EXPECTED_DENSE_CHECKPOINT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

(
    screening_path,
    pair_path,
    cofitok_path,
    dense_path,
    revision,
    cofitok_sha,
    dense_sha,
) = sys.argv[1:]
screening = json.loads(Path(screening_path).read_text(encoding="utf-8"))
pair = json.loads(Path(pair_path).read_text(encoding="utf-8"))
if (
    screening["status"] != "pass"
    or screening["protocol"]["weights"] != "model"
    or screening["protocol"]["rollout"]["num_images"] != 8
    or screening["identity"]["git_revision"] != revision
):
    raise SystemExit("raw n=8 screening did not pass with the expected identity")
if (
    pair["status"] != "completed"
    or pair["stage"] != "matched_1k_qualification_probe"
    or pair["git_revision"] != revision
):
    raise SystemExit("matched 1K pair identity is invalid")
for method, path, expected_sha in (
    ("cofitok", cofitok_path, cofitok_sha),
    ("dense", dense_path, dense_sha),
):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 1000
        or report["git"]["revision"] != revision
        or report["git"]["dirty"]
        or report["latest_checkpoint"]["checkpoint_sha256"] != expected_sha
    ):
        raise SystemExit(f"{method} training identity is invalid")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher raw n64 evaluation while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$DECISION_ROOT"
for seed in 2029 2039; do
  output_root="${OUTPUT_BASE}_seed${seed}"
  qualification_root="${QUALIFICATION_BASE}_seed${seed}"
  test ! -e "$output_root"
  test ! -e "$qualification_root"
  mkdir -p "$output_root"

  "$PYTHON" scripts/evaluate_generation_rollout_stability.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$output_root/cofitok_rollout" \
    --num-images 64 \
    --batch-size 8 \
    --sample-steps 100 \
    --teacher-timesteps 999,900,750,500,250,100,10 \
    --seed "$seed" \
    --weights model \
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
    --weights model \
    --precision bf16 \
    --guidance-scale 1.5 \
    --teacher-guidance-scale 1.0 \
    --cfg-batch-mode batched \
    --clip-x0

  "$PYTHON" scripts/build_generation_stability_qualification.py \
    --cofitok-training "$COFITOK_RUN/training_report.json" \
    --dense-training "$DENSE_RUN/training_report.json" \
    --cofitok-checkpoint "$SCREENING_ROOT/model/cofitok_checkpoint/checkpoint_evaluation_report.json" \
    --dense-checkpoint "$SCREENING_ROOT/model/dense_checkpoint/checkpoint_evaluation_report.json" \
    --cofitok-rollout "$output_root/cofitok_rollout/rollout_stability_report.json" \
    --dense-rollout "$output_root/dense_rollout/rollout_stability_report.json" \
    --output-dir "$qualification_root"
done

"$PYTHON" scripts/build_generation_stability_scaling_decision.py \
  --screening-report "$SCREENING_QUALIFICATION" \
  --robust-report "${QUALIFICATION_BASE}_seed2029/qualification_report.json" \
  --robust-report "${QUALIFICATION_BASE}_seed2039/qualification_report.json" \
  --output-dir "$DECISION_ROOT" \
  --min-robust-reports 2 \
  --min-robust-images 64 \
  --next-stage matched_5k

"$PYTHON" - \
  "$QUALIFICATION_BASE" \
  "$DECISION_ROOT/scaling_decision.json" \
  "$PAIR_ROOT/pair_summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def source(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


base = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
pair_path = Path(sys.argv[3])
summary = {
    "schema_version": 1,
    "status": "completed",
    "stage": "matched_1k_robust_qualification",
    "seeds": {},
    "decision": source(decision_path),
    "pair_summary": source(pair_path),
}
for seed in (2029, 2039):
    path = Path(f"{base}_seed{seed}") / "qualification_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    summary["seeds"][str(seed)] = {
        "status": report["status"],
        "failed_gates": [
            name for name, gate in report["gates"].items() if not gate["passed"]
        ],
        "metrics": report["metrics"],
        "source": source(path),
    }
decision = json.loads(decision_path.read_text(encoding="utf-8"))
summary["scaling_decision"] = {
    "status": decision["status"],
    "decision": decision["decision"],
    "authorized_next_stage": decision.get("authorized_next_stage"),
}
summary_path = Path(f"{base}_summary.json")
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
