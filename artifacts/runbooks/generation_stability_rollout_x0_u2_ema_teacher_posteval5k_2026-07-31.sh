#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-5k-59db142
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair5k_rollout_x0_u2_ema_teacher"
SCREENING_ROOT="$ROOT/evaluation5k_rollout_x0_u2_ema_teacher"
SCREENING_QUALIFICATION="$ROOT/qualification5k_rollout_x0_u2_ema_teacher_n8"
ROBUST_OUTPUT_BASE="$ROOT/evaluation5k_rollout_x0_u2_ema_teacher_n64"
ROBUST_QUALIFICATION_BASE="$ROOT/qualification5k_rollout_x0_u2_ema_teacher_n64"
SCALING_DECISION="$ROOT/scaling_decision5k_rollout_x0_u2_ema_teacher_to_50k"
EXPECTED_CODE_REVISION=59db142fc45d69dc92bb0333be5ac2d0162d9dc4
EXPECTED_SOURCE_DECISION_SHA256=7f2e6e691e26ef24f18d42e0f337229a35f2eec1a774e82f12141bb15dc48c9d
EXPECTED_BENCHMARK_SHA256=066a03a0bf9f6a7d40cd98de41468974d5d4e0d629a259c4a550026e6e399305
EXPECTED_PAIR_SUMMARY_SHA256=${EXPECTED_PAIR_SUMMARY_SHA256:?set EXPECTED_PAIR_SUMMARY_SHA256 to the completed pair summary SHA256}
COFITOK_RUN="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$PAIR_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00005000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00005000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$PAIR_ROOT/pair_summary.json" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]
export PYTHONPATH=src

"$PYTHON" - \
  "$PAIR_ROOT" \
  "$EXPECTED_CODE_REVISION" \
  "$EXPECTED_SOURCE_DECISION_SHA256" \
  "$EXPECTED_BENCHMARK_SHA256" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_revision = sys.argv[2]
expected_decision_sha = sys.argv[3]
expected_benchmark_sha = sys.argv[4]
summary = json.loads((root / "pair_summary.json").read_text(encoding="utf-8"))
if (
    summary["status"] != "completed"
    or summary["stage"] != "matched_5k_stability_gate"
    or summary["formal_scaling_authorization_allowed"] is not False
    or summary["git_revision"] != expected_revision
    or summary["completed_steps"] != 5000
    or summary["images_seen"] != 320000
    or summary["generation_pair_contract"]["valid"] is not True
):
    raise SystemExit("matched 5K summary identity mismatch")
if (
    summary["sources"]["scaling_decision"]["sha256"] != expected_decision_sha
    or summary["sources"]["active_5k_benchmark"]["sha256"]
    != expected_benchmark_sha
):
    raise SystemExit("matched 5K source provenance mismatch")
if summary["ema_teacher_contract"] != {
    "weight": 0.25,
    "start_step": 3000,
    "warmup_steps": 1000,
    "batch_fraction": 0.0625,
}:
    raise SystemExit("matched 5K teacher contract mismatch")
for method, run_name in (
    ("cofitok", "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"),
    ("dense", "dense_rollout_x0_u2_ema_teacher"),
):
    report = json.loads(
        (root / run_name / "training_report.json").read_text(encoding="utf-8")
    )
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 5000
        or report["target_steps"] != 5000
        or report["final_metrics"]["samples_seen"] != 320000
        or report["git"]["revision"] != expected_revision
        or report["git"]["dirty"]
    ):
        raise SystemExit(f"{method} exact 5K completion mismatch")
    latest = report["latest_checkpoint"]
    if (
        latest["checkpoint_sha256"] != summary[method]["checkpoint_sha256"]
        or latest["step"] != 5000
        or latest["git_revision"] != expected_revision
        or not (root / run_name / latest["integrity_manifest"]).is_file()
    ):
        raise SystemExit(f"{method} checkpoint identity mismatch")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing EMA-teacher 5K post-evaluation while the GPU is busy\n' >&2
  exit 9
fi

test ! -e "$SCREENING_ROOT"
test ! -e "$SCREENING_QUALIFICATION"
test ! -e "${ROBUST_OUTPUT_BASE}_seed2029"
test ! -e "${ROBUST_OUTPUT_BASE}_seed2039"
test ! -e "${ROBUST_QUALIFICATION_BASE}_seed2029"
test ! -e "${ROBUST_QUALIFICATION_BASE}_seed2039"
test ! -e "$SCALING_DECISION"
mkdir -p "$SCREENING_ROOT"

for weights in model ema; do
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$COFITOK_CHECKPOINT" \
    --output-dir "$SCREENING_ROOT/$weights/cofitok_checkpoint" \
    --num-images 256 \
    --timestep 500 \
    --random-orders 16 \
    --seed 2029 \
    --weights "$weights" \
    --precision bf16

  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output-dir "$SCREENING_ROOT/$weights/dense_checkpoint" \
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
      --output-dir "$SCREENING_ROOT/$weights/${method}_rollout" \
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
  --cofitok-checkpoint "$SCREENING_ROOT/model/cofitok_checkpoint/checkpoint_evaluation_report.json" \
  --dense-checkpoint "$SCREENING_ROOT/model/dense_checkpoint/checkpoint_evaluation_report.json" \
  --cofitok-rollout "$SCREENING_ROOT/model/cofitok_rollout/rollout_stability_report.json" \
  --dense-rollout "$SCREENING_ROOT/model/dense_rollout/rollout_stability_report.json" \
  --output-dir "$SCREENING_QUALIFICATION"

"$PYTHON" - "$SCREENING_QUALIFICATION/qualification_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
failed = sorted(
    name for name, gate in report["gates"].items() if gate["passed"] is not True
)
unexpected = sorted(set(failed) - {"reconstruction_regression"})
if unexpected:
    raise SystemExit(
        "n=8 screening has failures beyond reconstruction: "
        + ", ".join(unexpected)
    )
PY

for seed in 2029 2039; do
  output_root="${ROBUST_OUTPUT_BASE}_seed${seed}"
  qualification_root="${ROBUST_QUALIFICATION_BASE}_seed${seed}"
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
  --screening-report "$SCREENING_QUALIFICATION/qualification_report.json" \
  --robust-report "${ROBUST_QUALIFICATION_BASE}_seed2029/qualification_report.json" \
  --robust-report "${ROBUST_QUALIFICATION_BASE}_seed2039/qualification_report.json" \
  --output-dir "$SCALING_DECISION" \
  --next-stage fresh_matched_50k_preparation \
  --require-pass
