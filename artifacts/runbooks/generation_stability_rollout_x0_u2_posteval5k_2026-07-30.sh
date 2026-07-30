#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-5k-2521d87
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2
SCREENING_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation5k_rollout_x0_u2
SCREENING_QUALIFICATION=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/qualification5k_rollout_x0_u2_n8
ROBUST_OUTPUT_BASE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation5k_rollout_x0_u2_n64
ROBUST_QUALIFICATION_BASE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/qualification5k_rollout_x0_u2_n64
SCALING_DECISION=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/scaling_decision5k_rollout_x0_u2_to_50k
EXPECTED_CODE_REVISION=2521d874a82898a7a2a527d824ea1df285df221d
EXPECTED_PAIR_SUMMARY_SHA256=${EXPECTED_PAIR_SUMMARY_SHA256:?set EXPECTED_PAIR_SUMMARY_SHA256 to the completed pair summary SHA256}
COFITOK_RUN="$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2"
DENSE_RUN="$PAIR_ROOT/dense_rollout_x0_u2"
COFITOK_CHECKPOINT="$COFITOK_RUN/checkpoint_step_00005000.pt"
DENSE_CHECKPOINT="$DENSE_RUN/checkpoint_step_00005000.pt"

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$PAIR_ROOT/pair_summary.json" | awk '{print $1}')" == "$EXPECTED_PAIR_SUMMARY_SHA256" ]]
export PYTHONPATH=src

"$PYTHON" - "$PAIR_ROOT" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_revision = sys.argv[2]
summary = json.loads((root / "pair_summary.json").read_text(encoding="utf-8"))
if summary["status"] != "completed":
    raise SystemExit("matched 5K summary is incomplete")
if summary["git_revision"] != expected_revision:
    raise SystemExit("matched 5K summary revision mismatch")
if summary["completed_steps"] != 5000 or summary["images_seen"] != 320000:
    raise SystemExit("matched 5K summary budget mismatch")
for method, run_name in (
    ("cofitok", "cofitok_rgbtail3_rollout_x0_u2"),
    ("dense", "dense_rollout_x0_u2"),
):
    report = json.loads(
        (root / run_name / "training_report.json").read_text(encoding="utf-8")
    )
    if not report["training_complete"]:
        raise SystemExit(f"{method} training is incomplete")
    if report["completed_steps"] != 5000 or report["target_steps"] != 5000:
        raise SystemExit(f"{method} exact 5K budget mismatch")
    if report["final_metrics"]["samples_seen"] != 320000:
        raise SystemExit(f"{method} images-seen mismatch")
    if report["git"]["revision"] != expected_revision or report["git"]["dirty"]:
        raise SystemExit(f"{method} Git provenance mismatch")
    latest = report["latest_checkpoint"]
    if latest["checkpoint_sha256"] != summary[method]["checkpoint_sha256"]:
        raise SystemExit(f"{method} checkpoint identity differs from pair summary")
    if latest["step"] != 5000 or latest["git_revision"] != expected_revision:
        raise SystemExit(f"{method} latest checkpoint provenance mismatch")
    if not (root / run_name / latest["integrity_manifest"]).is_file():
        raise SystemExit(f"{method} final integrity sidecar is missing")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing two-step 5K post-evaluation while the GPU is busy\n' >&2
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
    name for name, gate in report["gates"].items() if not gate["passed"]
)
unexpected = sorted(set(failed) - {"reconstruction_regression"})
if unexpected:
    raise SystemExit(
        "n=8 screening has non-reconstruction failures: " + ", ".join(unexpected)
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
