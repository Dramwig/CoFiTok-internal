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

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
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

  "$PYTHON" scripts/build_generation_stability_qualification.py \
    --cofitok-training "$COFITOK_RUN/training_report.json" \
    --dense-training "$DENSE_RUN/training_report.json" \
    --cofitok-checkpoint "$OUTPUT_ROOT/$weights/cofitok_checkpoint/checkpoint_evaluation_report.json" \
    --dense-checkpoint "$OUTPUT_ROOT/$weights/dense_checkpoint/checkpoint_evaluation_report.json" \
    --cofitok-rollout "$OUTPUT_ROOT/$weights/cofitok_rollout/rollout_stability_report.json" \
    --dense-rollout "$OUTPUT_ROOT/$weights/dense_rollout/rollout_stability_report.json" \
    --output-dir "$QUALIFICATION_ROOT/$weights"
done

"$PYTHON" - "$QUALIFICATION_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {"schema_version": 1, "weights": {}}
for weights in ("model", "ema"):
    report = json.loads(
        (root / weights / "qualification_report.json").read_text(encoding="utf-8")
    )
    summary["weights"][weights] = {
        "status": report["status"],
        "failed_gates": [
            name for name, gate in report["gates"].items() if not gate["passed"]
        ],
        "metrics": report["metrics"],
    }
(root / "qualification_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
