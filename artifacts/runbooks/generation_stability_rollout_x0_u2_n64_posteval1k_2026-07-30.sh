#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-u2-6b77ef7
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
PAIR_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2
SCREENING_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation1k_rollout_x0_u2
SCREENING_QUALIFICATION=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2/model/qualification_report.json
OUTPUT_BASE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/evaluation1k_rollout_x0_u2_n64
QUALIFICATION_BASE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2_n64
EXPECTED_CODE_REVISION=6b77ef7254356d551b2e392aba07df1c96ed067c
EXPECTED_PAIR_SUMMARY_SHA256=bd6f1575f6250e58d710e7fa58c8cdcf3f0eadc729d867535a9bf46aa764e5d7
EXPECTED_SCREENING_QUALIFICATION_SHA256=f4a0b344041b777bd74202daf6661ce5db271e76022e5a5c9710407160a594ea
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
[[ "$(sha256sum "$SCREENING_QUALIFICATION" | awk '{print $1}')" == "$EXPECTED_SCREENING_QUALIFICATION_SHA256" ]]

"$PYTHON" - "$SCREENING_QUALIFICATION" "$EXPECTED_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "pass":
    raise SystemExit("n=8 raw-model screening did not pass")
if report["protocol"]["weights"] != "model":
    raise SystemExit("n=8 screening did not use raw model weights")
if report["identity"]["git_revision"] != sys.argv[2]:
    raise SystemExit("n=8 screening revision mismatch")
if report["protocol"]["rollout"]["num_images"] != 8:
    raise SystemExit("n=8 screening image count mismatch")
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing two-step 1K n64 evaluation while the GPU is busy\n' >&2
  exit 9
fi

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

"$PYTHON" - "$QUALIFICATION_BASE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
summary = {"schema_version": 1, "status": "completed", "seeds": {}}
for seed in (2029, 2039):
    path = Path(f"{base}_seed{seed}") / "qualification_report.json"
    content = path.read_bytes()
    report = json.loads(content)
    summary["seeds"][str(seed)] = {
        "status": report["status"],
        "failed_gates": [
            name for name, gate in report["gates"].items() if not gate["passed"]
        ],
        "metrics": report["metrics"],
        "source": {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    }
summary_path = Path(f"{base}_summary.json")
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
