#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
CLASSIFIER_CHECKPOINT=${CLASSIFIER_CHECKPOINT:-/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:-2c2c1f5166b73d4f28df93b276901671ac1a7836}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:-scale/generation-stability-50k-preflight}
EXPECTED_FROZEN_EVALUATION_REVISION=${EXPECTED_FROZEN_EVALUATION_REVISION:-c1efb12c6640f2d2d62ac7e9982c8804d96e7289}
EXPECTED_FROZEN_EVALUATION_BRANCH=${EXPECTED_FROZEN_EVALUATION_BRANCH:-scale/generation-stability-50k-posteval-v4}
EXPECTED_CLASS_FIDELITY_REVISION=${EXPECTED_CLASS_FIDELITY_REVISION:?set the clean class-fidelity evaluator revision}
EXPECTED_CLASS_FIDELITY_BRANCH=${EXPECTED_CLASS_FIDELITY_BRANCH:-scale/generation-large-capacity}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_ema_teacher"
REPORT_ROOT="$OUTPUT_ROOT/reports"
COFITOK_RUN="$OUTPUT_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DENSE_RUN="$OUTPUT_ROOT/dense_rollout_x0_u2_ema_teacher"
COFITOK_SAMPLES="$COFITOK_RUN/samples_gate10k_ddim100_cfg15"
DENSE_SAMPLES="$DENSE_RUN/samples_gate10k_ddim100_cfg15"
POSTEVAL_STATUS="$REPORT_ROOT/posteval_waiter.json"
SUPPLEMENTAL_WAITER="$REPORT_ROOT/frozen_posteval_supplemental/supplemental_waiter.json"
SUPPLEMENTAL_REPORT="$REPORT_ROOT/frozen_posteval_supplemental/supplemental_qualification.json"
PROMOTION_GATE="$REPORT_ROOT/promotion_gate.json"
CLASS_FIDELITY_ROOT="$REPORT_ROOT/frozen_posteval_class_fidelity"
COFITOK_CLASS_FIDELITY="$CLASS_FIDELITY_ROOT/cofitok"
DENSE_CLASS_FIDELITY="$CLASS_FIDELITY_ROOT/dense"
QUALIFICATION="$CLASS_FIDELITY_ROOT/qualification_report.json"
STAGE_STATE_ROOT="$CLASS_FIDELITY_ROOT/stage_receipts"
CLASS_FIDELITY_LOCK="$OUTPUT_ROOT/frozen_posteval_class_fidelity.lock"

cd "$PROJECT"
export PYTHONPATH=src
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CLASS_FIDELITY_REVISION" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_CLASS_FIDELITY_BRANCH" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
for path in \
  "$POSTEVAL_STATUS" \
  "$SUPPLEMENTAL_WAITER" \
  "$SUPPLEMENTAL_REPORT" \
  "$PROMOTION_GATE" \
  "$COFITOK_SAMPLES/sampling_manifest.json" \
  "$COFITOK_SAMPLES/sampling_progress.json" \
  "$COFITOK_SAMPLES/sampling_report.json" \
  "$DENSE_SAMPLES/sampling_manifest.json" \
  "$DENSE_SAMPLES/sampling_progress.json" \
  "$DENSE_SAMPLES/sampling_report.json" \
  "$CLASSIFIER_CHECKPOINT"; do
  [[ -f "$path" ]]
done
[[ -d "$COFITOK_SAMPLES/prefix_8" ]]
[[ -d "$DENSE_SAMPLES/prefix_1" ]]
[[ "$(stat -c '%s' "$CLASSIFIER_CHECKPOINT")" == "102540417" ]]
[[ "$(sha256sum "$CLASSIFIER_CHECKPOINT" | awk '{print $1}')" == "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca" ]]

mkdir -p "$CLASS_FIDELITY_ROOT" "$STAGE_STATE_ROOT"
command -v flock >/dev/null
exec 6>"$CLASS_FIDELITY_LOCK"
if ! flock -n 6; then
  printf 'refusing concurrent frozen class-fidelity evaluation\n' >&2
  exit 15
fi

"$PYTHON" - \
  "$POSTEVAL_STATUS" \
  "$SUPPLEMENTAL_WAITER" \
  "$SUPPLEMENTAL_REPORT" \
  "$EXPECTED_TRAINING_REVISION" \
  "$EXPECTED_TRAINING_BRANCH" \
  "$EXPECTED_FROZEN_EVALUATION_REVISION" \
  "$EXPECTED_FROZEN_EVALUATION_BRANCH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

posteval = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
supplemental = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
supplemental_report_path = Path(sys.argv[3]).resolve()
training_revision, training_branch = sys.argv[4:6]
evaluation_revision, evaluation_branch = sys.argv[6:8]
posteval_expected = {
    "training_revision": training_revision,
    "training_branch": training_branch,
    "evaluation_revision": evaluation_revision,
    "evaluation_branch": evaluation_branch,
    "formal_300k_allowed": False,
}
supplemental_expected = supplemental.get("expected")
supplemental_required = {
    "training_revision": training_revision,
    "training_branch": training_branch,
    "frozen_evaluation_revision": evaluation_revision,
    "frozen_evaluation_branch": evaluation_branch,
    "supplemental_non_authorizing": True,
    "full_training_launch_allowed": False,
}
nested = supplemental.get("supplemental")
source = nested.get("source") if isinstance(nested, dict) else None
digest = hashlib.sha256(supplemental_report_path.read_bytes()).hexdigest()
physical_source = {
    "path": supplemental_report_path.as_posix(),
    "bytes": supplemental_report_path.stat().st_size,
    "sha256": digest,
}
if (
    posteval.get("schema_version") != 1
    or posteval.get("role") != "generation_stability_50k_posteval_waiter"
    or posteval.get("status") != "pass"
    or posteval.get("detail") != "formal_ema_postevaluation_completed"
    or posteval.get("child_exit_code") != 0
    or posteval.get("expected") != posteval_expected
    or supplemental.get("schema_version") != 1
    or supplemental.get("role")
    != "generation_stability_frozen_supplemental_waiter"
    or supplemental.get("status") != "pass"
    or supplemental.get("child_exit_code") != 0
    or supplemental.get("full_training_launch_allowed") is not False
    or supplemental.get("supplemental_non_authorizing") is not True
    or not isinstance(supplemental_expected, dict)
    or any(
        supplemental_expected.get(name) != value
        for name, value in supplemental_required.items()
    )
    or not isinstance(nested, dict)
    or nested.get("status") != "pass"
    or nested.get("supplemental_non_authorizing") is not True
    or nested.get("full_training_launch_allowed") is not False
    or source != physical_source
):
    raise SystemExit("frozen class fidelity requires terminal post-eval and supplemental coordination")
PY

"$PYTHON" scripts/validate_generation_gate_report.py \
  --gate "$PROMOTION_GATE" \
  --stage scaling >/dev/null

SUPPLEMENTAL_SHA256="$(sha256sum "$SUPPLEMENTAL_REPORT" | awk '{print $1}')"
"$PYTHON" scripts/verify_generation_stability_frozen_supplemental.py \
  --report "$SUPPLEMENTAL_REPORT" \
  --expected-report-sha256 "$SUPPLEMENTAL_SHA256" \
  --promotion-gate "$PROMOTION_GATE" >/dev/null

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing frozen class-fidelity evaluation while the GPU is busy\n' >&2
  exit 9
fi

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/cofitok.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_SAMPLES/sampling_manifest.json" \
  --input-file "$COFITOK_SAMPLES/sampling_progress.json" \
  --input-file "$COFITOK_SAMPLES/sampling_report.json" \
  --input-file "$CLASSIFIER_CHECKPOINT" \
  --input-tree "$COFITOK_SAMPLES/prefix_8" \
  --output-tree "$COFITOK_CLASS_FIDELITY" \
  -- \
  "$PYTHON" scripts/evaluate_generation_class_fidelity.py \
  --generated-dir "$COFITOK_SAMPLES/prefix_8" \
  --sampling-report "$COFITOK_SAMPLES/sampling_report.json" \
  --output-dir "$COFITOK_CLASS_FIDELITY" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --batch-size 64 \
  --min-samples 10000 \
  --resume

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/dense.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$DENSE_SAMPLES/sampling_manifest.json" \
  --input-file "$DENSE_SAMPLES/sampling_progress.json" \
  --input-file "$DENSE_SAMPLES/sampling_report.json" \
  --input-file "$CLASSIFIER_CHECKPOINT" \
  --input-tree "$DENSE_SAMPLES/prefix_1" \
  --output-tree "$DENSE_CLASS_FIDELITY" \
  -- \
  "$PYTHON" scripts/evaluate_generation_class_fidelity.py \
  --generated-dir "$DENSE_SAMPLES/prefix_1" \
  --sampling-report "$DENSE_SAMPLES/sampling_report.json" \
  --output-dir "$DENSE_CLASS_FIDELITY" \
  --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
  --batch-size 64 \
  --min-samples 10000 \
  --resume

"$PYTHON" scripts/run_generation_stage_once.py \
  --state "$STAGE_STATE_ROOT/qualification.json" \
  --project "$PROJECT" \
  --cwd "$PROJECT" \
  --input-file "$COFITOK_CLASS_FIDELITY/class_fidelity_report.json" \
  --input-file "$DENSE_CLASS_FIDELITY/class_fidelity_report.json" \
  --output-file "$QUALIFICATION" \
  -- \
  "$PYTHON" scripts/build_generation_class_fidelity_qualification.py \
  --cofitok-report "$COFITOK_CLASS_FIDELITY/class_fidelity_report.json" \
  --dense-report "$DENSE_CLASS_FIDELITY/class_fidelity_report.json" \
  --output "$QUALIFICATION" \
  --stage scaling \
  --expected-revision "$EXPECTED_CLASS_FIDELITY_REVISION" \
  --expected-branch "$EXPECTED_CLASS_FIDELITY_BRANCH" \
  --min-top1 0.01 \
  --min-top5 0.05 \
  --min-predicted-class-fraction 0.25 \
  --min-normalized-predicted-entropy 0.50 \
  --max-top1-regression 0.05 \
  --max-top5-regression 0.05 \
  --allow-hold

QUALIFICATION_SHA256="$(sha256sum "$QUALIFICATION" | awk '{print $1}')"
"$PYTHON" scripts/verify_generation_stability_frozen_class_fidelity.py \
  --report "$QUALIFICATION" \
  --expected-report-sha256 "$QUALIFICATION_SHA256" \
  --promotion-gate "$PROMOTION_GATE" \
  --expected-evaluator-revision "$EXPECTED_CLASS_FIDELITY_REVISION" \
  --expected-evaluator-branch "$EXPECTED_CLASS_FIDELITY_BRANCH" >/dev/null

printf 'frozen scaling class fidelity: %s  %s\n' \
  "$QUALIFICATION_SHA256" "$QUALIFICATION"
