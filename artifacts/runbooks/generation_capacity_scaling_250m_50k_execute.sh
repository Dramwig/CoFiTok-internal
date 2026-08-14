#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact capacity-scaling execution checkout}
DECISION_PROJECT=${DECISION_PROJECT:?set DECISION_PROJECT to the exact decision checkout}
TRAINING_PROJECT=${TRAINING_PROJECT:?set TRAINING_PROJECT to the exact capacity-probe training checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set STANDING_AUTHORIZATION}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}
EXPECTED_DECISION_SHA256=${EXPECTED_DECISION_SHA256:?set EXPECTED_DECISION_SHA256}
EXPECTED_CAPACITY_RESULT_SHA256=${EXPECTED_CAPACITY_RESULT_SHA256:?set EXPECTED_CAPACITY_RESULT_SHA256}
EXPECTED_CAPACITY_LAUNCH_SHA256=${EXPECTED_CAPACITY_LAUNCH_SHA256:?set EXPECTED_CAPACITY_LAUNCH_SHA256}
EXPECTED_DECISION_REVISION=${EXPECTED_DECISION_REVISION:?set EXPECTED_DECISION_REVISION}
EXPECTED_DECISION_TREE=${EXPECTED_DECISION_TREE:?set EXPECTED_DECISION_TREE}
EXPECTED_DECISION_BRANCH=${EXPECTED_DECISION_BRANCH:?set EXPECTED_DECISION_BRANCH}
EXPECTED_EXECUTION_REVISION=${EXPECTED_EXECUTION_REVISION:?set EXPECTED_EXECUTION_REVISION}
EXPECTED_EXECUTION_TREE=${EXPECTED_EXECUTION_TREE:?set EXPECTED_EXECUTION_TREE}
EXPECTED_EXECUTION_BRANCH=${EXPECTED_EXECUTION_BRANCH:?set EXPECTED_EXECUTION_BRANCH}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_LAUNCH_RECEIPT_SHA256=${EXPECTED_LAUNCH_RECEIPT_SHA256:-}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
DECISION="$REPORT_ROOT/capacity_scaling_decision.json"
CAPACITY_RESULT="$REPORT_ROOT/capacity_probe_result.json"
CAPACITY_LAUNCH="$REPORT_ROOT/launch_receipt.json"
STORAGE="$REPORT_ROOT/storage_capacity_50k_launch.json"
LAUNCH_RECEIPT="$REPORT_ROOT/capacity_scaling_50k_launch_receipt.json"
EXECUTION_STATUS="$REPORT_ROOT/capacity_scaling_50k_execution_status.json"
LOCK="$OUTPUT_ROOT/capacity_scaling_50k_execution.lock"
COFITOK_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
DENSE_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rollout_x0_u2_ema_teacher_dense_100k.json"
COFITOK_RUN="$OUTPUT_ROOT/base256_cofitok"
DENSE_RUN="$OUTPUT_ROOT/base256_dense_identity"
VALIDATION_ROOT="$REPORT_ROOT/capacity_scaling_50k/training"
COFITOK_VALIDATION="$VALIDATION_ROOT/cofitok.json"
DENSE_VALIDATION="$VALIDATION_ROOT/dense_identity.json"
MILESTONE="$REPORT_ROOT/capacity_scaling_50k/milestone_step_00050000.json"
REAL_DATA=${REAL_DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_EXECUTION_REVISION" ]]
[[ "$(git rev-parse 'HEAD^{tree}')" == "$EXPECTED_EXECUTION_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_EXECUTION_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ "$(git -C "$DECISION_PROJECT" rev-parse HEAD)" == "$EXPECTED_DECISION_REVISION" ]]
[[ "$(git -C "$DECISION_PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_DECISION_TREE" ]]
[[ "$(git -C "$DECISION_PROJECT" branch --show-current)" == "$EXPECTED_DECISION_BRANCH" ]]
[[ -z "$(git -C "$DECISION_PROJECT" status --porcelain)" ]]
[[ "$(git -C "$TRAINING_PROJECT" rev-parse HEAD)" == "$EXPECTED_TRAINING_REVISION" ]]
[[ "$(git -C "$TRAINING_PROJECT" rev-parse 'HEAD^{tree}')" == "$EXPECTED_TRAINING_TREE" ]]
[[ "$(git -C "$TRAINING_PROJECT" branch --show-current)" == "$EXPECTED_TRAINING_BRANCH" ]]
[[ -z "$(git -C "$TRAINING_PROJECT" status --porcelain)" ]]
[[ -f "$DECISION" ]]
[[ -f "$CAPACITY_RESULT" ]]
[[ -f "$CAPACITY_LAUNCH" ]]
[[ -f "$STANDING_AUTHORIZATION" ]]
[[ -f "$COFITOK_CONFIG" ]]
[[ -f "$DENSE_CONFIG" ]]
[[ -d "$REAL_DATA" ]]
[[ "$(sha256sum "$DECISION" | awk '{print $1}')" == "$EXPECTED_DECISION_SHA256" ]]
[[ "$(sha256sum "$CAPACITY_RESULT" | awk '{print $1}')" == "$EXPECTED_CAPACITY_RESULT_SHA256" ]]
[[ "$(sha256sum "$CAPACITY_LAUNCH" | awk '{print $1}')" == "$EXPECTED_CAPACITY_LAUNCH_SHA256" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
mkdir -p "$VALIDATION_ROOT" "$(dirname "$MILESTONE")"
exec 7>"$LOCK"
flock -n 7 || {
  printf 'refusing concurrent capacity-scaling 50K execution\n' >&2
  exit 75
}

CURRENT_STAGE=initialization
completed=false
write_status() {
  local state="$1"
  local detail="$2"
  local exit_code="${3:-}"
  "$PYTHON" - "$EXECUTION_STATUS" "$state" "$detail" "$exit_code" \
    "$CURRENT_STAGE" "$EXPECTED_EXECUTION_REVISION" "$EXPECTED_EXECUTION_TREE" \
    "$EXPECTED_EXECUTION_BRANCH" "$$" "$LAUNCH_RECEIPT" "$MILESTONE" \
    "$COFITOK_VALIDATION" "$DENSE_VALIDATION" <<'PY'
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

def identity(raw):
    path=Path(raw)
    if not path.is_file(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return {'path':path.resolve().as_posix(),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

path=Path(sys.argv[1])
payload={
 'schema_version':1,
 'role':'stability_full_data_capacity_scaling_50k_execution',
 'status':sys.argv[2],
 'detail':sys.argv[3],
 'exit_code':int(sys.argv[4]) if sys.argv[4] else None,
 'stage':sys.argv[5],
 'git':{'revision':sys.argv[6],'tree':sys.argv[7],'branch':sys.argv[8],'tracked_dirty':False},
 'pid':int(sys.argv[9]),
 'hostname':socket.gethostname(),
 'updated_at':datetime.now(timezone.utc).isoformat(),
 'launch_receipt':identity(sys.argv[10]),
 'milestone':identity(sys.argv[11]),
 'training_validations':{'cofitok':identity(sys.argv[12]),'dense_identity':identity(sys.argv[13])},
 'authorization_boundary':{
   'matched_250m_resume_to_50000_allowed':True,
   'configured_100k_completion_allowed':False,
   'full_300k_launch_allowed':False,
   'promotion_or_release_allowed':False,
 },
}
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_name(f'{path.name}.tmp.{os.getpid()}')
with tmp.open('w',encoding='utf-8',newline='\n') as f:
 json.dump(payload,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
tmp.replace(path)
PY
}

on_exit() {
  local code=$?
  if [[ "$completed" != true && $code -ne 0 ]]; then
    write_status failed "capacity scaling failed at $CURRENT_STAGE" "$code" || true
  fi
}
trap on_exit EXIT
set_stage() {
  CURRENT_STAGE="$1"
  write_status running "$2"
}

set_stage decision_replay "replaying the exact source-compatible capacity decision"
(
  cd "$DECISION_PROJECT"
  export PYTHONPATH="$DECISION_PROJECT:$DECISION_PROJECT/src"
  "$PYTHON" scripts/verify_generation_capacity_scaling_decision.py \
    --capacity-probe-result "$CAPACITY_RESULT" \
    --expected-capacity-probe-result-sha256 "$EXPECTED_CAPACITY_RESULT_SHA256" \
    --standing-authorization "$STANDING_AUTHORIZATION" \
    --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
    --expected-capacity-revision "$EXPECTED_TRAINING_REVISION" \
    --expected-capacity-branch "$EXPECTED_TRAINING_BRANCH" \
    --expected-decision-revision "$EXPECTED_DECISION_REVISION" \
    --expected-decision-branch "$EXPECTED_DECISION_BRANCH" \
    --decision "$DECISION" \
    --expected-decision-sha256 "$EXPECTED_DECISION_SHA256" >/dev/null
)
"$PYTHON" - "$DECISION" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
if (
 d.get('execution_authorization',{}).get('matched_250m_resume_allowed') is not True
 or d.get('execution_authorization',{}).get('stop_after_step') != 50000
 or d.get('execution_authorization',{}).get('configured_100k_completion_allowed') is not False
): raise SystemExit('capacity decision does not authorize the exact 50K segment')
PY

receipt_args=(
  --decision "$DECISION"
  --expected-decision-sha256 "$EXPECTED_DECISION_SHA256"
  --capacity-probe-result "$CAPACITY_RESULT"
  --expected-capacity-probe-result-sha256 "$EXPECTED_CAPACITY_RESULT_SHA256"
  --capacity-probe-launch-receipt "$CAPACITY_LAUNCH"
  --expected-capacity-probe-launch-receipt-sha256 "$EXPECTED_CAPACITY_LAUNCH_SHA256"
  --standing-authorization "$STANDING_AUTHORIZATION"
  --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256"
  --cofitok-config "$COFITOK_CONFIG"
  --dense-config "$DENSE_CONFIG"
  --storage-capacity "$STORAGE"
  --training-project "$TRAINING_PROJECT"
  --output-root "$OUTPUT_ROOT"
  --storage-path "$CHECKPOINT_ROOT"
  --expected-capacity-revision "$EXPECTED_TRAINING_REVISION"
  --expected-capacity-branch "$EXPECTED_TRAINING_BRANCH"
  --expected-decision-revision "$EXPECTED_DECISION_REVISION"
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH"
  --expected-execution-revision "$EXPECTED_EXECUTION_REVISION"
  --expected-execution-tree "$EXPECTED_EXECUTION_TREE"
  --expected-execution-branch "$EXPECTED_EXECUTION_BRANCH"
  --expected-training-revision "$EXPECTED_TRAINING_REVISION"
  --expected-training-tree "$EXPECTED_TRAINING_TREE"
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH"
)

if [[ -f "$LAUNCH_RECEIPT" ]]; then
  set_stage launch_receipt_replay "replaying immutable capacity-scaling launch receipt"
  observed_receipt_sha=$(sha256sum "$LAUNCH_RECEIPT" | awk '{print $1}')
  if [[ -n "$EXPECTED_LAUNCH_RECEIPT_SHA256" \
    && "$observed_receipt_sha" != "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'capacity-scaling launch receipt SHA256 differs\n' >&2
    exit 10
  fi
  storage_sha=$(sha256sum "$STORAGE" | awk '{print $1}')
  "$PYTHON" scripts/verify_generation_capacity_scaling_launch_receipt.py \
    "${receipt_args[@]}" \
    --expected-storage-capacity-sha256 "$storage_sha" \
    --receipt "$LAUNCH_RECEIPT" \
    --expected-receipt-sha256 "$observed_receipt_sha" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256="$observed_receipt_sha"
else
  set_stage launch_guard "checking GPU idleness and duplicate training processes"
  [[ -z "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]
  if pgrep -af '[s]cripts/train_generation.py.*stability_capacity_probe.*100k' >/dev/null; then
    printf 'refusing duplicate capacity-scaling trainer\n' >&2
    exit 12
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    printf 'refusing capacity-scaling launch while GPU is busy\n' >&2
    exit 9
  fi
  set_stage storage_preflight "checking storage for the exact 50K segment"
  "$PYTHON" scripts/check_generation_storage_capacity.py \
    --path "$CHECKPOINT_ROOT" \
    --output "$STORAGE" \
    --stage stability_full_data_capacity_scaling_250m_50k_execution \
    --reference-checkpoint "$COFITOK_RUN/checkpoint_step_00010000.pt" \
    --reference-checkpoint "$DENSE_RUN/checkpoint_step_00010000.pt" \
    --checkpoint-count 6 \
    --checkpoint-size-multiplier 1.0 \
    --sample-count 4096 \
    --estimated-sample-kib 256 \
    --additional-gib 8 \
    --safety-margin-gib 64 >/dev/null
  storage_sha=$(sha256sum "$STORAGE" | awk '{print $1}')
  set_stage launch_receipt_build "building immutable capacity-scaling launch receipt"
  "$PYTHON" scripts/build_generation_capacity_scaling_launch_receipt.py \
    "${receipt_args[@]}" \
    --expected-storage-capacity-sha256 "$storage_sha" \
    --output "$LAUNCH_RECEIPT" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256=$(sha256sum "$LAUNCH_RECEIPT" | awk '{print $1}')
  "$PYTHON" scripts/verify_generation_capacity_scaling_launch_receipt.py \
    "${receipt_args[@]}" \
    --expected-storage-capacity-sha256 "$storage_sha" \
    --receipt "$LAUNCH_RECEIPT" \
    --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
fi

read -r MICRO_BATCH ACCUMULATION < <(
  "$PYTHON" - "$CAPACITY_LAUNCH" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())['runtime_selection']
print(r['micro_batch_size'],r['gradient_accumulation_steps'])
PY
)
(( MICRO_BATCH * ACCUMULATION == 64 ))

latest_step() {
  "$PYTHON" - "$1/latest.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
print(int(json.loads(p.read_text())['step']) if p.is_file() else 0)
PY
}

verify_or_build_training_validation() {
  local report="$1" config="$2" parameters="$3" validation="$4"
  if [[ -f "$validation" ]]; then
    local sha
    sha=$(sha256sum "$validation" | awk '{print $1}')
    "$PYTHON" scripts/verify_generation_capacity_scaling_training.py \
      --validation "$validation" --expected-validation-sha256 "$sha" \
      --training-report "$report" --config "$config" \
      --expected-revision "$EXPECTED_TRAINING_REVISION" \
      --expected-branch "$EXPECTED_TRAINING_BRANCH" \
      --expected-parameter-count "$parameters" \
      --expected-micro-batch-size "$MICRO_BATCH" \
      --expected-gradient-accumulation-steps "$ACCUMULATION" >/dev/null
  else
    "$PYTHON" scripts/validate_generation_capacity_scaling_training.py \
      --training-report "$report" --config "$config" \
      --expected-revision "$EXPECTED_TRAINING_REVISION" \
      --expected-branch "$EXPECTED_TRAINING_BRANCH" \
      --expected-parameter-count "$parameters" \
      --expected-micro-batch-size "$MICRO_BATCH" \
      --expected-gradient-accumulation-steps "$ACCUMULATION" \
      --output "$validation" >/dev/null
  fi
}

train_to_50k() {
  local method="$1" config="$2" run_dir="$3" parameters="$4" validation="$5"
  local current
  current=$(latest_step "$run_dir")
  if (( current < 10000 || current > 50000 )); then
    printf '%s capacity-scaling current step is invalid: %s\n' "$method" "$current" >&2
    exit 14
  fi
  if (( current < 50000 )); then
    local delta=$((50000-current))
    (
      cd "$TRAINING_PROJECT"
      export PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src"
      "$PYTHON" scripts/train_generation.py \
        --config "$config" \
        --output-dir "$run_dir" \
        --micro-batch-size "$MICRO_BATCH" \
        --gradient-accumulation-steps "$ACCUMULATION" \
        --stop-after-steps "$delta" \
        --resume auto
    )
  fi
  [[ "$(latest_step "$run_dir")" == 50000 ]]
  verify_or_build_training_validation \
    "$run_dir/training_report.json" "$config" "$parameters" "$validation"
}

evaluate_50k() {
  local method="$1" run_dir="$2" prefix="$3" random_orders="$4"
  local checkpoint="$run_dir/checkpoint_step_00050000.pt"
  PROJECT="$TRAINING_PROJECT" PYTHON="$PYTHON" OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    DATA="$REAL_DATA" bash "$TRAINING_PROJECT/artifacts/runbooks/generation_full_milestone_eval.sh" \
      "$method" "$run_dir" "$checkpoint" 50000 "$prefix" "$random_orders"
}

milestone_valid() {
  [[ -f "$MILESTONE" ]] || return 1
  "$PYTHON" scripts/validate_generation_milestone_report.py \
    --report "$MILESTONE" --expected-step 50000 \
    --source-profile capacity_scaling >/dev/null
}

if ! milestone_valid; then
  set_stage cofitok_training "resuming exact base256 CoFiTok trajectory to step 50K"
  train_to_50k cofitok "$COFITOK_CONFIG" "$COFITOK_RUN" 250153763 "$COFITOK_VALIDATION"
  set_stage cofitok_evaluation "evaluating base256 CoFiTok at step 50K"
  evaluate_50k cofitok "$COFITOK_RUN" 8 4

  set_stage dense_training "resuming exact base256 dense trajectory to step 50K"
  train_to_50k dense_identity "$DENSE_CONFIG" "$DENSE_RUN" 250135043 "$DENSE_VALIDATION"
  set_stage dense_evaluation "evaluating base256 dense at step 50K"
  evaluate_50k dense_identity "$DENSE_RUN" 1 0

  set_stage milestone_build "building the paired non-claim capacity-scaling milestone"
  [[ ! -e "$MILESTONE" ]]
  "$PYTHON" scripts/build_generation_milestone_report.py \
    --cofitok-generation "$COFITOK_RUN/milestones/step_00050000/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --dense-generation "$DENSE_RUN/milestones/step_00050000/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
    --cofitok-checkpoint-eval "$COFITOK_RUN/milestones/step_00050000/checkpoint_eval/checkpoint_evaluation_report.json" \
    --dense-checkpoint-eval "$DENSE_RUN/milestones/step_00050000/checkpoint_eval/checkpoint_evaluation_report.json" \
    --milestone-step 50000 --expected-samples 2048 \
    --source-profile capacity_scaling --output "$MILESTONE" >/dev/null
fi

set_stage completion_replay "replaying the complete paired 50K milestone"
storage_sha=$(sha256sum "$STORAGE" | awk '{print $1}')
"$PYTHON" scripts/verify_generation_capacity_scaling_launch_receipt.py \
  "${receipt_args[@]}" \
  --expected-storage-capacity-sha256 "$storage_sha" \
  --receipt "$LAUNCH_RECEIPT" \
  --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
verify_or_build_training_validation \
  "$COFITOK_RUN/training_report.json" "$COFITOK_CONFIG" 250153763 "$COFITOK_VALIDATION"
verify_or_build_training_validation \
  "$DENSE_RUN/training_report.json" "$DENSE_CONFIG" 250135043 "$DENSE_VALIDATION"
milestone_valid
CURRENT_STAGE=complete
write_status completed "matched 250M step-50K milestone completed; new decision required"
completed=true
printf 'capacity-scaling 50K launch receipt: %s  %s\n' \
  "$EXPECTED_LAUNCH_RECEIPT_SHA256" "$LAUNCH_RECEIPT"
printf 'capacity-scaling 50K milestone: %s\n' "$MILESTONE"
