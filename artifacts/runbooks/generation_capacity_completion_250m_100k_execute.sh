#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT to the exact capacity-completion execution checkout}
DECISION_PROJECT=${DECISION_PROJECT:?set DECISION_PROJECT to the exact completion-decision checkout}
TRAINING_PROJECT=${TRAINING_PROJECT:?set TRAINING_PROJECT to the exact capacity-probe training checkout}
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
STANDING_AUTHORIZATION=${STANDING_AUTHORIZATION:?set STANDING_AUTHORIZATION}
EXPECTED_STANDING_AUTHORIZATION_SHA256=${EXPECTED_STANDING_AUTHORIZATION_SHA256:?set EXPECTED_STANDING_AUTHORIZATION_SHA256}
EXPECTED_COMPLETION_DECISION_SHA256=${EXPECTED_COMPLETION_DECISION_SHA256:?set EXPECTED_COMPLETION_DECISION_SHA256}
EXPECTED_SCALING_RESULT_SHA256=${EXPECTED_SCALING_RESULT_SHA256:?set EXPECTED_SCALING_RESULT_SHA256}
EXPECTED_DECISION_REVISION=${EXPECTED_DECISION_REVISION:?set EXPECTED_DECISION_REVISION}
EXPECTED_DECISION_TREE=${EXPECTED_DECISION_TREE:?set EXPECTED_DECISION_TREE}
EXPECTED_DECISION_BRANCH=${EXPECTED_DECISION_BRANCH:?set EXPECTED_DECISION_BRANCH}
EXPECTED_SCALING_DECISION_REVISION=${EXPECTED_SCALING_DECISION_REVISION:?set EXPECTED_SCALING_DECISION_REVISION}
EXPECTED_SCALING_DECISION_BRANCH=${EXPECTED_SCALING_DECISION_BRANCH:?set EXPECTED_SCALING_DECISION_BRANCH}
EXPECTED_SCALING_EXECUTION_REVISION=${EXPECTED_SCALING_EXECUTION_REVISION:?set EXPECTED_SCALING_EXECUTION_REVISION}
EXPECTED_SCALING_EXECUTION_TREE=${EXPECTED_SCALING_EXECUTION_TREE:?set EXPECTED_SCALING_EXECUTION_TREE}
EXPECTED_SCALING_EXECUTION_BRANCH=${EXPECTED_SCALING_EXECUTION_BRANCH:?set EXPECTED_SCALING_EXECUTION_BRANCH}
EXPECTED_EXECUTION_REVISION=${EXPECTED_EXECUTION_REVISION:?set EXPECTED_EXECUTION_REVISION}
EXPECTED_EXECUTION_TREE=${EXPECTED_EXECUTION_TREE:?set EXPECTED_EXECUTION_TREE}
EXPECTED_EXECUTION_BRANCH=${EXPECTED_EXECUTION_BRANCH:?set EXPECTED_EXECUTION_BRANCH}
EXPECTED_TRAINING_REVISION=${EXPECTED_TRAINING_REVISION:?set EXPECTED_TRAINING_REVISION}
EXPECTED_TRAINING_TREE=${EXPECTED_TRAINING_TREE:?set EXPECTED_TRAINING_TREE}
EXPECTED_TRAINING_BRANCH=${EXPECTED_TRAINING_BRANCH:?set EXPECTED_TRAINING_BRANCH}
EXPECTED_LAUNCH_RECEIPT_SHA256=${EXPECTED_LAUNCH_RECEIPT_SHA256:-}

OUTPUT_ROOT="$CHECKPOINT_ROOT/stability_full_data_100k_capacity_probe_250m_10k_v1"
REPORT_ROOT="$OUTPUT_ROOT/reports"
CAPACITY_RESULT="$REPORT_ROOT/capacity_probe_result.json"
SCALING_DECISION="$REPORT_ROOT/capacity_scaling_decision.json"
SCALING_LAUNCH="$REPORT_ROOT/capacity_scaling_50k_launch_receipt.json"
SCALING_STATUS="$REPORT_ROOT/capacity_scaling_50k_execution_status.json"
SCALING_COFITOK_VALIDATION="$REPORT_ROOT/capacity_scaling_50k/training/cofitok.json"
SCALING_DENSE_VALIDATION="$REPORT_ROOT/capacity_scaling_50k/training/dense_identity.json"
MILESTONE_50000="$REPORT_ROOT/capacity_scaling_50k/milestone_step_00050000.json"
SCALING_RESULT="$REPORT_ROOT/capacity_scaling_50k_result.json"
COMPLETION_DECISION="$REPORT_ROOT/capacity_completion_100k_decision.json"
COMPLETION_ROOT="$REPORT_ROOT/capacity_completion_100k"
SOURCE_ARCHIVE="$COMPLETION_ROOT/source_checkpoint_archive.json"
STORAGE="$COMPLETION_ROOT/storage_capacity.json"
LAUNCH_RECEIPT="$COMPLETION_ROOT/launch_receipt.json"
EXECUTION_STATUS="$COMPLETION_ROOT/execution_status.json"
COFITOK_VALIDATION="$COMPLETION_ROOT/training/cofitok.json"
DENSE_VALIDATION="$COMPLETION_ROOT/training/dense_identity.json"
MILESTONE_100000="$COMPLETION_ROOT/milestone_step_00100000.json"
CLASS_FIDELITY_QUALIFICATION="$COMPLETION_ROOT/class_fidelity/qualification_report.json"
LOCK="$OUTPUT_ROOT/capacity_completion_100k_execution.lock"
COFITOK_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
DENSE_CONFIG="$TRAINING_PROJECT/configs/generation/imagenet256_stability_capacity_probe_rollout_x0_u2_ema_teacher_dense_100k.json"
COFITOK_RUN="$OUTPUT_ROOT/base256_cofitok"
DENSE_RUN="$OUTPUT_ROOT/base256_dense_identity"
REAL_DATA=${REAL_DATA:-/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val}
EVAL_CACHE=${EVAL_CACHE:-$CHECKPOINT_ROOT/evaluation_cache}

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
for source in \
  "$CAPACITY_RESULT" "$SCALING_DECISION" "$SCALING_LAUNCH" \
  "$SCALING_STATUS" "$SCALING_COFITOK_VALIDATION" \
  "$SCALING_DENSE_VALIDATION" "$MILESTONE_50000" "$SCALING_RESULT" \
  "$COMPLETION_DECISION" "$STANDING_AUTHORIZATION" "$COFITOK_CONFIG" \
  "$DENSE_CONFIG"; do
  [[ -f "$source" ]]
done
[[ -d "$REAL_DATA" ]]
[[ "$(sha256sum "$STANDING_AUTHORIZATION" | awk '{print $1}')" == "$EXPECTED_STANDING_AUTHORIZATION_SHA256" ]]
[[ "$(sha256sum "$SCALING_RESULT" | awk '{print $1}')" == "$EXPECTED_SCALING_RESULT_SHA256" ]]
[[ "$(sha256sum "$COMPLETION_DECISION" | awk '{print $1}')" == "$EXPECTED_COMPLETION_DECISION_SHA256" ]]
mkdir -p "$COMPLETION_ROOT/training" "$(dirname "$CLASS_FIDELITY_QUALIFICATION")"
exec 7>"$LOCK"
flock -n 7 || {
  printf 'refusing concurrent capacity-completion 100K execution\n' >&2
  exit 75
}

CURRENT_STAGE=initialization
completed=false
write_status() {
  local state="$1" detail="$2" exit_code="${3:-}"
  "$PYTHON" - "$EXECUTION_STATUS" "$state" "$detail" "$exit_code" \
    "$CURRENT_STAGE" "$EXPECTED_EXECUTION_REVISION" "$EXPECTED_EXECUTION_TREE" \
    "$EXPECTED_EXECUTION_BRANCH" "$$" "$OUTPUT_ROOT" <<'PY'
import hashlib,json,os,socket,sys
from datetime import datetime,timezone
from pathlib import Path

def identity(path):
    path=Path(path)
    if not path.is_file(): return None
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return {'path':path.resolve().as_posix(),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}

path=Path(sys.argv[1]); state=sys.argv[2]; root=Path(sys.argv[10]); reports=root/'reports'; completion=reports/'capacity_completion_100k'
methods={}
for method,run_name in (('cofitok','base256_cofitok'),('dense_identity','base256_dense_identity')):
    run=root/run_name; terminal=run/'terminal_100k'; samples=terminal/'samples_10000_ddim100_cfg15'
    methods[method]={
      'training_validation':identity(completion/'training'/f"{'cofitok' if method=='cofitok' else 'dense_identity'}.json"),
      'sampling_preflight':identity(terminal/'sampling_preflight.json'),
      'generation_metrics':identity(samples/'metrics/generation_metrics_report.json'),
      'checkpoint_evaluation':identity(terminal/'checkpoint_eval/checkpoint_evaluation_report.json'),
      'class_fidelity':identity(samples/'class_fidelity/class_fidelity_report.json'),
    }
payload={
 'schema_version':1,
 'role':'stability_full_data_capacity_completion_100k_execution',
 'status':state,
 'detail':sys.argv[3],
 'exit_code':int(sys.argv[4]) if sys.argv[4] else None,
 'stage':sys.argv[5],
 'git':{'revision':sys.argv[6],'tree':sys.argv[7],'branch':sys.argv[8],'tracked_dirty':False},
 'pid':int(sys.argv[9]),
 'hostname':socket.gethostname(),
 'updated_at':datetime.now(timezone.utc).isoformat(),
 'output_root':root.resolve().as_posix(),
 'launch_receipt':identity(completion/'launch_receipt.json'),
 'source_checkpoint_archive':identity(completion/'source_checkpoint_archive.json'),
 'milestone_100000':identity(completion/'milestone_step_00100000.json'),
 'class_fidelity_qualification':identity(completion/'class_fidelity/qualification_report.json'),
 'methods':methods,
 'authorization_boundary':{
   'matched_250m_resume_to_100000_allowed':True,
   'configured_100k_training_complete':state=='completed',
   'terminal_evidence_complete':state=='completed',
   'additional_training_allowed':False,
   'full_300k_launch_allowed':False,
   'report_is_promotion_gate':False,
   'formal_generation_claim_allowed':False,
   'release_authorization_allowed':False,
   'new_source_compatible_result_required':True,
 },
}
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
os.replace(tmp,path)
PY
}
on_exit() {
  local code=$?
  if [[ "$completed" != true && $code -ne 0 ]]; then
    write_status failed "capacity completion failed at $CURRENT_STAGE" "$code" || true
  fi
}
trap on_exit EXIT
set_stage() {
  CURRENT_STAGE="$1"
  write_status running "$2"
}
sha() { sha256sum "$1" | awk '{print $1}'; }

archive_args=(
  --decision "$COMPLETION_DECISION"
  --expected-decision-sha256 "$EXPECTED_COMPLETION_DECISION_SHA256"
  --expected-decision-revision "$EXPECTED_DECISION_REVISION"
  --expected-decision-tree "$EXPECTED_DECISION_TREE"
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH"
  --output-root "$OUTPUT_ROOT"
)
if [[ -f "$SOURCE_ARCHIVE" ]]; then
  archive_sha=$(sha "$SOURCE_ARCHIVE")
  "$PYTHON" scripts/restore_generation_capacity_completion_sources.py \
    --archive "$SOURCE_ARCHIVE" --expected-archive-sha256 "$archive_sha" >/dev/null
fi

set_stage decision_replay "physically replaying the exact capacity-completion decision"
capacity_result_sha=$(sha "$CAPACITY_RESULT")
scaling_decision_sha=$(sha "$SCALING_DECISION")
scaling_launch_sha=$(sha "$SCALING_LAUNCH")
scaling_status_sha=$(sha "$SCALING_STATUS")
scaling_cofitok_sha=$(sha "$SCALING_COFITOK_VALIDATION")
scaling_dense_sha=$(sha "$SCALING_DENSE_VALIDATION")
milestone_50000_sha=$(sha "$MILESTONE_50000")
(
  cd "$DECISION_PROJECT"
  export PYTHONPATH="$DECISION_PROJECT:$DECISION_PROJECT/src"
  "$PYTHON" scripts/verify_generation_capacity_completion_decision.py \
    --capacity-probe-result "$CAPACITY_RESULT" \
    --expected-capacity-probe-result-sha256 "$capacity_result_sha" \
    --capacity-scaling-decision "$SCALING_DECISION" \
    --expected-capacity-scaling-decision-sha256 "$scaling_decision_sha" \
    --capacity-scaling-launch-receipt "$SCALING_LAUNCH" \
    --expected-capacity-scaling-launch-receipt-sha256 "$scaling_launch_sha" \
    --execution-status "$SCALING_STATUS" \
    --expected-execution-status-sha256 "$scaling_status_sha" \
    --cofitok-training-validation "$SCALING_COFITOK_VALIDATION" \
    --expected-cofitok-training-validation-sha256 "$scaling_cofitok_sha" \
    --dense-training-validation "$SCALING_DENSE_VALIDATION" \
    --expected-dense-training-validation-sha256 "$scaling_dense_sha" \
    --milestone-50000 "$MILESTONE_50000" \
    --expected-milestone-50000-sha256 "$milestone_50000_sha" \
    --standing-authorization "$STANDING_AUTHORIZATION" \
    --expected-standing-authorization-sha256 "$EXPECTED_STANDING_AUTHORIZATION_SHA256" \
    --training-project "$TRAINING_PROJECT" \
    --expected-capacity-revision "$EXPECTED_TRAINING_REVISION" \
    --expected-capacity-tree "$EXPECTED_TRAINING_TREE" \
    --expected-capacity-branch "$EXPECTED_TRAINING_BRANCH" \
    --expected-decision-revision "$EXPECTED_SCALING_DECISION_REVISION" \
    --expected-decision-branch "$EXPECTED_SCALING_DECISION_BRANCH" \
    --expected-execution-revision "$EXPECTED_SCALING_EXECUTION_REVISION" \
    --expected-execution-tree "$EXPECTED_SCALING_EXECUTION_TREE" \
    --expected-execution-branch "$EXPECTED_SCALING_EXECUTION_BRANCH" \
    --expected-result-revision "$EXPECTED_DECISION_REVISION" \
    --expected-result-tree "$EXPECTED_DECISION_TREE" \
    --expected-result-branch "$EXPECTED_DECISION_BRANCH" \
    --capacity-scaling-50k-result "$SCALING_RESULT" \
    --expected-capacity-scaling-50k-result-sha256 "$EXPECTED_SCALING_RESULT_SHA256" \
    --expected-completion-decision-revision "$EXPECTED_DECISION_REVISION" \
    --expected-completion-decision-tree "$EXPECTED_DECISION_TREE" \
    --expected-completion-decision-branch "$EXPECTED_DECISION_BRANCH" \
    --decision "$COMPLETION_DECISION" \
    --expected-decision-sha256 "$EXPECTED_COMPLETION_DECISION_SHA256" >/dev/null
)
"$PYTHON" - "$COMPLETION_DECISION" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
a=d.get('execution_authorization',{})
if (a.get('matched_250m_resume_allowed') is not True or a.get('resume_from_step') != 50000 or a.get('stop_after_step') != 100000 or a.get('full_300k_launch_allowed') is not False):
    raise SystemExit('completion decision does not authorize the exact 50K-to-100K segment')
PY

if [[ -f "$SOURCE_ARCHIVE" ]]; then
  set_stage source_archive_replay "replaying preserved step-50K source checkpoint hard links"
  archive_sha=$(sha "$SOURCE_ARCHIVE")
  "$PYTHON" scripts/verify_generation_capacity_completion_source_archive.py \
    "${archive_args[@]}" --archive "$SOURCE_ARCHIVE" \
    --expected-archive-sha256 "$archive_sha" >/dev/null
else
  set_stage source_archive_build "archiving exact step-50K source checkpoints by hard link"
  "$PYTHON" scripts/archive_generation_capacity_completion_sources.py \
    "${archive_args[@]}" --output "$SOURCE_ARCHIVE" >/dev/null
  archive_sha=$(sha "$SOURCE_ARCHIVE")
  "$PYTHON" scripts/verify_generation_capacity_completion_source_archive.py \
    "${archive_args[@]}" --archive "$SOURCE_ARCHIVE" \
    --expected-archive-sha256 "$archive_sha" >/dev/null
fi

if pgrep -af '[s]cripts/train_generation.py.*stability_capacity_probe.*100k' >/dev/null; then
  printf 'refusing duplicate capacity-completion trainer\n' >&2
  exit 12
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing capacity-completion launch while GPU is busy\n' >&2
  exit 9
fi

receipt_args=(
  --decision "$COMPLETION_DECISION"
  --expected-decision-sha256 "$EXPECTED_COMPLETION_DECISION_SHA256"
  --capacity-scaling-result "$SCALING_RESULT"
  --expected-capacity-scaling-result-sha256 "$EXPECTED_SCALING_RESULT_SHA256"
  --capacity-scaling-launch-receipt "$SCALING_LAUNCH"
  --expected-capacity-scaling-launch-receipt-sha256 "$scaling_launch_sha"
  --source-checkpoint-archive "$SOURCE_ARCHIVE"
  --expected-source-checkpoint-archive-sha256 "$archive_sha"
  --storage-capacity "$STORAGE"
  --training-project "$TRAINING_PROJECT"
  --output-root "$OUTPUT_ROOT"
  --storage-path "$CHECKPOINT_ROOT"
  --expected-decision-revision "$EXPECTED_DECISION_REVISION"
  --expected-decision-tree "$EXPECTED_DECISION_TREE"
  --expected-decision-branch "$EXPECTED_DECISION_BRANCH"
  --expected-scaling-execution-revision "$EXPECTED_SCALING_EXECUTION_REVISION"
  --expected-scaling-execution-tree "$EXPECTED_SCALING_EXECUTION_TREE"
  --expected-scaling-execution-branch "$EXPECTED_SCALING_EXECUTION_BRANCH"
  --expected-training-revision "$EXPECTED_TRAINING_REVISION"
  --expected-training-tree "$EXPECTED_TRAINING_TREE"
  --expected-training-branch "$EXPECTED_TRAINING_BRANCH"
  --expected-execution-revision "$EXPECTED_EXECUTION_REVISION"
  --expected-execution-tree "$EXPECTED_EXECUTION_TREE"
  --expected-execution-branch "$EXPECTED_EXECUTION_BRANCH"
)
if [[ -f "$LAUNCH_RECEIPT" ]]; then
  set_stage launch_receipt_replay "replaying immutable capacity-completion launch receipt"
  observed_receipt_sha=$(sha "$LAUNCH_RECEIPT")
  if [[ -n "$EXPECTED_LAUNCH_RECEIPT_SHA256" && "$observed_receipt_sha" != "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]; then
    printf 'capacity-completion launch receipt SHA256 differs\n' >&2
    exit 10
  fi
  storage_sha=$(sha "$STORAGE")
  "$PYTHON" scripts/verify_generation_capacity_completion_launch_receipt.py \
    "${receipt_args[@]}" --expected-storage-capacity-sha256 "$storage_sha" \
    --receipt "$LAUNCH_RECEIPT" --expected-receipt-sha256 "$observed_receipt_sha" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256="$observed_receipt_sha"
else
  [[ -z "$EXPECTED_LAUNCH_RECEIPT_SHA256" ]]
  set_stage storage_preflight "checking storage for 100K training and terminal evidence"
  "$PYTHON" scripts/check_generation_storage_capacity.py \
    --path "$CHECKPOINT_ROOT" --output "$STORAGE" \
    --stage stability_full_data_capacity_completion_250m_100k_execution \
    --reference-checkpoint "$COFITOK_RUN/checkpoint_step_00050000.pt" \
    --reference-checkpoint "$DENSE_RUN/checkpoint_step_00050000.pt" \
    --checkpoint-count 8 --checkpoint-size-multiplier 1.0 \
    --sample-count 24096 --estimated-sample-kib 256 \
    --additional-gib 16 --safety-margin-gib 64 >/dev/null
  storage_sha=$(sha "$STORAGE")
  set_stage launch_receipt_build "building immutable capacity-completion launch receipt"
  "$PYTHON" scripts/build_generation_capacity_completion_launch_receipt.py \
    "${receipt_args[@]}" --expected-storage-capacity-sha256 "$storage_sha" \
    --output "$LAUNCH_RECEIPT" >/dev/null
  EXPECTED_LAUNCH_RECEIPT_SHA256=$(sha "$LAUNCH_RECEIPT")
  "$PYTHON" scripts/verify_generation_capacity_completion_launch_receipt.py \
    "${receipt_args[@]}" --expected-storage-capacity-sha256 "$storage_sha" \
    --receipt "$LAUNCH_RECEIPT" \
    --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
fi

read -r MICRO_BATCH ACCUMULATION < <(
  "$PYTHON" - "$LAUNCH_RECEIPT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())['selection']['runtime_selection']
print(r['micro_batch_size'],r['gradient_accumulation_steps'])
PY
)
(( MICRO_BATCH * ACCUMULATION == 64 ))
latest_step() {
  "$PYTHON" - "$1/latest.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); print(int(json.loads(p.read_text())['step']) if p.is_file() else 0)
PY
}
restore_sources() {
  "$PYTHON" scripts/restore_generation_capacity_completion_sources.py \
    --archive "$SOURCE_ARCHIVE" --expected-archive-sha256 "$archive_sha" >/dev/null
}
verify_or_build_training_validation() {
  local method="$1" report="$2" config="$3" parameters="$4" validation="$5"
  local args=(
    --training-report "$report" --config "$config"
    --source-archive "$SOURCE_ARCHIVE"
    --expected-source-archive-sha256 "$archive_sha"
    --method "$method"
    --expected-revision "$EXPECTED_TRAINING_REVISION"
    --expected-branch "$EXPECTED_TRAINING_BRANCH"
    --expected-parameter-count "$parameters"
    --expected-micro-batch-size "$MICRO_BATCH"
    --expected-gradient-accumulation-steps "$ACCUMULATION"
  )
  if [[ -f "$validation" ]]; then
    local validation_sha
    validation_sha=$(sha "$validation")
    "$PYTHON" scripts/verify_generation_capacity_completion_training.py \
      "${args[@]}" --validation "$validation" \
      --expected-validation-sha256 "$validation_sha" >/dev/null
  else
    "$PYTHON" scripts/validate_generation_capacity_completion_training.py \
      "${args[@]}" --output "$validation" >/dev/null
  fi
}
train_to_100k() {
  local method="$1" config="$2" run_dir="$3" parameters="$4" validation="$5"
  local current
  current=$(latest_step "$run_dir")
  if (( current < 50000 || current > 100000 )); then
    printf '%s capacity-completion current step is invalid: %s\n' "$method" "$current" >&2
    exit 14
  fi
  if (( current < 100000 )); then
    local delta=$((100000-current))
    (
      cd "$TRAINING_PROJECT"
      export PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src"
      "$PYTHON" scripts/train_generation.py \
        --config "$config" --output-dir "$run_dir" \
        --micro-batch-size "$MICRO_BATCH" \
        --gradient-accumulation-steps "$ACCUMULATION" \
        --stop-after-steps "$delta" --resume auto
    )
  fi
  [[ "$(latest_step "$run_dir")" == 100000 ]]
  restore_sources
  verify_or_build_training_validation "$method" "$run_dir/training_report.json" \
    "$config" "$parameters" "$validation"
}
evaluate_milestone_100k() {
  local method="$1" run_dir="$2" prefix="$3" random_orders="$4"
  PROJECT="$TRAINING_PROJECT" PYTHON="$PYTHON" OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    DATA="$REAL_DATA" bash "$TRAINING_PROJECT/artifacts/runbooks/generation_full_milestone_eval.sh" \
      "$method" "$run_dir" "$run_dir/checkpoint_step_00100000.pt" 100000 "$prefix" "$random_orders"
}
milestone_valid() {
  [[ -f "$MILESTONE_100000" ]] || return 1
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/validate_generation_milestone_report.py" \
      --report "$MILESTONE_100000" --expected-step 100000 \
      --source-profile capacity_scaling >/dev/null
}

if ! milestone_valid; then
  set_stage cofitok_training "resuming exact base256 CoFiTok trajectory to step 100K"
  train_to_100k cofitok "$COFITOK_CONFIG" "$COFITOK_RUN" 250153763 "$COFITOK_VALIDATION"
  set_stage cofitok_milestone_evaluation "evaluating base256 CoFiTok step-100K milestone"
  evaluate_milestone_100k cofitok "$COFITOK_RUN" 8 4
  set_stage dense_training "resuming exact base256 dense trajectory to step 100K"
  train_to_100k dense_identity "$DENSE_CONFIG" "$DENSE_RUN" 250135043 "$DENSE_VALIDATION"
  set_stage dense_milestone_evaluation "evaluating base256 dense step-100K milestone"
  evaluate_milestone_100k dense_identity "$DENSE_RUN" 1 0
  set_stage milestone_build "building paired non-claim step-100K milestone"
  [[ ! -e "$MILESTONE_100000" ]]
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/build_generation_milestone_report.py" \
      --cofitok-generation "$COFITOK_RUN/milestones/step_00100000/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
      --dense-generation "$DENSE_RUN/milestones/step_00100000/samples_2048_ddim50_cfg15/metrics/generation_metrics_report.json" \
      --cofitok-checkpoint-eval "$COFITOK_RUN/milestones/step_00100000/checkpoint_eval/checkpoint_evaluation_report.json" \
      --dense-checkpoint-eval "$DENSE_RUN/milestones/step_00100000/checkpoint_eval/checkpoint_evaluation_report.json" \
      --milestone-step 100000 --expected-samples 2048 \
      --source-profile capacity_scaling --output "$MILESTONE_100000" >/dev/null
fi

terminal_eval() {
  local run_dir="$1" prefix="$2" random_orders="$3"
  local terminal="$run_dir/terminal_100k"
  local samples="$terminal/samples_10000_ddim100_cfg15"
  local checkpoint="$run_dir/checkpoint_step_00100000.pt"
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/preflight_generation_sampling.py" \
      --checkpoint "$checkpoint" --output "$terminal/sampling_preflight.json" \
      --batch-size 32 --prefix-budget "$prefix" --guidance-scale 1.5 \
      --guidance-rescale 0.0 --cfg-batch-mode batched --weights ema \
      --precision bf16 >/dev/null
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/evaluate_generation_checkpoint.py" \
      --checkpoint "$checkpoint" --output-dir "$terminal/checkpoint_eval" \
      --num-images 256 --timestep 500 --random-orders "$random_orders" \
      --weights ema --precision bf16 --resume >/dev/null
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/generate_samples.py" \
      --checkpoint "$checkpoint" --output-dir "$samples" --num-samples 10000 \
      --batch-size 32 --sample-steps 100 --prefix-budgets "$prefix" \
      --guidance-scale 1.5 --guidance-rescale 0.0 --cfg-batch-mode batched \
      --weights ema --precision bf16 --resume >/dev/null
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/evaluate_generation_metrics.py" \
      --real-dir "$REAL_DATA" --generated-dir "$samples/prefix_$prefix" \
      --sampling-report "$samples/sampling_report.json" \
      --output-dir "$samples/metrics" --cache-root "$EVAL_CACHE" \
      --min-samples 10000 --resume >/dev/null
  PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
    "$PYTHON" "$TRAINING_PROJECT/scripts/evaluate_generation_class_fidelity.py" \
      --generated-dir "$samples/prefix_$prefix" \
      --sampling-report "$samples/sampling_report.json" \
      --output-dir "$samples/class_fidelity" --min-samples 10000 --resume >/dev/null
}

set_stage cofitok_terminal_evaluation "building CoFiTok 10K-sample terminal evidence"
terminal_eval "$COFITOK_RUN" 8 4
set_stage dense_terminal_evaluation "building dense 10K-sample terminal evidence"
terminal_eval "$DENSE_RUN" 1 0
set_stage class_fidelity_qualification "building matched terminal class-fidelity qualification"
PYTHONPATH="$TRAINING_PROJECT:$TRAINING_PROJECT/src" \
  "$PYTHON" "$TRAINING_PROJECT/scripts/build_generation_class_fidelity_qualification.py" \
    --cofitok-report "$COFITOK_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json" \
    --dense-report "$DENSE_RUN/terminal_100k/samples_10000_ddim100_cfg15/class_fidelity/class_fidelity_report.json" \
    --output "$CLASS_FIDELITY_QUALIFICATION" --stage scaling \
    --expected-revision "$EXPECTED_TRAINING_REVISION" \
    --expected-branch "$EXPECTED_TRAINING_BRANCH" \
    --min-top1 0.01 --min-top5 0.05 --min-predicted-class-fraction 0.25 \
    --min-normalized-predicted-entropy 0.50 --max-top1-regression 0.05 \
    --max-top5-regression 0.05 --allow-hold >/dev/null

set_stage completion_replay "replaying all step-100K training and terminal evidence"
restore_sources
"$PYTHON" scripts/verify_generation_capacity_completion_source_archive.py \
  "${archive_args[@]}" --archive "$SOURCE_ARCHIVE" \
  --expected-archive-sha256 "$archive_sha" >/dev/null
storage_sha=$(sha "$STORAGE")
"$PYTHON" scripts/verify_generation_capacity_completion_launch_receipt.py \
  "${receipt_args[@]}" --expected-storage-capacity-sha256 "$storage_sha" \
  --receipt "$LAUNCH_RECEIPT" \
  --expected-receipt-sha256 "$EXPECTED_LAUNCH_RECEIPT_SHA256" >/dev/null
verify_or_build_training_validation cofitok "$COFITOK_RUN/training_report.json" \
  "$COFITOK_CONFIG" 250153763 "$COFITOK_VALIDATION"
verify_or_build_training_validation dense_identity "$DENSE_RUN/training_report.json" \
  "$DENSE_CONFIG" 250135043 "$DENSE_VALIDATION"
milestone_valid
"$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_TRAINING_REVISION" "$EXPECTED_TRAINING_BRANCH" <<'PY'
import json,sys
from pathlib import Path
from cofitok.generation_class_fidelity import validate_class_fidelity_qualification
root=Path(sys.argv[1]); completion=root/'reports/capacity_completion_100k'
for method,run_name,budget in (('cofitok','base256_cofitok',8),('dense_identity','base256_dense_identity',1)):
    terminal=root/run_name/'terminal_100k'; samples=terminal/'samples_10000_ddim100_cfg15'
    required=[terminal/'sampling_preflight.json',terminal/'checkpoint_eval/checkpoint_evaluation_report.json',samples/'sampling_report.json',samples/'sampling_progress.json',samples/'metrics/generation_metrics_report.json',samples/'class_fidelity/class_fidelity_report.json']
    if any(not path.is_file() for path in required): raise SystemExit(f'{method} terminal evidence is incomplete')
    metrics=json.loads(required[4].read_text()); provenance=metrics.get('sample_provenance',{})
    if (metrics.get('status')!='completed' or int(metrics.get('counts',{}).get('generated_image_count',-1))!=10000 or provenance.get('selected_prefix_budget')!=budget or int(provenance.get('checkpoint_step',-1))!=100000 or metrics.get('parameters',{}).get('precision_recall_enabled') is not True):
        raise SystemExit(f'{method} terminal metric contract differs')
qualification=json.loads((completion/'class_fidelity/qualification_report.json').read_text())
validate_class_fidelity_qualification(qualification,expected_stage='scaling',expected_revision=sys.argv[2],expected_branch=sys.argv[3],require_pass=False)
PY
CURRENT_STAGE=complete
write_status completed "matched 250M step-100K terminal evidence completed; new result required"
completed=true
printf 'capacity-completion 100K launch receipt: %s  %s\n' \
  "$EXPECTED_LAUNCH_RECEIPT_SHA256" "$LAUNCH_RECEIPT"
printf 'capacity-completion 100K milestone: %s\n' "$MILESTONE_100000"
