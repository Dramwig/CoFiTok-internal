#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}
EXPECTED_REVISION=${EXPECTED_REVISION:?set immutable pilot revision}
EXPECTED_TREE=${EXPECTED_TREE:?set immutable pilot tree}
EXPECTED_BRANCH=${EXPECTED_BRANCH:?set immutable pilot branch}
EXPECTED_PREPARATION_SHA256=${EXPECTED_PREPARATION_SHA256:?set immutable preparation SHA256}
EXPECTED_EXECUTION_GATE_SHA256=${EXPECTED_EXECUTION_GATE_SHA256:?set immutable execution-gate SHA256}

BRIDGE_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1
REPORT_ROOT="$BRIDGE_ROOT/reports/min_snr_matched_50k_pilot_v1_20260825"
PREPARATION="$REPORT_ROOT/preparation.json"
EXECUTION_GATE="$REPORT_ROOT/execution_gate.json"
EXECUTION_LOCK=/tmp/cofitok-min-snr-matched-50k-pilot-v1.lock
STATUS="$OUTPUT_ROOT/controller_status.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
AUDIT_ROOT="$OUTPUT_ROOT/reports/training_audits"
RESULT="$OUTPUT_ROOT/reports/min_snr_pilot_result.json"
REAL_DATA=/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val
EVAL_CACHE=/root/autodl-tmp/CoFiTok/checkpoints/generation/eval_cache/torch_fidelity
COFITOK_CONFIG=configs/generation/imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json
DENSE_CONFIG=configs/generation/imagenet256_min_snr_gamma5_quality_repair_rollout_x0_u2_ema_teacher_dense_100k_horizon_50k_pilot.json
COFITOK_RUN="$OUTPUT_ROOT/cofitok_gamma5"
DENSE_RUN="$OUTPUT_ROOT/dense_gamma5"
LEGACY_COFITOK=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt
LEGACY_DENSE=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/dense_rollout_x0_u2_ema_teacher/checkpoint_step_00050000.pt

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ -x "$PYTHON" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]]
[[ "$(git rev-parse HEAD^{tree})" == "$EXPECTED_TREE" ]]
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -d "$REAL_DATA" ]]
[[ "$(sha256sum "$PREPARATION" | awk '{print $1}')" == "$EXPECTED_PREPARATION_SHA256" ]]
[[ "$(sha256sum "$EXECUTION_GATE" | awk '{print $1}')" == "$EXPECTED_EXECUTION_GATE_SHA256" ]]
"$PYTHON" scripts/validate_generation_min_snr_pilot_execution_gate.py \
  --gate "$EXECUTION_GATE" \
  --expected-gate-sha256 "$EXPECTED_EXECUTION_GATE_SHA256" \
  --preparation "$PREPARATION" \
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256" >/dev/null

command -v flock >/dev/null
exec 9>"$EXECUTION_LOCK"
if ! flock -n 9; then
  printf 'refusing duplicate Min-SNR pilot controller\n' >&2
  exit 20
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$AUDIT_ROOT"

write_status() {
  local status="$1"
  local phase="$2"
  local detail="$3"
  local exit_code="${4:-}"
  "$PYTHON" - "$STATUS" "$status" "$phase" "$detail" "$exit_code" \
    "$EXPECTED_REVISION" "$EXPECTED_TREE" "$EXPECTED_BRANCH" "$$" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1])
payload={
  "schema_version":1,
  "role":"generation_matched_min_snr_50k_pilot_controller",
  "status":sys.argv[2],
  "phase":sys.argv[3],
  "detail":sys.argv[4],
  "exit_code":int(sys.argv[5]) if sys.argv[5] else None,
  "git":{"revision":sys.argv[6],"tree":sys.argv[7],"branch":sys.argv[8],"tracked_dirty":False},
  "pid":int(sys.argv[9]),
  "parent_pid":os.getppid(),
  "hostname":socket.gethostname(),
  "updated_at":datetime.now(timezone.utc).isoformat(),
  "generation_advantage_proven":False,
  "continuation_beyond_50000_allowed":False,
  "full_300k_launch_allowed":False,
  "promotion_allowed":False,
  "release_allowed":False,
  "process_signals_allowed":False,
}
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
with tmp.open("w",encoding="utf-8") as f:
  json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
}

completed=false
on_exit() {
  local code=$?
  if [[ "$completed" != true && $code -ne 0 ]]; then
    write_status failed failed "controller exited before bounded result replay" "$code" || true
  fi
}
trap on_exit EXIT

require_idle_gpu() {
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    printf 'refusing Min-SNR pilot GPU stage while compute processes are active\n' >&2
    exit 21
  fi
}

latest_step() {
  "$PYTHON" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"latest.json"
print(0 if not p.is_file() else int(json.loads(p.read_text())["step"]))
PY
}

train_arm() {
  local method="$1"
  local config="$2"
  local run_dir="$3"
  local current
  current="$(latest_step "$run_dir")"
  if (( current > 50000 )); then
    printf 'pilot arm exceeded the authorized 50K boundary: %s\n' "$run_dir" >&2
    exit 22
  fi
  if (( current < 50000 )); then
    local resume=()
    if (( current > 0 )); then
      resume=(--resume auto)
    elif [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      printf 'refusing fresh pilot arm with unbound existing state: %s\n' "$run_dir" >&2
      exit 22
    fi
    require_idle_gpu
    write_status running "training_${method}" "serial fresh/same-config Min-SNR training to 50K"
    "$PYTHON" scripts/train_generation.py \
      --config "$config" \
      --output-dir "$run_dir" \
      --micro-batch-size 64 \
      --gradient-accumulation-steps 1 \
      --stop-after-steps "$((50000-current))" \
      "${resume[@]}" \
      >"$LOG_ROOT/train_${method}_to_50k.log" 2>&1
  fi
  require_idle_gpu
  "$PYTHON" scripts/audit_generation_min_snr_pilot_training.py \
    --method "$method" \
    --run-dir "$run_dir" \
    --config "$config" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-tree "$EXPECTED_TREE" \
    --expected-branch "$EXPECTED_BRANCH" \
    --output "$AUDIT_ROOT/${method}_50k_physical_audit.json" \
    >"$LOG_ROOT/audit_${method}_50k.log" 2>&1
}

sampling_preflight_valid() {
  local arm="$1"
  local root="$2"
  local checkpoint="$3"
  local prefix="$4"
  "$PYTHON" scripts/validate_generation_min_snr_pilot_sampling_preflight.py \
    --arm "$arm" \
    --expected-prefix-budget "$prefix" \
    --sampling-preflight "$root/sampling_preflight.json" \
    --checkpoint "$checkpoint" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-branch "$EXPECTED_BRANCH" >/dev/null
}

evaluation_complete() {
  local arm="$1"
  local root="$2"
  local prefix="$3"
  [[ -f "$root/sampling_preflight.json" \
    && -f "$root/samples/sampling_report.json" \
    && -f "$root/samples/metrics/generation_metrics_report.json" \
    && -f "$root/samples/class_fidelity/class_fidelity_report.json" \
    && -f "$root/checkpoint_eval/checkpoint_evaluation_report.json" ]] || return 1
  "$PYTHON" scripts/validate_generation_min_snr_pilot_evaluation_arm.py \
    --arm "$arm" \
    --expected-prefix-budget "$prefix" \
    --sampling-preflight "$root/sampling_preflight.json" \
    --generation "$root/samples/metrics/generation_metrics_report.json" \
    --class-fidelity "$root/samples/class_fidelity/class_fidelity_report.json" \
    --checkpoint-eval "$root/checkpoint_eval/checkpoint_evaluation_report.json" \
    >/dev/null
}

evaluate_arm() {
  local arm="$1"
  local checkpoint="$2"
  local prefix="$3"
  local random_orders="$4"
  local root="$OUTPUT_ROOT/evaluations/$arm"
  local samples="$root/samples"
  if evaluation_complete "$arm" "$root" "$prefix"; then
    printf 'evaluation arm already complete: %s\n' "$arm"
    return
  fi
  mkdir -p "$root"
  write_status running "evaluating_${arm}" "serial 10K DDIM-100 quality/support/class/mechanism evaluation"
  require_idle_gpu
  if [[ -f "$root/sampling_preflight.json" ]] \
    && ! sampling_preflight_valid "$arm" "$root" "$checkpoint" "$prefix"; then
    mkdir -p "$root/preflight_attempts"
    FAILED_PREFLIGHT_SHA256="$(sha256sum "$root/sampling_preflight.json" | awk '{print $1}')"
    mv -- "$root/sampling_preflight.json" \
      "$root/preflight_attempts/sampling_preflight.rejected.${FAILED_PREFLIGHT_SHA256}.$(date +%s).$$.json"
  fi
  if [[ ! -f "$root/sampling_preflight.json" ]]; then
    "$PYTHON" scripts/preflight_generation_sampling.py \
      --checkpoint "$checkpoint" \
      --output "$root/sampling_preflight.json" \
      --batch-size 32 \
      --prefix-budget "$prefix" \
      --guidance-scale 1.5 \
      --guidance-rescale 0.0 \
      --cfg-batch-mode batched \
      --weights ema \
      --precision bf16 \
      >"$LOG_ROOT/preflight_${arm}.log" 2>&1
  fi
  sampling_preflight_valid "$arm" "$root" "$checkpoint" "$prefix"
  require_idle_gpu
  "$PYTHON" scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$checkpoint" \
    --output-dir "$root/checkpoint_eval" \
    --num-images 256 \
    --timestep 500 \
    --random-orders "$random_orders" \
    --seed 2027 \
    --weights ema \
    --precision bf16 \
    --resume \
    >"$LOG_ROOT/checkpoint_eval_${arm}.log" 2>&1
  require_idle_gpu
  "$PYTHON" scripts/generate_samples.py \
    --checkpoint "$checkpoint" \
    --output-dir "$samples" \
    --num-samples 10000 \
    --batch-size 32 \
    --sample-steps 100 \
    --prefix-budgets "$prefix" \
    --guidance-scale 1.5 \
    --guidance-rescale 0.0 \
    --cfg-batch-mode batched \
    --eta 0.0 \
    --seed 20260825 \
    --start-index 0 \
    --weights ema \
    --precision bf16 \
    --resume \
    >"$LOG_ROOT/sampling_${arm}.log" 2>&1
  require_idle_gpu
  "$PYTHON" scripts/evaluate_generation_metrics.py \
    --real-dir "$REAL_DATA" \
    --generated-dir "$samples/prefix_$prefix" \
    --sampling-report "$samples/sampling_report.json" \
    --output-dir "$samples/metrics" \
    --cache-root "$EVAL_CACHE" \
    --min-samples 10000 \
    --resume \
    >"$LOG_ROOT/metrics_${arm}.log" 2>&1
  require_idle_gpu
  "$PYTHON" scripts/evaluate_generation_class_fidelity.py \
    --generated-dir "$samples/prefix_$prefix" \
    --sampling-report "$samples/sampling_report.json" \
    --output-dir "$samples/class_fidelity" \
    --min-samples 10000 \
    --resume \
    >"$LOG_ROOT/class_fidelity_${arm}.log" 2>&1
  require_idle_gpu
  evaluation_complete "$arm" "$root" "$prefix"
}

result_args=(
  --preparation "$PREPARATION"
  --expected-preparation-sha256 "$EXPECTED_PREPARATION_SHA256"
  --execution-gate "$EXECUTION_GATE"
  --expected-execution-gate-sha256 "$EXPECTED_EXECUTION_GATE_SHA256"
  --pilot-cofitok-training-audit "$AUDIT_ROOT/cofitok_50k_physical_audit.json"
  --pilot-dense-training-audit "$AUDIT_ROOT/dense_identity_50k_physical_audit.json"
  --legacy-gamma0-cofitok-generation "$OUTPUT_ROOT/evaluations/legacy_gamma0_cofitok/samples/metrics/generation_metrics_report.json"
  --legacy-gamma0-cofitok-class-fidelity "$OUTPUT_ROOT/evaluations/legacy_gamma0_cofitok/samples/class_fidelity/class_fidelity_report.json"
  --legacy-gamma0-cofitok-checkpoint-eval "$OUTPUT_ROOT/evaluations/legacy_gamma0_cofitok/checkpoint_eval/checkpoint_evaluation_report.json"
  --legacy-gamma0-cofitok-sampling-preflight "$OUTPUT_ROOT/evaluations/legacy_gamma0_cofitok/sampling_preflight.json"
  --legacy-gamma0-dense-identity-generation "$OUTPUT_ROOT/evaluations/legacy_gamma0_dense_identity/samples/metrics/generation_metrics_report.json"
  --legacy-gamma0-dense-identity-class-fidelity "$OUTPUT_ROOT/evaluations/legacy_gamma0_dense_identity/samples/class_fidelity/class_fidelity_report.json"
  --legacy-gamma0-dense-identity-checkpoint-eval "$OUTPUT_ROOT/evaluations/legacy_gamma0_dense_identity/checkpoint_eval/checkpoint_evaluation_report.json"
  --legacy-gamma0-dense-identity-sampling-preflight "$OUTPUT_ROOT/evaluations/legacy_gamma0_dense_identity/sampling_preflight.json"
  --pilot-gamma5-cofitok-generation "$OUTPUT_ROOT/evaluations/pilot_gamma5_cofitok/samples/metrics/generation_metrics_report.json"
  --pilot-gamma5-cofitok-class-fidelity "$OUTPUT_ROOT/evaluations/pilot_gamma5_cofitok/samples/class_fidelity/class_fidelity_report.json"
  --pilot-gamma5-cofitok-checkpoint-eval "$OUTPUT_ROOT/evaluations/pilot_gamma5_cofitok/checkpoint_eval/checkpoint_evaluation_report.json"
  --pilot-gamma5-cofitok-sampling-preflight "$OUTPUT_ROOT/evaluations/pilot_gamma5_cofitok/sampling_preflight.json"
  --pilot-gamma5-dense-identity-generation "$OUTPUT_ROOT/evaluations/pilot_gamma5_dense_identity/samples/metrics/generation_metrics_report.json"
  --pilot-gamma5-dense-identity-class-fidelity "$OUTPUT_ROOT/evaluations/pilot_gamma5_dense_identity/samples/class_fidelity/class_fidelity_report.json"
  --pilot-gamma5-dense-identity-checkpoint-eval "$OUTPUT_ROOT/evaluations/pilot_gamma5_dense_identity/checkpoint_eval/checkpoint_evaluation_report.json"
  --pilot-gamma5-dense-identity-sampling-preflight "$OUTPUT_ROOT/evaluations/pilot_gamma5_dense_identity/sampling_preflight.json"
)

write_status running training_pair "starting sole serial matched Min-SNR pilot"
train_arm cofitok "$COFITOK_CONFIG" "$COFITOK_RUN"
train_arm dense_identity "$DENSE_CONFIG" "$DENSE_RUN"

evaluate_arm legacy_gamma0_cofitok "$LEGACY_COFITOK" 8 4
evaluate_arm legacy_gamma0_dense_identity "$LEGACY_DENSE" 1 0
evaluate_arm pilot_gamma5_cofitok "$COFITOK_RUN/checkpoint_step_00050000.pt" 8 4
evaluate_arm pilot_gamma5_dense_identity "$DENSE_RUN/checkpoint_step_00050000.pt" 1 0

write_status running result_replay "building bounded non-authorizing pilot result"
if [[ -f "$RESULT" ]]; then
  RESULT_SHA256="$(sha256sum "$RESULT" | awk '{print $1}')"
else
  "$PYTHON" scripts/build_generation_min_snr_pilot_result.py \
    "${result_args[@]}" \
    --output "$RESULT" \
    >"$LOG_ROOT/build_result.log" 2>&1
  RESULT_SHA256="$(sha256sum "$RESULT" | awk '{print $1}')"
fi
"$PYTHON" scripts/validate_generation_min_snr_pilot_result.py \
  "${result_args[@]}" \
  --result "$RESULT" \
  --expected-result-sha256 "$RESULT_SHA256" \
  >"$LOG_ROOT/validate_result.log" 2>&1

completed=true
write_status completed completed "50K pilot and four-arm 10K evaluation replayed; no continuation authorized"
printf '%s  %s\n' "$RESULT_SHA256" "$RESULT"

