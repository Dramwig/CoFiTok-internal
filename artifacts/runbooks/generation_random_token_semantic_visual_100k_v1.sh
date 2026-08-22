#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10}"
QUALITY_ROOT="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1"
CHECKPOINT="${QUALITY_ROOT}/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/checkpoint_step_00100000.pt"
QUALITY_RESULT="${QUALITY_ROOT}/reports/quality_bridge_result.json"
TERMINAL_SYSTEM_GUARD="${QUALITY_ROOT}/reports/terminal_system_claim_guard_v1/terminal_system_claim_guard.json"
FACTORIZATION_SUPERVISOR_STATUS="${QUALITY_ROOT}/reports/factorization_quality_regression_supervisor_v1/supervisor_status.json"
OUTPUT_DIR="/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_random_token_semantic_visual_v1"
RUNBOOK_LOCK="${OUTPUT_DIR}.runbook.lock"
EXPECTED_CHECKPOINT_SHA256="b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e"
EXPECTED_CHECKPOINT_STEP="100000"
REQUIRED_IDLE_GPU_POLLS="5"
IDLE_GPU_POLL_SECONDS="60"

: "${EXPECTED_EVALUATOR_REVISION:?set exact evaluator revision}"
: "${EXPECTED_EVALUATOR_TREE:?set exact evaluator tree}"
: "${EXPECTED_EVALUATOR_BRANCH:?set exact evaluator branch}"

if [[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" != "${EXPECTED_EVALUATOR_REVISION}" ]]; then
  echo "evaluator revision mismatch" >&2
  exit 2
fi
if [[ "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')" != "${EXPECTED_EVALUATOR_TREE}" ]]; then
  echo "evaluator tree mismatch" >&2
  exit 2
fi
if [[ "$(git -C "${PROJECT_ROOT}" branch --show-current)" != "${EXPECTED_EVALUATOR_BRANCH}" ]]; then
  echo "evaluator branch mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "evaluator checkout has tracked changes" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python runtime is missing: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is unavailable; refusing random-token diagnostic" >&2
  exit 2
fi
exec 8>"${RUNBOOK_LOCK}"
if ! flock -n 8; then
  echo "another random-token diagnostic owns the runbook lock" >&2
  exit 75
fi
if [[ ! -f "${CHECKPOINT}" || ! -f "${CHECKPOINT}.integrity.json" ]]; then
  echo "exact CoFiTok 100K checkpoint evidence is missing" >&2
  exit 2
fi
if [[ ! -f "${QUALITY_RESULT}" ]]; then
  echo "terminal quality bridge result is not complete" >&2
  exit 75
fi
if [[ ! -f "${TERMINAL_SYSTEM_GUARD}" ]]; then
  echo "terminal system guard is not complete" >&2
  exit 75
fi
if [[ ! -f "${FACTORIZATION_SUPERVISOR_STATUS}" ]]; then
  echo "factorization supervisor is not terminal" >&2
  exit 75
fi

"${PYTHON_BIN}" - \
  "${QUALITY_RESULT}" \
  "${TERMINAL_SYSTEM_GUARD}" \
  "${FACTORIZATION_SUPERVISOR_STATUS}" <<'PY'
import json
import sys

def read_object(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"terminal evidence is not a JSON object: {path}")
    return value


result = read_object(sys.argv[1])
if result.get("status") != "completed":
    raise SystemExit(75)
boundary = result.get("authorization_boundary", {})
expected_false = (
    "quality_bridge_execution_allowed",
    "full_training_launch_allowed",
    "full_300k_launch_allowed",
    "report_is_promotion_gate",
    "release_authorization_allowed",
)
if any(boundary.get(key) is not False for key in expected_false):
    raise SystemExit("terminal quality bridge authorization boundary differs")

guard = read_object(sys.argv[2])
policy = guard.get("claim_policy", {})
if (
    guard.get("status") not in {"pass", "hold"}
    or policy.get("terminal_system_evidence_complete") is not True
):
    raise SystemExit(75)
evidence = guard.get("evidence", {})
classifier_integrity = evidence.get("class_fidelity_classifier_integrity", {})
if (
    policy.get("class_fidelity_classifier_physical_integrity_verified") is not True
    or classifier_integrity.get("status") != "verified"
):
    raise SystemExit("terminal classifier physical-integrity evidence differs")
guard_false = (
    "larger_training_launch_allowed",
    "inference_export_authorization_allowed",
    "release_authorization_allowed",
    "broad_generation_superiority_claim_allowed",
    "sota_claim_allowed",
)
if any(policy.get(key) is not False for key in guard_false):
    raise SystemExit("terminal system guard claim boundary differs")

supervisor = read_object(sys.argv[3])
if supervisor.get("status") != "completed":
    raise SystemExit(75)
if supervisor.get("detail") != (
    "matched_factorization_quality_regression_diagnostic_completed"
):
    raise SystemExit("factorization supervisor terminal detail differs")
child_pid = supervisor.get("child_pid")
if isinstance(child_pid, bool) or not isinstance(child_pid, int) or child_pid < 1:
    raise SystemExit("factorization supervisor child identity differs")
required_sources = {
    "deployment_receipt",
    "quality_bridge_followup_decision",
    "source_binding",
    "execution_authorization",
    "diagnostic_report",
}
if not required_sources.issubset(supervisor.get("sources", {})):
    raise SystemExit("factorization supervisor source set differs")
supervisor_boundary = supervisor.get("authorization_boundary", {})
supervisor_false = (
    "training_launch_allowed",
    "checkpoint_promotion_allowed",
    "followup_experiment_launch_allowed",
    "full_300k_launch_allowed",
    "release_authorization_allowed",
)
if any(supervisor_boundary.get(key) is not False for key in supervisor_false):
    raise SystemExit("factorization supervisor authorization boundary differs")
PY

for ((idle_poll = 1; idle_poll <= REQUIRED_IDLE_GPU_POLLS; idle_poll++)); do
  mapfile -t GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d'
  )
  if (( ${#GPU_PIDS[@]} != 0 )); then
    echo "GPU is not idle; refusing random-token visual diagnostic: ${GPU_PIDS[*]}" >&2
    exit 75
  fi
  if (( idle_poll < REQUIRED_IDLE_GPU_POLLS )); then
    sleep "${IDLE_GPU_POLL_SECONDS}"
  fi
done
RESUME_ARGS=()
if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
  echo "random-token output root exists but is not a directory" >&2
  exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]]; then
  RESUME_ARGS=(--resume)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/evaluate_generation_random_token_semantics.py \
  --checkpoint "${CHECKPOINT}" \
  --terminal-quality-result "${QUALITY_RESULT}" \
  --terminal-system-guard "${TERMINAL_SYSTEM_GUARD}" \
  --factorization-supervisor-status "${FACTORIZATION_SUPERVISOR_STATUS}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-images 16 \
  --seed 20260822 \
  --weights ema \
  --precision bf16 \
  --prefix-budgets 1,2,4,8 \
  --clip-sigma 3.0 \
  --expected-revision "${EXPECTED_EVALUATOR_REVISION}" \
  --expected-tree "${EXPECTED_EVALUATOR_TREE}" \
  --expected-branch "${EXPECTED_EVALUATOR_BRANCH}" \
  --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}" \
  --expected-checkpoint-step "${EXPECTED_CHECKPOINT_STEP}" \
  --expected-training-revision "cf0e5faa94bf4ab38d947b921935b3b765b5537a" \
  --expected-training-branch "scale/generation-stability-quality-bridge-100k" \
  "${RESUME_ARGS[@]}"
