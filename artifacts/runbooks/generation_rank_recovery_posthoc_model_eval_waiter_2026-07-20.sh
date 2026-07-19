#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
EXPECTED_REVISION=05bbb4af63a9f1d9b7f11bc4222d50875e382e1d
AGGREGATE="$PROJECT/artifacts/reports/generation/rank_recovery_probe_2026-07-20_v2/rank_recovery_probe.json"
AUDIT_OUTPUT="$PROJECT/artifacts/reports/generation/rank_recovery_probe_2026-07-20_v2/ema_lag_audit.json"
AUDIT_SCRIPT=/tmp/audit_generation_probe_ema.py
MONITOR="$OUTPUT_ROOT/generation_rank_recovery_probe_v2_monitor.json"
DENOISE_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_denoise_path_k8_probe5k_v2"
BAND_RUN="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_epsilon_band_k8_probe5k_v2"
TIMEOUT_SECONDS=43200
POLL_SECONDS=60

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
cd "$PROJECT"
export PYTHONPATH=src

deadline=$((SECONDS + TIMEOUT_SECONDS))
while [[ ! -f "$AGGREGATE" ]] || \
      pgrep -af '[g]eneration_rank_recovery_probe_2026-07-19.sh' >/dev/null; do
  if (( SECONDS >= deadline )); then
    printf 'timed out waiting for rank-recovery aggregate\n' >&2
    exit 74
  fi
  python - "$MONITOR" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    monitor = json.load(handle)
if monitor.get("status") in {"failed", "stalled"}:
    raise SystemExit("rank-recovery monitor reached a terminal failure")
PY
  sleep "$POLL_SECONDS"
done

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_REVISION" ]]; then
  printf 'remote revision changed before posthoc model evaluation\n' >&2
  exit 75
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'tracked worktree changed before posthoc model evaluation\n' >&2
  exit 76
fi
if [[ ! -f "$AUDIT_SCRIPT" ]]; then
  printf 'EMA audit script is missing: %s\n' "$AUDIT_SCRIPT" >&2
  exit 77
fi
if pgrep -af '[s]cripts/train_generation.py' >/dev/null; then
  printf 'generation training is still active after aggregate creation\n' >&2
  exit 78
fi

evaluate_model_weights() {
  local run_dir="$1"
  local checkpoint="$run_dir/checkpoint_step_00005000.pt"
  local output="$run_dir/checkpoint_eval_model_t500_512"
  if [[ -f "$output/checkpoint_evaluation_report.json" ]]; then
    return
  fi
  python scripts/evaluate_generation_checkpoint.py \
    --checkpoint "$checkpoint" --output-dir "$output" \
    --num-images 512 --timestep 500 --random-orders 16 \
    --weights model --precision bf16
}

evaluate_model_weights "$DENOISE_RUN"
evaluate_model_weights "$BAND_RUN"

python "$AUDIT_SCRIPT" \
  --candidate "denoise_path=$DENOISE_RUN" \
  --candidate "epsilon_band=$BAND_RUN" \
  --output "$AUDIT_OUTPUT"

printf 'wrote posthoc EMA-lag audit %s\n' "$AUDIT_OUTPUT"
