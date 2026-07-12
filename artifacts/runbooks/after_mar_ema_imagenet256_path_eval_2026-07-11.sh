#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
MAR_STATUS="$ROOT/artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.status"
QUEUE_LOG=/root/autodl-tmp/CoFiTok/checkpoints/after_mar_ema_imagenet256_path_eval_2026-07-11.log
QUEUE_STATUS="$ROOT/artifacts/runbooks/after_mar_ema_imagenet256_path_eval_2026-07-11.status"

exec > >(tee -a "$QUEUE_LOG") 2>&1
finish() {
  local code=$?
  if [[ "$code" == 0 ]]; then
    printf 'completed\n' > "$QUEUE_STATUS"
  else
    printf 'failed:%s\n' "$code" > "$QUEUE_STATUS"
  fi
  date --iso-8601=seconds
}
trap finish EXIT
printf 'waiting\n' > "$QUEUE_STATUS"

while true; do
  state=$(cat "$MAR_STATUS" 2>/dev/null || true)
  case "$state" in
    completed)
      printf '[dependency-finished] MAR status=%s\n' "$state"
      break
      ;;
    failed:*)
      printf '[dependency-failed] MAR status=%s\n' "$state"
      exit 1
      ;;
    *)
      printf '[waiting] MAR status=%s\n' "${state:-missing}"
      sleep 60
      ;;
  esac
done

printf 'running\n' > "$QUEUE_STATUS"
cd "$ROOT"
bash artifacts/runbooks/dense_monolithic_p0_2026-07-11.sh
bash artifacts/runbooks/imagenet256_confirmatory_path_eval_2026-07-11.sh
