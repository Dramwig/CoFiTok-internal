#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="/root/autodl-tmp/CoFiTok/CoFiTok-internal"
MAR_STATUS="${CODE_DIR}/artifacts/runbooks/mar_hf_official_50k_2026-07-11.status"
QUEUE_STATUS="${CODE_DIR}/artifacts/runbooks/imagenet256_long_budget_repeat_2026-07-11.status"

if [[ -f "${QUEUE_STATUS}" ]] && [[ "$(cat "${QUEUE_STATUS}")" == "completed" ]]; then
  exit 0
fi

while [[ -f "${MAR_STATUS}" ]] && [[ "$(cat "${MAR_STATUS}")" == "running" ]]; do
  sleep 60
done

if [[ ! -f "${MAR_STATUS}" ]] || [[ "$(cat "${MAR_STATUS}")" != "completed" ]]; then
  printf 'MAR did not complete; ImageNet-256 20k queue not launched.\n' >&2
  exit 3
fi

cd "${CODE_DIR}"
bash artifacts/runbooks/imagenet256_long_budget_repeat_2026-07-11.sh
