#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}

[[ "$FULL_OUTPUT_ROOT" == "$CHECKPOINT_ROOT/stability_capacity_full_300k_v1" ]]

export FULL_ROOT="$FULL_OUTPUT_ROOT"
export COFITOK_RUN_ID=cofitok
export DENSE_RUN_ID=dense_identity
export EXPORT_ROOT="$CHECKPOINT_ROOT/exports/stability_capacity_full_300k_v1"

exec bash "$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_export_inference_artifacts.sh"
