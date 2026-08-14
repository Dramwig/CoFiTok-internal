#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/generation}
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-$CHECKPOINT_ROOT/stability_capacity_full_300k_v1}
EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256=${EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256:?set the immutable capacity-full training launch receipt SHA256}

[[ "$FULL_OUTPUT_ROOT" == "$CHECKPOINT_ROOT/stability_capacity_full_300k_v1" ]]

export OUTPUT_ROOT="$FULL_OUTPUT_ROOT"
export COFITOK_RUN_ID=cofitok
export DENSE_RUN_ID=dense_identity
export SOURCE_PROFILE=capacity_full
export TRAINING_AUTHORIZATION="$FULL_OUTPUT_ROOT/reports/capacity_full_300k_training_launch_receipt.json"
export EXPECTED_TRAINING_AUTHORIZATION_SHA256="$EXPECTED_TRAINING_LAUNCH_RECEIPT_SHA256"
export EXPECTED_TRAINING_AUTHORIZATION_STAGE=capacity_full_experimental
export EXPECTED_TRAINING_AUTHORIZATION_DECISION=authorize_fresh_matched_300k_training

exec bash "$PROJECT/artifacts/runbooks/generation_stability_ema_teacher_full_posteval_50k.sh"
