#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
REPO="${REPO:-$PROJECT_ROOT/baselines/repos/improved_diffusion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/baselines/adapters/improved_diffusion:$REPO:$CODE_DIR/src"
mkdir -p "$CHECKPOINT_ROOT/baselines/improved_diffusion" artifacts/reports/baselines/improved_diffusion

# CPU-only integration smoke. It proves the adapter and upstream train loop can
# execute without consuming the GPU that may be running CoFiTok queues.
export CUDA_VISIBLE_DEVICES=""
export OPENAI_LOGDIR="$CHECKPOINT_ROOT/baselines/improved_diffusion/smoke_ffhq64_cpu_${DATE_TAG}"
export DIFFUSION_BLOB_LOGDIR="$OPENAI_LOGDIR"
export DIFFUSION_TRAINING_TEST=1
mkdir -p "$OPENAI_LOGDIR"

"$PYTHON" "$REPO/scripts/image_train.py" \
  --data_dir "$PROJECT_ROOT/datasets/ffhq_64/extracted/images/00000" \
  --image_size 64 \
  --num_channels 32 \
  --num_res_blocks 1 \
  --num_heads 1 \
  --attention_resolutions 16 \
  --diffusion_steps 100 \
  --noise_schedule linear \
  --lr 1e-4 \
  --batch_size 1 \
  --microbatch 1 \
  --log_interval 1 \
  --save_interval 1 \
  --lr_anneal_steps 1

cat > "artifacts/reports/baselines/improved_diffusion/smoke_ffhq64_cpu_${DATE_TAG}.json" <<JSON
{
  "baseline": "improved_diffusion",
  "date_tag": "$DATE_TAG",
  "status": "completed",
  "mode": "cpu_integration_smoke",
  "output_dir": "$OPENAI_LOGDIR",
  "data_dir": "$PROJECT_ROOT/datasets/ffhq_64/extracted/images/00000",
  "notes": "This is not a paper result. It validates the adapter and upstream finite training loop only."
}
JSON

echo "improved_diffusion_formal64_smoke_done"
