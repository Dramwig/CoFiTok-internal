#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
REPO="${REPO:-$PROJECT_ROOT/baselines/repos/improved_diffusion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"
BASELINE_REPORT_ROOT="${BASELINE_REPORT_ROOT:-$CODE_DIR/artifacts/reports/baselines}"

DATASETS="${DATASETS:-downsampled_imagenet_64 ffhq_64 afhqv2_64}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICROBATCH="${MICROBATCH:-8}"
LR="${LR:-1e-4}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NUM_CHANNELS="${NUM_CHANNELS:-64}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-1}"
ATTENTION_RESOLUTIONS="${ATTENTION_RESOLUTIONS:-16}"
EMA_RATE="${EMA_RATE:-0.9999}"
USE_EMA="${USE_EMA:-1}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1024}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
REAL_COUNT="${REAL_COUNT:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/baselines/adapters/improved_diffusion:$REPO:$CODE_DIR/src"

dataset_train_dir() {
  case "$1" in
    cifar10)
      printf "%s\n" "$PROJECT_ROOT/datasets/cifar10/derived/imagefolder/train"
      ;;
    tiny_imagenet_200)
      printf "%s\n" "$PROJECT_ROOT/datasets/tiny_imagenet_200/extracted/tiny-imagenet-200/train"
      ;;
    imagenet_1k_64x64_hf)
      printf "%s\n" "$PROJECT_ROOT/datasets/imagenet_1k_64x64_hf/extracted/train"
      ;;
    imagenet_256_10pct)
      printf "%s\n" "$PROJECT_ROOT/datasets/imagenet_256_10pct/extracted/train"
      ;;
    imagenet_256)
      printf "%s\n" "$PROJECT_ROOT/datasets/imagenet_256/extracted/train"
      ;;
    downsampled_imagenet_64)
      printf "%s\n" "$PROJECT_ROOT/datasets/downsampled_imagenet_64/extracted/train_64x64"
      ;;
    ffhq_64)
      printf "%s\n" "$PROJECT_ROOT/datasets/ffhq_64/extracted/images"
      ;;
    afhqv2_64)
      printf "%s\n" "$PROJECT_ROOT/datasets/afhqv2_64/extracted/train"
      ;;
    *)
      printf "unsupported dataset: %s\n" "$1" >&2
      return 2
      ;;
  esac
}

for DATASET in $DATASETS; do
  DATA_DIR="$(dataset_train_dir "$DATASET")"
  CONFIG="$CODE_DIR/configs/baselines/improved_diffusion/${DATASET}.json"
  RUN_NAME="improved_diffusion_${DATASET}_${TRAIN_STEPS}steps_ddim${SAMPLE_STEPS}_${DATE_TAG}"
  RUN_ROOT="$CHECKPOINT_ROOT/baselines/improved_diffusion/$RUN_NAME"
  TRAIN_DIR="$RUN_ROOT/train"
  SAMPLE_DIR="$RUN_ROOT/sample"
  PNG_DIR="$RUN_ROOT/samples_png"
  REPORT_DIR="$BASELINE_REPORT_ROOT/improved_diffusion/$RUN_NAME"

  if [ ! -d "$DATA_DIR" ]; then
    printf "missing data dir: %s\n" "$DATA_DIR" >&2
    exit 1
  fi
  if [ ! -f "$CONFIG" ]; then
    printf "missing config: %s\n" "$CONFIG" >&2
    exit 1
  fi

  mkdir -p "$TRAIN_DIR" "$SAMPLE_DIR" "$PNG_DIR" "$REPORT_DIR"

  export OPENAI_LOGDIR="$TRAIN_DIR"
  export DIFFUSION_BLOB_LOGDIR="$TRAIN_DIR"
  TRAIN_START="$(date -Iseconds)"
  "$PYTHON" "$REPO/scripts/image_train.py" \
    --data_dir "$DATA_DIR" \
    --image_size "$IMAGE_SIZE" \
    --num_channels "$NUM_CHANNELS" \
    --num_res_blocks "$NUM_RES_BLOCKS" \
    --num_heads "$NUM_HEADS" \
    --attention_resolutions "$ATTENTION_RESOLUTIONS" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --noise_schedule linear \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --microbatch "$MICROBATCH" \
    --ema_rate "$EMA_RATE" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$TRAIN_STEPS" \
    --lr_anneal_steps "$TRAIN_STEPS"
  TRAIN_END="$(date -Iseconds)"

  MODEL_CHECKPOINT="$(find "$TRAIN_DIR" -maxdepth 1 -name 'model*.pt' | sort | tail -n 1)"
  EMA_CHECKPOINT="$(find "$TRAIN_DIR" -maxdepth 1 -name "ema_${EMA_RATE}_*.pt" | sort | tail -n 1 || true)"
  SAMPLE_CHECKPOINT="$MODEL_CHECKPOINT"
  if [ "$USE_EMA" = "1" ] && [ -n "$EMA_CHECKPOINT" ]; then
    SAMPLE_CHECKPOINT="$EMA_CHECKPOINT"
  fi
  if [ -z "$SAMPLE_CHECKPOINT" ]; then
    printf "no checkpoint found in %s\n" "$TRAIN_DIR" >&2
    exit 1
  fi

  BASELINE_ALIAS="improved_diffusion" \
  DATASET_ALIAS="$DATASET" \
  RUN_NAME="$RUN_NAME" \
  REPORT_DIR="$REPORT_DIR" \
  REPO_PATH="$REPO" \
  DATA_DIR="$DATA_DIR" \
  TRAIN_DIR="$TRAIN_DIR" \
  MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
  EMA_CHECKPOINT="$EMA_CHECKPOINT" \
  SAMPLE_CHECKPOINT="$SAMPLE_CHECKPOINT" \
  TRAIN_START="$TRAIN_START" \
  TRAIN_END="$TRAIN_END" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  BATCH_SIZE="$BATCH_SIZE" \
  MICROBATCH="$MICROBATCH" \
  LR="$LR" \
  DIFFUSION_STEPS="$DIFFUSION_STEPS" \
  NUM_CHANNELS="$NUM_CHANNELS" \
  NUM_RES_BLOCKS="$NUM_RES_BLOCKS" \
  "$PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])
repo = os.environ["REPO_PATH"]
try:
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = ""
payload = {
    "report_type": "baseline_train",
    "baseline": os.environ["BASELINE_ALIAS"],
    "dataset": os.environ["DATASET_ALIAS"],
    "status": "completed",
    "run_name": os.environ["RUN_NAME"],
    "repo": repo,
    "repo_commit": commit,
    "data_dir": os.environ["DATA_DIR"],
    "output_dir": os.environ["TRAIN_DIR"],
    "checkpoints": {
        "model": os.environ["MODEL_CHECKPOINT"],
        "ema": os.environ["EMA_CHECKPOINT"],
        "sample_checkpoint": os.environ["SAMPLE_CHECKPOINT"],
    },
    "parameters": {
        "train_steps": int(os.environ["TRAIN_STEPS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "microbatch": int(os.environ["MICROBATCH"]),
        "learning_rate": os.environ["LR"],
        "diffusion_steps": int(os.environ["DIFFUSION_STEPS"]),
        "num_channels": int(os.environ["NUM_CHANNELS"]),
        "num_res_blocks": int(os.environ["NUM_RES_BLOCKS"]),
    },
    "runtime": {
        "started_at": os.environ["TRAIN_START"],
        "finished_at": os.environ["TRAIN_END"],
    },
}
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "baseline_train_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  export OPENAI_LOGDIR="$SAMPLE_DIR"
  export DIFFUSION_BLOB_LOGDIR="$SAMPLE_DIR"
  "$PYTHON" "$REPO/scripts/image_sample.py" \
    --model_path "$SAMPLE_CHECKPOINT" \
    --image_size "$IMAGE_SIZE" \
    --num_channels "$NUM_CHANNELS" \
    --num_res_blocks "$NUM_RES_BLOCKS" \
    --num_heads "$NUM_HEADS" \
    --attention_resolutions "$ATTENTION_RESOLUTIONS" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --noise_schedule linear \
    --num_samples "$SAMPLE_COUNT" \
    --batch_size "$SAMPLE_BATCH_SIZE" \
    --use_ddim True \
    --timestep_respacing "ddim${SAMPLE_STEPS}"

  SAMPLE_NPZ="$(find "$SAMPLE_DIR" -maxdepth 1 -name 'samples_*.npz' | sort | tail -n 1)"
  if [ -z "$SAMPLE_NPZ" ]; then
    printf "no sample npz found in %s\n" "$SAMPLE_DIR" >&2
    exit 1
  fi
  "$PYTHON" scripts/baselines/npz_to_png.py \
    --input "$SAMPLE_NPZ" \
    --output-dir "$PNG_DIR" \
    --limit "$SAMPLE_COUNT" \
    --prefix sample

  "$PYTHON" scripts/evaluate_generated_samples.py \
    --config "$CONFIG" \
    --samples-dir "$PNG_DIR" \
    --output-dir "$REPORT_DIR/generated_quality" \
    --split val \
    --max-real-images "$REAL_COUNT" \
    --max-sample-images "$SAMPLE_COUNT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --enable-inception-fid

  BASELINE_ALIAS="improved_diffusion" \
  DATASET_ALIAS="$DATASET" \
  RUN_NAME="$RUN_NAME" \
  REPORT_DIR="$REPORT_DIR" \
  SAMPLE_NPZ="$SAMPLE_NPZ" \
  PNG_DIR="$PNG_DIR" \
  SAMPLE_CHECKPOINT="$SAMPLE_CHECKPOINT" \
  SAMPLE_COUNT="$SAMPLE_COUNT" \
  SAMPLE_STEPS="$SAMPLE_STEPS" \
  REAL_COUNT="$REAL_COUNT" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])
generated_path = report_dir / "generated_quality" / "generated_quality_report.json"
generated = json.loads(generated_path.read_text(encoding="utf-8"))
payload = {
    "report_type": "baseline_eval",
    "baseline": os.environ["BASELINE_ALIAS"],
    "dataset": os.environ["DATASET_ALIAS"],
    "status": "completed",
    "run_name": os.environ["RUN_NAME"],
    "checkpoint": os.environ["SAMPLE_CHECKPOINT"],
    "sample_npz": os.environ["SAMPLE_NPZ"],
    "samples_dir": os.environ["PNG_DIR"],
    "generated_quality_report": generated_path.as_posix(),
    "parameters": {
        "sample_count": int(os.environ["SAMPLE_COUNT"]),
        "sample_steps": int(os.environ["SAMPLE_STEPS"]),
        "sampler_nfe": int(os.environ["SAMPLE_STEPS"]),
        "sampler": "ddim",
        "real_count": int(os.environ["REAL_COUNT"]),
    },
    "metrics": generated.get("metrics", {}),
    "evaluation": generated.get("evaluation", {}),
}
(report_dir / "baseline_eval_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  printf "improved_diffusion_completed %s %s\n" "$DATASET" "$REPORT_DIR"
done
