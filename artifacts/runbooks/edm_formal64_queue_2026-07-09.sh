#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
REPO="${REPO:-$PROJECT_ROOT/baselines/repos/edm}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}"
DATE_TAG="${DATE_TAG:-2026-07-09}"
BASELINE_REPORT_ROOT="${BASELINE_REPORT_ROOT:-$CODE_DIR/artifacts/reports/baselines}"

DATASETS="${DATASETS:-downsampled_imagenet_64 ffhq_64 afhqv2_64}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_GPU="${BATCH_GPU:-16}"
ARCH="${ARCH:-ddpmpp}"
PRECOND="${PRECOND:-edm}"
CBASE="${CBASE:-64}"
CRES="${CRES:-1,1,1}"
LR="${LR:-1e-4}"
EMA_MIMG="${EMA_MIMG:-0.001}"
DROPOUT="${DROPOUT:-0}"
AUGMENT="${AUGMENT:-0}"
WORKERS="${WORKERS:-1}"
TICK_KIMG="${TICK_KIMG:-20}"
SNAP_TICKS="${SNAP_TICKS:-999}"
DUMP_TICKS="${DUMP_TICKS:-999}"
SEED="${SEED:-139}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1024}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
REAL_COUNT="${REAL_COUNT:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

cd "$CODE_DIR"
export PYTHONPATH="$REPO:$CODE_DIR/src:${PYTHONPATH:-}"

dataset_train_dir() {
  case "$1" in
    cifar10)
      printf "%s\n" "$PROJECT_ROOT/datasets/cifar10/derived/imagefolder/train"
      ;;
    tiny_imagenet_200)
      printf "%s\n" "$PROJECT_ROOT/datasets/tiny_imagenet_200/derived/edm_rgb_train"
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

TRAIN_MIMG="$("$PYTHON" -c "print(${TRAIN_STEPS} * ${BATCH_SIZE} / 1000000)")"
SAMPLER_NFE="$("$PYTHON" -c "print(2 * ${SAMPLE_STEPS} - 1)")"

for DATASET in $DATASETS; do
  DATA_DIR="$(dataset_train_dir "$DATASET")"
  CONFIG="$CODE_DIR/configs/baselines/edm/${DATASET}.json"
  RUN_NAME="edm_${DATASET}_${TRAIN_STEPS}steps_edm${SAMPLE_STEPS}_${DATE_TAG}"
  RUN_ROOT="$CHECKPOINT_ROOT/baselines/edm/$RUN_NAME"
  TRAIN_DIR="$RUN_ROOT/train"
  SAMPLE_DIR="$RUN_ROOT/samples_png"
  REPORT_DIR="$BASELINE_REPORT_ROOT/edm/$RUN_NAME"

  if [ ! -d "$DATA_DIR" ]; then
    printf "missing data dir: %s\n" "$DATA_DIR" >&2
    exit 1
  fi
  if [ ! -f "$CONFIG" ]; then
    printf "missing config: %s\n" "$CONFIG" >&2
    exit 1
  fi

  mkdir -p "$TRAIN_DIR" "$SAMPLE_DIR" "$REPORT_DIR"

  TRAIN_START="$(date -Iseconds)"
  "$PYTHON" "$REPO/train.py" \
    --outdir="$TRAIN_DIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch="$ARCH" \
    --precond="$PRECOND" \
    --duration="$TRAIN_MIMG" \
    --batch="$BATCH_SIZE" \
    --batch-gpu="$BATCH_GPU" \
    --cbase="$CBASE" \
    --cres="$CRES" \
    --lr="$LR" \
    --ema="$EMA_MIMG" \
    --dropout="$DROPOUT" \
    --augment="$AUGMENT" \
    --cache=0 \
    --workers="$WORKERS" \
    --tick="$TICK_KIMG" \
    --snap="$SNAP_TICKS" \
    --dump="$DUMP_TICKS" \
    --seed="$SEED" \
    --desc="$RUN_NAME" \
    --nosubdir
  TRAIN_END="$(date -Iseconds)"

  SNAPSHOT="$(find "$TRAIN_DIR" -maxdepth 1 -name 'network-snapshot-*.pkl' | sort | tail -n 1)"
  if [ -z "$SNAPSHOT" ]; then
    printf "no EDM snapshot found in %s\n" "$TRAIN_DIR" >&2
    exit 1
  fi

  BASELINE_ALIAS="edm" \
  DATASET_ALIAS="$DATASET" \
  RUN_NAME="$RUN_NAME" \
  REPORT_DIR="$REPORT_DIR" \
  REPO_PATH="$REPO" \
  DATA_DIR="$DATA_DIR" \
  TRAIN_DIR="$TRAIN_DIR" \
  SNAPSHOT="$SNAPSHOT" \
  TRAIN_START="$TRAIN_START" \
  TRAIN_END="$TRAIN_END" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  TRAIN_MIMG="$TRAIN_MIMG" \
  BATCH_SIZE="$BATCH_SIZE" \
  BATCH_GPU="$BATCH_GPU" \
  ARCH="$ARCH" \
  PRECOND="$PRECOND" \
  CBASE="$CBASE" \
  CRES="$CRES" \
  LR="$LR" \
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
    "checkpoints": {"snapshot": os.environ["SNAPSHOT"]},
    "parameters": {
        "train_steps": int(os.environ["TRAIN_STEPS"]),
        "train_mimg": float(os.environ["TRAIN_MIMG"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "batch_gpu": int(os.environ["BATCH_GPU"]),
        "arch": os.environ["ARCH"],
        "precond": os.environ["PRECOND"],
        "cbase": int(os.environ["CBASE"]),
        "cres": os.environ["CRES"],
        "learning_rate": os.environ["LR"],
    },
    "runtime": {"started_at": os.environ["TRAIN_START"], "finished_at": os.environ["TRAIN_END"]},
}
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "baseline_train_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  "$PYTHON" "$REPO/generate.py" \
    --network="$SNAPSHOT" \
    --outdir="$SAMPLE_DIR" \
    --seeds="0-$((SAMPLE_COUNT - 1))" \
    --batch="$SAMPLE_BATCH_SIZE" \
    --steps="$SAMPLE_STEPS"

  "$PYTHON" scripts/evaluate_generated_samples.py \
    --config "$CONFIG" \
    --samples-dir "$SAMPLE_DIR" \
    --output-dir "$REPORT_DIR/generated_quality" \
    --split val \
    --max-real-images "$REAL_COUNT" \
    --max-sample-images "$SAMPLE_COUNT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --enable-inception-fid

  BASELINE_ALIAS="edm" \
  DATASET_ALIAS="$DATASET" \
  RUN_NAME="$RUN_NAME" \
  REPORT_DIR="$REPORT_DIR" \
  SNAPSHOT="$SNAPSHOT" \
  SAMPLE_DIR="$SAMPLE_DIR" \
  SAMPLE_COUNT="$SAMPLE_COUNT" \
  SAMPLE_STEPS="$SAMPLE_STEPS" \
  SAMPLER_NFE="$SAMPLER_NFE" \
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
    "checkpoint": os.environ["SNAPSHOT"],
    "samples_dir": os.environ["SAMPLE_DIR"],
    "generated_quality_report": generated_path.as_posix(),
    "parameters": {
        "sample_count": int(os.environ["SAMPLE_COUNT"]),
        "sample_steps": int(os.environ["SAMPLE_STEPS"]),
        "sampler_nfe": int(os.environ["SAMPLER_NFE"]),
        "sampler": "edm_heun",
        "real_count": int(os.environ["REAL_COUNT"]),
    },
    "metrics": generated.get("metrics", {}),
    "evaluation": generated.get("evaluation", {}),
}
(report_dir / "baseline_eval_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  printf "edm_completed %s %s\n" "$DATASET" "$REPORT_DIR"
done
