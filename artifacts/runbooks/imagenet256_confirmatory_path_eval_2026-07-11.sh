#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/CoFiTok
INTERNAL_ROOT="$PROJECT_ROOT/CoFiTok-internal"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
STATUS="$INTERNAL_ROOT/artifacts/runbooks/imagenet256_confirmatory_path_eval_2026-07-11.status"
LOG="$CHECKPOINT_ROOT/imagenet256_confirmatory_path_eval_2026-07-11.log"
RUN_TAG=2026-07-11_imagenet256_20k_repeat

exec > >(tee -a "$LOG") 2>&1
finish() {
  local code=$?
  if [[ "$code" == 0 ]]; then
    printf 'completed\n' > "$STATUS"
  else
    printf 'failed:%s\n' "$code" > "$STATUS"
  fi
  date --iso-8601=seconds
}
trap finish EXIT
printf 'running\n' > "$STATUS"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
export TORCH_HOME="$PROJECT_ROOT/checkpoints/torch_cache"
export PYTHONPATH="$INTERNAL_ROOT/src:$INTERNAL_ROOT:${PYTHONPATH:-}"
cd "$INTERNAL_ROOT"

run_quality() {
  local method=$1
  local seed=$2
  local order=$3
  local config=$4
  local train_dir=$5
  local budgets=$6
  local output="$CHECKPOINT_ROOT/quality_path_imagenet256_${method}_20k_seed${seed}_${order}_1024_t500_${RUN_TAG}"
  local extras=()
  local order_seed=0
  if [[ "$order" == ordered ]]; then
    extras=(--enable-lpips --enable-inception-fid)
  fi
  if [[ "$order" == random ]]; then
    # K=4 seed 1 gives [1,3,2,0]; seed 0 only swaps the last two components.
    order_seed=1
  fi
  if [[ -f "$output/quality_report.json" ]]; then
    return
  fi
  python scripts/evaluate_quality.py \
    --config "$config" \
    --checkpoint "$CHECKPOINT_ROOT/$train_dir/checkpoint_final.pt" \
    --output-dir "$output" \
    --split val \
    --max-batches 256 \
    --max-images 1024 \
    --timestep 500 \
    --prefix-budgets "$budgets" \
    --component-order "$order" \
    --random-order-seed "$order_seed" \
    "${extras[@]}"
}

for seed in 103 139; do
  run_quality \
    endpoint_only "$seed" ordered \
    "configs/train_imagenet256_k4_epsilononly_p150eval_20k_seed${seed}_cuda.json" \
    "train_imagenet256_k4_epsilononly_p150eval_20k_seed${seed}_cuda_${RUN_TAG}" \
    1,2,3,4

  run_quality \
    dense_monolithic "$seed" ordered \
    "configs/train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda.json" \
    "train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda_${RUN_TAG}" \
    1

  for order in ordered random reverse; do
    run_quality \
      cofitok "$seed" "$order" \
      "configs/train_imagenet256_k4_denoisepath_p150_light_20k_seed${seed}_cuda.json" \
      "train_imagenet256_k4_denoisepath_p150_light_20k_seed${seed}_cuda_${RUN_TAG}" \
      1,2,3,4
  done

  permutation_output="$CHECKPOINT_ROOT/order_permutations_imagenet256_cofitok_20k_seed${seed}_1024_t500_${RUN_TAG}"
  if [[ ! -f "$permutation_output/order_permutation_report.json" ]]; then
    python scripts/evaluate_order_permutations.py \
      --config "configs/train_imagenet256_k4_denoisepath_p150_light_20k_seed${seed}_cuda.json" \
      --checkpoint "$CHECKPOINT_ROOT/train_imagenet256_k4_denoisepath_p150_light_20k_seed${seed}_cuda_${RUN_TAG}/checkpoint_final.pt" \
      --output-dir "$permutation_output" \
      --split val --max-batches 256 --max-images 1024 --timestep 500 \
      --bootstrap-repetitions 10000 --bootstrap-seed 0
  fi

  dense_generated="$CHECKPOINT_ROOT/generated_quality_stream_imagenet256_dense_monolithic_20k_seed${seed}_4096_ddim50_${RUN_TAG}"
  if [[ ! -f "$dense_generated/generated_quality_report.json" ]]; then
    python scripts/evaluate_generated_samples_stream.py \
      --config "configs/train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda.json" \
      --checkpoint "$CHECKPOINT_ROOT/train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda_${RUN_TAG}/checkpoint_final.pt" \
      --output-dir "$dense_generated" \
      --split val --sample-count 4096 --max-real-images 10000 \
      --batch-size 4 --sample-steps 50 --enable-inception-fid
  fi
done

python scripts/build_imagenet256_confirmatory_report.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/imagenet256_confirmatory_2026-07-11

bash artifacts/runbooks/finalize_paper_evidence_2026-07-11.sh
