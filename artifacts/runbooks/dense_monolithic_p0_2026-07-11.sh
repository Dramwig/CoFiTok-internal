#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/CoFiTok
INTERNAL_ROOT="$PROJECT_ROOT/CoFiTok-internal"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
RUN_TAG=2026-07-11_dense_monolithic
STATUS="$INTERNAL_ROOT/artifacts/runbooks/dense_monolithic_p0_2026-07-11.status"
LOG="$CHECKPOINT_ROOT/dense_monolithic_p0_2026-07-11.log"

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

run_p0_suite() {
  local config=$1
  local dataset_tag=$2
  local steps_tag=$3
  local sample_count=$4
  local real_count=$5
  local sample_batch=$6
  local quality_count=$7
  local config_stem
  config_stem=$(basename "$config" .json)
  local train_dir="$CHECKPOINT_ROOT/${config_stem}_${RUN_TAG}"
  local quality_dir="$CHECKPOINT_ROOT/quality_${dataset_tag}_dense_monolithic_${steps_tag}_${quality_count}_t500_${RUN_TAG}"
  local generated_dir="$CHECKPOINT_ROOT/generated_quality_stream_${dataset_tag}_dense_monolithic_${steps_tag}_${sample_count}_ddim50_${RUN_TAG}"

  if [[ ! -f "$train_dir/checkpoint_final.pt" ]]; then
    python scripts/train_short.py --config "$config" --output-dir "$train_dir"
  fi
  if [[ ! -f "$quality_dir/quality_report.json" ]]; then
    python scripts/evaluate_quality.py \
      --config "$config" \
      --checkpoint "$train_dir/checkpoint_final.pt" \
      --output-dir "$quality_dir" \
      --split val --max-batches 256 --max-images "$quality_count" --timestep 500 \
      --enable-lpips --enable-inception-fid
  fi
  if [[ ! -f "$generated_dir/generated_quality_report.json" ]]; then
    python scripts/evaluate_generated_samples_stream.py \
      --config "$config" \
      --checkpoint "$train_dir/checkpoint_final.pt" \
      --output-dir "$generated_dir" \
      --split val --sample-count "$sample_count" --max-real-images "$real_count" \
      --batch-size "$sample_batch" --sample-steps 50 --enable-inception-fid
  fi
}

run_existing_generation() {
  local config=$1
  local train_dir=$2
  local output_tag=$3
  local sample_count=$4
  local real_count=$5
  local sample_batch=$6
  local output="$CHECKPOINT_ROOT/generated_quality_stream_${output_tag}_${sample_count}_ddim50_2026-07-11_fair_p0"
  if [[ ! -f "$output/generated_quality_report.json" ]]; then
    python scripts/evaluate_generated_samples_stream.py \
      --config "$config" \
      --checkpoint "$CHECKPOINT_ROOT/$train_dir/checkpoint_final.pt" \
      --output-dir "$output" \
      --split val --sample-count "$sample_count" --max-real-images "$real_count" \
      --batch-size "$sample_batch" --sample-steps 50 --enable-inception-fid
  fi
}

run_existing_quality() {
  local config=$1
  local train_dir=$2
  local output_tag=$3
  local quality_count=$4
  local budgets=${5:-}
  local output="$CHECKPOINT_ROOT/quality_${output_tag}_${quality_count}_t500_2026-07-11_fair_p0"
  local budget_args=()
  if [[ -n "$budgets" ]]; then
    budget_args=(--prefix-budgets "$budgets")
  fi
  if [[ ! -f "$CHECKPOINT_ROOT/$train_dir/checkpoint_final.pt" ]]; then
    python scripts/train_short.py \
      --config "$config" \
      --output-dir "$CHECKPOINT_ROOT/$train_dir"
  fi
  if [[ ! -f "$output/quality_report.json" ]]; then
    python scripts/evaluate_quality.py \
      --config "$config" \
      --checkpoint "$CHECKPOINT_ROOT/$train_dir/checkpoint_final.pt" \
      --output-dir "$output" \
      --split val --max-batches 256 --max-images "$quality_count" --timestep 500 \
      "${budget_args[@]}" \
      --enable-lpips --enable-inception-fid
  fi
}

# Lock all 32/64-resolution internal quality rows to 512 val images at t=500.
run_existing_quality \
  configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json \
  train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08 \
  cifar10_cofitok_3k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_cifar10_k8_epsilononly_p150eval_3k_cuda.json \
  train_cifar10_k8_epsilononly_p150eval_3k_2026-07-08 \
  cifar10_endpoint_only_3k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_cifar10_k8_channelmask_p150eval_3k_cuda.json \
  train_cifar10_k8_channelmask_p150eval_3k_2026-07-08 \
  cifar10_channelmask_3k 512

run_existing_quality \
  configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json \
  train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08 \
  tiny_imagenet_cofitok_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_tiny_imagenet_k8_epsilononly_p150eval_5k_cuda.json \
  train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08 \
  tiny_imagenet_endpoint_only_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_tiny_imagenet_k8_channelmask_5k_cuda.json \
  train_tiny_imagenet_k8_channelmask_5k_2026-07-08 \
  tiny_imagenet_channelmask_5k 512

run_existing_quality \
  configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_2026-07-08 \
  imagenet_hf_cofitok_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_2026-07-08 \
  imagenet_hf_endpoint_only_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_imagenet_1k_64x64_hf_k8_channelmask_p150eval_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_channelmask_p150eval_5k_2026-07-08 \
  imagenet_hf_channelmask_5k 512
run_existing_quality \
  configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda.json \
  imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda \
  imagenet_hf_deepsk_5k 512
run_existing_quality \
  configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_2026-07-08 \
  imagenet_hf_simultaneous_5k 512

run_existing_quality \
  configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json \
  train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda_2026-07-09 \
  downsampled_imagenet64_cofitok_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda.json \
  train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda_2026-07-09 \
  downsampled_imagenet64_endpoint_only_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_ffhq64_k8_denoisepath_p150_light_5k_cuda.json \
  train_ffhq64_k8_denoisepath_p150_light_5k_cuda_2026-07-09 \
  ffhq64_cofitok_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_ffhq64_k8_epsilononly_p150eval_5k_cuda.json \
  train_ffhq64_k8_epsilononly_p150eval_5k_cuda_2026-07-09 \
  ffhq64_endpoint_only_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda.json \
  train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda_2026-07-09 \
  afhqv2_64_cofitok_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_afhqv2_64_k8_epsilononly_p150eval_5k_cuda.json \
  train_afhqv2_64_k8_epsilononly_p150eval_5k_cuda_2026-07-09 \
  afhqv2_64_endpoint_only_5k 512 1,2,3,4,5,6,7,8
run_existing_quality \
  configs/train_imagenet256_10pct_k4_denoisepath_p150_light_5k_cuda.json \
  train_imagenet256_10pct_k4_denoisepath_p150_light_5k_cuda_2026-07-10_imagenet256_10pct_p0 \
  imagenet256_10pct_cofitok_5k 256 1,2,3,4
run_existing_quality \
  configs/train_imagenet256_10pct_k4_epsilononly_p150eval_5k_cuda.json \
  train_imagenet256_10pct_k4_epsilononly_p150eval_5k_cuda_2026-07-10_imagenet256_10pct_p0 \
  imagenet256_10pct_endpoint_only_5k 256 1,2,3,4
run_existing_quality \
  configs/train_imagenet256_k4_denoisepath_p150_light_5k_cuda.json \
  train_imagenet256_k4_denoisepath_p150_light_5k_cuda_2026-07-10_imagenet256_p0 \
  imagenet256_cofitok_5k 256 1,2,3,4
run_existing_quality \
  configs/train_imagenet256_k4_epsilononly_p150eval_5k_cuda.json \
  train_imagenet256_k4_epsilononly_p150eval_5k_cuda_2026-07-10_imagenet256_p0 \
  imagenet256_endpoint_only_5k 256 1,2,3,4

run_long_budget_quality() {
  local config=$1
  local train_dir=$2
  local dataset_tag=$3
  local method_tag=$4
  local seed=$5
  local output="$CHECKPOINT_ROOT/quality_path_${dataset_tag}_${method_tag}_20k_seed${seed}_1024_t500_2026-07-11_confirmatory"
  if [[ ! -f "$output/quality_report.json" ]]; then
    python scripts/evaluate_quality.py \
      --config "$config" \
      --checkpoint "$CHECKPOINT_ROOT/$train_dir/checkpoint_final.pt" \
      --output-dir "$output" \
      --split val --max-batches 256 --max-images 1024 --timestep 500 \
      --prefix-budgets 1,2,3,4,5,6,7,8
  fi
}

run_long_budget_quality configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_cuda.json train_tiny_imagenet_k8_epsilononly_p150eval_20k_2026-07-08 tiny endpoint_only 103
run_long_budget_quality configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda.json train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda_2026-07-08 tiny endpoint_only 139
run_long_budget_quality configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_cuda.json train_tiny_imagenet_k8_denoisepath_p150_light_20k_2026-07-08 tiny cofitok 103
run_long_budget_quality configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda_2026-07-08 tiny cofitok 139
run_long_budget_quality configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_cuda.json train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_2026-07-08 imagenet_hf endpoint_only 103
run_long_budget_quality configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda_2026-07-08 imagenet_hf endpoint_only 139
run_long_budget_quality configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_cuda.json train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_2026-07-08 imagenet_hf cofitok 103
run_long_budget_quality configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda_2026-07-08 imagenet_hf cofitok 139

# Replace legacy 64-image/DDIM-20 smoke rows with the locked P0 generation protocol.
run_existing_generation \
  configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json \
  train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08 \
  cifar10_cofitok_3k 1024 4096 64
run_existing_generation \
  configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json \
  train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08 \
  tiny_imagenet_cofitok_5k 1024 4096 64
run_existing_generation \
  configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_2026-07-08 \
  imagenet_hf_cofitok_5k 1024 4096 64
run_existing_generation \
  configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_cuda.json \
  train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_2026-07-08 \
  imagenet_hf_endpoint_only_5k 1024 4096 64

run_p0_suite configs/train_cifar10_k8_densehead_p150eval_3k_cuda.json cifar10 3k 1024 4096 64 512
run_p0_suite configs/train_tiny_imagenet_k8_densehead_p150eval_5k_cuda.json tiny_imagenet 5k 1024 4096 64 512
run_p0_suite configs/train_imagenet_1k_64x64_hf_k8_densehead_p150eval_5k_cuda.json imagenet_hf 5k 1024 4096 64 512
run_p0_suite configs/train_downsampled_imagenet64_k8_densehead_p150eval_5k_cuda.json downsampled_imagenet64 5k 1024 4096 64 512
run_p0_suite configs/train_ffhq64_k8_densehead_p150eval_5k_cuda.json ffhq64 5k 1024 4096 64 512
run_p0_suite configs/train_afhqv2_64_k8_densehead_p150eval_5k_cuda.json afhqv2_64 5k 1024 4096 64 512
run_p0_suite configs/train_imagenet256_10pct_k4_densehead_p150eval_5k_cuda.json imagenet256_10pct 5k 512 2048 4 256
run_p0_suite configs/train_imagenet256_k4_densehead_p150eval_5k_cuda.json imagenet256 5k 512 2048 4 256

for seed in 103 139; do
  config="configs/train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda.json"
  train_dir="$CHECKPOINT_ROOT/train_imagenet256_k4_densehead_p150eval_20k_seed${seed}_cuda_2026-07-11_imagenet256_20k_repeat"
  if [[ ! -f "$train_dir/checkpoint_final.pt" ]]; then
    python scripts/train_short.py --config "$config" --output-dir "$train_dir"
  fi
done

python scripts/summarize_experiments.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/summary_2026-07-11_dense_monolithic_p0

printf 'dense_monolithic_p0_done\n'
