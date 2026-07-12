# CoFiTok Publication-Readiness Gap Report

Status: `ready`

This gate is intentionally stricter than the MVP evidence gate. It tracks whether current local and remote summaries support publication-facing claims.

## Criteria

| criterion | value |
|---|---:|
| `main_steps` | 20000 |
| `min_generated_samples` | 2048 |
| `min_main_train_seeds` | 2 |
| `min_quality_images` | 1024 |
| `min_real_images` | 8192 |
| `min_sample_steps` | 50 |
| `multiscale_steps` | 10000 |
| `token_count` | 8 |

## Checks

| check | status | missing | evidence |
|---|---|---|---|
| `main_train_seed_coverage` | `ok` | none | tiny_imagenet_200/epsilon_only: seed_count=2<br>seeds=2 items<br>tiny_imagenet_200/light_denoise_path: seed_count=2<br>seeds=2 items<br>imagenet_1k_64x64_hf/epsilon_only: seed_count=2<br>seeds=2 items<br>imagenet_1k_64x64_hf/light_denoise_path: seed_count=2<br>seeds=2 items |
| `quality_scale` | `ok` | none | tiny_imagenet_200/epsilon_only: matching_rows=1<br>max_image_count=10000<br>tiny_imagenet_200/light_denoise_path: matching_rows=1<br>max_image_count=10000<br>imagenet_1k_64x64_hf/epsilon_only: matching_rows=1<br>max_image_count=50000<br>imagenet_1k_64x64_hf/light_denoise_path: matching_rows=1<br>max_image_count=50000 |
| `main_20k_order_diagnostics` | `ok` | none | tiny_imagenet_200: complete_seed=103<br>path_auc=3 keys<br>imagenet_1k_64x64_hf: complete_seed=103<br>path_auc=3 keys |
| `sampling_scale` | `ok` | none | tiny_imagenet_200/epsilon_only: matching_rows=1<br>max_num_samples=2048<br>max_sample_steps=50<br>tiny_imagenet_200/light_denoise_path: matching_rows=1<br>max_num_samples=2048<br>max_sample_steps=50<br>imagenet_1k_64x64_hf/epsilon_only: matching_rows=1<br>max_num_samples=2048<br>max_sample_steps=50<br>imagenet_1k_64x64_hf/light_denoise_path: matching_rows=1<br>max_num_samples=2048<br>max_sample_steps=50 |
| `generated_quality_scale` | `ok` | none | tiny_imagenet_200/epsilon_only: matching_rows=3<br>max_sample_image_count=10000<br>max_real_image_count=10000<br>tiny_imagenet_200/light_denoise_path: matching_rows=3<br>max_sample_image_count=10000<br>max_real_image_count=10000<br>imagenet_1k_64x64_hf/epsilon_only: matching_rows=4<br>max_sample_image_count=50000<br>max_real_image_count=50000<br>imagenet_1k_64x64_hf/light_denoise_path: matching_rows=4<br>max_sample_image_count=50000<br>max_real_image_count=50000 |
| `multiscale_pilots` | `ok` | none | tiny_imagenet_200: train_rows=2<br>quality_rows=1<br>sampling_rows=1<br>generated_quality_rows=2<br>imagenet_1k_64x64_hf: train_rows=2<br>quality_rows=1<br>sampling_rows=1<br>generated_quality_rows=2 |
| `loss_ablation_pilots` | `ok` | none | tiny_imagenet_200/no_prefix_loss_ablation: train_rows=1<br>quality_rows=1<br>orders=3 items<br>tiny_imagenet_200/clean_monotonic_ablation: train_rows=1<br>quality_rows=1<br>orders=3 items<br>tiny_imagenet_200/no_monotonic_reference: train_rows=2<br>imagenet_1k_64x64_hf/no_prefix_loss_ablation: train_rows=1<br>quality_rows=1<br>orders=3 items<br>imagenet_1k_64x64_hf/clean_monotonic_ablation: train_rows=1<br>quality_rows=1<br>orders=3 items<br>imagenet_1k_64x64_hf/no_monotonic_reference: train_rows=2 |

## Next Action

The publication-readiness gate is satisfied. Freeze the summary, figures, and paper-facing tables before changing claims.
