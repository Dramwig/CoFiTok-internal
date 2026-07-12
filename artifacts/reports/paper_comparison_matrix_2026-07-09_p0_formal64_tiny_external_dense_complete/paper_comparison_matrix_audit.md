# Paper Comparison Matrix Audit

This file audits current evidence only. It is not a completed result table.

## Status Counts

| status | count |
| --- | --- |
| completed | 45 |
| missing | 26 |
| needs_adapter | 32 |
| partial | 1 |
| protocol_blocked | 8 |

## Priority Counts

| priority | status | count |
| --- | --- | --- |
| P0 | completed | 45 |
| P0 | missing | 26 |
| P0 | partial | 1 |
| P0 | protocol_blocked | 8 |
| P1 | needs_adapter | 32 |

## Dataset Counts

| dataset | status | count |
| --- | --- | --- |
| afhqv2_64 | completed | 9 |
| afhqv2_64 | needs_adapter | 4 |
| afhqv2_64 | protocol_blocked | 1 |
| cifar10 | completed | 2 |
| cifar10 | missing | 6 |
| cifar10 | needs_adapter | 4 |
| cifar10 | partial | 1 |
| cifar10 | protocol_blocked | 1 |
| downsampled_imagenet_64 | completed | 9 |
| downsampled_imagenet_64 | needs_adapter | 4 |
| downsampled_imagenet_64 | protocol_blocked | 1 |
| ffhq_64 | completed | 9 |
| ffhq_64 | needs_adapter | 4 |
| ffhq_64 | protocol_blocked | 1 |
| imagenet_1k_64x64_hf | completed | 7 |
| imagenet_1k_64x64_hf | missing | 2 |
| imagenet_1k_64x64_hf | needs_adapter | 4 |
| imagenet_1k_64x64_hf | protocol_blocked | 1 |
| imagenet_256 | missing | 9 |
| imagenet_256 | needs_adapter | 4 |
| imagenet_256 | protocol_blocked | 1 |
| imagenet_256_10pct | missing | 9 |
| imagenet_256_10pct | needs_adapter | 4 |
| imagenet_256_10pct | protocol_blocked | 1 |
| tiny_imagenet_200 | completed | 9 |
| tiny_imagenet_200 | needs_adapter | 4 |
| tiny_imagenet_200 | protocol_blocked | 1 |

## Open Cells

| dataset | method | priority | status | missing evidence |
| --- | --- | --- | --- | --- |
| cifar10 | same_backbone_dense | P0 | partial | generated_quality |
| cifar10 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| cifar10 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| cifar10 | cofitok_simultaneous | P0 | missing | train,quality |
| cifar10 | cofitok_deep_synthesis | P0 | missing | train,quality |
| cifar10 | edm | P0 | missing | baseline_train,baseline_eval |
| cifar10 | improved_diffusion | P0 | missing | baseline_train,baseline_eval |
| cifar10 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| cifar10 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| cifar10 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| cifar10 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| cifar10 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| tiny_imagenet_200 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | edm | P0 | missing | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | improved_diffusion | P0 | missing | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| downsampled_imagenet_64 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| ffhq_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| ffhq_64 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| ffhq_64 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| ffhq_64 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| ffhq_64 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| afhqv2_64 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | cofitok_light | P0 | missing | train,quality,generated_quality |
| imagenet_256_10pct | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| imagenet_256_10pct | channel_mask | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_no_prefix_loss | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_simultaneous | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_deep_synthesis | P0 | missing | train,quality |
| imagenet_256_10pct | edm | P0 | missing | baseline_train,baseline_eval |
| imagenet_256_10pct | improved_diffusion | P0 | missing | baseline_train,baseline_eval |
| imagenet_256_10pct | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_256_10pct | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256 | cofitok_light | P0 | missing | train,quality,generated_quality |
| imagenet_256 | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| imagenet_256 | channel_mask | P0 | missing | train,quality |
| imagenet_256 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| imagenet_256 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| imagenet_256 | cofitok_simultaneous | P0 | missing | train,quality |
| imagenet_256 | cofitok_deep_synthesis | P0 | missing | train,quality |
| imagenet_256 | edm | P0 | missing | baseline_train,baseline_eval |
| imagenet_256 | improved_diffusion | P0 | missing | baseline_train,baseline_eval |
| imagenet_256 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_256 | ml_flextok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256 | mar | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256 | titok_1d_tokenizer | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
| imagenet_256 | retok | P1 | needs_adapter | baseline_train_or_eval,baseline_eval |
