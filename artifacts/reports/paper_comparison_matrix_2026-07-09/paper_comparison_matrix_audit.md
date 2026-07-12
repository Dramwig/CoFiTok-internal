# Paper Comparison Matrix Audit

This file audits current evidence only. It is not a completed result table.

## Status Counts

| status | count |
| --- | --- |
| completed | 14 |
| missing | 41 |
| needs_adapter | 56 |
| partial | 1 |

## Priority Counts

| priority | status | count |
| --- | --- | --- |
| P0 | completed | 14 |
| P0 | missing | 41 |
| P0 | needs_adapter | 24 |
| P0 | partial | 1 |
| P1 | needs_adapter | 32 |

## Dataset Counts

| dataset | status | count |
| --- | --- | --- |
| afhqv2_64 | missing | 7 |
| afhqv2_64 | needs_adapter | 7 |
| cifar10 | completed | 2 |
| cifar10 | missing | 4 |
| cifar10 | needs_adapter | 7 |
| cifar10 | partial | 1 |
| downsampled_imagenet_64 | missing | 7 |
| downsampled_imagenet_64 | needs_adapter | 7 |
| ffhq_64 | missing | 7 |
| ffhq_64 | needs_adapter | 7 |
| imagenet_1k_64x64_hf | completed | 7 |
| imagenet_1k_64x64_hf | needs_adapter | 7 |
| imagenet_256 | missing | 7 |
| imagenet_256 | needs_adapter | 7 |
| imagenet_256_10pct | missing | 7 |
| imagenet_256_10pct | needs_adapter | 7 |
| tiny_imagenet_200 | completed | 5 |
| tiny_imagenet_200 | missing | 2 |
| tiny_imagenet_200 | needs_adapter | 7 |

## Open Cells

| dataset | method | priority | status | missing evidence |
| --- | --- | --- | --- | --- |
| cifar10 | same_backbone_dense | P0 | partial | generated_quality |
| cifar10 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| cifar10 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| cifar10 | cofitok_simultaneous | P0 | missing | train,quality |
| cifar10 | cofitok_deep_synthesis | P0 | missing | train,quality |
| cifar10 | edm | P0 | needs_adapter |  |
| cifar10 | improved_diffusion | P0 | needs_adapter |  |
| cifar10 | d_ar | P0 | needs_adapter |  |
| cifar10 | ml_flextok | P1 | needs_adapter |  |
| cifar10 | mar | P1 | needs_adapter |  |
| cifar10 | titok_1d_tokenizer | P1 | needs_adapter |  |
| cifar10 | retok | P1 | needs_adapter |  |
| tiny_imagenet_200 | cofitok_simultaneous | P0 | missing | train,quality |
| tiny_imagenet_200 | cofitok_deep_synthesis | P0 | missing | train,quality |
| tiny_imagenet_200 | edm | P0 | needs_adapter |  |
| tiny_imagenet_200 | improved_diffusion | P0 | needs_adapter |  |
| tiny_imagenet_200 | d_ar | P0 | needs_adapter |  |
| tiny_imagenet_200 | ml_flextok | P1 | needs_adapter |  |
| tiny_imagenet_200 | mar | P1 | needs_adapter |  |
| tiny_imagenet_200 | titok_1d_tokenizer | P1 | needs_adapter |  |
| tiny_imagenet_200 | retok | P1 | needs_adapter |  |
| imagenet_1k_64x64_hf | edm | P0 | needs_adapter |  |
| imagenet_1k_64x64_hf | improved_diffusion | P0 | needs_adapter |  |
| imagenet_1k_64x64_hf | d_ar | P0 | needs_adapter |  |
| imagenet_1k_64x64_hf | ml_flextok | P1 | needs_adapter |  |
| imagenet_1k_64x64_hf | mar | P1 | needs_adapter |  |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | P1 | needs_adapter |  |
| imagenet_1k_64x64_hf | retok | P1 | needs_adapter |  |
| downsampled_imagenet_64 | cofitok_light | P0 | missing | train,quality,generated_quality |
| downsampled_imagenet_64 | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| downsampled_imagenet_64 | channel_mask | P0 | missing | train,quality |
| downsampled_imagenet_64 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| downsampled_imagenet_64 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| downsampled_imagenet_64 | cofitok_simultaneous | P0 | missing | train,quality |
| downsampled_imagenet_64 | cofitok_deep_synthesis | P0 | missing | train,quality |
| downsampled_imagenet_64 | edm | P0 | needs_adapter |  |
| downsampled_imagenet_64 | improved_diffusion | P0 | needs_adapter |  |
| downsampled_imagenet_64 | d_ar | P0 | needs_adapter |  |
| downsampled_imagenet_64 | ml_flextok | P1 | needs_adapter |  |
| downsampled_imagenet_64 | mar | P1 | needs_adapter |  |
| downsampled_imagenet_64 | titok_1d_tokenizer | P1 | needs_adapter |  |
| downsampled_imagenet_64 | retok | P1 | needs_adapter |  |
| ffhq_64 | cofitok_light | P0 | missing | train,quality,generated_quality |
| ffhq_64 | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| ffhq_64 | channel_mask | P0 | missing | train,quality |
| ffhq_64 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| ffhq_64 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| ffhq_64 | cofitok_simultaneous | P0 | missing | train,quality |
| ffhq_64 | cofitok_deep_synthesis | P0 | missing | train,quality |
| ffhq_64 | edm | P0 | needs_adapter |  |
| ffhq_64 | improved_diffusion | P0 | needs_adapter |  |
| ffhq_64 | d_ar | P0 | needs_adapter |  |
| ffhq_64 | ml_flextok | P1 | needs_adapter |  |
| ffhq_64 | mar | P1 | needs_adapter |  |
| ffhq_64 | titok_1d_tokenizer | P1 | needs_adapter |  |
| ffhq_64 | retok | P1 | needs_adapter |  |
| afhqv2_64 | cofitok_light | P0 | missing | train,quality,generated_quality |
| afhqv2_64 | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| afhqv2_64 | channel_mask | P0 | missing | train,quality |
| afhqv2_64 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| afhqv2_64 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| afhqv2_64 | cofitok_simultaneous | P0 | missing | train,quality |
| afhqv2_64 | cofitok_deep_synthesis | P0 | missing | train,quality |
| afhqv2_64 | edm | P0 | needs_adapter |  |
| afhqv2_64 | improved_diffusion | P0 | needs_adapter |  |
| afhqv2_64 | d_ar | P0 | needs_adapter |  |
| afhqv2_64 | ml_flextok | P1 | needs_adapter |  |
| afhqv2_64 | mar | P1 | needs_adapter |  |
| afhqv2_64 | titok_1d_tokenizer | P1 | needs_adapter |  |
| afhqv2_64 | retok | P1 | needs_adapter |  |
| imagenet_256_10pct | cofitok_light | P0 | missing | train,quality,generated_quality |
| imagenet_256_10pct | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| imagenet_256_10pct | channel_mask | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_no_prefix_loss | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_simultaneous | P0 | missing | train,quality |
| imagenet_256_10pct | cofitok_deep_synthesis | P0 | missing | train,quality |
| imagenet_256_10pct | edm | P0 | needs_adapter |  |
| imagenet_256_10pct | improved_diffusion | P0 | needs_adapter |  |
| imagenet_256_10pct | d_ar | P0 | needs_adapter |  |
| imagenet_256_10pct | ml_flextok | P1 | needs_adapter |  |
| imagenet_256_10pct | mar | P1 | needs_adapter |  |
| imagenet_256_10pct | titok_1d_tokenizer | P1 | needs_adapter |  |
| imagenet_256_10pct | retok | P1 | needs_adapter |  |
| imagenet_256 | cofitok_light | P0 | missing | train,quality,generated_quality |
| imagenet_256 | same_backbone_dense | P0 | missing | train,quality,generated_quality |
| imagenet_256 | channel_mask | P0 | missing | train,quality |
| imagenet_256 | cofitok_no_prefix_loss | P0 | missing | train,quality |
| imagenet_256 | cofitok_no_monotonic_loss | P0 | missing | train,quality |
| imagenet_256 | cofitok_simultaneous | P0 | missing | train,quality |
| imagenet_256 | cofitok_deep_synthesis | P0 | missing | train,quality |
| imagenet_256 | edm | P0 | needs_adapter |  |
| imagenet_256 | improved_diffusion | P0 | needs_adapter |  |
| imagenet_256 | d_ar | P0 | needs_adapter |  |
| imagenet_256 | ml_flextok | P1 | needs_adapter |  |
| imagenet_256 | mar | P1 | needs_adapter |  |
| imagenet_256 | titok_1d_tokenizer | P1 | needs_adapter |  |
| imagenet_256 | retok | P1 | needs_adapter |  |
