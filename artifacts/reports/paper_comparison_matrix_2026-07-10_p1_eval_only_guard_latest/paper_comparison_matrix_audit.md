# Paper Comparison Matrix Audit

This file audits current evidence only. It is not a completed result table.

## Status Counts

| status | count |
| --- | --- |
| completed | 72 |
| completed_eval_only | 2 |
| needs_eval | 14 |
| protocol_blocked | 24 |

## Priority Counts

| priority | status | count |
| --- | --- | --- |
| P0 | completed | 72 |
| P0 | protocol_blocked | 8 |
| P1 | completed_eval_only | 2 |
| P1 | needs_eval | 14 |
| P1 | protocol_blocked | 16 |

## Dataset Counts

| dataset | status | count |
| --- | --- | --- |
| afhqv2_64 | completed | 9 |
| afhqv2_64 | needs_eval | 2 |
| afhqv2_64 | protocol_blocked | 3 |
| cifar10 | completed | 9 |
| cifar10 | completed_eval_only | 2 |
| cifar10 | protocol_blocked | 3 |
| downsampled_imagenet_64 | completed | 9 |
| downsampled_imagenet_64 | needs_eval | 2 |
| downsampled_imagenet_64 | protocol_blocked | 3 |
| ffhq_64 | completed | 9 |
| ffhq_64 | needs_eval | 2 |
| ffhq_64 | protocol_blocked | 3 |
| imagenet_1k_64x64_hf | completed | 9 |
| imagenet_1k_64x64_hf | needs_eval | 2 |
| imagenet_1k_64x64_hf | protocol_blocked | 3 |
| imagenet_256 | completed | 9 |
| imagenet_256 | needs_eval | 2 |
| imagenet_256 | protocol_blocked | 3 |
| imagenet_256_10pct | completed | 9 |
| imagenet_256_10pct | needs_eval | 2 |
| imagenet_256_10pct | protocol_blocked | 3 |
| tiny_imagenet_200 | completed | 9 |
| tiny_imagenet_200 | needs_eval | 2 |
| tiny_imagenet_200 | protocol_blocked | 3 |

## Open Cells

| dataset | method | priority | status | missing evidence |
| --- | --- | --- | --- | --- |
| cifar10 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| cifar10 | ml_flextok | P1 | completed_eval_only |  |
| cifar10 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| cifar10 | titok_1d_tokenizer | P1 | completed_eval_only |  |
| cifar10 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| tiny_imagenet_200 | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| tiny_imagenet_200 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_1k_64x64_hf | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| downsampled_imagenet_64 | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| downsampled_imagenet_64 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| ffhq_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| ffhq_64 | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| ffhq_64 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| ffhq_64 | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| ffhq_64 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| afhqv2_64 | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| afhqv2_64 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_256_10pct | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_256_10pct | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_256 | d_ar | P0 | protocol_blocked | baseline_train,baseline_eval |
| imagenet_256 | ml_flextok | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_256 | mar | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
| imagenet_256 | titok_1d_tokenizer | P1 | needs_eval | baseline_train_or_eval,baseline_eval |
| imagenet_256 | retok | P1 | protocol_blocked | baseline_train_or_eval,baseline_eval |
