# Paper Evidence Report

This report is generated from current artifacts. Lower Frechet-style values are better.

## Completion State

| status | count |
| --- | --- |
| completed | 72 |
| completed_eval_only | 16 |
| protocol_blocked | 24 |

Interpretation: P0 train/eval evidence is complete for current runnable protocols. P1 tokenizer rows are eval-only. D-AR/MAR/ReTok remain protocol-blocked and should not be reported as completed baselines.

## P0 Generation Comparison

| dataset | CoFiTok lowres | Dense lowres | DDPM lowres | EDM lowres | lowres winner | CoFiTok rank | CoFiTok Inc | Dense Inc | DDPM Inc | EDM Inc | Inc winner | CoFiTok rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 7.2663 | 6.4290 | 7.8610 | 2.8681 | EDM | 3 | 313.8881 | 320.1376 | 498.6459 | 351.1658 | CoFiTok | 1 |
| tiny_imagenet_200 | 8.2197 | 8.2676 | 9.0747 | 3.0996 | EDM | 2 | 373.2350 | 385.5078 | 381.2383 | 268.7190 | EDM | 2 |
| imagenet_1k_64x64_hf | 8.5337 | 7.6747 | 8.8830 | 3.2285 | EDM | 3 | 374.3523 | 388.1463 | 378.3325 | 258.4005 | EDM | 2 |
| downsampled_imagenet_64 | 8.3554 | 7.5778 | 8.8044 | 2.7540 | EDM | 3 | 374.6833 | 377.1191 | 355.1061 | 248.3076 | EDM | 3 |
| ffhq_64 | 8.7379 | 8.0881 | 9.4360 | 2.5077 | EDM | 3 | 425.7089 | 424.1108 | 369.0054 | 325.4898 | EDM | 4 |
| afhqv2_64 | 7.0781 | 6.5792 | 8.2918 | 1.9621 | EDM | 3 | 331.0574 | 345.9386 | 335.1814 | 230.3979 | EDM | 2 |
| imagenet_256_10pct | 10.4423 | 10.5062 | 10.3989 | 10.3918 | EDM | 3 | 414.1863 | 337.8743 | 419.5981 | 368.6788 | Dense epsilon | 3 |
| imagenet_256 | 10.5431 | 10.2803 | 10.3998 | 10.3975 | Dense epsilon | 4 | 417.2551 | 354.4296 | 414.2991 | 368.4282 | Dense epsilon | 4 |

## CoFiTok Diagnostic Evidence

| dataset | CoFiTok PSNR | Dense PSNR | Channel PSNR | CoFiTok path AUC | Dense path AUC | CoFiTok eff K | Dense eff K | CoFiTok zero | Deep-S zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 14.118 | 14.390 | 12.938 | 0.0960 | 1.3022 | 6.847 | 5.159 | 0.0000 | 0.0619 |
| tiny_imagenet_200 | 14.250 | 14.375 | 12.955 | 0.0763 | 0.8459 | 6.843 | 5.377 | 0.0000 | 0.0505 |
| imagenet_1k_64x64_hf | 15.062 | 15.131 | 13.593 | 0.0641 | 0.8837 | 6.768 | 5.400 | 0.0000 | 0.0450 |
| downsampled_imagenet_64 | 14.521 | 14.651 | 13.250 | 0.0681 | 1.0403 | 6.817 | 5.376 | 0.0000 | 0.0535 |
| ffhq_64 | 15.024 | 15.144 | 13.285 | 0.0682 | 1.0807 | 6.818 | 5.415 | 0.0000 | 0.0518 |
| afhqv2_64 | 15.167 | 15.196 | 13.384 | 0.0643 | 1.0118 | 6.819 | 5.462 | 0.0000 | 0.0561 |
| imagenet_256_10pct | 13.627 | 13.867 | 13.807 | 0.1347 | 1.5609 | 3.354 | 3.664 | 0.0000 | 0.0179 |
| imagenet_256 | 13.809 | 13.877 | 13.761 | 0.1255 | 1.4917 | 3.367 | 3.663 | 0.0000 | 0.0185 |

## P1 Tokenizer Reconstruction

| dataset | baseline | images | src res | eval res | NFE | PSNR | lowres Frechet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afhqv2_64 | ml_flextok | 64 | 64 | 256 | 4 | 23.8092 | 0.1007 |
| afhqv2_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.2349 | 0.5780 |
| cifar10 | ml_flextok | 64 | 32 | 256 | 4 | 26.4198 | 0.0969 |
| cifar10 | titok_1d_tokenizer | 64 | 32 | 256 | 1 | 19.3168 | 0.4682 |
| downsampled_imagenet_64 | ml_flextok | 64 | 64 | 256 | 4 | 22.8808 | 0.1042 |
| downsampled_imagenet_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.0297 | 0.5557 |
| ffhq_64 | ml_flextok | 64 | 64 | 256 | 4 | 23.9493 | 0.1158 |
| ffhq_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.4036 | 0.6161 |
| imagenet_1k_64x64_hf | ml_flextok | 64 | 64 | 256 | 4 | 24.1108 | 0.0973 |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 18.4670 | 0.4658 |
| imagenet_256 | ml_flextok | 64 | 256 | 256 | 4 | 19.5459 | 0.1071 |
| imagenet_256 | titok_1d_tokenizer | 64 | 256 | 256 | 1 | 16.0468 | 0.4646 |
| imagenet_256_10pct | ml_flextok | 64 | 256 | 256 | 4 | 19.5465 | 0.1072 |
| imagenet_256_10pct | titok_1d_tokenizer | 64 | 256 | 256 | 1 | 16.0468 | 0.4646 |
| tiny_imagenet_200 | ml_flextok | 64 | 64 | 256 | 4 | 21.8345 | 0.1153 |
| tiny_imagenet_200 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.1144 | 0.4924 |

## Machine Summary

| claim | evidence |
| --- | --- |
| Generation quality | CoFiTok is best on 0/8 lowres rows and 1/8 Inception rows; average ranks 3.00 lowres and 2.62 Inception. |
| Prefix/factorization behavior | CoFiTok path AUC is lower than dense on 8/8 datasets; mean PSNR gap over channel-mask ablation is 1.076 dB. |
| Restricted synthesis safety | CoFiTok zero-token ratio remains zero in current rows, while deep-S_k has nonzero zero-token ratio on 8/8 datasets. |
| Top-tier claim stance | Support is strongest for a scoped method paper about ordered restricted dense-noise factorization and prefix-controllable denoising. Current evidence does not support claiming broad unconditional generation SOTA. |
