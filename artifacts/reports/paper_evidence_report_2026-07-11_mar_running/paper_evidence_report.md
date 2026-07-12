# Paper Evidence Report

This report is generated from current artifacts. Lower Frechet-style values are better.

## Completion State

| status | count |
| --- | --- |
| completed | 72 |
| completed_eval_only | 16 |
| protocol_blocked | 24 |

Interpretation: P0 train/eval evidence is complete for current runnable protocols. P1 tokenizer rows are eval-only. Official related-method rows remain secondary and must not be reported as completed fair-training baselines in the all-dataset same-budget matrix.

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

## Matched 20k Two-Seed Evidence

| dataset | method | seeds | final MSE | path AUC | effective K |
| --- | --- | --- | --- | --- | --- |
| Tiny ImageNet-200 | Dense epsilon | 2 | 0.1403 +/- 0.0014 | 1.0100 +/- 0.1444 | 5.314 +/- 0.032 |
| Tiny ImageNet-200 | CoFiTok | 2 | 0.1442 +/- 0.0022 | 0.0395 +/- 0.0009 | 6.771 +/- 0.013 |
| ImageNet-64 HF | Dense epsilon | 2 | 0.0836 +/- 0.0008 | 0.9881 +/- 0.1223 | 5.170 +/- 0.099 |
| ImageNet-64 HF | CoFiTok | 2 | 0.0861 +/- 0.0013 | 0.0333 +/- 0.0039 | 6.748 +/- 0.013 |

## P1 Tokenizer Reconstruction

| dataset | baseline | images | src res | eval res | NFE | PSNR | lowres Frechet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afhqv2_64 | ml_flextok | 256 | 64 | 256 | 4 | 24.0421 | 0.0358 |
| afhqv2_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.2868 | 0.2168 |
| cifar10 | ml_flextok | 256 | 32 | 256 | 4 | 26.7679 | 0.0373 |
| cifar10 | titok_1d_tokenizer | 256 | 32 | 256 | 1 | 19.3461 | 0.1678 |
| downsampled_imagenet_64 | ml_flextok | 256 | 64 | 256 | 4 | 22.5683 | 0.0418 |
| downsampled_imagenet_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 16.7253 | 0.1956 |
| ffhq_64 | ml_flextok | 256 | 64 | 256 | 4 | 24.0572 | 0.0422 |
| ffhq_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.3911 | 0.2307 |
| imagenet_1k_64x64_hf | ml_flextok | 256 | 64 | 256 | 4 | 23.5966 | 0.0390 |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 18.0363 | 0.1817 |
| imagenet_256 | ml_flextok | 256 | 256 | 256 | 4 | 19.1226 | 0.0438 |
| imagenet_256 | titok_1d_tokenizer | 256 | 256 | 256 | 1 | 15.5439 | 0.1959 |
| imagenet_256_10pct | ml_flextok | 256 | 256 | 256 | 4 | 19.1230 | 0.0438 |
| imagenet_256_10pct | titok_1d_tokenizer | 256 | 256 | 256 | 1 | 15.5439 | 0.1959 |
| tiny_imagenet_200 | ml_flextok | 256 | 64 | 256 | 4 | 21.8110 | 0.0442 |
| tiny_imagenet_200 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.3008 | 0.1600 |

## Official Related-Method Eval-Only Rows

| method | dataset | protocol | status | samples | FID | sFID | IS | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-AR | imagenet_256 | official pretrained eval-only | completed_eval_only_50k | 50000 | 2.6281 | 6.7195 | 285.8914 | 0.8029 | 0.5884 |
| MAR | imagenet_256 | official pretrained eval-only | sampling_or_eval_pending | 3072 |  |  |  |  |  |
| ReTok | imagenet_256 | official pretrained eval-only | completed_eval_only_50k | 50000 | 2.2189 | 5.8956 | 245.9392 | 0.8159 | 0.5994 |

## Machine Summary

| claim | evidence |
| --- | --- |
| Generation quality | CoFiTok is best on 0/8 lowres rows and 1/8 Inception rows; average ranks 3.00 lowres and 2.62 Inception. |
| Prefix/factorization behavior | CoFiTok path AUC is lower than dense on 8/8 datasets; mean PSNR gap over channel-mask ablation is 1.076 dB. |
| Repeated long-budget prefix control | CoFiTok path AUC is lower in 4/4 paired dataset-seed runs, with mean relative reduction 96.31%; mean endpoint-MSE change is +2.89%. |
| Restricted synthesis safety | CoFiTok zero-token ratio remains zero in current rows, while deep-S_k has nonzero zero-token ratio on 8/8 datasets. |
| Related-method coverage | completed official 50K: D-AR, ReTok; running/eval pending: MAR. All rows remain secondary and outside P0 fair training. |
| Top-tier claim stance | Support is strongest for a scoped method paper about ordered restricted dense-noise factorization and prefix-controllable denoising. Current evidence does not support claiming broad unconditional generation SOTA. |
