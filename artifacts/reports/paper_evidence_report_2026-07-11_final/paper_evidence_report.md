# Paper Evidence Report

This report is generated from current artifacts. Lower Frechet-style values are better.

## Completion State

| status | count |
| --- | --- |
| completed | 80 |
| completed_eval_only | 16 |
| protocol_blocked | 24 |

Interpretation: P0 train/eval evidence is complete for the locked matched-dataset/optimizer-step protocols. Batch size, nominal images seen, parameter count, and NFE remain explicit because this is not a compute-matched SOTA table. P1 tokenizer rows and official related-method rows are eval-only secondary evidence.

## P0 Generation Comparison

| dataset | CoFiTok lowres | Direct dense lowres | DDPM lowres | EDM lowres | lowres winner | CoFiTok rank | CoFiTok Inc | Direct dense Inc | DDPM Inc | EDM Inc | Inc winner | CoFiTok rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 6.4656 | 5.7996 | 7.8610 | 2.8681 | EDM | 3 | 312.0042 | 359.5046 | 498.6459 | 351.1658 | CoFiTok | 1 |
| tiny_imagenet_200 | 8.4871 | 6.9867 | 9.0747 | 3.0996 | EDM | 3 | 373.6340 | 392.3954 | 381.2383 | 268.7190 | EDM | 2 |
| imagenet_1k_64x64_hf | 8.0915 | 6.8916 | 8.8830 | 3.2285 | EDM | 3 | 390.9865 | 417.7856 | 378.3325 | 258.4005 | EDM | 3 |
| downsampled_imagenet_64 | 8.3554 | 6.8707 | 8.8044 | 2.7540 | EDM | 3 | 374.6833 | 383.7239 | 355.1061 | 248.3076 | EDM | 3 |
| ffhq_64 | 8.7379 | 6.8419 | 9.4360 | 2.5077 | EDM | 3 | 425.7089 | 427.0490 | 369.0054 | 325.4898 | EDM | 3 |
| afhqv2_64 | 7.0781 | 5.6786 | 8.2918 | 1.9621 | EDM | 3 | 331.0574 | 333.4181 | 335.1814 | 230.3979 | EDM | 2 |
| imagenet_256_10pct | 10.4423 | 12.5863 | 10.3989 | 10.3918 | EDM | 3 | 414.1863 | 366.4175 | 419.5981 | 368.6788 | Direct dense epsilon | 3 |
| imagenet_256 | 10.5431 | 10.2398 | 10.3998 | 10.3975 | Direct dense epsilon | 4 | 417.2551 | 354.1265 | 414.2991 | 368.4282 | Direct dense epsilon | 4 |

## CoFiTok Diagnostic Evidence

| dataset | CoFiTok PSNR | Direct dense PSNR | Channel PSNR | CoFiTok path AUC | Endpoint-only path AUC | CoFiTok eff K | Endpoint-only eff K | CoFiTok zero | Deep-S zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 14.245 | 14.407 | 13.063 | 0.0967 | 0.7906 | 6.837 | 5.244 | 0.0000 | 0.0619 |
| tiny_imagenet_200 | 14.348 | 14.453 | 13.015 | 0.0731 | 1.1177 | 6.800 | 5.567 | 0.0000 | 0.0505 |
| imagenet_1k_64x64_hf | 14.897 | 15.004 | 13.548 | 0.0672 | 1.1120 | 6.791 | 5.329 | 0.0000 | 0.0522 |
| downsampled_imagenet_64 | 14.521 | 14.656 | 13.250 | 0.0705 | 1.0502 | 6.818 | 5.362 | 0.0000 | 0.0535 |
| ffhq_64 | 15.024 | 15.126 | 13.285 | 0.0676 | 1.0830 | 6.818 | 5.408 | 0.0000 | 0.0518 |
| afhqv2_64 | 15.167 | 15.154 | 13.384 | 0.0653 | 1.0185 | 6.818 | 5.453 | 0.0000 | 0.0561 |
| imagenet_256_10pct | 13.627 | 13.490 | 13.807 | 0.1416 | 1.5686 | 3.355 | 3.665 | 0.0000 | 0.0179 |
| imagenet_256 | 13.809 | 13.770 | 13.761 | 0.1325 | 1.5003 | 3.369 | 3.667 | 0.0000 | 0.0185 |

## Matched 20k Two-Seed Evidence

| dataset | method | seeds | endpoint x0 MSE (t=500) | path AUC | effective K |
| --- | --- | --- | --- | --- | --- |
| Tiny ImageNet-200 | Endpoint-only factorized | 2 | 0.1253 +/- 0.0001 | 1.0098 +/- 0.1430 | 5.312 +/- 0.049 |
| Tiny ImageNet-200 | CoFiTok | 2 | 0.1296 +/- 0.0015 | 0.0378 +/- 0.0007 | 6.766 +/- 0.015 |
| ImageNet-64 HF | Endpoint-only factorized | 2 | 0.1096 +/- 0.0006 | 0.9984 +/- 0.1210 | 5.169 +/- 0.108 |
| ImageNet-64 HF | CoFiTok | 2 | 0.1151 +/- 0.0016 | 0.0379 +/- 0.0037 | 6.752 +/- 0.012 |

## ImageNet-256 20k Confirmatory Scaling

| seed | endpoint-only path AUC | ordered | random | reverse | all non-ID mean | ordered rank/24 | non-ID delta CI low | direct-dense endpoint MSE | CoFiTok endpoint MSE | zero ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 103 | 1.78791 | 0.06642 | 0.51300 | 0.71921 | 0.31573 | 1 | 0.24891 | 0.12339 | 0.11327 | 0.0000 |
| 139 | 1.44015 | 0.06784 | 0.47117 | 0.68214 | 0.30820 | 1 | 0.24002 | 0.12438 | 0.11785 | 0.0000 |

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
| MAR | imagenet_256 | official LTH14 PTH model_ema eval-only | completed_eval_only_50k | 50000 | 2.3385 | 4.6999 | 62.8979 | 0.8216 | 0.5716 |
| ReTok | imagenet_256 | official pretrained eval-only | completed_eval_only_50k | 50000 | 2.2189 | 5.8956 | 245.9392 | 0.8159 | 0.5994 |

## Machine Summary

| claim | evidence |
| --- | --- |
| Generation quality | CoFiTok is best on 0/8 lowres rows and 1/8 Inception rows; average ranks 3.12 lowres and 2.62 Inception. |
| Prefix/factorization behavior | CoFiTok path AUC is lower than the endpoint-only factorized control on 8/8 datasets; mean PSNR gap over channel-mask ablation is 1.066 dB. |
| Repeated long-budget prefix control | CoFiTok path AUC is lower in 4/4 paired dataset-seed runs, with mean relative reduction 96.18%; mean endpoint-MSE change is +4.20%. |
| ImageNet-256 confirmatory scaling | ImageNet-256 K4 20k confirmatory gates: 7/7 passed; overall PASS; mean endpoint-MSE change -6.73%; ordered ranks among all 24 permutations 1/1. |
| Restricted synthesis safety | CoFiTok zero-token ratio remains zero in current rows, while deep-S_k has nonzero zero-token ratio on 8/8 datasets. |
| Related-method coverage | completed official 50K: D-AR, MAR, ReTok. All rows remain secondary and outside P0 matched-dataset/step training. |
| Top-tier claim stance | Support is strongest for a scoped method paper about ordered restricted dense-noise factorization and prefix-controllable denoising. Current evidence does not support claiming broad unconditional generation SOTA. |
