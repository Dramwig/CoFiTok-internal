# P0 64x64 Comparison

CIFAR10 uses the existing 32x32 3k smoke protocol. Other rows use 64x64 5k protocol. Lower Frechet-style metrics are better.

| dataset | method | status | steps | NFE | sample lowres | sample Inception | denoise PSNR | path AUC | eff K | zero ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | CoFiTok | completed | 3000 | 50 | 7.2663 | 313.8881 | 14.118 | 0.0960 | 6.847 | 0.0000 | train,quality,generated_quality |
| cifar10 | Dense epsilon | completed | 3000 | 50 | 6.4290 | 320.1376 | 14.390 | 1.3022 | 5.159 | 0.0000 | train,quality,generated_quality |
| cifar10 | Channel mask | completed | 3000 |  |  |  | 12.938 | 5.0198 | 1.096 | 0.0000 | train,quality |
| cifar10 | No prefix loss | completed | 3000 |  |  |  | 14.286 | 0.1472 | 6.723 | 0.0000 | train,quality |
| cifar10 | Clean monotonic | completed | 3000 |  |  |  | 14.353 | 0.1032 | 6.740 | 0.0000 | train,quality |
| cifar10 | Simultaneous components | completed | 3000 |  |  |  | 14.417 | 0.0941 | 6.924 | 0.0000 | train,quality |
| cifar10 | Deep S_k | completed | 3000 |  |  |  | 14.102 | 0.0583 | 6.790 | 0.0619 | train,quality |
| cifar10 | Improved DDPM | completed | 3000 | 50 | 7.8610 | 498.6459 |  |  |  |  | baseline_train,baseline_eval |
| cifar10 | EDM | completed | 3000 | 79 | 2.8681 | 351.1658 |  |  |  |  | baseline_train,baseline_eval |
| cifar10 | D-AR | protocol_blocked | 3000 |  |  |  |  |  |  |  | feasibility_report |
| tiny_imagenet_200 | CoFiTok | completed | 5000 | 50 | 8.2197 | 373.2350 | 14.250 | 0.0763 | 6.843 | 0.0000 | train,quality,generated_quality |
| tiny_imagenet_200 | Dense epsilon | completed | 5000 | 50 | 8.2676 | 385.5078 | 14.375 | 0.8459 | 5.377 | 0.0000 | train,quality,generated_quality |
| tiny_imagenet_200 | Channel mask | completed | 5000 |  |  |  | 12.955 | 4.9461 | 1.164 |  | train,quality |
| tiny_imagenet_200 | No prefix loss | completed | 5000 |  |  |  | 14.494 | 0.1122 | 6.571 | 0.0000 | train,quality |
| tiny_imagenet_200 | Clean monotonic | completed | 5000 |  |  |  | 14.327 | 0.0727 | 6.765 | 0.0000 | train,quality |
| tiny_imagenet_200 | Simultaneous components | completed | 5000 |  |  |  | 14.278 | 0.0752 | 6.872 | 0.0000 | train,quality |
| tiny_imagenet_200 | Deep S_k | completed | 5000 |  |  |  | 14.398 | 0.0477 | 6.889 | 0.0505 | train,quality |
| tiny_imagenet_200 | Improved DDPM | completed | 5000 | 50 | 9.0747 | 381.2383 |  |  |  |  | baseline_train,baseline_eval |
| tiny_imagenet_200 | EDM | completed | 5000 | 79 | 3.0996 | 268.7190 |  |  |  |  | baseline_train,baseline_eval |
| tiny_imagenet_200 | D-AR | protocol_blocked | 5000 |  |  |  |  |  |  |  | feasibility_report |
| imagenet_1k_64x64_hf | CoFiTok | completed | 5000 | 50 | 8.5337 | 374.3523 | 15.062 | 0.0641 | 6.768 | 0.0000 | train,quality,generated_quality |
| imagenet_1k_64x64_hf | Dense epsilon | completed | 5000 | 50 | 7.6747 | 388.1463 | 15.131 | 0.8837 | 5.400 | 0.0000 | train,quality,generated_quality |
| imagenet_1k_64x64_hf | Channel mask | completed | 5000 |  |  |  | 13.593 | 4.8844 | 1.111 | 0.0000 | train,quality |
| imagenet_1k_64x64_hf | No prefix loss | completed | 5000 |  |  |  | 14.932 | 0.0939 | 6.560 | 0.0000 | train,quality |
| imagenet_1k_64x64_hf | Clean monotonic | completed | 5000 |  |  |  | 14.904 | 0.0658 | 6.773 | 0.0000 | train,quality |
| imagenet_1k_64x64_hf | Simultaneous components | completed | 5000 |  |  |  | 14.904 | 0.0723 | 6.801 | 0.0000 | train,quality |
| imagenet_1k_64x64_hf | Deep S_k | completed | 5000 |  |  |  | 13.779 | 0.0433 | 6.748 | 0.0450 | train,quality |
| imagenet_1k_64x64_hf | Improved DDPM | completed | 5000 | 50 | 8.8830 | 378.3325 |  |  |  |  | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | EDM | completed | 5000 | 79 | 3.2285 | 258.4005 |  |  |  |  | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | D-AR | protocol_blocked | 5000 |  |  |  |  |  |  |  | feasibility_report |
| downsampled_imagenet_64 | CoFiTok | completed | 5000 | 50 | 8.3554 | 374.6833 | 14.521 | 0.0681 | 6.817 | 0.0000 | train,quality,generated_quality |
| downsampled_imagenet_64 | Dense epsilon | completed | 5000 | 50 | 7.5778 | 377.1191 | 14.651 | 1.0403 | 5.376 | 0.0000 | train,quality,generated_quality |
| downsampled_imagenet_64 | Channel mask | completed | 5000 |  |  |  | 13.250 | 4.9031 | 1.112 | 0.0000 | train,quality |
| downsampled_imagenet_64 | No prefix loss | completed | 5000 |  |  |  | 14.474 | 0.0953 | 6.573 | 0.0000 | train,quality |
| downsampled_imagenet_64 | Clean monotonic | completed | 5000 |  |  |  | 14.664 | 0.0644 | 6.826 | 0.0000 | train,quality |
| downsampled_imagenet_64 | Simultaneous components | completed | 5000 |  |  |  | 14.586 | 0.0681 | 6.887 | 0.0000 | train,quality |
| downsampled_imagenet_64 | Deep S_k | completed | 5000 |  |  |  | 14.502 | 0.0397 | 6.754 | 0.0535 | train,quality |
| downsampled_imagenet_64 | Improved DDPM | completed | 5000 | 50 | 8.8044 | 355.1061 |  |  |  |  | baseline_train,baseline_eval |
| downsampled_imagenet_64 | EDM | completed | 5000 | 79 | 2.7540 | 248.3076 |  |  |  |  | baseline_train,baseline_eval |
| downsampled_imagenet_64 | D-AR | protocol_blocked | 5000 |  |  |  |  |  |  |  | feasibility_report |
| ffhq_64 | CoFiTok | completed | 5000 | 50 | 8.7379 | 425.7089 | 15.024 | 0.0682 | 6.818 | 0.0000 | train,quality,generated_quality |
| ffhq_64 | Dense epsilon | completed | 5000 | 50 | 8.0881 | 424.1108 | 15.144 | 1.0807 | 5.415 | 0.0000 | train,quality,generated_quality |
| ffhq_64 | Channel mask | completed | 5000 |  |  |  | 13.285 | 4.8775 | 1.113 | 0.0000 | train,quality |
| ffhq_64 | No prefix loss | completed | 5000 |  |  |  | 14.948 | 0.0996 | 6.523 | 0.0000 | train,quality |
| ffhq_64 | Clean monotonic | completed | 5000 |  |  |  | 15.135 | 0.0672 | 6.770 | 0.0000 | train,quality |
| ffhq_64 | Simultaneous components | completed | 5000 |  |  |  | 15.157 | 0.0637 | 6.861 | 0.0000 | train,quality |
| ffhq_64 | Deep S_k | completed | 5000 |  |  |  | 14.951 | 0.0456 | 6.787 | 0.0518 | train,quality |
| ffhq_64 | Improved DDPM | completed | 5000 | 50 | 9.4360 | 369.0054 |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | EDM | completed | 5000 | 79 | 2.5077 | 325.4898 |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | D-AR | protocol_blocked | 5000 |  |  |  |  |  |  |  | feasibility_report |
| afhqv2_64 | CoFiTok | completed | 5000 | 50 | 7.0781 | 331.0574 | 15.167 | 0.0643 | 6.819 | 0.0000 | train,quality,generated_quality |
| afhqv2_64 | Dense epsilon | completed | 5000 | 50 | 6.5792 | 345.9386 | 15.196 | 1.0118 | 5.462 | 0.0000 | train,quality,generated_quality |
| afhqv2_64 | Channel mask | completed | 5000 |  |  |  | 13.384 | 4.7483 | 1.112 | 0.0000 | train,quality |
| afhqv2_64 | No prefix loss | completed | 5000 |  |  |  | 15.081 | 0.0936 | 6.538 | 0.0000 | train,quality |
| afhqv2_64 | Clean monotonic | completed | 5000 |  |  |  | 15.161 | 0.0665 | 6.753 | 0.0000 | train,quality |
| afhqv2_64 | Simultaneous components | completed | 5000 |  |  |  | 15.105 | 0.0604 | 6.846 | 0.0000 | train,quality |
| afhqv2_64 | Deep S_k | completed | 5000 |  |  |  | 15.110 | 0.0417 | 6.794 | 0.0561 | train,quality |
| afhqv2_64 | Improved DDPM | completed | 5000 | 50 | 8.2918 | 335.1814 |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | EDM | completed | 5000 | 79 | 1.9621 | 230.3979 |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | D-AR | protocol_blocked | 5000 |  |  |  |  |  |  |  | feasibility_report |
