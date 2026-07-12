# P0 Current Comparison

This table audits current P0 evidence across available datasets. Lower Frechet-style metrics are better. It is not a formal SOTA FID table.

| dataset | protocol | method | status | steps | NFE | quality n | sample/real n | sample lowres | sample Inception | denoise PSNR | denoise Inception | LPIPS | path AUC | eff K | zero ratio | params | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 32x32 | CoFiTok | partial | 3000 |  | 256 | / |  |  | 14.170 | 327.8888 | 0.1674 | 0.1003 | 6.834 | 0.0000 |  | train,quality |
| cifar10 | 32x32 | Direct dense epsilon | missing | 3000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| cifar10 | 32x32 | Endpoint-only factorized | partial | 3000 |  | 256 | / |  |  | 14.390 | 334.0694 | 0.1639 | 0.8030 | 5.240 | 0.0000 |  | train,quality |
| cifar10 | 32x32 | Channel mask | completed | 3000 |  | 256 | / |  |  | 12.938 |  |  | 4.9586 | 1.086 | 0.0000 |  | train,quality |
| cifar10 | 32x32 | No prefix loss | partial | 3000 |  |  | / |  |  |  |  |  | 0.1472 | 6.723 | 0.0000 |  | train |
| cifar10 | 32x32 | No monotonic loss | partial | 3000 |  |  | / |  |  |  |  |  | 0.1032 | 6.740 | 0.0000 |  | train |
| cifar10 | 32x32 | Simultaneous | partial | 3000 |  |  | / |  |  |  |  |  | 0.0941 | 6.924 | 0.0000 |  | train |
| cifar10 | 32x32 | Deep S_k | partial | 3000 |  |  | / |  |  |  |  |  | 0.0583 | 6.790 | 0.0619 |  | train |
| cifar10 | 32x32 | Improved DDPM | completed | 3000 | 50 |  | 1024/4096 | 7.8610 | 498.6459 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| cifar10 | 32x32 | EDM | completed | 3000 | 79 |  | 1024/4096 | 2.8681 | 351.1658 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| cifar10 | 32x32 | D-AR | protocol_blocked | 3000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| tiny_imagenet_200 | 64x64 | CoFiTok | partial | 5000 |  | 256 | / |  |  | 14.297 | 343.6019 | 0.5652 | 0.0759 | 6.804 | 0.0000 |  | train,quality |
| tiny_imagenet_200 | 64x64 | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| tiny_imagenet_200 | 64x64 | Endpoint-only factorized | partial | 5000 |  | 256 | / |  |  | 14.375 | 346.7260 | 0.5689 | 1.1171 | 5.568 | 0.0000 |  | train,quality |
| tiny_imagenet_200 | 64x64 | Channel mask | completed | 5000 |  | 256 | / |  |  | 12.955 |  |  |  |  |  |  | train,quality |
| tiny_imagenet_200 | 64x64 | No prefix loss | partial | 5000 |  |  | / |  |  |  |  |  | 0.1122 | 6.571 | 0.0000 |  | train |
| tiny_imagenet_200 | 64x64 | No monotonic loss | partial | 5000 |  |  | / |  |  |  |  |  | 0.0727 | 6.765 | 0.0000 |  | train |
| tiny_imagenet_200 | 64x64 | Simultaneous | partial | 5000 |  |  | / |  |  |  |  |  | 0.0752 | 6.872 | 0.0000 |  | train |
| tiny_imagenet_200 | 64x64 | Deep S_k | partial | 5000 |  |  | / |  |  |  |  |  | 0.0477 | 6.889 | 0.0505 |  | train |
| tiny_imagenet_200 | 64x64 | Improved DDPM | completed | 5000 | 50 |  | 1024/4096 | 9.0747 | 381.2383 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| tiny_imagenet_200 | 64x64 | EDM | completed | 5000 | 79 |  | 1024/4096 | 3.0996 | 268.7190 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| tiny_imagenet_200 | 64x64 | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_1k_64x64_hf | 64x64 HF | CoFiTok | partial | 5000 |  | 256 | / |  |  | 14.917 | 321.7677 | 0.5495 | 0.0620 | 6.787 | 0.0000 |  | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_1k_64x64_hf | 64x64 HF | Endpoint-only factorized | partial | 5000 |  | 256 | / |  |  | 15.011 | 331.3044 | 0.5354 | 1.1000 | 5.339 | 0.0000 |  | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | Channel mask | completed | 5000 |  | 256 | / |  |  | 13.593 |  |  | 4.8844 | 1.111 | 0.0000 |  | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | No prefix loss | partial | 5000 |  |  | / |  |  |  |  |  | 0.0939 | 6.560 | 0.0000 |  | train |
| imagenet_1k_64x64_hf | 64x64 HF | No monotonic loss | partial | 5000 |  |  | / |  |  |  |  |  | 0.0658 | 6.773 | 0.0000 |  | train |
| imagenet_1k_64x64_hf | 64x64 HF | Simultaneous | partial | 5000 |  |  | / |  |  |  |  |  | 0.0600 | 6.845 | 0.0000 |  | train |
| imagenet_1k_64x64_hf | 64x64 HF | Deep S_k | completed | 5000 |  | 256 | / |  |  | 15.034 |  |  | 0.0366 | 6.792 | 0.0522 |  | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | Improved DDPM | completed | 5000 | 50 |  | 1024/4096 | 8.8830 | 378.3325 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | 64x64 HF | EDM | completed | 5000 | 79 |  | 1024/4096 | 3.2285 | 258.4005 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | 64x64 HF | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| downsampled_imagenet_64 | 64x64 strict | CoFiTok | partial | 5000 |  | 512 | / |  |  | 14.521 | 307.8558 | 0.5415 | 0.0681 | 6.817 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| downsampled_imagenet_64 | 64x64 strict | Endpoint-only factorized | partial | 5000 |  | 512 | / |  |  | 14.651 | 283.7165 | 0.5554 | 1.0403 | 5.376 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Channel mask | completed | 5000 |  | 512 | / |  |  | 13.250 | 257.7474 | 0.6823 | 4.9031 | 1.112 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | No prefix loss | completed | 5000 |  | 512 | / |  |  | 14.474 | 306.5019 | 0.5471 | 0.0953 | 6.573 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | No monotonic loss | completed | 5000 |  | 512 | / |  |  | 14.664 | 319.4717 | 0.5510 | 0.0644 | 6.826 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Simultaneous | completed | 5000 |  | 512 | / |  |  | 14.586 | 306.6071 | 0.5448 | 0.0681 | 6.887 | 0.0000 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Deep S_k | completed | 5000 |  | 512 | / |  |  | 14.502 | 289.4443 | 0.5502 | 0.0397 | 6.754 | 0.0535 |  | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Improved DDPM | completed | 5000 | 50 |  | 1024/4096 | 8.8044 | 355.1061 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| downsampled_imagenet_64 | 64x64 strict | EDM | completed | 5000 | 79 |  | 1024/4096 | 2.7540 | 248.3076 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| downsampled_imagenet_64 | 64x64 strict | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| ffhq_64 | 64x64 | CoFiTok | partial | 5000 |  | 512 | / |  |  | 15.024 | 355.7610 | 0.5027 | 0.0682 | 6.818 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| ffhq_64 | 64x64 | Endpoint-only factorized | partial | 5000 |  | 512 | / |  |  | 15.144 | 344.1628 | 0.4925 | 1.0807 | 5.415 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | Channel mask | completed | 5000 |  | 512 | / |  |  | 13.285 | 307.0517 | 0.6535 | 4.8775 | 1.113 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | No prefix loss | completed | 5000 |  | 512 | / |  |  | 14.948 | 365.6998 | 0.4794 | 0.0996 | 6.523 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | No monotonic loss | completed | 5000 |  | 512 | / |  |  | 15.135 | 358.2656 | 0.4709 | 0.0672 | 6.770 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | Simultaneous | completed | 5000 |  | 512 | / |  |  | 15.157 | 341.5909 | 0.4884 | 0.0637 | 6.861 | 0.0000 |  | train,quality |
| ffhq_64 | 64x64 | Deep S_k | completed | 5000 |  | 512 | / |  |  | 14.951 | 328.1341 | 0.5211 | 0.0456 | 6.787 | 0.0518 |  | train,quality |
| ffhq_64 | 64x64 | Improved DDPM | completed | 5000 | 50 |  | 1024/4096 | 9.4360 | 369.0054 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | 64x64 | EDM | completed | 5000 | 79 |  | 1024/4096 | 2.5077 | 325.4898 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | 64x64 | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| afhqv2_64 | 64x64 | CoFiTok | partial | 5000 |  | 512 | / |  |  | 15.167 | 265.0387 | 0.4366 | 0.0643 | 6.819 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| afhqv2_64 | 64x64 | Endpoint-only factorized | partial | 5000 |  | 512 | / |  |  | 15.196 | 251.5074 | 0.4547 | 1.0118 | 5.462 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | Channel mask | completed | 5000 |  | 512 | / |  |  | 13.384 | 262.9012 | 0.6255 | 4.7483 | 1.112 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | No prefix loss | completed | 5000 |  | 512 | / |  |  | 15.081 | 257.0476 | 0.4444 | 0.0936 | 6.538 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | No monotonic loss | completed | 5000 |  | 512 | / |  |  | 15.161 | 282.3491 | 0.4510 | 0.0665 | 6.753 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | Simultaneous | completed | 5000 |  | 512 | / |  |  | 15.105 | 245.0019 | 0.4629 | 0.0604 | 6.846 | 0.0000 |  | train,quality |
| afhqv2_64 | 64x64 | Deep S_k | completed | 5000 |  | 512 | / |  |  | 15.110 | 255.6228 | 0.4552 | 0.0417 | 6.794 | 0.0561 |  | train,quality |
| afhqv2_64 | 64x64 | Improved DDPM | completed | 5000 | 50 |  | 1024/4096 | 8.2918 | 335.1814 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | 64x64 | EDM | completed | 5000 | 79 |  | 1024/4096 | 1.9621 | 230.3979 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | 64x64 | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_256_10pct | 256x256 10pct | CoFiTok | partial | 5000 |  | 256 | / |  |  | 13.627 | 330.2744 | 0.8962 | 0.1347 | 3.354 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_256_10pct | 256x256 10pct | Endpoint-only factorized | partial | 5000 |  | 256 | / |  |  | 13.867 | 330.7862 | 0.8920 | 1.5609 | 3.664 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | Channel mask | completed | 5000 |  | 256 | / |  |  | 13.807 | 338.2386 | 0.8782 | 3.4933 | 1.068 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | No prefix loss | completed | 5000 |  | 256 | / |  |  | 13.865 | 323.1635 | 0.8814 | 0.1700 | 3.398 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | No monotonic loss | completed | 5000 |  | 256 | / |  |  | 13.853 | 308.7236 | 0.8142 | 0.1157 | 3.398 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | Simultaneous | completed | 5000 |  | 256 | / |  |  | 13.850 | 312.5193 | 0.8626 | 0.1315 | 3.412 | 0.0000 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | Deep S_k | completed | 5000 |  | 256 | / |  |  | 14.483 | 325.1934 | 0.8572 | 0.0545 | 3.394 | 0.0179 |  | train,quality |
| imagenet_256_10pct | 256x256 10pct | Improved DDPM | completed | 5000 | 50 |  | 512/2048 | 10.3989 | 419.5981 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_256_10pct | 256x256 10pct | EDM | completed | 5000 | 79 |  | 512/2048 | 10.3918 | 368.6788 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_256_10pct | 256x256 10pct | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_256 | 256x256 full | CoFiTok | partial | 5000 |  | 256 | / |  |  | 13.809 | 329.2057 | 0.8438 | 0.1255 | 3.367 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | Direct dense epsilon | missing | 5000 |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_256 | 256x256 full | Endpoint-only factorized | partial | 5000 |  | 256 | / |  |  | 13.877 | 323.9321 | 0.8360 | 1.4917 | 3.663 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | Channel mask | completed | 5000 |  | 256 | / |  |  | 13.761 | 340.7570 | 0.8794 | 3.5299 | 1.068 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | No prefix loss | completed | 5000 |  | 256 | / |  |  | 13.388 | 324.3095 | 0.8812 | 0.1731 | 3.397 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | No monotonic loss | completed | 5000 |  | 256 | / |  |  | 13.654 | 312.4116 | 0.8209 | 0.1210 | 3.404 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | Simultaneous | completed | 5000 |  | 256 | / |  |  | 13.974 | 313.5437 | 0.8475 | 0.1292 | 3.417 | 0.0000 |  | train,quality |
| imagenet_256 | 256x256 full | Deep S_k | completed | 5000 |  | 256 | / |  |  | 14.653 | 322.0835 | 0.8278 | 0.0521 | 3.372 | 0.0185 |  | train,quality |
| imagenet_256 | 256x256 full | Improved DDPM | completed | 5000 | 50 |  | 512/2048 | 10.3998 | 414.2991 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_256 | 256x256 full | EDM | completed | 5000 | 79 |  | 512/2048 | 10.3975 | 368.4282 |  |  |  |  |  |  |  | baseline_train,baseline_eval |
| imagenet_256 | 256x256 full | D-AR | protocol_blocked | 5000 |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
