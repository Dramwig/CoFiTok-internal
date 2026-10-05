# P0 Current Comparison

This table audits the locked matched-dataset/optimizer-step P0 evidence. Batch size, nominal images seen, parameter count, and NFE remain explicit because external methods are not compute matched. Lower Frechet-style metrics are better; this is not a formal SOTA FID table.

| dataset | protocol | method | status | steps | batch | nominal train imgs | NFE | quality n | sample/real n | sample lowres | sample Inception | denoise PSNR | denoise Inception | LPIPS | path AUC | eff K | zero ratio | params | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | 32x32 | CoFiTok | partial | 3000 | 64 | 192000 |  |  | / |  |  |  |  |  |  | 6.834 | 0.0000 | 81808 | train |
| cifar10 | 32x32 | Direct dense epsilon | missing | 3000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| cifar10 | 32x32 | Endpoint-only factorized | partial | 3000 | 64 | 192000 | 50 |  | 1024/4096 | 6.4290 | 320.1376 |  |  |  |  | 5.240 | 0.0000 | 81808 | train,generated_quality |
| cifar10 | 32x32 | Channel mask | partial | 3000 | 64 | 192000 |  |  | / |  |  |  |  |  |  | 1.086 | 0.0000 | 81808 | train |
| cifar10 | 32x32 | No prefix loss | completed | 3000 | 64 | 192000 |  | 512 | / |  |  | 14.286 | 310.8179 | 0.1645 |  | 6.723 | 0.0000 | 81808 | train,quality |
| cifar10 | 32x32 | No monotonic loss | completed | 3000 | 64 | 192000 |  | 512 | / |  |  | 14.353 | 310.4246 | 0.1518 |  | 6.740 | 0.0000 | 81808 | train,quality |
| cifar10 | 32x32 | Simultaneous | completed | 3000 | 64 | 192000 |  | 512 | / |  |  | 14.417 | 311.6254 | 0.1580 |  | 6.924 | 0.0000 | 76096 | train,quality |
| cifar10 | 32x32 | Deep S_k | completed | 3000 | 64 | 192000 |  | 512 | / |  |  | 14.102 | 295.6173 | 0.1691 |  | 6.790 | 0.0619 | 272800 | train,quality |
| cifar10 | 32x32 | Improved DDPM | completed | 3000 | 32 | 96000 | 50 |  | 1024/4096 | 7.8610 | 498.6459 |  |  |  |  |  |  | 9593539 | baseline_train,baseline_eval |
| cifar10 | 32x32 | EDM | completed | 3000 | 32 | 96000 | 79 |  | 1024/4096 | 2.8681 | 351.1658 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| cifar10 | 32x32 | D-AR | protocol_blocked | 3000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| tiny_imagenet_200 | 64x64 | CoFiTok | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 6.804 | 0.0000 | 81808 | train |
| tiny_imagenet_200 | 64x64 | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| tiny_imagenet_200 | 64x64 | Endpoint-only factorized | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.2676 | 385.5078 |  |  |  |  | 5.568 | 0.0000 | 81808 | train,generated_quality |
| tiny_imagenet_200 | 64x64 | Channel mask | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  |  |  | 81808 | train |
| tiny_imagenet_200 | 64x64 | No prefix loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.494 | 329.9761 | 0.5560 |  | 6.571 | 0.0000 | 81808 | train,quality |
| tiny_imagenet_200 | 64x64 | No monotonic loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.327 | 349.1998 | 0.5468 |  | 6.765 | 0.0000 | 81808 | train,quality |
| tiny_imagenet_200 | 64x64 | Simultaneous | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.278 | 338.9335 | 0.5596 |  | 6.872 | 0.0000 | 76096 | train,quality |
| tiny_imagenet_200 | 64x64 | Deep S_k | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.398 | 321.9253 | 0.5748 |  | 6.889 | 0.0505 | 272800 | train,quality |
| tiny_imagenet_200 | 64x64 | Improved DDPM | completed | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 9.0747 | 381.2383 |  |  |  |  |  |  | 21715907 | baseline_train,baseline_eval |
| tiny_imagenet_200 | 64x64 | EDM | completed | 5000 | 32 | 160000 | 79 |  | 1024/4096 | 3.0996 | 268.7190 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| tiny_imagenet_200 | 64x64 | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_1k_64x64_hf | 64x64 HF | CoFiTok | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 6.787 | 0.0000 | 81808 | train |
| imagenet_1k_64x64_hf | 64x64 HF | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_1k_64x64_hf | 64x64 HF | Endpoint-only factorized | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 5.339 | 0.0000 | 81808 | train |
| imagenet_1k_64x64_hf | 64x64 HF | Channel mask | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 1.111 | 0.0000 | 81808 | train |
| imagenet_1k_64x64_hf | 64x64 HF | No prefix loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.932 | 305.8822 | 0.5417 |  | 6.560 | 0.0000 | 81808 | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | No monotonic loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.904 | 339.3832 | 0.5301 |  | 6.773 | 0.0000 | 81808 | train,quality |
| imagenet_1k_64x64_hf | 64x64 HF | Simultaneous | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 6.845 | 0.0000 | 76096 | train |
| imagenet_1k_64x64_hf | 64x64 HF | Deep S_k | partial | 5000 | 32 | 160000 |  |  | / |  |  |  |  |  |  | 6.792 | 0.0522 | 272800 | train |
| imagenet_1k_64x64_hf | 64x64 HF | Improved DDPM | completed | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.8830 | 378.3325 |  |  |  |  |  |  | 21715907 | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | 64x64 HF | EDM | completed | 5000 | 32 | 160000 | 79 |  | 1024/4096 | 3.2285 | 258.4005 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| imagenet_1k_64x64_hf | 64x64 HF | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| downsampled_imagenet_64 | 64x64 strict | CoFiTok | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.3554 | 374.6833 |  |  |  |  | 6.817 | 0.0000 | 81808 | train,generated_quality |
| downsampled_imagenet_64 | 64x64 strict | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| downsampled_imagenet_64 | 64x64 strict | Endpoint-only factorized | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 7.5778 | 377.1191 |  |  |  |  | 5.376 | 0.0000 | 81808 | train,generated_quality |
| downsampled_imagenet_64 | 64x64 strict | Channel mask | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 13.250 | 257.7474 | 0.6823 |  | 1.112 | 0.0000 | 81808 | train,quality |
| downsampled_imagenet_64 | 64x64 strict | No prefix loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.474 | 306.5019 | 0.5471 |  | 6.573 | 0.0000 | 81808 | train,quality |
| downsampled_imagenet_64 | 64x64 strict | No monotonic loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.664 | 319.4717 | 0.5510 |  | 6.826 | 0.0000 | 81808 | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Simultaneous | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.586 | 306.6071 | 0.5448 |  | 6.887 | 0.0000 | 76096 | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Deep S_k | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.502 | 289.4443 | 0.5502 |  | 6.754 | 0.0535 | 272800 | train,quality |
| downsampled_imagenet_64 | 64x64 strict | Improved DDPM | completed | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.8044 | 355.1061 |  |  |  |  |  |  | 21715907 | baseline_train,baseline_eval |
| downsampled_imagenet_64 | 64x64 strict | EDM | completed | 5000 | 32 | 160000 | 79 |  | 1024/4096 | 2.7540 | 248.3076 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| downsampled_imagenet_64 | 64x64 strict | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| ffhq_64 | 64x64 | CoFiTok | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.7379 | 425.7089 |  |  |  |  | 6.818 | 0.0000 | 81808 | train,generated_quality |
| ffhq_64 | 64x64 | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| ffhq_64 | 64x64 | Endpoint-only factorized | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.0881 | 424.1108 |  |  |  |  | 5.415 | 0.0000 | 81808 | train,generated_quality |
| ffhq_64 | 64x64 | Channel mask | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 13.285 | 307.0517 | 0.6535 |  | 1.113 | 0.0000 | 81808 | train,quality |
| ffhq_64 | 64x64 | No prefix loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.948 | 365.6998 | 0.4794 |  | 6.523 | 0.0000 | 81808 | train,quality |
| ffhq_64 | 64x64 | No monotonic loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.135 | 358.2656 | 0.4709 |  | 6.770 | 0.0000 | 81808 | train,quality |
| ffhq_64 | 64x64 | Simultaneous | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.157 | 341.5909 | 0.4884 |  | 6.861 | 0.0000 | 76096 | train,quality |
| ffhq_64 | 64x64 | Deep S_k | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 14.951 | 328.1341 | 0.5211 |  | 6.787 | 0.0518 | 272800 | train,quality |
| ffhq_64 | 64x64 | Improved DDPM | completed | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 9.4360 | 369.0054 |  |  |  |  |  |  | 21715907 | baseline_train,baseline_eval |
| ffhq_64 | 64x64 | EDM | completed | 5000 | 32 | 160000 | 79 |  | 1024/4096 | 2.5077 | 325.4898 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| ffhq_64 | 64x64 | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| afhqv2_64 | 64x64 | CoFiTok | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 7.0781 | 331.0574 |  |  |  |  | 6.819 | 0.0000 | 81808 | train,generated_quality |
| afhqv2_64 | 64x64 | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| afhqv2_64 | 64x64 | Endpoint-only factorized | partial | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 6.5792 | 345.9386 |  |  |  |  | 5.462 | 0.0000 | 81808 | train,generated_quality |
| afhqv2_64 | 64x64 | Channel mask | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 13.384 | 262.9012 | 0.6255 |  | 1.112 | 0.0000 | 81808 | train,quality |
| afhqv2_64 | 64x64 | No prefix loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.081 | 257.0476 | 0.4444 |  | 6.538 | 0.0000 | 81808 | train,quality |
| afhqv2_64 | 64x64 | No monotonic loss | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.161 | 282.3491 | 0.4510 |  | 6.753 | 0.0000 | 81808 | train,quality |
| afhqv2_64 | 64x64 | Simultaneous | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.105 | 245.0019 | 0.4629 |  | 6.846 | 0.0000 | 76096 | train,quality |
| afhqv2_64 | 64x64 | Deep S_k | completed | 5000 | 32 | 160000 |  | 512 | / |  |  | 15.110 | 255.6228 | 0.4552 |  | 6.794 | 0.0561 | 272800 | train,quality |
| afhqv2_64 | 64x64 | Improved DDPM | completed | 5000 | 32 | 160000 | 50 |  | 1024/4096 | 8.2918 | 335.1814 |  |  |  |  |  |  | 21715907 | baseline_train,baseline_eval |
| afhqv2_64 | 64x64 | EDM | completed | 5000 | 32 | 160000 | 79 |  | 1024/4096 | 1.9621 | 230.3979 |  |  |  |  |  |  | 3870531 | baseline_train,baseline_eval |
| afhqv2_64 | 64x64 | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_256_10pct | 256x256 10pct | CoFiTok | partial | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.4423 | 414.1863 |  |  |  |  | 3.354 | 0.0000 | 15776 | train,generated_quality |
| imagenet_256_10pct | 256x256 10pct | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_256_10pct | 256x256 10pct | Endpoint-only factorized | partial | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.5062 | 337.8743 |  |  |  |  | 3.664 | 0.0000 | 15776 | train,generated_quality |
| imagenet_256_10pct | 256x256 10pct | Channel mask | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.807 | 338.2386 | 0.8782 |  | 1.068 | 0.0000 | 15776 | train,quality |
| imagenet_256_10pct | 256x256 10pct | No prefix loss | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.865 | 323.1635 | 0.8814 |  | 3.398 | 0.0000 | 15776 | train,quality |
| imagenet_256_10pct | 256x256 10pct | No monotonic loss | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.853 | 308.7236 | 0.8142 |  | 3.398 | 0.0000 | 15776 | train,quality |
| imagenet_256_10pct | 256x256 10pct | Simultaneous | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.850 | 312.5193 | 0.8626 |  | 3.412 | 0.0000 | 14552 | train,quality |
| imagenet_256_10pct | 256x256 10pct | Deep S_k | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 14.483 | 325.1934 | 0.8572 |  | 3.394 | 0.0179 | 52616 | train,quality |
| imagenet_256_10pct | 256x256 10pct | Improved DDPM | completed | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.3989 | 419.5981 |  |  |  |  |  |  | 7200579 | baseline_train,baseline_eval |
| imagenet_256_10pct | 256x256 10pct | EDM | completed | 5000 | 4 | 20000 | 79 |  | 512/2048 | 10.3918 | 368.6788 |  |  |  |  |  |  | 952291 | baseline_train,baseline_eval |
| imagenet_256_10pct | 256x256 10pct | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
| imagenet_256 | 256x256 full | CoFiTok | partial | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.5431 | 417.2551 |  |  |  |  | 3.367 | 0.0000 | 15776 | train,generated_quality |
| imagenet_256 | 256x256 full | Direct dense epsilon | missing | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  |  |
| imagenet_256 | 256x256 full | Endpoint-only factorized | partial | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.2803 | 354.4296 |  |  |  |  | 3.663 | 0.0000 | 15776 | train,generated_quality |
| imagenet_256 | 256x256 full | Channel mask | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.761 | 340.7570 | 0.8794 |  | 1.068 | 0.0000 | 15776 | train,quality |
| imagenet_256 | 256x256 full | No prefix loss | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.388 | 324.3095 | 0.8812 |  | 3.397 | 0.0000 | 15776 | train,quality |
| imagenet_256 | 256x256 full | No monotonic loss | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.654 | 312.4116 | 0.8209 |  | 3.404 | 0.0000 | 15776 | train,quality |
| imagenet_256 | 256x256 full | Simultaneous | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 13.974 | 313.5437 | 0.8475 |  | 3.417 | 0.0000 | 14552 | train,quality |
| imagenet_256 | 256x256 full | Deep S_k | completed | 5000 | 4 | 20000 |  | 256 | / |  |  | 14.653 | 322.0835 | 0.8278 |  | 3.372 | 0.0185 | 52616 | train,quality |
| imagenet_256 | 256x256 full | Improved DDPM | completed | 5000 | 4 | 20000 | 50 |  | 512/2048 | 10.3998 | 414.2991 |  |  |  |  |  |  | 7200579 | baseline_train,baseline_eval |
| imagenet_256 | 256x256 full | EDM | completed | 5000 | 4 | 20000 | 79 |  | 512/2048 | 10.3975 | 368.4282 |  |  |  |  |  |  | 952291 | baseline_train,baseline_eval |
| imagenet_256 | 256x256 full | D-AR | protocol_blocked | 5000 |  |  |  |  | / |  |  |  |  |  |  |  |  |  | feasibility_report |
