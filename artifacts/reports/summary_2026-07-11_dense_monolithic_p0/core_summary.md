# CoFiTok Experiment Summary

Generated from report JSON files. The low-res Frechet proxy is not formal Inception FID.

## ImageNet-64 HF Training Matrix

| variant | seed | final MSE | path AUC | effective K | zero ratio | shuffle ratio |
| --- | --- | --- | --- | --- | --- | --- |
| channel_mask | 139 | 0.1230 | 4.8844 | 1.111 | 0.0000 | 200.1 |
| epsilon_only | 139 | 0.0842 | 1.0746 | 5.240 | 0.0000 | 278.0 |
| epsilon_only | 103 | 0.0830 | 0.9016 | 5.100 | 0.0000 | 287.0 |
| epsilon_only | 139 | 0.0998 | 1.1000 | 5.339 | 0.0000 | 238.0 |
| epsilon_only | 103 | 0.0940 | 0.8837 | 5.400 | 0.0000 | 250.2 |
| epsilon_only | 151 | 0.0628 | 0.9146 | 6.035 | 0.0000 | 373.8 |
| full_denoise_path | 139 | 0.1074 | 0.0417 | 6.758 | 0.0000 | 219.7 |
| light_denoise_path | 139 | 0.0852 | 0.0305 | 6.758 | 0.0000 | 275.7 |
| light_denoise_path | 103 | 0.0870 | 0.0361 | 6.739 | 0.0000 | 271.7 |
| light_denoise_path | 139 | 0.1018 | 0.0620 | 6.787 | 0.0000 | 233.5 |
| light_denoise_path | 103 | 0.0963 | 0.0641 | 6.768 | 0.0000 | 244.3 |
| light_denoise_path | 151 | 0.0776 | 0.0364 | 6.749 | 0.0000 | 300.0 |
| light_denoise_path | 151 | 0.0654 | 0.0253 | 6.721 | 0.0000 | 357.0 |
| simultaneous_predictor | 139 | 0.0995 | 0.0600 | 6.845 | 0.0000 | 237.4 |
| simultaneous_predictor | 103 | 0.1024 | 0.0723 | 6.801 | 0.0000 | 233.1 |
| deep_synthesis_ablation | 139 | 0.0977 | 0.0366 | 6.792 | 0.0522 | 242.0 |
| deep_synthesis_ablation | 103 | 0.1327 | 0.0433 | 6.748 | 0.0450 | 177.3 |

## ImageNet-64 HF Order Matrix

| K | order | final MSE | path AUC | zero ratio |
| --- | --- | --- | --- | --- |
| 4 | ordered | 0.1029 | 0.0674 | 0.0000 |
| 4 | random | 0.1029 | 0.0727 | 0.0000 |
| 4 | reverse | 0.1029 | 0.7516 | 0.0000 |
| 8 | ordered | 0.1041 | 0.0635 | 0.0000 |
| 8 | random | 0.1041 | 0.2441 | 0.0000 |
| 8 | reverse | 0.1041 | 0.6947 | 0.0000 |
| 16 | ordered | 0.1021 | 0.0680 | 0.0000 |
| 16 | random | 0.1021 | 0.4680 | 0.0000 |
| 16 | reverse | 0.1021 | 0.7020 | 0.0000 |

## Cross-Dataset Quality Matrix

| dataset | variant | steps | final MSE | final PSNR | final proxy | final Inception | final LPIPS | MSE AUC | proxy AUC | Inception AUC | LPIPS AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | channel_mask | 3000 | 0.1976 | 13.063 | 3.9912 | 318.0934 | 0.2587 | 0.2454 | 5.0303 | 334.7035 | 0.3040 |
| cifar10 | epsilon_only | 3000 | 0.1432 | 14.460 | 1.3377 | 318.1592 | 0.1614 | 6.4738 | 5.4849 | 477.1410 | 0.3396 |
| cifar10 | light_denoise_path | 3000 | 0.1505 | 14.245 | 1.6109 | 312.2101 | 0.1662 | 4.6818 | 4.4195 | 473.8411 | 0.3632 |
| imagenet_1k_64x64_hf | channel_mask | 5000 | 0.1767 | 13.548 | 3.9352 | 288.3849 | 0.6505 | 0.2406 | 5.1146 | 295.7223 | 0.8236 |
| imagenet_1k_64x64_hf | epsilon_only | 5000 | 0.1265 | 15.000 | 0.9974 | 317.2139 | 0.5329 | 6.9916 | 6.2107 | 362.8906 | 0.9719 |
| imagenet_1k_64x64_hf | light_denoise_path | 5000 | 0.1295 | 14.897 | 0.9866 | 307.0050 | 0.5482 | 4.6766 | 4.3426 | 340.0534 | 0.9919 |
| tiny_imagenet_200 | channel_mask | 5000 | 0.1998 | 13.015 | 4.2813 | 290.0460 | 0.6634 | 0.2667 | 5.5572 | 302.1776 | 0.7942 |
| tiny_imagenet_200 | epsilon_only | 20000 | 0.1272 | 14.976 | 0.8355 | 344.1619 | 0.5479 | 8.1900 | 5.2462 | 382.9905 | 0.9284 |
| tiny_imagenet_200 | light_denoise_path | 5000 | 0.1470 | 14.348 | 1.0661 | 332.2910 | 0.5605 | 4.6911 | 4.2058 | 350.9371 | 0.9334 |

## Prefix-Aware Sampling Smoke Matrix

| dataset | variant | train steps | samples | DDIM steps | prefix budgets | eta | seed | artifact count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | light_denoise_path | 3000 | 16 | 20 | [1, 4, 8] | 0.0 | 20260708 | 3 |
| cifar10 | light_denoise_path | 3000 | 64 | 20 | [8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | light_denoise_path | 5000 | 8 | 20 | [1, 4, 8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | light_denoise_path | 5000 | 64 | 20 | [8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | light_denoise_path | 20000 | 64 | 20 | [8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | epsilon_only | 20000 | 256 | 20 | [8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | light_denoise_path | 20000 | 256 | 20 | [8] | 0.0 | 20260708 | 3 |
| imagenet_1k_64x64_hf | epsilon_only | 20000 | 1024 | 20 | [8] | 0.0 | 2601024 | 3 |
| imagenet_1k_64x64_hf | light_denoise_path | 20000 | 1024 | 20 | [8] | 0.0 | 2601024 | 3 |
| tiny_imagenet_200 | light_denoise_path | 5000 | 8 | 20 | [1, 4, 8] | 0.0 | 20260708 | 3 |
| tiny_imagenet_200 | light_denoise_path | 5000 | 64 | 20 | [8] | 0.0 | 20260708 | 3 |
| tiny_imagenet_200 | light_denoise_path | 20000 | 64 | 20 | [8] | 0.0 | 20260708 | 3 |
| tiny_imagenet_200 | epsilon_only | 20000 | 256 | 20 | [8] | 0.0 | 20260708 | 3 |
| tiny_imagenet_200 | light_denoise_path | 20000 | 256 | 20 | [8] | 0.0 | 20260708 | 3 |
| tiny_imagenet_200 | epsilon_only | 20000 | 1024 | 20 | [8] | 0.0 | 2601024 | 3 |
| tiny_imagenet_200 | light_denoise_path | 20000 | 1024 | 20 | [8] | 0.0 | 2601024 | 3 |

## Generated Sample Quality Smoke Matrix

| dataset | variant | backbone | train steps | samples | real images | sample proxy | sample Inception |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afhqv2_64 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 6.5792 | 345.9386 |
| afhqv2_64 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 7.0781 | 331.0574 |
| cifar10 | light_denoise_path | tiny_conv | 3000 | 64 | 256 | 7.2663 | 313.8881 |
| cifar10 | light_denoise_path | tiny_conv | 3000 | 1024 | 4096 | 6.4656 | 312.0042 |
| cifar10 | epsilon_only | tiny_conv | 3000 | 1024 | 4096 | 6.4290 | 320.1376 |
| downsampled_imagenet_64 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 7.5778 | 377.1191 |
| downsampled_imagenet_64 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.3554 | 374.6833 |
| ffhq_64 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 8.0881 | 424.1108 |
| ffhq_64 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.7379 | 425.7089 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 5000 | 64 | 256 | 8.5337 | 374.3523 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.0915 | 390.9865 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 6.9468 | 393.0053 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 7.6747 | 388.1463 |
| imagenet_1k_64x64_hf | light_denoise_path | multiscale_unet | 10000 | 2048 | 8192 | 6.0108 | 189.0491 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 64 | 256 | 7.9744 | 321.2626 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 20000 | 256 | 1024 | 5.2657 | 262.0618 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 256 | 1024 | 6.6183 | 276.0005 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 20000 | 1024 | 4096 | 5.0077 | 230.2794 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 1024 | 4096 | 6.3373 | 241.9877 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 20000 | 2048 | 8192 | 6.5585 | 246.0469 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 2048 | 8192 | 6.6856 | 274.0251 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 20000 | 8192 | 8192 | 6.5490 | 240.8919 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 8192 | 8192 | 6.6993 | 269.2785 |
| imagenet_1k_64x64_hf | epsilon_only | tiny_conv | 20000 | 50000 | 50000 | 6.4496 | 238.0329 |
| imagenet_1k_64x64_hf | epsilon_only | multiscale_unet | 20000 | 50000 | 50000 | 5.5593 | 134.2443 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 20000 | 50000 | 50000 | 6.6096 | 266.8045 |
| imagenet_1k_64x64_hf | light_denoise_path | multiscale_unet | 20000 | 50000 | 50000 | 6.2099 | 136.8376 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 5000 | 64 | 256 | 8.2197 | 373.2350 |
| tiny_imagenet_200 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 8.2676 | 385.5078 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.4871 | 373.6340 |
| tiny_imagenet_200 | light_denoise_path | multiscale_unet | 10000 | 2048 | 8192 | 6.2118 | 174.4824 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 20000 | 64 | 256 | 7.9563 | 326.7515 |
| tiny_imagenet_200 | epsilon_only | tiny_conv | 20000 | 256 | 1024 | 5.7540 | 263.9912 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 20000 | 256 | 1024 | 6.9828 | 281.2646 |
| tiny_imagenet_200 | epsilon_only | tiny_conv | 20000 | 1024 | 4096 | 5.5385 | 227.2814 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 20000 | 1024 | 4096 | 6.7311 | 245.2619 |
| tiny_imagenet_200 | epsilon_only | tiny_conv | 20000 | 2048 | 8192 | 7.5941 | 276.9332 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 20000 | 2048 | 8192 | 7.3397 | 274.7529 |
| tiny_imagenet_200 | epsilon_only | tiny_conv | 20000 | 8192 | 8192 | 7.5761 | 272.9955 |
| tiny_imagenet_200 | light_denoise_path | tiny_conv | 20000 | 8192 | 8192 | 7.3264 | 271.0260 |
| tiny_imagenet_200 | epsilon_only | multiscale_unet | 20000 | 10000 | 10000 | 5.2057 | 136.7132 |
| tiny_imagenet_200 | light_denoise_path | multiscale_unet | 20000 | 10000 | 10000 | 5.7001 | 141.4577 |

## Official FID Matrix

| dataset | variant | backbone | train steps | generated | real | DDIM steps | prefix budget | official FID | implementation | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imagenet_1k_64x64_hf | epsilon_only | multiscale_unet | 20000 | 50000 | 50000 | 50 | 8 | 140.7213 | pytorch-fid | ok |
| imagenet_1k_64x64_hf | light_denoise_path | multiscale_unet | 20000 | 50000 | 50000 | 50 | 8 | 134.0818 | pytorch-fid | ok |
| tiny_imagenet_200 | epsilon_only | multiscale_unet | 20000 | 10000 | 10000 | 50 | 8 | 148.8414 | pytorch-fid | ok |
| tiny_imagenet_200 | light_denoise_path | multiscale_unet | 20000 | 10000 | 10000 | 50 | 8 | 152.2474 | pytorch-fid | ok |
