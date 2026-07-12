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
| cifar10 | channel_mask | 3000 | 0.2034 | 12.938 | 4.3115 |  |  | 0.2516 | 5.3941 |  |  |
| cifar10 | epsilon_only | 3000 | 0.1456 | 14.390 | 1.5634 | 334.0694 | 0.1639 | 8.1429 | 6.0239 | 493.3578 | 0.3585 |
| cifar10 | light_denoise_path | 3000 | 0.1531 | 14.170 | 1.8491 | 327.8888 | 0.1674 | 6.7221 | 5.5970 | 493.2344 | 0.3685 |
| imagenet_1k_64x64_hf | channel_mask | 5000 | 0.1749 | 13.593 | 3.9121 |  |  | 0.2382 | 5.0606 |  |  |
| imagenet_1k_64x64_hf | epsilon_only | 20000 | 0.1066 | 15.743 | 0.7160 | 330.6432 | 0.5204 | 8.2138 | 5.4477 | 371.2769 | 0.9828 |
| imagenet_1k_64x64_hf | light_denoise_path | 20000 | 0.1112 | 15.558 | 0.8874 | 360.7951 | 0.5150 | 6.6272 | 5.2239 | 378.2376 | 0.9735 |
| tiny_imagenet_200 | channel_mask | 5000 | 0.2026 | 12.955 | 4.3987 |  |  | 0.2700 | 5.6925 |  |  |
| tiny_imagenet_200 | epsilon_only | 20000 | 0.1272 | 14.976 | 0.8355 | 344.1619 | 0.5479 | 8.1900 | 5.2462 | 382.9905 | 0.9284 |
| tiny_imagenet_200 | light_denoise_path | 20000 | 0.1305 | 14.865 | 0.9674 | 346.0830 | 0.5505 | 6.6039 | 5.1926 | 384.5499 | 0.9242 |

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
| downsampled_imagenet_64 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 7.5778 | 377.1191 |
| downsampled_imagenet_64 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.3554 | 374.6833 |
| ffhq_64 | epsilon_only | tiny_conv | 5000 | 1024 | 4096 | 8.0881 | 424.1108 |
| ffhq_64 | light_denoise_path | tiny_conv | 5000 | 1024 | 4096 | 8.7379 | 425.7089 |
| imagenet_1k_64x64_hf | light_denoise_path | tiny_conv | 5000 | 64 | 256 | 8.5337 | 374.3523 |
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
