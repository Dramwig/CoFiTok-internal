# P1 Tokenizer Reconstruction Table

Eval-only official pretrained tokenizer reconstruction. This is not a P0 diffusion generation table.

Run filter: `eval256_2026-07-10`

| dataset | baseline | images | src res | eval res | NFE | PSNR | MSE | lowres Frechet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afhqv2_64 | ml_flextok | 256 | 64 | 256 | 4 | 24.0421 | 0.0158 | 0.0358 |
| afhqv2_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.2868 | 0.0747 | 0.2168 |
| cifar10 | ml_flextok | 256 | 32 | 256 | 4 | 26.7679 | 0.0084 | 0.0373 |
| cifar10 | titok_1d_tokenizer | 256 | 32 | 256 | 1 | 19.3461 | 0.0465 | 0.1678 |
| downsampled_imagenet_64 | ml_flextok | 256 | 64 | 256 | 4 | 22.5683 | 0.0221 | 0.0418 |
| downsampled_imagenet_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 16.7253 | 0.0850 | 0.1956 |
| ffhq_64 | ml_flextok | 256 | 64 | 256 | 4 | 24.0572 | 0.0157 | 0.0422 |
| ffhq_64 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.3911 | 0.0729 | 0.2307 |
| imagenet_1k_64x64_hf | ml_flextok | 256 | 64 | 256 | 4 | 23.5966 | 0.0175 | 0.0390 |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 18.0363 | 0.0629 | 0.1817 |
| imagenet_256 | ml_flextok | 256 | 256 | 256 | 4 | 19.1226 | 0.0490 | 0.0438 |
| imagenet_256 | titok_1d_tokenizer | 256 | 256 | 256 | 1 | 15.5439 | 0.1116 | 0.1959 |
| imagenet_256_10pct | ml_flextok | 256 | 256 | 256 | 4 | 19.1230 | 0.0490 | 0.0438 |
| imagenet_256_10pct | titok_1d_tokenizer | 256 | 256 | 256 | 1 | 15.5439 | 0.1116 | 0.1959 |
| tiny_imagenet_200 | ml_flextok | 256 | 64 | 256 | 4 | 21.8110 | 0.0264 | 0.0442 |
| tiny_imagenet_200 | titok_1d_tokenizer | 256 | 64 | 256 | 1 | 17.3008 | 0.0745 | 0.1600 |
