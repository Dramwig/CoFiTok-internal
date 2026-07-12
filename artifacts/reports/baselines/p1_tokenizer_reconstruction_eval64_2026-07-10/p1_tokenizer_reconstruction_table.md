# P1 Tokenizer Reconstruction Table

Eval-only official pretrained tokenizer reconstruction. This is not a P0 diffusion generation table.

Run filter: `eval64_2026-07-10`

| dataset | baseline | images | src res | eval res | NFE | PSNR | MSE | lowres Frechet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afhqv2_64 | ml_flextok | 64 | 64 | 256 | 4 | 23.8092 | 0.0166 | 0.1007 |
| afhqv2_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.2349 | 0.0756 | 0.5780 |
| cifar10 | ml_flextok | 64 | 32 | 256 | 4 | 26.4198 | 0.0091 | 0.0969 |
| cifar10 | titok_1d_tokenizer | 64 | 32 | 256 | 1 | 19.3168 | 0.0468 | 0.4682 |
| downsampled_imagenet_64 | ml_flextok | 64 | 64 | 256 | 4 | 22.8808 | 0.0206 | 0.1042 |
| downsampled_imagenet_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.0297 | 0.0793 | 0.5557 |
| ffhq_64 | ml_flextok | 64 | 64 | 256 | 4 | 23.9493 | 0.0161 | 0.1158 |
| ffhq_64 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.4036 | 0.0727 | 0.6161 |
| imagenet_1k_64x64_hf | ml_flextok | 64 | 64 | 256 | 4 | 24.1108 | 0.0155 | 0.0973 |
| imagenet_1k_64x64_hf | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 18.4670 | 0.0569 | 0.4658 |
| imagenet_256 | ml_flextok | 64 | 256 | 256 | 4 | 19.5459 | 0.0444 | 0.1071 |
| imagenet_256 | titok_1d_tokenizer | 64 | 256 | 256 | 1 | 16.0468 | 0.0994 | 0.4646 |
| imagenet_256_10pct | ml_flextok | 64 | 256 | 256 | 4 | 19.5465 | 0.0444 | 0.1072 |
| imagenet_256_10pct | titok_1d_tokenizer | 64 | 256 | 256 | 1 | 16.0468 | 0.0994 | 0.4646 |
| tiny_imagenet_200 | ml_flextok | 64 | 64 | 256 | 4 | 21.8345 | 0.0262 | 0.1153 |
| tiny_imagenet_200 | titok_1d_tokenizer | 64 | 64 | 256 | 1 | 17.1144 | 0.0777 | 0.4924 |
