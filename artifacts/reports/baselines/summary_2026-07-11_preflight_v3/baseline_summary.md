# External Baseline Summary

| baseline | dataset | train steps | batch | params | samples | real | DDIM steps | NFE | lowres Frechet | Inception Frechet | recon PSNR | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d_ar |  |  |  |  |  |  |  |  |  |  |  | train+eval |
| d_ar |  |  |  |  |  |  |  |  |  |  |  | train+eval |
| edm | afhqv2_64 | 5000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 1.9621 | 230.3979 |  | train+eval |
| edm | cifar10 | 3000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 2.8681 | 351.1658 |  | train+eval |
| edm | downsampled_imagenet_64 | 5000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 2.7540 | 248.3076 |  | train+eval |
| edm | ffhq_64 | 5000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 2.5077 | 325.4898 |  | train+eval |
| edm | imagenet_1k_64x64_hf | 5000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 3.2285 | 258.4005 |  | train+eval |
| edm | imagenet_256_10pct | 5000 | 4 | 952291 | 512 | 2048 | 40 | 79 | 10.3918 | 368.6788 |  | train+eval |
| edm | imagenet_256 | 5000 | 4 | 952291 | 512 | 2048 | 40 | 79 | 10.3975 | 368.4282 |  | train+eval |
| edm | tiny_imagenet_200 | 5000 | 32 | 3870531 | 1024 | 4096 | 40 | 79 | 3.0996 | 268.7190 |  | train+eval |
| improved_diffusion | afhqv2_64 | 5000 | 32 | 21715907 | 1024 | 4096 | 50 | 50 | 8.2918 | 335.1814 |  | train+eval |
| improved_diffusion | cifar10 | 3000 | 32 | 9593539 | 1024 | 4096 | 50 | 50 | 7.8610 | 498.6459 |  | train+eval |
| improved_diffusion | downsampled_imagenet_64 | 5000 | 32 | 21715907 | 1024 | 4096 | 50 | 50 | 8.8044 | 355.1061 |  | train+eval |
| improved_diffusion | ffhq_64 | 5000 | 32 | 21715907 | 1024 | 4096 | 50 | 50 | 9.4360 | 369.0054 |  | train+eval |
| improved_diffusion | imagenet_1k_64x64_hf | 5000 | 32 | 21715907 | 1024 | 4096 | 50 | 50 | 8.8830 | 378.3325 |  | train+eval |
| improved_diffusion | imagenet_256_10pct | 5000 | 4 | 7200579 | 512 | 2048 | 50 | 50 | 10.3989 | 419.5981 |  | train+eval |
| improved_diffusion | imagenet_256 | 5000 | 4 | 7200579 | 512 | 2048 | 50 | 50 | 10.3998 | 414.2991 |  | train+eval |
| improved_diffusion | tiny_imagenet_200 | 5000 | 32 | 21715907 | 1024 | 4096 | 50 | 50 | 9.0747 | 381.2383 |  | train+eval |
| mar | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| mar | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| mar | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| mar |  |  |  |  |  |  |  |  |  |  |  | train+eval |
| ml_flextok | afhqv2_64 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0358 |  | 24.0421 | eval-only |
| ml_flextok | afhqv2_64 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1007 |  | 23.8092 | eval-only |
| ml_flextok | cifar10 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0373 |  | 26.7679 | eval-only |
| ml_flextok | cifar10 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.0969 |  | 26.4198 | eval-only |
| ml_flextok | downsampled_imagenet_64 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0418 |  | 22.5683 | eval-only |
| ml_flextok | downsampled_imagenet_64 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1042 |  | 22.8808 | eval-only |
| ml_flextok | ffhq_64 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0422 |  | 24.0572 | eval-only |
| ml_flextok | ffhq_64 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1158 |  | 23.9493 | eval-only |
| ml_flextok | imagenet_1k_64x64_hf | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0390 |  | 23.5966 | eval-only |
| ml_flextok | imagenet_1k_64x64_hf | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.0973 |  | 24.1108 | eval-only |
| ml_flextok | imagenet_256_10pct | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0438 |  | 19.1230 | eval-only |
| ml_flextok | imagenet_256_10pct | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1072 |  | 19.5465 | eval-only |
| ml_flextok | imagenet_256 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0438 |  | 19.1226 | eval-only |
| ml_flextok | imagenet_256 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1071 |  | 19.5459 | eval-only |
| ml_flextok | cifar10 | 0 | 1 |  | 2 | 2 | 1 | 1 | 0.2912 |  | 26.7589 | eval-only |
| ml_flextok | cifar10 | 0 | 1 |  | 2 | 2 | 1 | 1 | 0.2930 |  | 26.7195 | eval-only |
| ml_flextok | tiny_imagenet_200 | 0 | 1 |  | 256 | 256 | 4 | 4 | 0.0442 |  | 21.8110 | eval-only |
| ml_flextok | tiny_imagenet_200 | 0 | 1 |  | 64 | 64 | 4 | 4 | 0.1153 |  | 21.8345 | eval-only |
| retok | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| retok | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| retok | imagenet_256 |  |  |  |  |  |  |  |  |  |  | train+eval |
| titok_1d_tokenizer | afhqv2_64 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.2168 |  | 17.2868 | eval-only |
| titok_1d_tokenizer | afhqv2_64 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.5780 |  | 17.2349 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1678 |  | 19.3461 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.4682 |  | 19.3168 | eval-only |
| titok_1d_tokenizer | downsampled_imagenet_64 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1956 |  | 16.7253 | eval-only |
| titok_1d_tokenizer | downsampled_imagenet_64 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.5557 |  | 17.0297 | eval-only |
| titok_1d_tokenizer | ffhq_64 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.2307 |  | 17.3911 | eval-only |
| titok_1d_tokenizer | ffhq_64 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.6161 |  | 17.4036 | eval-only |
| titok_1d_tokenizer | imagenet_1k_64x64_hf | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1817 |  | 18.0363 | eval-only |
| titok_1d_tokenizer | imagenet_1k_64x64_hf | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.4658 |  | 18.4670 | eval-only |
| titok_1d_tokenizer | imagenet_256_10pct | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1959 |  | 15.5439 | eval-only |
| titok_1d_tokenizer | imagenet_256_10pct | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.4646 |  | 16.0468 | eval-only |
| titok_1d_tokenizer | imagenet_256 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1959 |  | 15.5439 | eval-only |
| titok_1d_tokenizer | imagenet_256 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.4646 |  | 16.0468 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 2 |  | 4 | 4 | 1 | 1 | 1.7601 |  | 19.2482 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 2 |  | 4 | 4 | 1 | 1 | 1.7600 |  | 19.2483 | eval-only |
| titok_1d_tokenizer | tiny_imagenet_200 | 0 | 4 |  | 256 | 256 | 1 | 1 | 0.1600 |  | 17.3008 | eval-only |
| titok_1d_tokenizer | tiny_imagenet_200 | 0 | 4 |  | 64 | 64 | 1 | 1 | 0.4924 |  | 17.1144 | eval-only |
