# External Baseline Summary

| baseline | dataset | train steps | samples | real | DDIM steps | NFE | lowres Frechet | Inception Frechet | recon PSNR | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edm | afhqv2_64 | 5000 | 1024 | 4096 | 40 | 79 | 1.9621 | 230.3979 |  | train+eval |
| edm | cifar10 | 3000 | 1024 | 4096 | 40 | 79 | 2.8681 | 351.1658 |  | train+eval |
| edm | downsampled_imagenet_64 | 5000 | 1024 | 4096 | 40 | 79 | 2.7540 | 248.3076 |  | train+eval |
| edm | ffhq_64 | 5000 | 1024 | 4096 | 40 | 79 | 2.5077 | 325.4898 |  | train+eval |
| edm | imagenet_1k_64x64_hf | 5000 | 1024 | 4096 | 40 | 79 | 3.2285 | 258.4005 |  | train+eval |
| edm | imagenet_256_10pct | 5000 | 512 | 2048 | 40 | 79 | 10.3918 | 368.6788 |  | train+eval |
| edm | imagenet_256 | 5000 | 512 | 2048 | 40 | 79 | 10.3975 | 368.4282 |  | train+eval |
| edm | tiny_imagenet_200 | 5000 | 1024 | 4096 | 40 | 79 | 3.0996 | 268.7190 |  | train+eval |
| improved_diffusion | afhqv2_64 | 5000 | 1024 | 4096 | 50 | 50 | 8.2918 | 335.1814 |  | train+eval |
| improved_diffusion | cifar10 | 3000 | 1024 | 4096 | 50 | 50 | 7.8610 | 498.6459 |  | train+eval |
| improved_diffusion | downsampled_imagenet_64 | 5000 | 1024 | 4096 | 50 | 50 | 8.8044 | 355.1061 |  | train+eval |
| improved_diffusion | ffhq_64 | 5000 | 1024 | 4096 | 50 | 50 | 9.4360 | 369.0054 |  | train+eval |
| improved_diffusion | imagenet_1k_64x64_hf | 5000 | 1024 | 4096 | 50 | 50 | 8.8830 | 378.3325 |  | train+eval |
| improved_diffusion | imagenet_256_10pct | 5000 | 512 | 2048 | 50 | 50 | 10.3989 | 419.5981 |  | train+eval |
| improved_diffusion | imagenet_256 | 5000 | 512 | 2048 | 50 | 50 | 10.3998 | 414.2991 |  | train+eval |
| improved_diffusion | tiny_imagenet_200 | 5000 | 1024 | 4096 | 50 | 50 | 9.0747 | 381.2383 |  | train+eval |
| ml_flextok | cifar10 | 0 | 2 | 2 | 1 | 1 | 0.2912 |  | 26.7589 | eval-only |
| ml_flextok | cifar10 | 0 | 2 | 2 | 1 | 1 | 0.2930 |  | 26.7195 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 4 | 4 | 1 | 1 | 1.7601 |  | 19.2482 | eval-only |
| titok_1d_tokenizer | cifar10 | 0 | 4 | 4 | 1 | 1 | 1.7600 |  | 19.2483 | eval-only |
