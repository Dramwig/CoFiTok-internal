# Formal64 Paper Comparison

Lower Frechet-style metrics are better. Prefix/path metrics apply only to CoFiTok-style internal methods.

| dataset | method | steps | NFE | sample lowres | sample Inception | denoise PSNR | path AUC | eff K | zero ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downsampled_imagenet_64 | CoFiTok | 5000 | 50 | 8.3554 | 374.6833 | 14.521 | 0.0681 | 6.817 | 0.0000 | train,quality,generated_quality |
| downsampled_imagenet_64 | Dense epsilon | 5000 | 50 | 7.5778 | 377.1191 | 14.651 | 1.0403 | 5.376 | 0.0000 | train,quality,generated_quality |
| downsampled_imagenet_64 | Channel mask | 5000 |  |  |  | 13.250 | 4.9031 | 1.112 | 0.0000 | train,quality |
| downsampled_imagenet_64 | improved_diffusion | 5000 | 50 | 8.8044 | 355.1061 |  |  |  |  | baseline_train,baseline_eval |
| downsampled_imagenet_64 | edm | 5000 | 79 | 2.7540 | 248.3076 |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | CoFiTok | 5000 | 50 | 8.7379 | 425.7089 | 15.024 | 0.0682 | 6.818 | 0.0000 | train,quality,generated_quality |
| ffhq_64 | Dense epsilon | 5000 | 50 | 8.0881 | 424.1108 | 15.144 | 1.0807 | 5.415 | 0.0000 | train,quality,generated_quality |
| ffhq_64 | Channel mask | 5000 |  |  |  | 13.285 | 4.8775 | 1.113 | 0.0000 | train,quality |
| ffhq_64 | improved_diffusion | 5000 | 50 | 9.4360 | 369.0054 |  |  |  |  | baseline_train,baseline_eval |
| ffhq_64 | edm | 5000 | 79 | 2.5077 | 325.4898 |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | CoFiTok | 5000 | 50 | 7.0781 | 331.0574 | 15.167 | 0.0643 | 6.819 | 0.0000 | train,quality,generated_quality |
| afhqv2_64 | Dense epsilon | 5000 | 50 | 6.5792 | 345.9386 | 15.196 | 1.0118 | 5.462 | 0.0000 | train,quality,generated_quality |
| afhqv2_64 | Channel mask | 5000 |  |  |  | 13.384 | 4.7483 | 1.112 | 0.0000 | train,quality |
| afhqv2_64 | improved_diffusion | 5000 | 50 | 8.2918 | 335.1814 |  |  |  |  | baseline_train,baseline_eval |
| afhqv2_64 | edm | 5000 | 79 | 1.9621 | 230.3979 |  |  |  |  | baseline_train,baseline_eval |
