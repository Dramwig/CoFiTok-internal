# Matched 20k Two-Seed Comparison

All rows use K=8, 20,000 training steps, seeds 103/139, and 1,024-image full-prefix evaluation at t=500. The endpoint-only control retains token factorization and is not a monolithic dense head.

| dataset | method | endpoint x0 MSE at t=500 | path AUC | effective K | zero ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Tiny ImageNet-200 | Endpoint-only factorized | 0.1253 +/- 0.0001 | 1.0098 +/- 0.1430 | 5.312 +/- 0.049 | 0.0000 +/- 0.0000 |
| Tiny ImageNet-200 | CoFiTok | 0.1296 +/- 0.0015 | 0.0378 +/- 0.0007 | 6.766 +/- 0.015 | 0.0000 +/- 0.0000 |
| ImageNet-64 HF | Endpoint-only factorized | 0.1096 +/- 0.0006 | 0.9984 +/- 0.1210 | 5.169 +/- 0.108 | 0.0000 +/- 0.0000 |
| ImageNet-64 HF | CoFiTok | 0.1151 +/- 0.0016 | 0.0379 +/- 0.0037 | 6.752 +/- 0.012 | 0.0000 +/- 0.0000 |

CoFiTok has lower path AUC in 4/4 paired dataset-seed comparisons, with a mean relative reduction of 96.18%.
The mean endpoint x0-MSE change at t=500 is +4.20%; endpoint MSE is better in 0/4 pairs.
