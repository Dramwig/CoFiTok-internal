# ImageNet-256 20k Confirmatory Report

All confirmatory path metrics use 1,024 validation images at t=500.

| seed | endpoint-only AUC | ordered | random | reverse | all non-ID mean | rank/24 | dense endpoint MSE | CoFiTok endpoint MSE | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 103 | 1.78791 | 0.06642 | 0.51300 | 0.71921 | 0.31573 | 1/24 | 0.12339 | 0.11327 | -8.20% |
| 139 | 1.44015 | 0.06784 | 0.47117 | 0.68214 | 0.30820 | 1/24 | 0.12438 | 0.11785 | -5.25% |

## Preregistered decision

- `path_auc_lower_than_endpoint_only_both_seeds`: PASS
- `mean_endpoint_mse_within_plus_5pct_of_dense_monolithic`: PASS
- `random_and_reverse_worse_both_seeds`: PASS
- `exhaustive_nonidentity_and_reverse_ci_positive_both_seeds`: PASS
- `all_24_permutation_endpoint_sums_preserved`: PASS
- `endpoint_sum_preserved_across_orders`: PASS
- `zero_token_ratio_zero_both_seeds`: PASS

Overall confirmatory result: **PASS**.
Mean paired endpoint-MSE change: -6.73%.

## Exploratory generation controls

| seed | method | samples | lowres Frechet | Inception Frechet |
| ---: | --- | ---: | ---: | ---: |
| 103 | endpoint_only_factorized | 4096 | 11.1619 | 370.3672 |
| 103 | dense_monolithic | 4096 | 9.7396 | 449.3913 |
| 103 | cofitok | 4096 | 10.8232 | 349.4076 |
| 139 | endpoint_only_factorized | 4096 | 8.9753 | 404.6116 |
| 139 | dense_monolithic | 4096 | 10.5597 | 384.8126 |
| 139 | cofitok | 4096 | 11.0839 | 356.9311 |

Generation metrics are exploratory and must not be used to claim broad generation-quality superiority.
