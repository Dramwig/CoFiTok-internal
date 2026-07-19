# Fresh scaling augmentation contract

Date: 2026-07-20

Branch: `scale/generative-system`

## Problem

The failed rank-deficient 10% pair and the fixed 5K recovery probes used no
random horizontal flip. The full ImageNet-256 300K recipe already used the
standard matched flip probability of `0.5`. Keeping the future promotion pair
at `0.0` would test a weaker data recipe than the authorized full run and would
carry the legacy setting into a fresh versioned experiment for no fairness
reason. The failed matched dense control reached FID 114.8773 under that legacy
setting, already above the unchanged scaling absolute-FID threshold of 100.

## Decision

The authoritative fresh rank-complete 10% CoFiTok and dense configs now both
set `random_horizontal_flip_prob=0.5`. The scaling and full recipe contracts
require that value for both methods. `legacy_scaling` continues to resolve an
implicit or explicit `0.0`, preserving validation of the immutable historical
pair. The two active 5K objective probes also remain at their launched `0.0`
configuration and are used only to choose the factorization objective.

This is a matched shared-data change, not a CoFiTok-only advantage. Exact
resume already binds the probability in the fully resolved config and restores
sampler/RNG state, so a changed augmentation probability cannot be introduced
mid-run.

The new formal run IDs remain:

```text
imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2
imagenet256_10pct_rankcomplete_dense_50k_v2
imagenet256_10pct_rankcomplete_matched_50k_v2
```
