# Generation-scale configurations

These configs belong to the `scale/generative-system` branch and do not replace
the locked short-budget paper runs.

- `imagenet256_10pct_cofitok_k8_50k.json`: completed legacy full-resolution-field run.
- `imagenet256_10pct_dense_50k.json`: completed legacy matched dense control.
- `imagenet256_10pct_compressed_cofitok_k8_50k.json`: historical v2
  rank-complete learned-synthesis recipe; its failed promotion evidence is
  immutable.
- `imagenet256_10pct_compressed_dense_50k.json`: historical v2 matched dense
  recipe.
- `imagenet256_10pct_fixed_basis_cofitok_k8_50k.json`: authoritative v3
  fixed-basis scaling recipe; run identity
  `imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3`.
- `imagenet256_10pct_fixed_basis_dense_50k.json`: authoritative v3 same-revision
  dense recipe; run identity `imagenet256_10pct_fixed_basis_dense_50k_v3`.
- `imagenet256_cofitok_k8_300k.json`: full-data long-budget run after the 10% gate passes.
- `imagenet256_dense_300k.json`: full-data matched dense control.

The fresh fixed-basis scaling pair and the full 300K pair both use matched
random horizontal flips with probability `0.5`. Legacy failed scaling evidence
and the fixed 5K objective probes retain their recorded `0.0` setting; they are
not silently relabeled as the augmented recipe.

All runs are class-conditional ImageNet-256 with classifier-free guidance
dropout inside `T_k`. Formal v3 CoFiTok uses a deterministic, parameter-free,
bias-free, token-only fixed RGB basis for every `S_k`; `S_k(0)=0` exactly.
The production K8 layout uses spatial strides `[16,16,8,8,4,4,1,1]` and true
channel counts `[4,4,8,8,8,8,1,2]`; every individual token field is strictly
smaller than a dense RGB epsilon field, and the final two tokens jointly span
RGB. Historical 10% runs cannot authorize the v3 full-data topology.
