# Generation-scale configurations

These configs belong to the `scale/generative-system` branch and do not replace
the locked short-budget paper runs.

- `imagenet256_10pct_cofitok_k8_50k.json`: completed legacy full-resolution-field run.
- `imagenet256_10pct_dense_50k.json`: completed legacy matched dense control.
- `imagenet256_10pct_compressed_cofitok_k8_50k.json`: authoritative compressed-token scaling gate.
- `imagenet256_10pct_compressed_dense_50k.json`: same-revision dense control for the authoritative gate.
- `imagenet256_cofitok_k8_300k.json`: full-data long-budget run after the 10% gate passes.
- `imagenet256_dense_300k.json`: full-data matched dense control.

All runs are class-conditional ImageNet-256 with classifier-free guidance
dropout inside `T_k`. CoFiTok configs retain restricted, token-only `S_k`.
The production K8 layout uses spatial strides `[16,16,8,8,4,4,2,1]` and
true channel counts `[4,4,8,8,8,8,4,2]`; every token field is strictly smaller
than a dense RGB epsilon field. The legacy 10% run is not allowed to authorize
the compressed full-data topology.
