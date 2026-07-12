# Generation-scale configurations

These configs belong to the `scale/generative-system` branch and do not replace
the locked short-budget paper runs.

- `imagenet256_10pct_cofitok_k8_50k.json`: architecture and throughput gate.
- `imagenet256_10pct_dense_50k.json`: same-backbone dense epsilon control.
- `imagenet256_cofitok_k8_300k.json`: full-data long-budget run after the 10% gate passes.
- `imagenet256_dense_300k.json`: full-data matched dense control.

All runs are class-conditional ImageNet-256 with classifier-free guidance
dropout inside `T_k`. CoFiTok configs retain restricted, token-only `S_k`.
