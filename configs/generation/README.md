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
- `imagenet256_terminal_snr_endpoint0975_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json`:
  fresh base-256 CoFiTok candidate selected only after a passing terminal-SNR
  frozen-checkpoint confirmation.
- `imagenet256_terminal_snr_endpoint0975_rollout_x0_u2_ema_teacher_dense_300k.json`:
  fresh base-256 matched dense candidate with the identical endpoint-0.975
  diffusion schedule and 300K budget.

The terminal-SNR endpoint-0.975 files are a candidate pair, not execution
authority. Their non-authorizing preparation must physically replay the exact
passing 10K-per-arm frozen confirmation and its adjacent validation receipt.
GPU/runtime/storage/live-snapshot evidence, explicit user stage authorization,
a separate source-bound execution authorization, and an immutable launch
receipt are still required before either 300K run may be created or launched.

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
