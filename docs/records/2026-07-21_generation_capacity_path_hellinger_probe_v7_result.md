# Capacity-path Hellinger probe v7 result

Date: 2026-07-21

## Purpose

V6 restored ordered-prefix rank and endpoint quality, but tokens 1-6 carried
only about 0.56% of normalized component energy. V7 tested whether squared
Hellinger distance against the token-capacity energy target could prevent this
collapse without materially weakening the dense epsilon prediction.

V7 changed only the energy objective relative to v6:

- `denoise_path_progress_mode="token_capacity"`
- `denoise_path_prefix_weight=0.05`
- `denoise_path_component_weight=0.1`
- `denoise_path_energy_weight=0.1`
- `denoise_path_energy_mode="hellinger"`

Data, true-compressed K8 layout, backbone, seed, optimizer, EMA, 5K horizon,
and all evaluation commands remained fixed.

## Provenance

- Git revision: `94a3bb3a11da94ef91f2fe165aa092f5ba5ea129`
- Run: `/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k_v7`
- Checkpoint: `checkpoint_step_00005000.pt`
- Checkpoint bytes: `1,006,396,450`
- Checkpoint SHA256: `0fd5a005adc9e39301f598edd33f1d013c85f5c165369b73107843a857e00996`
- Runtime environment SHA256: `51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57`
- Dataset identity SHA256: `97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741`
- Local report archive: `artifacts/reports/generation/rank_recovery_probe_2026-07-21_v7/`

## Results

Final validation epsilon MSE was `0.02439082`, 1.1% above v6 and 1.1% below
v5. EMA `t=500` endpoint MSE was `0.02072175`; this is 3.67% above v6 and
4.50% below v5. Ordered path AUC was `0.09709960`, ranked 1 of 18 tested
orders. Zero-token synthesis was exactly zero and shuffled/ordered endpoint MSE
ratio was `101.2445`.

EMA token-1-6 energy ratio and order rank across timesteps:

| timestep | rank | token-1-6 energy | endpoint MSE |
|---:|---:|---:|---:|
| 50 | 1/18 | 10.56% | 0.0012090 |
| 250 | 1/18 | 7.92% | 0.0084057 |
| 500 | 1/18 | 7.48% | 0.0207218 |
| 750 | 1/18 | 7.45% | 0.0514318 |
| 950 | 1/18 | 7.46% | 0.6180999 |

The raw model at `t=500` also ranked 1/18 with 8.33% token-1-6 energy and
endpoint MSE `0.01865786`. The 512-image DDIM-50/CFG-1.5 diagnostic produced
FID `296.3751` and IS `1.6684`; FID was only 0.80% above v6. This sample count
is directional evidence only and is not a formal FID claim.

Fixed-index visual review showed that prefix 4 contains repeatable low-frequency
spatial structure before prefix 8 adds high-frequency detail. The same v6
indices kept prefixes 1-4 nearly indistinguishable from noise.

## Decision

V7 is the selected formal factorization objective. Formal scaling and full
configs must use the exact objective above. Training recipe schema v2 locks it,
and generation gate schema v2 requires at least 5% of per-sample-normalized
component energy in tokens 1-6 in both scaling and full evaluation.

The v7 checkpoint remains a non-formal mechanism probe. It does not authorize
50K or 300K training. The next authorized step is a fresh same-revision matched
10% ImageNet-256 50K pair after the selected target revision passes deployment
attestation.
