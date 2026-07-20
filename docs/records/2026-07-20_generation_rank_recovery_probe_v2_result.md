# Generation rank-recovery probe v2 result

Date: 2026-07-20

Role: non-formal mechanism and directional-quality probe. These 512-image
results cannot authorize the 10K promotion gate, formal 50K sampling, or full
300K training.

## Immutable inputs

- Git revision: `05bbb4af63a9f1d9b7f11bc4222d50875e382e1d`
- Dataset: `imagenet_256_10pct`
- Layout: channels `[4,4,8,8,8,8,1,2]`, spatial strides
  `[16,16,8,8,4,4,1,1]`
- Training: 5,000 updates, effective batch 64, seed 2027
- Mechanism eval: 512 validation images, timestep 500, ordered/reverse plus 16
  fixed random orders
- Sampling: 512 images, deterministic DDIM-50, CFG 1.5, EMA weights

## Result

The rank-complete layout repaired endpoint expressivity but did not recover the
ordered mechanism:

| candidate | endpoint MSE | ordered path AUC | rank / 18 | FID-512 | IS-512 |
|---|---:|---:|---:|---:|---:|
| denoise path | 0.0199395 | 0.1995274 | 6 | 275.1347 | 2.2749 |
| epsilon band | 0.0200607 | 0.2939404 | 18 | 281.8623 | 2.1310 |
| old rank-deficient reference | 0.1695224 | n/a | 3 | n/a | n/a |

The endpoint MSE improved by about 88% relative to the old rank-deficient
layout, proving that three aggregate full-resolution channels remove the RGB
rank bottleneck. The ordered rank requirement remains unmet, so neither
candidate may launch a fresh matched 50K pair.

Both candidates collapsed most component energy into the last two tokens:

- denoise path: token 7/8 energy ratios `0.3440 / 0.6727`
- epsilon band: token 7/8 energy ratios `0.3403 / 0.6823`

The prefix samples agree with the mechanism report: budgets 1, 2, and 4 remain
near-noise, while budget 8 introduces the first substantial semantic structure.
This is endpoint-heavy decoding rather than usable progressive denoising.

## Raw versus EMA audit

Raw model weights improve endpoint MSE relative to EMA at this short horizon,
but do not recover ordering:

| candidate | EMA MSE | raw MSE | EMA rank | raw rank |
|---|---:|---:|---:|---:|
| denoise path | 0.0199395 | 0.0186604 | 6 | 7 |
| epsilon band | 0.0200607 | 0.0186982 | 18 | 18 |

Therefore EMA lag is not the cause of the failed ordered mechanism. Formal
sampling remains EMA-only.

## Next probe

The next candidate keeps the rank-complete compressed layout and changes only
the factorization objective:

- equal denoising progress across the eight prefixes (`progress_power=1.0`),
- the established low-resolution path weights (`prefix=0.15`, `component=0.3`),
- a balanced component-energy target with weight 0.5.

This objective preserves low-to-high spatial targets while preventing the two
largest token fields from absorbing nearly all denoising work. It does not train
against shuffled orders directly, so shuffle remains an independent diagnostic.

Authoritative machine-readable sources are stored remotely under
`artifacts/reports/generation/rank_recovery_probe_2026-07-20_v2/`; selected
copies and representative samples are mirrored to the local archive without
entering the deployment bundle.
