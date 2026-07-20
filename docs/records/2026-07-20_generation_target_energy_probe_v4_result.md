# Generation target-energy probe v4 result

Date: 2026-07-20

Role: non-formal mechanism and directional-quality probe. This run cannot
authorize the fresh matched 50K pair or full 300K training.

## Immutable inputs

- Git revision: `2a45b27dd197f5c014bded11f0590520e35c3527`
- Dataset: `imagenet_256_10pct`
- Checkpoint SHA256:
  `c73ba02a5e793d0daf9232c0e7f5b1222b0df2f555ab0b60e530c9490d4d81f3`
- Layout: channels `[4,4,8,8,8,8,1,2]`, spatial strides
  `[16,16,8,8,4,4,1,1]`
- Training: 5,000 updates, 320,000 images seen, effective batch 64, seed 2027
- Objective: equal denoising progress, denoise-path prefix/component weights
  `0.15/0.3`, and per-sample exact-target energy-ratio weight `0.5`
- Sampling: 512 images, deterministic DDIM-50, CFG 1.5, EMA weights

Training, checkpoint integrity, EMA/raw mechanism evaluation, five-timestep
diagnostics, sampling, directional metrics, and aggregation all completed. The
read-only monitor and training watchdog passed.

## Result

The target-energy term preserved endpoint learning but did not recover the
ordered path:

| weights | endpoint MSE at t=500 | path AUC | ordered rank / 18 |
|---|---:|---:|---:|
| EMA | 0.0225144 | 0.2884318 | 17 |
| raw model | 0.0204233 | 0.2836512 | 16 |

EMA lag is therefore not the cause. Directional generation also remained
unusable: FID-512 was `337.1272` and IS-512 was `1.6390`. These small-sample
values are diagnostics only, but they agree with the mechanism failure and do
not authorize scaling.

The failure persists across the diffusion schedule:

| timestep | endpoint MSE | ordered path AUC | rank / 18 | learned tail-2 energy | equal-progress target tail-2 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.001287 | 0.009660 | 2 | 65.73% | 41.22% |
| 250 | 0.008848 | 0.056769 | 14 | 87.18% | 33.67% |
| 500 | 0.022514 | 0.288432 | 17 | 89.84% | 27.74% |
| 750 | 0.060621 | 1.630933 | 17 | 90.88% | 25.53% |
| 950 | 0.863465 | 46.154339 | 17 | 91.38% | 25.02% |

Zero-token synthesis remained exact and shuffled tokens strongly mismatched
the endpoint. Those diagnostics prove that the restricted synthesis contract
is intact, but they do not compensate for the missing learned order.

## Visual audit

Four shared sample indices were inspected at prefix budgets 1/2/4/8. Prefix 1
is almost pure noise, prefix 2 introduces broad color regions, prefix 4 creates
strong false contours and texture, and prefix 8 returns to high-frequency
noise instead of resolving a class-conditional image. The visual trajectory
matches the rank and FID diagnostics.

The small reports and 16 reviewed PNGs are mirrored in the untracked local
archive:

```text
artifacts/reports/generation/rank_recovery_probe_2026-07-20_v4/
```

## Root cause and next probe

The equal-progress target is inconsistent with the heterogeneous compressed
token layout. At high noise, each equal progress increment asks every token to
predict a comparable full-resolution noise component. The early stride-16,
stride-8, and stride-4 tokens cannot span that target, while the final
stride-1 pair can, so the model rationally moves most work to tokens 7 and 8.
Increasing the same energy penalty would fight the representation rather than
repair its ordering.

The v5 capacity-path probe adds a backward-compatible
`denoise_path_progress_mode="token_capacity"`. For token `k`, its restricted
synthesis rank proxy is

```text
r_k = min(C_k, C_image) * H_k * W_k
```

and its denoising-progress increment is proportional to `sqrt(r_k)`. Because
component energy is quadratic in amplitude, this makes high-noise target
energy proportional to the representable subspace. Spatial targets use each
token's actual resolution. For the current layout the cumulative progress is

```text
[0.027547, 0.055094, 0.110188, 0.165282,
 0.275470, 0.385658, 0.640127, 1.000000]
```

and the asymptotic target energy ratio is approximately

```text
[0.0034, 0.0034, 0.0134, 0.0134, 0.0537, 0.0537, 0.2864, 0.5728]
```

Training and checkpoint evaluation share one target implementation. The v5
runbook remains non-formal, includes raw/EMA and five-timestep audits, requires
manual visual review, and cannot automatically launch 50K or 300K.
