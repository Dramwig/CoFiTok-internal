# Fixed-basis rank-recovery probe v8 result (2026-07-25)

## Decision

Promote the deterministic fixed-basis synthesis operator into the formal v3
10% and full ImageNet-256 recipes. This decision authorizes a fresh
same-revision 10% CoFiTok/dense 50K pair only. The 5K probe is non-formal and
does not authorize full 300K training or a generation-quality claim.

The v8 probe changed only the synthesis operator relative to v7:

- `synthesis_mode=fixed_basis`
- `synthesis_kernel_size=1`
- `gamma_mode=fixed_one`

Data, U-Net, token layout, feedback, objective, optimizer, seed, effective
batch, precision, and 5K training budget remained fixed. Each `S_k` is
condition-free, parameter-free, bias-free, and satisfies `S_k(0)=0` exactly.

## Source identity

- Git revision: `5a93c7061e4df237234b19b052331fbc3ec10425`
- Branch: `scale/generative-system`
- Tracked worktree: clean
- Run:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_fixed_basis_hellinger_k8_probe5k_v8`
- Config:
  `configs/generation/imagenet256_10pct_rankcomplete_fixed_basis_hellinger_k8_probe5k.json`
- Dataset identity:
  `97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741`
- Final checkpoint:
  `checkpoint_step_00005000.pt`
- Checkpoint bytes: `1,006,351,466`
- Checkpoint SHA256:
  `fd69048f442896010333e892f77094f64e223a3e72bdb89ce0aecf5c656035a5`
- Parameters: `62,836,011`
- Images seen: `320,000`
- Peak VRAM: `15,100,217,856` bytes
- Training elapsed: `11,281.76` seconds
- Pair monitor: terminal `pass`, no issues

## Training trajectory

Matched validation events consistently favored v8 over v7:

| Step | v8 validation epsilon MSE | v7 | Relative change |
|---:|---:|---:|---:|
| 1,000 | 0.035887 | 0.096986 | -63.0% |
| 2,000 | 0.037563 | 0.040769 | -7.9% |
| 3,000 | 0.023297 | 0.025816 | -9.8% |
| 4,000 | 0.018506 | 0.020514 | -9.8% |
| 5,000 | 0.022716 | 0.024391 | -6.9% |

The final training state was finite, with gradient norm `0.1001`.

## Mechanism evidence

EMA learned order ranked first among the ordered, reverse, and 16 random
permutations at all audited timesteps:

| Timestep | Endpoint clean MSE | Ordered path AUC | Token 1-6 energy | Rank |
|---:|---:|---:|---:|---:|
| 50 | 0.001151 | 0.001731 | 11.74% | 1 |
| 250 | 0.007731 | 0.016916 | 8.57% | 1 |
| 500 | 0.017635 | 0.089170 | 8.11% | 1 |
| 750 | 0.034896 | 0.505867 | 8.02% | 1 |
| 950 | 0.154867 | 14.321498 | 8.00% | 1 |

At `t=500`, v8 versus v7:

- endpoint clean MSE: `0.017635` versus `0.020722` (`-14.90%`)
- ordered path AUC: `0.089170` versus `0.097100` (`-8.17%`)
- token 1-6 energy: `8.11%` versus `7.48%`
- zero-token max absolute output: `0.0`
- shuffled/ordered endpoint MSE ratio: `114.01`

Raw model and EMA agreed on rank 1. EMA endpoint MSE was only 2.20% above raw,
so short-horizon EMA lag is not the main result driver.

## Directional generation evidence

Protocol: 512 EMA samples, deterministic DDIM-50, CFG 1.5, bf16, balanced
classes, prefix budgets 1/2/4/8. These metrics are directional only.

| Candidate | FID-512 | IS | Endpoint MSE | Ordered path AUC | Rank |
|---|---:|---:|---:|---:|---:|
| v2 denoise path | 275.13 | 2.275 | 0.019940 | 0.199527 | 6 |
| v7 Hellinger | 296.38 | 1.668 | 0.020722 | 0.097100 | 1 |
| v8 fixed basis | **260.31** | **3.175** | **0.017635** | **0.089170** | **1** |

V8 reduced directional FID by 12.17% versus v7 and 5.39% versus the prior best
v2 candidate. Fixed-index endpoint samples show materially stronger global
structure than v7, while remaining visibly undertrained at 5K. Prefix 1/2
remain noise-like and prefix 4 introduces coarse structure; no 5K sample is
treated as deployable output.

## Formal v3 handoff

The new authoritative scaling identities are:

```text
CoFiTok: imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3
dense:   imagenet256_10pct_fixed_basis_dense_50k_v3
report:  imagenet256_10pct_fixed_basis_matched_50k_v3
```

Formal preflight passes with CoFiTok `62,836,011` parameters and dense
`62,824,707`, a relative gap of `+0.017993%`. Recipe schema v3 requires the
fixed basis for scaling and full training. Historical v2 configs, runs, and
failed promotion evidence remain separately named and must not be overwritten.

The next valid sequence is:

1. deploy and attest one clean target revision;
2. train the fresh v3 CoFiTok and dense 50K pair at that exact revision;
3. generate matched 10K DDIM-100 EMA sample sets and run the unchanged
   promotion gate;
4. launch full matched 300K only if that gate passes.

## Local evidence archive

Bounded reports and fixed sample subsets are stored under:

```text
artifacts/reports/generation/rank_recovery_probe_2026-07-24_v8/
```

The archive includes aggregate reports, training/sampling/checkpoint source
reports, integrity metadata, the complete 101-row training metrics JSONL, one
four-prefix trajectory, and eight endpoint samples. The checkpoint and full
512-sample sets remain remote.
