# 2026-07-20 capacity-path generation probe v5 result

## Decision

The v5 capacity-path probe recovered the required prefix ordering, but it is not
the final objective for formal scaling. It is retained as mechanism evidence and
is followed by a light-capacity v6 probe before any fresh 50K launch.

V5 is the first rank-complete 5K candidate to place the ordered path first at
the formal checkpoint-audit timestep. Its directional generation quality and
early-token utilization are nevertheless weaker than required for selecting the
heavy objective unchanged.

## Locked identity

- Git revision: `3f98a4e106a1251d5242a41e260bd9fc4686c5f0`
- Run: `imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k_v5`
- Completed steps/images: `5,000 / 320,000`
- Checkpoint SHA256:
  `51138d35be612d94b507bfd20a80ef4f68ff94693c216803e9deeff006fc1644`
- Parameter count: `62,836,796`
- Final validation epsilon MSE: `0.02465395`
- Objective: prefix/component/energy weights `0.15/0.30/0.50`, with
  `denoise_path_progress_mode="token_capacity"`

The checkpoint, integrity sidecar, `latest.json`, training report, EMA state,
optimizer state, scheduler state, sampler state, and RNG state remain
server-only under `/root/autodl-tmp/CoFiTok/checkpoints/generation/`.

## Mechanism result

| weights / timestep | endpoint MSE | path AUC | ordered rank |
|---|---:|---:|---:|
| raw / 500 | 0.0187210 | 0.0841978 | 1 / 18 |
| EMA / 50 | 0.0012280 | 0.0016512 | 2 / 18 |
| EMA / 250 | 0.0085754 | 0.0165463 | 1 / 18 |
| EMA / 500 | 0.0216982 | 0.0872809 | 1 / 18 |
| EMA / 750 | 0.0575545 | 0.4952355 | 1 / 18 |
| EMA / 950 | 0.7982778 | 14.0224521 | 1 / 18 |

EMA and raw agree on rank 1 at timestep 500. Exact zero-token synthesis passes,
and the shuffled-to-ordered endpoint ratio is `92.68` at timestep 500.

The learned EMA per-sample energy ratios at timestep 500 are approximately:

```text
[0.00001, 0.00002, 0.00002, 0.00002, 0.00376, 0.00510, 0.32976, 0.66131]
```

The corresponding target ratios are:

```text
[0.00348, 0.00348, 0.01365, 0.01365, 0.05410, 0.05386, 0.29456, 0.56323]
```

Capacity-aware targets therefore fix global order, but the first six learned
components still receive only about `0.89%` of prediction energy instead of the
target's `14.22%`. The plain per-sample energy-ratio MSE is too weakly scaled to
close that gap at weight `0.5`.

## Directional generation result

The deterministic 512-image EMA DDIM-50 / CFG-1.5 probe produced:

| candidate | endpoint MSE | path AUC | rank | FID-512 | IS-512 |
|---|---:|---:|---:|---:|---:|
| denoise path v2 | 0.0199395 | 0.1995274 | 6 | 275.1347 | 2.2749 |
| equal progress v3 | 0.0224811 | 0.2858201 | 17 | 375.8400 | 1.4768 |
| target energy v4 | 0.0225144 | 0.2884318 | 17 | 337.1272 | 1.6390 |
| capacity path v5 | 0.0216982 | 0.0872809 | 1 | 370.5892 | 1.3818 |

The FID/IS values are directional 512-image diagnostics only. They are not a
formal generation claim and cannot replace the 10K promotion gate.

Four shared fish-class indices and additional cross-class indices were reviewed
at prefix budgets 1/2/4/8. Prefixes 1/2/4 remain structured noise. Prefix 8 is
less saturated than the failed old 50K layout and contains faint class-related
contours, but it is still dominated by high-frequency texture. V5 therefore
passes mechanism ordering and fails final objective selection on directional
quality and prefix utilization.

## V6 decision

V2 used lighter path weights `0.05/0.10` and had the best directional FID but an
incorrect power-based path. V5 proves that the capacity schedule fixes the
ordering, while its heavier auxiliary weights and ineffective energy MSE damage
directional quality. V6 combines only the supported pieces:

- keep the v2 prefix/component weights `0.05/0.10`;
- use `token_capacity` progress;
- disable the ineffective target-energy MSE;
- keep model, data, seed, optimizer, EMA, runtime, and evaluation protocol
  identical.

V6 remains a non-formal 5K probe and cannot launch 50K or 300K automatically.
