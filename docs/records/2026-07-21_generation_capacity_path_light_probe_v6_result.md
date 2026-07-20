# 2026-07-21 light capacity-path generation probe v6 result

## Decision

V6 recovers both the short-budget endpoint trajectory and the required ordered
rank, but it is rejected as the formal scaling objective because the first six
tokens remain nearly unused and directional generation quality still regresses
against the best v2 probe. It cannot authorize 50K or 300K training.

## Locked identity

- Git revision: `ead85dcb304bea8d5b23611add8e5397c65b5e73`
- Run: `imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k_v6`
- Completed steps/images: `5,000 / 320,000`
- Checkpoint SHA256:
  `1032cd8b0702c41f5da64d4df5dae7a628dfec20a1b00fc345150456c62aa53f`
- Parameter count: `62,836,796`
- Final validation epsilon MSE: `0.02412383`
- Peak CUDA memory: `15,105,196,544` bytes
- Runtime-environment SHA256:
  `51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57`
- Objective: prefix/component weights `0.05/0.10`, zero target-energy weight,
  and `denoise_path_progress_mode="token_capacity"`

The exact-resume checkpoint, integrity sidecar, `latest.json`, optimizer/EMA/RNG
state, and full sample set remain server-only under the generation checkpoint
root. Small reports and reviewed samples are mirrored in the untracked local
archive:

```text
artifacts/reports/generation/rank_recovery_probe_2026-07-20_v6/
```

## Training and mechanism result

V6 was better than or essentially tied with v2 at every scheduled validation:

| step | v2 validation MSE | v5 validation MSE | v6 validation MSE |
|---:|---:|---:|---:|
| 1,000 | 0.0731234 | 0.1009903 | **0.0708840** |
| 2,000 | 0.0398666 | 0.0419158 | **0.0396485** |
| 3,000 | 0.0254631 | 0.0261320 | **0.0252384** |
| 4,000 | 0.0201745 | 0.0213824 | **0.0199722** |
| 5,000 | 0.0244053 | 0.0246540 | **0.0241238** |

The final checkpoint audit produced:

| weights / timestep | endpoint MSE | path AUC | ordered rank |
|---|---:|---:|---:|
| raw / 500 | 0.0185548 | 0.0853947 | 1 / 18 |
| EMA / 50 | 0.0011960 | 0.0016972 | 2 / 18 |
| EMA / 250 | 0.0082816 | 0.0171071 | 1 / 18 |
| EMA / 500 | 0.0199877 | 0.0907247 | 1 / 18 |
| EMA / 750 | 0.0471426 | 0.5160589 | 1 / 18 |
| EMA / 950 | 0.4987320 | 14.6293026 | 1 / 18 |

Zero-token synthesis is exact (`max_abs=0`) and the timestep-500
shuffled-to-ordered endpoint ratio is `104.50`. EMA and raw therefore agree that
capacity-aware ordering is learned rather than being an EMA artifact.

## Prefix-utilization failure

At EMA timestep 500, the learned per-sample component-energy ratios are:

```text
[0.0000026, 0.0000061, 0.0000122, 0.0000065,
 0.0015122, 0.0041045, 0.3344358, 0.6599200]
```

The capacity-derived targets are:

```text
[0.0034784, 0.0034784, 0.0136454, 0.0136495,
 0.0540973, 0.0538573, 0.2945592, 0.5632345]
```

Only about `0.56%` of learned energy is assigned to tokens 1-6, versus a target
of about `14.22%`. Prefixes 1 and 2 are visually almost identical noise, prefix
4 introduces only weak low-frequency structure, and prefix 8 remains heavily
contaminated by high-frequency noise. Rank 1 alone is therefore insufficient:
the model behaves like an effective two-token predictor.

## Directional generation result

The deterministic 512-image EMA DDIM-50 / CFG-1.5 probe produced:

| candidate | endpoint MSE | path AUC | rank | FID-512 | IS-512 |
|---|---:|---:|---:|---:|---:|
| denoise path v2 | 0.0199395 | 0.1995274 | 6 | **275.1347** | **2.2749** |
| capacity path v5 | 0.0216982 | **0.0872809** | 1 | 370.5892 | 1.3818 |
| light capacity path v6 | **0.0199877** | 0.0907247 | 1 | 294.0330 | 1.9437 |

The 512-image metrics are directional diagnostics only. V6 improves strongly
over v5 but remains about `6.87%` worse than v2 in FID and has lower IS. Together
with the utilization failure, this rejects v6 as the formal objective.

## V7 hypothesis

The plain per-sample ratio MSE underweights small target probabilities. On the
v6 timestep-500 distributions, squared Hellinger distance is `0.05161`, about
`24.9x` the ratio-MSE value `0.002072`, and has much stronger sensitivity near a
collapsed component. V7 therefore keeps every v6 condition fixed and changes
only the energy-distribution term:

- `denoise_path_energy_weight=0.1`;
- `denoise_path_energy_mode="hellinger"`;
- all data, layout, backbone, seed, optimizer, EMA, path weights, runtime, and
  evaluation settings remain identical.

The new mode is opt-in and defaults to historical `mse`. V7 remains a non-formal
5K probe with no automatic 50K or 300K authorization.
