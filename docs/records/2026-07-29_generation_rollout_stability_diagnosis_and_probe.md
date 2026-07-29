# Generation rollout stability diagnosis and matched probe

Date: 2026-07-29

## Scope

This record starts corrective work after the formal fixed-basis v3 10% ImageNet-256
promotion gate failed generation quality while passing the endpoint and mechanism
checks. The locked 50K checkpoints and acceptance pack remain immutable.

The work addresses two observed failures:

1. Error amplification during the 100-step DDIM rollout.
2. Energy concentration in the final two denoising tokens.

## Read-only diagnosis

Evaluator revision:

```text
6688652746e0bd8c2b02acc03493b4f52bf381be
```

Protocol:

```text
dataset: imagenet_256_10pct validation
images: 8, deterministic order
weights: EMA
teacher timesteps: 999,900,750,500,250,100,10
rollout: DDIM-100, eta=0, x0 clipping enabled
CFG: 1.5 for rollout, 1.0 for teacher-forced diagnosis
precision: bf16
seed: 2029
```

Source reports:

```text
artifacts/reports/generation/rollout_stability_2026-07-29/formal_v3_cofitok/rollout_stability_report.json
sha256: ac1bb154b0b864e70dc104c3a0661ca56e7797a2bc9a64a5714f63bfaaa94083

artifacts/reports/generation/rollout_stability_2026-07-29/formal_v3_dense/rollout_stability_report.json
sha256: cc7f57bbd416fd7f0c716443bc69762a2c209860e607fccd8de57f8bfc08c47a
```

The formal CoFiTok target already assigned 79.35% to 85.91% of component
energy to the final two tokens across the teacher-forced timestep sweep. The
trained model concentrated this further to 83.41% to 91.98%. At `t=500`, the
target and prediction were:

```text
target:     [0.00344, 0.00344, 0.01357, 0.01357, 0.05401, 0.05388, 0.29422, 0.56387]
prediction: [0.00178, 0.00216, 0.00775, 0.00851, 0.03053, 0.03200, 0.30493, 0.61235]
```

The free rollout exposed amplification that endpoint-only evaluation did not:

| DDIM timestep | CoFiTok / dense epsilon HF | CoFiTok / dense predicted-x0 HF | CoFiTok / dense clip fraction |
|---:|---:|---:|---:|
| 898 | 1.002x | 8.787x | 1.18x |
| 595 | 1.004x | 10.032x | 4.61x |
| 394 | 1.014x | 8.616x | 7.52x |
| 192 | 1.072x | 6.882x | 6.69x |
| 91 | 1.106x | 6.060x | 5.36x |

The diagnosis is therefore:

- The token-capacity path target itself is tail-heavy because high-frequency
  residual recovery is deferred to the only full-resolution tail bases.
- The learned model exaggerates that target by another four to six percentage
  points.
- Small high-frequency epsilon differences are magnified by the epsilon-to-x0
  conversion as alpha decreases, producing excessive x0 high-frequency energy
  and clipping during rollout.

## Candidate correction

Implementation revisions:

```text
c95d707afe6b230a141e2143dc7b9f51828ab651
a608b5e484707ed4a9fe6890c7cfc0277add7643
```

The correction keeps every synthesis operator restricted, bias-free,
condition-free, and linear:

- RGB full-resolution rank is spread over tokens 6, 7, and 8, with one fixed
  basis channel per token.
- The layout changes from channels/strides
  `[4,4,8,8,8,8,1,2] / [16,16,8,8,4,4,1,1]` to
  `[4,4,8,8,8,1,1,1] / [16,16,8,8,4,1,1,1]`.
- Aggregate token-to-dense scalar ratio decreases from 1.427x to 1.260x.
- Each full-resolution token remains compressed by 3x.
- A capacity-prior mixture gives the path-energy objective an explicit
  non-tail-collapse target.
- A bounded low-SNR high-frequency epsilon loss directly penalizes the error
  mode amplified into x0 during rollout.
- `hellinger_stable` uses an additive `1e-4` floor to bound near-zero
  component gradients. The old `hellinger` implementation is unchanged.

Candidate CoFiTok parameters are 62,834,083 versus dense 62,824,707, a
`+0.014924%` difference.

## CUDA preflight

Preflight revision:

```text
a608b5e484707ed4a9fe6890c7cfc0277add7643
```

The first strong-prior attempt was rejected before training because its
four-step preflight reached a gradient norm of 1,175.6. After bounded
Hellinger, the eight-step candidate preflight reached 139.0, compared with
686.6 for the locked formal recipe under the same eight-step check.

| recipe | images/s | peak VRAM | step-8 grad norm |
|---|---:|---:|---:|
| locked formal CoFiTok | 28.21 | approximately 14 GiB | 686.6 |
| corrected RGB-tail3 CoFiTok | 28.23 | 14.06 GiB | 139.0 |

Preflight reports:

```text
artifacts/reports/generation/stability_probe_2026-07-29/runtime_v2/formal_old_benchmark.json
artifacts/reports/generation/stability_probe_2026-07-29/runtime_v2/cofitok_benchmark.json
```

## Matched 1K qualification

The first matched pair established that the RGB tail-3 layout solved energy
concentration but did not by itself solve raw-model rollout amplification:

| candidate | tail-two | t=500 endpoint vs dense | peak selected x0 HF vs dense | final reconstruction vs dense |
|---|---:|---:|---:|---:|
| RGB tail-3, HF 0.1 | 57.03% | +1.44% | 2.551x | +21.29% |
| RGB tail-3, HF 0.5 | 56.94% | +0.40% | 2.166x | +19.90% |
| epsilon consistency | 56.93% | +0.72% | 2.682x | +18.16% |

Increasing only the high-frequency weight had diminishing returns. Matching
epsilon after a generated step also failed because epsilon-scale agreement
does not directly bound the low-alpha epsilon-to-x0 amplification.

The accepted correction uses one detached generated state and performs a
second forward pass on a 25% micro-batch subset. Its loss directly matches
clipped predicted x0 to the bounded clean image. CoFiTok and dense use the
same consistency weight `0.1`, timestep delta `10`, clipping, subset, and
warmup.

Implementation revision:

```text
9e7ed713cd5bc46d3722e177cae343f0c0be483b
```

Matched 1K outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0
```

Final checkpoints:

```text
CoFiTok:
6867beb7f698e5ad5eea790fef8e4c8d14ace9879bcaccefc30d8a2f0d046d63

dense:
f913d3e740b03b75790294cabc950dda70e5dc270fcfde148d30919a24e9290e
```

The final validation epsilon MSE was `0.0224283542` for CoFiTok and
`0.0221664011` for dense, a `+1.1818%` relative difference. The raw-model
qualification passed all nine fixed gates:

| gate | value | threshold |
|---|---:|---:|
| tail-two component energy | 0.570061 | <= 0.65 |
| largest single-token energy | 0.286358 | <= 0.35 |
| ordered path rank | 1 | = 1 |
| endpoint CoFiTok/dense | 1.013961 | <= 1.05 |
| validation CoFiTok/dense | 1.011818 | <= 1.05 |
| peak predicted-x0 HF ratio | 1.453409 | <= 1.50 |
| final reconstruction CoFiTok/dense | 0.977907 | <= 1.05 |
| zero-token max abs | 0 | <= 1e-8 |
| shuffle/ordered endpoint | 103.367x | >= 2x |

The four selected predicted-x0 high-frequency ratios at timesteps
`595/394/192/91` were `1.2150/1.4534/1.3863/1.3512`. The former formal-v3
peak was `10.032x`, so the peak amplification ratio fell by `85.51%`.
Tail-two energy fell from `91.72%` at formal-v3 t=500 to `57.01%`.

The CoFiTok reconstruction ended `2.21%` below matched dense, with
final-to-best amplification `1.0848` versus dense `1.1004`. EMA also had
stable high-frequency and reconstruction behavior, but its short-run
endpoint ratio `1.1031` and ordered rank `2` expose expected 1K EMA lag;
EMA is not the authoritative short-probe gate.

Authoritative report:

```text
artifacts/reports/generation/stability_probe_2026-07-29/qualification1k_rollout_x0/qualification_report.json
sha256: 225839d69a239483852b40413fce17e0ff56467c6be3bf3d2c5a4b26228715ef
```

The reusable builder is:

```text
scripts/build_generation_stability_qualification.py
```

It binds all six source reports by bytes/SHA and rejects mixed revisions,
checkpoint identities, steps, or weight types.

## Matched 5K scaling gate

The 1K pass authorizes only a 5K scaling check. It does not authorize a new
formal 50K pair or full 300K.

Training revision:

```text
68ca820d5d901eff2623b97b86c4b88bf68ef500
```

Isolated checkout:

```text
/tmp/cofitok-generation-stability-scale-68ca820
```

The official remote repository remains unchanged at:

```text
1ebcc15210e63a776a2ba448481cbd8bb94a4066
```

Runbook:

```text
artifacts/runbooks/generation_stability_rollout_x0_probe5k_2026-07-29.sh
sha256: 6e05871189342e6d37db18102ec93b85bb022fda36181a34ae42514fea1eb3d3
```

The incremental deployment bundle was `10,838` bytes with SHA256
`66f78489066f85a8e63d3b0693e7616be012189eca34bfa90277581bd8bc4c08`.
The runbook binds the passing 1K qualification SHA and both 1K checkpoint
SHAs before launch.

Remote output:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0
```

Launch state:

```text
runbook PID: 362616
initial CoFiTok trainer PID: 362632
order: CoFiTok, then dense identity
steps per method: 5,000
status at record update: running, CoFiTok step >= 50
```

The next decision requires the same raw-model checkpoint and rollout gate at
5K. Only a 5K pass can justify preparing a fresh matched 50K rerun.
