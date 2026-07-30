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

## One-step 5K result and hold

The original one-step matched pair completed both methods at revision
`68ca820d5d901eff2623b97b86c4b88bf68ef500`. CoFiTok/dense final validation
MSE was `0.0184789896 / 0.0184153821`, a ratio of `1.00345404`.

The raw-model rollout evidence did not satisfy the multi-seed scaling gate:

| protocol | reconstruction ratio | result |
|---|---:|---|
| n=8, seed 2029 | 1.17100471 | fail |
| n=64, seed 2029 | 1.03208552 | pass |
| n=64, seed 2039 | 1.05197590 | fail |

The second robust seed missed the `1.05` threshold by about `0.198` percentage
points. A CFG `1.0` rerun also failed (`1.08092112`), while teacher-forced
ratios remained near one. The residual was therefore free-state multi-step
drift, not CFG. EMA passed the same reconstruction checks but remained
diagnostic only and could not erase the raw-model failure.

The fail-closed decision is:

```text
artifacts/reports/generation/stability_probe_2026-07-29/scaling_decision5k_rollout_x0_v1/scaling_decision.json
status: fail
decision: hold_for_stability_correction
```

No 50K or 300K run was authorized from that result.

## Detached two-step correction

Revision `6b77ef7254356d551b2e392aba07df1c96ed067c` generalized the one-step loss
to a detached multi-step rollout:

- every generated state and previous epsilon is detached;
- each unrolled depth receives its own bounded `clipped_x0` loss;
- `unroll_steps=2` and `batch_fraction=0.125` replace one step at `0.25`;
- CoFiTok and dense share every rollout field through the generation-pair
  contract;
- the expected extra sample-forward budget per optimizer step remains
  approximately unchanged.

Local full pytest and Linux targeted tests passed. The 1K CUDA benchmark
measured `22.4459 / 24.7608 images/s` and `15.232 / 15.036 GB` peak allocated
VRAM for CoFiTok/dense.

The fresh matched 1K pair completed at the same revision:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2
```

| method | validation MSE | final checkpoint SHA256 |
|---|---:|---|
| CoFiTok | 0.0224639922 | `9885273a3bd3773274d140db433af047c7cefeb87e460b9868406b45b7e1bb05` |
| dense | 0.0226641875 | `e27a03434ec3bef7f7528d6d7a0d61867845c5f95165cd1ccb51a07adafe0046` |

The validation ratio was `0.99116689`; each method completed exactly `1,000`
steps and `64,000` images. Both final checkpoints were re-hashed by the
runbook and matched their integrity sidecars. One initial runbook invocation
failed before output creation because `PYTHONPATH=src` was missing; its log is
preserved as `pair1k_rollout_x0_u2.preflight_failed_missing_pythonpath.log`.
The corrected invocation completed normally.

## Two-step 1K robust qualification

Raw-model n=8 passed all nine gates:

| metric | value |
|---|---:|
| tail-two energy | 0.57099337 |
| maximum single-token energy | 0.29130879 |
| ordered path rank | 1 |
| endpoint ratio | 0.99369001 |
| validation ratio | 0.99116689 |
| peak predicted-x0 HF ratio | 0.90376871 |
| reconstruction ratio | 0.99445521 |
| CoFiTok/dense reconstruction amplification | 1.07672227 / 1.10168261 |
| zero / shuffle | 0 / 103.289x |

EMA remained a non-authoritative 1K lag diagnostic: endpoint ratio
`1.10809945`, ordered rank `4`, reconstruction ratio `1.00518232`, and peak
HF ratio `0.92085161`.

Both independent n=64 raw-model reports passed:

| seed | reconstruction ratio | CoFiTok/dense amplification | peak HF ratio |
|---:|---:|---:|---:|
| 2029 | 0.97597848 | 1.04731459 / 1.05692856 | 0.96025456 |
| 2039 | 0.97530117 | 1.04200636 / 1.05223188 | 0.93034258 |

Authoritative sources:

```text
artifacts/reports/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2/model/qualification_report.json
sha256: f4a0b344041b777bd74202daf6661ce5db271e76022e5a5c9710407160a594ea

artifacts/reports/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2_n64_seed2029/qualification_report.json
sha256: 4585dd1122aebbd00eae56386d25d14309f6fe3534b7e1ac692d91d7f2ad679f

artifacts/reports/generation/stability_probe_2026-07-29/qualification1k_rollout_x0_u2_n64_seed2039/qualification_report.json
sha256: 4c15772716eaaea749cc0ccf4a023b8fb15041d8f82d634afade97b155f442d2
```

The stage-aware decision is SHA256
`d47c2518e3c7fff18e8c4a9d6a2c605e1c875b9100a4641d9b8250285daac53b`:

```text
status: pass
decision: authorize_fresh_matched_5k
authorized_next_stage: matched_5k
```

It explicitly does not authorize fresh 50K or full 300K.

## Two-step matched 5K rerun

The 5K target revision is
`2521d874a82898a7a2a527d824ea1df285df221d`; its isolated checkout is:

```text
/tmp/cofitok-generation-stability-u2-5k-2521d87
```

The official remote repository remains unchanged at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`. The target Linux tests passed,
and the 5K CUDA benchmark measured `22.4852 / 24.7537 images/s` for
CoFiTok/dense.

Runbook:

```text
artifacts/runbooks/generation_stability_rollout_x0_u2_probe5k_2026-07-30.sh
sha256: b4538d2929ec285cb4a3307f42d703e553d782df91e6ad33ada95275a4561709
```

Active queue:

```text
runbook PID: 494747
initial CoFiTok trainer PID: 494771
read-only monitor PID: 496061
output: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2
log: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2.log
monitor: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_monitor.json
order: CoFiTok 5K, then dense 5K
```

At the static record update, CoFiTok had reached step 75 with finite metrics.
Real-time status must be read from the remote `train_metrics.jsonl`,
`training_report.json`, process table, and GPU state. This fresh 5K pair must
complete and pass the same raw-model n=8 plus two-seed n=64 gate before any
new 50K preparation is allowed.

The monitor is the tracked generic read-only pair monitor at target revision
`2521d87`. It polls every 60 seconds, declares a stall after 1,800 seconds
without metric activity, allows 600 seconds for process transitions, checks
the 1,250-step checkpoint cadence with a 100-step grace, and exits terminally
only on `pass`, `failed`, or `stalled`. Its initial report was
`running / cofitok_training` with `issues=[]`, clean Git provenance, and about
274 GB free disk. It does not load or hash checkpoint payloads and does not
restart or terminate training.

The first CoFiTok milestone passed a read-only audit while training continued:

```text
step: 1,250 / 5,000
checkpoint bytes: 1,006,321,322
checkpoint SHA256: 8da625d6fb70dab7cab9a607029bd3c9efc9596b75a7acb53576872673836c3d
validation epsilon MSE: 0.03383226320147514
rollout consistency scale: 1.0
```

The checkpoint integrity sidecar and `latest.json` agree on filename, bytes,
SHA256, dataset identity, runtime-environment identity, step, and revision
`2521d874a82898a7a2a527d824ea1df285df221d`. The canonical metrics row includes
the scheduled validation event. No checkpoint payload was loaded or rehashed
during this audit.

The second CoFiTok milestone passed the same read-only integrity audit:

```text
step: 2,500 / 5,000
checkpoint bytes: 1,006,321,322
checkpoint SHA256: ad14dbd6ae47543fbe8d15a1ba86aed928b901bc97d0a15a776bdfefd74b1253
validation epsilon MSE: 0.0368012972176075
rollout consistency scale: 1.0
```

The sidecar, `latest.json`, canonical validation row, dataset identity,
runtime-environment identity, and Git revision all agree. The raw validation
MSE is 8.78% higher than the step-1,250 value even though the logged training
losses continued to decrease. This is not a like-for-like learning-curve
comparison: the deterministic evaluation iterator advances to a new image batch
at each milestone, while only the noise and timestep stream is reset. The value
is therefore retained as an unnormalized intermediate observation, not a
generalization warning. The matched CoFiTok/dense values at the same milestone
and the final free-state evaluation are the valid comparisons.

The third CoFiTok milestone also passed the read-only integrity audit:

```text
step: 3,750 / 5,000
checkpoint bytes: 1,006,321,322
checkpoint SHA256: 6513d12b9131a0276eed465d51b8f6908197d309463ea4d8ef43f7690d0f675e
validation epsilon MSE: 0.022937312722206116
rollout consistency scale: 1.0
```

The checkpoint, sidecar, `latest.json`, canonical validation row, dataset
identity, runtime-environment identity, and Git revision agree. Steps 1,250,
2,500, and 3,750 all remain present. The validation value belongs to the third
deterministic image batch and is not interpreted as a same-batch trend.

Post-training revision `9f435d4` closes this validation-observability gap for
future runs. Each validation row now records its zero-based event index, actual
DataLoader batch index, reset noise seed, and image count. Exact-resume tests
prove that uninterrupted and segmented runs preserve all four fields, and the
training auditor distinguishes complete metadata from legacy absence while
rejecting noncontiguous event indices or invalid counts. The targeted tests
passed `30/30`, and the full local suite passed with three existing skips.
This change is intentionally not deployed into the active `2521d87` checkout;
the current matched 5K pair remains one-revision evidence.

The tracked post-training evaluator is prepared but cannot run before the
pair summary exists:

```text
artifacts/runbooks/generation_stability_rollout_x0_u2_posteval5k_2026-07-30.sh
sha256: f6358383cabd61c2215dea748014b8d93e189d57c5690e307c4f34078f0a64cd
```

Its invocation requires `EXPECTED_PAIR_SUMMARY_SHA256` to bind the completed
pair. It runs raw and EMA n=8 diagnostics, permits robust expansion only when
n=8 has no failures beyond an optional reconstruction failure, then runs raw
n=64 seeds 2029 and 2039. Only two passing robust reports can produce
`authorize_fresh_matched_50k_preparation`; the runbook never launches 50K
training itself. Targeted tests, `git diff --check`, remote `bash -n`, and the
uploaded source SHA all passed.

A bounded follow-up waiter now keeps the post-evaluation connected to the
long-running pair:

```text
source: artifacts/runbooks/generation_stability_rollout_x0_u2_posteval5k_waiter_2026-07-30.sh
sha256: 17dc6a11482aa27e6b0518821708e03668df52dd2536726522adef487beb91c0
remote PID: 500048
log: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_posteval_waiter.log
```

It polls the authoritative monitor every 60 seconds for at most 16 hours and
fails closed on `failed`, `stalled`, timeout, revision drift, tracked checkout
changes, a busy GPU, or a changed post-evaluation SHA. It launches the prepared
post-evaluation only after monitor `pass`, an existing pair summary, and the
exit of both the pair runbook and training processes. The computed pair-summary
SHA is then passed through `EXPECTED_PAIR_SUMMARY_SHA256`. A one-second bounded
remote trial correctly observed `running / cofitok_training` and exited with
the dedicated timeout code without side effects.

## Matched two-step 5K result

The pair completed on revision
`2521d874a82898a7a2a527d824ea1df285df221d`:

```text
pair summary SHA256: ce604c2e9d2a1864bb6ba6fad31cdcee1cc92219256d361333abc00feaded28c
steps / images seen: 5,000 / 320,000 per method
CoFiTok validation MSE: 0.018510280176997185
dense validation MSE: 0.018453380092978477
validation ratio: 1.0030834504969828
CoFiTok final checkpoint SHA256: 47cfc77efac3835e57225cb14f0d6b78d94554e1c93d50e6efc46629979acbf4
dense final checkpoint SHA256: e7bf9d53cca26605842841285f9884dbd8cd549cd611296b5c0ea39ec03d4a35
CoFiTok elapsed / peak VRAM: 14,203.25 s / 15,237,613,056 bytes
dense elapsed / peak VRAM: 12,897.02 s / 15,043,470,848 bytes
```

Both methods retain checkpoints and integrity sidecars at steps 1,250, 2,500,
3,750, and 5,000. The terminal monitor is `pass / complete` with no issues.

The bound post-evaluation ran and stopped exactly at the raw n=8 fail-closed
screen:

```text
qualification SHA256: c4603cbcdfad7537e8a53d9cb5493cd0896eb17444a9cf04d7ef1362bf42f6a1
status: fail
failed gates: predicted_x0_high_frequency
peak raw high-frequency ratio: 2.0512487574980742
raw reconstruction ratio: 0.9384448969585206
raw endpoint ratio: 1.0041073640248266
raw validation ratio: 1.0030834504969828
tail-two energy ratio: 0.5670275926111694
maximum single-token energy ratio: 0.28473167502910013
```

All other gates passed, including ordered rank 1, zero-token, shuffle mismatch,
endpoint, validation, reconstruction, and both tail-energy checks. The raw
high-frequency ratio was already 1.60 at timestep 595 and rose to 2.05 at
timestep 91. Because this is a non-reconstruction screening failure, the
authoritative runbook correctly skipped raw n=64 and did not create a scaling
decision. Formal 50K preparation remains unauthorized.

The n=8 EMA diagnostic shows a weight-path split rather than tail-token energy
collapse: EMA CoFiTok reconstruction is `0.25409` versus dense `0.26379`, and
its four high-frequency values are lower than dense, while the raw CoFiTok
values are higher. The diagnostic-only EMA n=64 two-seed run completed using
tracked runbook
`generation_stability_rollout_x0_u2_ema_n64_diagnostic5k_2026-07-30.sh`
(SHA256
`091a2ca9dd4b4422d55eeb25d626fb79e0498ab47a690a7e4906d30f389a98cc`):

| seed | peak EMA HF ratio | reconstruction ratio |
|---:|---:|---:|
| 2029 | 0.36133072 | 0.96585160 |
| 2039 | 0.33558837 | 0.96539166 |

Every diagnostic threshold check passed. The summary SHA256 is
`78e14cd337ea43a24d621f04544a159b933bcc981c1ef82c49f7de7ab2ac7046`.
It permanently declares `scaling_authorization_allowed=false` and cannot
replace or override the failed raw qualification.

The next read-only diagnostic evaluated raw model weights at the immutable
1,250, 2,500, 3,750, and 5,000 checkpoints under the same n=8 seed-2029
rollout protocol. It bound the pair and failed qualification SHA256 values,
reused the final 5K reports, and recorded the first checkpoint whose peak
selected predicted-x0 high-frequency ratio exceeded `1.5`. It could not
authorize scaling:

```text
runbook:
artifacts/runbooks/generation_stability_rollout_x0_u2_raw_milestone_diagnostic5k_2026-07-30.sh
sha256: e3ba897cb2b2f16bde0fe19bd6b8d47807ffce6318eb0115b9f7857eaf8ccff0
output:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/diagnostic5k_rollout_x0_u2_raw_milestones/diagnostic_report.json
summary sha256:
e741e84b23c090e219b10b36ba0d403d067a9a30d567ef2aa0b80d1f3dbcf9ec
```

| step | peak raw HF ratio | reconstruction ratio |
|---:|---:|---:|
| 1,250 | 1.27977744 | 1.09487631 |
| 2,500 | 0.69351851 | 1.01795077 |
| 3,750 | 0.84661776 | 0.84107244 |
| 5,000 | 2.05124876 | 0.93844490 |

The HF gate first crosses at step 5,000. Steps 2,500 and 3,750 are stable in
both selected HF and reconstruction, while the final raw model abruptly
diverges from its stable EMA path. This rules out a monotonic capacity collapse
and localizes the correction to late raw-weight stability between 3,750 and
5,000. Formal 50K remains unauthorized until a fresh corrected matched pair
passes the raw gate.

## Late raw-weight EMA-teacher correction

Revision `4cb061d898aabe58e7b5542f214b6d7968e1b4e1` introduced an optional
late-training consistency loss between the raw model output and the existing
EMA shadow output on a bounded micro-batch subset. The teacher path is
no-gradient, uses evaluation mode, and never enters the synthesis operator.
The loss weight, start, warmup, and batch fraction are shared training fields
that the generation-pair contract requires to match exactly between CoFiTok and
dense.

The fresh 1K candidate uses:

```text
weight: 0.25
start step: 600
warmup steps: 300
batch fraction: 0.0625
two-step clipped-x0 rollout: unchanged
```

The first CUDA rehearsal correctly failed before any checkpoint was written.
PyTorch `functional_call(strict=True)` required sixteen nonpersistent fixed
synthesis buffers that are intentionally absent from both `state_dict()` and
the EMA shadow. Revision
`10f2f6bd9977fb1a63de4b2939ca641107a0ccaa` fixes the boundary by first
requiring exact equality between EMA keys and `model.state_dict()` keys, then
allowing only nonpersistent module constants to come from the live model.
Student gradients, EMA-state isolation, training-mode restoration, mismatched
state rejection, and nonpersistent-buffer behavior are covered by tests.

Local full pytest passed with three existing skips. The isolated Linux suite
passed after excluding only `test_aaai27_experiment_structure.py`, whose paper
sibling directory does not exist under the `/tmp` worktree layout; the relevant
Linux tests passed `59/59`. The official remote repository remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

The active-teacher CUDA benchmark v2 passed:

| method | images/s | peak VRAM | teacher loss | grad norm |
|---|---:|---:|---:|---:|
| CoFiTok | 22.1245 | 15,232,468,992 | 1.3413e-10 | 110.60 |
| dense | 24.3071 | 15,036,313,600 | 1.2310e-11 | 1.14 |

Both reports have teacher scale `1.0`, finite positive teacher loss, no
checkpoint, clean revision provenance, memory below 90%, and a valid matched
pair contract. The benchmark summary SHA256 is
`e5a88a1e7d18bef30de56ee43b46e94447c76302a84c0aa09d0928495fbaab49`.
The tracked runbook SHA256 is
`c9723579eb4bc432e898481beee8de63c8980a68fe3a3505d5dc69674030dc1f`.

This benchmark authorizes only a fresh matched 1K qualification probe. It does
not authorize 5K, 50K, or full training.

## EMA-teacher matched 1K launch

The fresh matched pair is running in the isolated revision
`10f2f6bd9977fb1a63de4b2939ca641107a0ccaa` checkout:

```text
checkout: /tmp/cofitok-generation-stability-ema-teacher-10f2f6b
output:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2_ema_teacher
runbook PID: 853818
initial CoFiTok trainer PID: 853839
monitor PID: 854173
post-eval waiter PID: 854722
raw n64 waiter PID: 856046
```

The training runbook SHA256 is
`4b2d5daa78d819edce64eb530506dc7a9eba323cfe9077d3378ce43b0a9def6c`.
Its inputs bind the active-teacher benchmark, failed raw 5K qualification, and
raw milestone diagnostic SHA256 values. The initial monitor state is
`running / cofitok_training / issues=[]` with clean target Git provenance.

The n=8 post-evaluation runbook SHA256 is
`dedef29410b6d76c91fc0a626e9f2e1f05e062cf6782d926e8082b8452a434a5`;
the waiter SHA256 is
`48eef1da04914e9cf86e19f6e8b9021bf1c2942e85a3fb522edf995908a9cfff`.
A one-second bounded waiter trial observed the active queue and timed out
without side effects. The real waiter requires monitor `pass / complete`, both
trainers and the pair runbook to exit, an idle GPU, exact pair/checkpoint
SHA256 identities, and the unchanged post-eval SHA before launch. The screening
summary always sets `scaling_authorization_allowed=false`; n=8 alone cannot
authorize 5K.

Real-time authority is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2_ema_teacher_monitor.json
```

The second-stage raw n=64 runbook SHA256 is
`04ff2aa80762a0e4a25c0608520ef4740c54ce49192d64380321141ee4ddf919`;
its waiter SHA256 is
`ceae6c4acdfec5c35d62d09c573f37a8a0cd50f1546206334365e71cdcb765d5`.
A bounded trial correctly timed out while the n=8 screening was absent. The
real waiter is PID `856046`: it exits without GPU work when authoritative raw
n=8 fails, and otherwise runs raw n=64 seeds 2029 and 2039 before invoking the
shared stage-aware decision builder with `next_stage=matched_5k`. It never
starts 5K training.

The first CoFiTok milestone passed a read-only integrity audit while training
continued:

```text
step: 250 / 1,000
checkpoint bytes: 1,006,321,770
checkpoint SHA256: f78653e9e6441b7b14d6067181b5529fecc5aef33ea417a65c117631c42cb1cf
validation epsilon MSE: 0.07069774717092514
EMA-teacher scale: 0.0
rollout-consistency scale: 1.0
```

The checkpoint sidecar and `latest.json` agree on filename, bytes, SHA256,
step, Git revision, dataset identity, and runtime-environment identity. The
canonical metric row records exactly 16,000 images. The checkpoint payload was
not loaded or rehashed during this audit.

The second CoFiTok milestone passed the same read-only integrity audit:

```text
step: 500 / 1,000
checkpoint bytes: 1,006,321,770
checkpoint SHA256: 09ab41200dcb4a783358c5705d455ba0fa7756db2204e40114badfcc3411fcd0
validation epsilon MSE: 0.042676351964473724
EMA-teacher scale: 0.0
rollout-consistency scale: 1.0
```

The sidecar and `latest.json` again agree on filename, bytes, SHA256, step,
revision, dataset identity, and runtime-environment identity. The canonical
step-500 row records exactly 32,000 images and the second scheduled validation
event. Validation MSE improved from the step-250 milestone. The teacher is
still correctly inactive before its step-600 boundary, and the checkpoint
payload was not loaded or rehashed.

The live training loop then exercised the active-teacher path at the expected
warmup values:

| step | teacher scale | teacher loss | total loss | gradient norm |
|---:|---:|---:|---:|---:|
| 600 | 0.00000000 | 0.00000000 | 0.07045249 | 0.86652696 |
| 625 | 0.08333334 | 0.15508846 | 0.06570209 | 0.18528393 |
| 650 | 0.16666667 | 0.15917403 | 0.07743759 | 1.65572608 |

This matches the configured `(step - 600) / 300` schedule. The active loss
remained finite and positive, the student gradient remained finite, and
monitoring reported no issue. GPU memory was 23,515 MiB at step 650 versus
about 23,507 MiB before activation.

The third CoFiTok milestone also passed the read-only integrity audit:

```text
step: 750 / 1,000
checkpoint bytes: 1,006,321,770
checkpoint SHA256: e50371e8bca0843d115826382200c2b651dc7dfb496a6c2cf17686fdd3ed7708
validation epsilon MSE: 0.029762834310531616
EMA-teacher loss / scale: 0.1346598118543625 / 0.5
rollout-consistency scale: 1.0
```

The sidecar, `latest.json`, checkpoint stat, revision, dataset identity, and
runtime-environment identity all agree. The canonical row records exactly
48,000 images and the third scheduled validation event. Validation MSE
continued to improve after teacher activation. The checkpoint payload was not
loaded or rehashed.

The teacher reached full scale at step 900 and remained stable through the
final row:

| step | teacher scale | teacher loss | total loss | gradient norm |
|---:|---:|---:|---:|---:|
| 900 | 1.0 | 0.11594299 | 0.09708235 | 0.33503842 |
| 925 | 1.0 | 0.11425493 | 0.09783078 | 0.19913450 |
| 1,000 | 1.0 | 0.12345801 | 0.10803740 | 0.88709414 |

The CoFiTok member then completed exactly:

```text
steps: 1,000 / 1,000
images seen: 64,000
canonical metric rows: 41
scheduled validation rows: 4
final validation epsilon MSE: 0.026219427585601807
final checkpoint bytes: 1,006,321,770
final checkpoint SHA256: 85c61f83a333f330dde42ab1df1f3eb462788164caf05950327f590ab732695b
segment elapsed seconds: 2,890.8206026554108
peak VRAM bytes: 15,234,796,032
parameters: 62,834,083
```

The final checkpoint stat, sidecar, and `latest.json` agree on all integrity
and provenance fields. `training_report.json` declares exact completion at the
target revision, and the final canonical metric row binds the image count and
fourth validation event. No payload was loaded or rehashed. The locked runbook
then launched the matched dense member as PID 862754; GPU utilization was 98%
with 18,370 MiB process memory during its first interval.

The first dense milestone passed the same read-only audit while training
continued:

```text
step: 250 / 1,000
images seen: 16,000
checkpoint bytes: 1,006,120,150
checkpoint SHA256: ca4386e700e4549b28f5ca50d8ad10552a7a924b05a173078302f6c181f73e94
validation epsilon MSE: 0.061262041330337524
EMA-teacher scale: 0.0
rollout-consistency scale: 1.0
```

The dense checkpoint stat, sidecar, and `latest.json` agree on filename,
bytes, SHA256, step, target revision, and the same dataset and runtime
identities used by CoFiTok. The checkpoint payload was not loaded or rehashed.

The second dense milestone passed the same audit:

```text
step: 500 / 1,000
images seen: 32,000
checkpoint bytes: 1,006,120,150
checkpoint SHA256: 9567e4c5a0be80d7c453161a3cd6daff1f3bbe072b36248fb15c8c9bf2972da1
validation epsilon MSE: 0.042720943689346313
EMA-teacher scale: 0.0
rollout-consistency scale: 1.0
```

The sidecar, `latest.json`, checkpoint stat, revision, dataset identity, and
runtime-environment identity all agree. The checkpoint payload was not loaded
or rehashed. At this matched step the CoFiTok and dense validation MSE values
were close (`0.04267635` and `0.04272094`), but this training-time diagnostic
does not replace the external rollout qualification.

The dense active-teacher path also ran at the configured boundary:

| step | logged teacher scale | teacher loss | total loss | gradient norm |
|---:|---:|---:|---:|---:|
| 600 | 0.00000000 | 0.00000000 | 0.05102846 | 0.46318033 |
| 625 | 0.08349609 | 0.14063581 | 0.04537644 | 0.22004163 |
| 650 | 0.16699219 | 0.14415441 | 0.05803266 | 0.85093874 |

The schedule function returned the same Python float values as CoFiTok. The
dense output is bf16, so `LossBreakdown` logged the nearest bf16 scalar values
for `1/12` and `1/6`; this is a bounded representation effect rather than a
step offset. The teacher MSE is explicitly computed in float32. Losses and
gradients remained finite and monitoring reported no issue.

The third dense milestone passed the read-only audit:

```text
step: 750 / 1,000
images seen: 48,000
checkpoint bytes: 1,006,120,150
checkpoint SHA256: 7036abd8076ea09d2ba3710d729da5fc35b78508a604c3fb74d9f05d3b6e2231
validation epsilon MSE: 0.02896382473409176
EMA-teacher loss / scale: 0.12420607171952724 / 0.5
rollout-consistency scale: 1.0
```

The checkpoint stat, sidecar, and `latest.json` agree on all integrity and
provenance fields. The canonical row records the third scheduled validation
event. The payload was not loaded or rehashed.

The dense teacher reached full scale and remained finite:

| step | teacher scale | teacher loss | total loss | gradient norm |
|---:|---:|---:|---:|---:|
| 900 | 1.0 | 0.10376317 | 0.07457095 | 0.23146720 |
| 925 | 1.0 | 0.09929421 | 0.07404177 | 0.18409708 |
| 1,000 | 1.0 | 0.10877534 | 0.08509394 | 0.48646978 |

The dense member completed exactly:

```text
steps: 1,000 / 1,000
images seen: 64,000
canonical metric rows: 41
scheduled validation rows: 4
final validation epsilon MSE: 0.025760110467672348
final checkpoint bytes: 1,006,120,150
final checkpoint SHA256: a0494f10d5f36cef707654b9bd5406d464ee1b894f32f302cfa7e4a9a2614aa2
segment elapsed seconds: 2,603.068747997284
peak VRAM bytes: 15,043,471,872
parameters: 62,824,707
```

The final dense checkpoint stat, sidecar, and `latest.json` agree on all
integrity and provenance fields. The report declares exact completion, and the
final metric row binds the fourth validation event. The payload was not loaded
or rehashed.

The read-only monitor reached `pass / complete / issues=[]`. The training
runbook then wrote a completed pair summary:

```text
pair summary SHA256: 94e857f0b537a03213e57079ef5a333d6508eaacbcdd067f85e71dbe0077f083
CoFiTok training report SHA256: e3db90536f019585ca264e22db6ed87e72c592a5a2857b4ace3f6c3ebe2ec94a
dense training report SHA256: c40ef1d8133b5abe328bece50b915c58e92aa795d76aab84dbbc96df3d053d41
parameter ratio delta: +0.014924064826926653%
validation ratio: 1.0178305569964803
generation-pair contract: valid
```

Both members use revision `10f2f6b`, the same dataset/runtime identities,
exactly 1,000 steps and 64,000 images, and identical data, diffusion, runtime,
optimization, rollout-consistency, and EMA-teacher fields. The pair summary
binds the active CUDA benchmark, failed raw 5K qualification, and raw
milestone diagnostic. Its completion authorized only the already queued raw
n=8 screening; it did not authorize 5K.

## EMA-teacher matched 1K raw n=8 screening

The automatic post-evaluation completed model and EMA checkpoint/rollout
reports before writing the screening summary. The authoritative raw model
qualification passed every gate:

| gate | value | threshold |
|---|---:|---:|
| predicted-x0 peak high-frequency ratio | 0.74367630 | <= 1.50 |
| reconstruction ratio | 1.04567489 | <= 1.05 |
| endpoint ratio | 1.00346327 | <= 1.05 |
| validation ratio | 1.01783056 | <= 1.05 |
| tail-two energy ratio | 0.56783742 | <= 0.65 |
| maximum single-token energy ratio | 0.28732264 | <= 0.35 |
| ordered rank by path AUC | 1 | == 1 |
| shuffle / ordered endpoint ratio | 82.24813305 | >= 2.0 |
| zero-token maximum absolute value | 0.0 | <= 1e-8 |

The selected raw high-frequency ratios were `0.65457`, `0.69905`, `0.72493`,
and `0.74368` at timesteps 595, 394, 192, and 91. This is a large reversal
from the failed raw 5K peak of `2.05125`, but n=8 remains screening evidence.
The raw qualification SHA256 is
`c719b1daef242157cb968bf0d2f49995f64ca6368bb8f8d355e12b96673ee651`;
the screening summary SHA256 is
`ccb70d103c9c565591c20c671e7b26c963cbdd830783856c7f842125c22c63ef`.
The summary still declares `scaling_authorization_allowed=false`.

The EMA diagnostic had peak high-frequency ratio `1.12792` and reconstruction
ratio `0.97055`; it remains diagnostic only. Because raw n=8 passed, the
pre-authorized waiter launched raw n=64 seed 2029. No 5K training was started.

All small 1K training, monitor, evaluation, and qualification artifacts were
synced under
`artifacts/reports/generation/stability_probe_2026-07-29/`. A source/local
SHA256 audit compared 29 files and reported zero missing, extra, or mismatched
files. No checkpoint payload was copied.
