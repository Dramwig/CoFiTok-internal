# Stability matched 12K schedule-transition trajectory

This evidence pack freezes the exact CoFiTok and dense-identity metrics prefixes
through optimizer step 12,000 from the active stability matched 50K pair. It is
a source-bound training diagnostic around the rollout-consistency warmup
boundary, not a sample-quality result, statistical significance claim, or
promotion gate.

## Result

Both methods have exactly 241 logged rows through step 12,000, exactly 768,000
training images, finite numeric metrics, strict expected logging steps, and 12
scheduled validation events. The resolved pair passes the matched data,
diffusion, runtime, optimization, shared-backbone, shared-schedule, Git,
dataset, runtime-environment, and parameter-gap contracts.

Across all 12 validation events, which are paired by step, event index, batch
index, 64-image count, and fixed noise seed 102030:

- mean epsilon MSE is `0.0325498291` for CoFiTok and `0.0327016859` for dense,
  a relative difference of `-0.464370%`;
- CoFiTok is lower on `6/12` events and dense is lower on `6/12`;
- the step-12,000 relative difference is `-4.507906%` for CoFiTok.

The report predeclares schedule regimes from both resolved manifests and also
checks that the observed per-event schedule scales agree between methods and
match the inferred phase:

| Shared schedule | Phase | Validation steps | CoFiTok mean | Dense mean | Relative delta of means |
|---|---|---|---:|---:|---:|
| rollout consistency | warmup | 1K--9K | `0.0327476596` | `0.0327698582` | `-0.067741%` |
| rollout consistency | full scale | 10K--12K | `0.0319563374` | `0.0324971688` | `-1.664242%` |
| EMA teacher consistency | inactive | 1K--12K | `0.0325498291` | `0.0327016859` | `-0.464370%` |

Only three validation events exist in the rollout full-scale regime, and each
event uses the next deterministic validation batch. The larger 12K endpoint
difference is therefore an exploratory observation, not evidence of a durable
trend or a generation-quality advantage. EMA-teacher consistency does not
start until step 30,000 and reaches full scale at step 40,000, so this pack says
nothing about that transition.

Total training loss remains incomparable because CoFiTok contains
factorization-only auxiliaries. Raw wall-clock throughput remains
observational-only because the dense run experienced unrelated FieldScope GPU
contention.

## Bound sources

| Source | Bytes | SHA256 |
|---|---:|---|
| CoFiTok metrics prefix | 227,039 | `6c1038815e530f2bd977c2fa46ea89111e6ef0350864118be42bb886b390559d` |
| Dense metrics prefix | 211,293 | `73aa12a6d4d7b82f7f46cdc1bbd1173aeef1c25853979182288747636001855e` |
| CoFiTok run manifest | 8,401 | `3b554762a5ee75d77fbf66af616c6eaab31ae24dd7a6f40f1f9d4f747bc06519` |
| Dense run manifest | 7,671 | `a049a97d8a4e9f4dc4761a9269e50ae55f53ec28b9971d14a7d968bfd7e9f9fe` |
| Report builder | 26,360 | `0491b733b7e4839f86e80c10964a9bf01be0be94ec7f75c95ec597c6cf127968` |
| `trajectory_report.json` | 21,443 | `64d291ae5730ce80d7b1656728e97dbf930aac68149332375fab8c2dbe8e7d62` |

The manifests bind training revision
`2c2c1f5166b73d4f28df93b276901671ac1a7836`, branch
`scale/generation-stability-50k-preflight`, dataset identity
`97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741`,
and runtime identity
`d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`.

## Rebuild

From `CoFiTok-internal`:

```powershell
$base = 'artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_12k_schedule_trajectory_2026-08-03'
.\.venv\Scripts\python.exe scripts\build_generation_matched_training_trajectory.py `
  --cofitok-metrics "$base/source/cofitok_train_metrics_through_step_00012000.jsonl" `
  --dense-metrics "$base/source/dense_train_metrics_through_step_00012000.jsonl" `
  --cofitok-manifest "$base/source/cofitok_run_manifest.json" `
  --dense-manifest "$base/source/dense_run_manifest.json" `
  --cutoff-step 12000 `
  --expected-revision 2c2c1f5166b73d4f28df93b276901671ac1a7836 `
  --expected-branch scale/generation-stability-50k-preflight `
  --cofitok-origin /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/train_metrics.jsonl `
  --dense-origin /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/dense_rollout_x0_u2_ema_teacher/train_metrics.jsonl `
  --output "$base/trajectory_report.json"
```

The report explicitly keeps schedule-regime significance, sample quality,
formal-gate substitution, promotion, and full-300K launch authorization false.
The required next evidence remains exact healthy matched 50K completion followed
by the existing formal EMA post-evaluation waiter.
