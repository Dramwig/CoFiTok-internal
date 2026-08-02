# Stability matched 10K training trajectory

This evidence pack freezes the exact CoFiTok and dense-identity metrics prefixes
through optimizer step 10,000 from the active stability matched 50K pair. It is
a training-trajectory diagnostic, not a sample-quality result or promotion gate.

## Result

Both methods have exactly 201 logged rows through step 10,000, exactly 640,000
training images, finite numeric metrics, strict expected logging steps, and ten
scheduled validation events. The resolved data, diffusion, runtime,
optimization, shared backbone, shared consistency schedule, Git revision,
dataset identity, and runtime-environment identity pass the matched-pair
contract. The parameter gap is `+0.014924%` for CoFiTok relative to dense.

The ten validation events are directly paired by step, event index, batch
index, image count (`64`), and fixed noise seed (`102030`). Across those events:

- mean validation epsilon MSE is `0.0326398859` for CoFiTok and `0.0326587601`
  for dense, a relative difference of `-0.057792%`;
- the step-10,000 values are `0.0316699222` and `0.0316588767`, a relative
  difference of `+0.034889%`;
- CoFiTok is lower on `4/10` events and dense on `6/10`;
- the largest absolute per-event relative difference is `1.444119%`.

This is evidence of a closely aligned shared validation trajectory through 10K,
not evidence that either method has better generation quality. Training total
loss is intentionally not compared because CoFiTok has factorization-only
auxiliaries. Raw wall-clock speed is intentionally not compared because the
dense run experienced externally observed GPU contention.

## Bound sources

| Source | Bytes | SHA256 |
|---|---:|---|
| CoFiTok metrics prefix | 189,765 | `2610dedcba63f70421fc4868fc0c845e6dd4ae8c2ce729330c9d7ce8dd19a893` |
| Dense metrics prefix | 176,675 | `40713026c7d4e39d814398f3fe9820db8d9671246500aa2100336e7b3d1464f6` |
| CoFiTok run manifest | 8,401 | `3b554762a5ee75d77fbf66af616c6eaab31ae24dd7a6f40f1f9d4f747bc06519` |
| Dense run manifest | 7,671 | `a049a97d8a4e9f4dc4761a9269e50ae55f53ec28b9971d14a7d968bfd7e9f9fe` |
| Report builder | 20,764 | `91c907d297c83a8b643ca615f071415d50ed25ed78355c4be84f46d9295e1057` |
| `trajectory_report.json` | 12,877 | `2576f53574667ca0e0a8172a3501e9e5baf8f0255a6b6cd5d961797e2a267476` |

The raw-prefix identities remain stable even as the authoritative dense JSONL
continues growing. The source manifests bind training revision
`2c2c1f5166b73d4f28df93b276901671ac1a7836`, branch
`scale/generation-stability-50k-preflight`, dataset identity
`97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741`,
and runtime identity
`d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`.

## Rebuild

From `CoFiTok-internal`:

```powershell
$base = 'artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_10k_training_trajectory_2026-08-02'
.\.venv\Scripts\python.exe scripts\build_generation_matched_training_trajectory.py `
  --cofitok-metrics "$base/source/cofitok_train_metrics_through_step_00010000.jsonl" `
  --dense-metrics "$base/source/dense_train_metrics_through_step_00010000.jsonl" `
  --cofitok-manifest "$base/source/cofitok_run_manifest.json" `
  --dense-manifest "$base/source/dense_run_manifest.json" `
  --cutoff-step 10000 `
  --expected-revision 2c2c1f5166b73d4f28df93b276901671ac1a7836 `
  --expected-branch scale/generation-stability-50k-preflight `
  --cofitok-origin /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/train_metrics.jsonl `
  --dense-origin /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/dense_rollout_x0_u2_ema_teacher/train_metrics.jsonl `
  --output "$base/trajectory_report.json"
```

The report explicitly keeps `quality_claim_allowed=false`,
`formal_50k_gate_substitute=false`, `promotion_authorization_allowed=false`,
and `full_training_launch_allowed=false`. Required next evidence remains exact
healthy matched 50K completion followed by the existing formal EMA post-eval.
