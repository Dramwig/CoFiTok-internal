# Full-data quality-bridge matched 19K training trajectory

Date: 2026-08-19

## Purpose

Freeze the largest currently available shared training prefix of the active
full-ImageNet-256 matched quality bridge.  This is a CPU-only optimization
trajectory diagnostic.  It does not load checkpoint payloads, generate
samples, replace the terminal 100K evaluation, or authorize any later stage.

## Exact source

- Output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1`
- Revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- Branch: `scale/generation-stability-quality-bridge-100k`
- Dataset: `imagenet_256`
- Dataset identity:
  `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
- Cutoff: step `19,000`
- Images seen per method: `1,216,000`
- Effective batch: `64`
- CoFiTok parameters: `62,834,083`
- Dense parameters: `62,824,707`
- Relative parameter gap: `+0.0149240648%`

The generation-pair contract passed with no mismatched shared model,
data/diffusion/runtime/optimization, or shared stabilization-loss fields.

## Matched fixed-validation result

All 19 events are paired by exact optimizer step, validation event and batch
indices, image count, and fixed noise seed.

| statistic | CoFiTok | dense identity |
|---|---:|---:|
| mean epsilon MSE | `0.0321993203` | `0.0321801003` |
| lower-MSE event count | `10/19` | `9/19` |

The ratio-of-means relative delta is `+0.0597265%`, so CoFiTok is slightly
higher on the shared primary epsilon metric.  The exact 19K endpoint delta is
`+0.0462920%`.  The largest absolute per-event relative difference is only
`1.088895%`.

During the rollout-consistency full-scale interval from 10K through 19K,
CoFiTok is lower on `6/10` events, but its mean remains `+0.0313072%` above
dense.  The 1K--9K rollout warmup mean delta is `+0.0899670%`.

The correct interpretation is close matched optimization with no visible
CoFiTok-specific collapse and no training-space advantage at this cutoff.
The near tie cannot establish endpoint generation quality, distribution
support, conditioning fidelity, or a CoFiTok quality win.

## Frozen evidence

Local evidence directory:

```text
artifacts/reports/generation/matched_training_trajectory_step_00019000_2026-08-19
```

Identities:

- `trajectory_report.json`: `25,897` bytes, SHA256
  `43788ae02587eff174f30bf4120fe5b3f3db974deaf78813525f6ea31ba5cc37`
- CoFiTok metrics through 19K: `357,450` bytes, SHA256
  `4badc1bf7910786feb297b7063b2a5188430abe3c61eda227ed45e9f0c654263`
- Dense metrics through 19K: `332,707` bytes, SHA256
  `8d1e9431e2701878c330ddc1d8d188b5deba034f9017a5a4b6aa1ddc567f85c6`
- Builder SHA256:
  `0491b733b7e4839f86e80c10964a9bf01be0be94ec7f75c95ec597c6cf127968`

An independent rebuild from the still-growing authoritative JSONL sources
produced byte-identical frozen 19K snapshots.  The report was structurally
identical after normalizing the intentionally different absolute temporary
snapshot directory.  A raw cross-directory byte comparison is not expected
because the report source descriptors bind absolute snapshot paths.

## Claim boundary

This diagnostic supports only:

- a matched resolved training contract through step 19K;
- a source-bound fixed-validation trajectory through step 19K;
- absence of an obvious method-specific optimization collapse in the shared
  primary loss.

It is not a sample-quality metric, significance claim, promotion gate,
full-training authorization, or substitute for exact 100K completion followed
by matched EMA sampling, FID/precision/recall, class fidelity, uncertainty,
runtime fairness, visual review, and terminal claim guards.
