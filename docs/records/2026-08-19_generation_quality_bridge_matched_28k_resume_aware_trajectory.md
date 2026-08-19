# Full-data quality-bridge matched 28K resume-aware trajectory

Date: 2026-08-19

## Purpose

Freeze and replay the largest complete 1,000-step shared prefix available after
the full-data CoFiTok run resumed from step 20,000. This is a CPU-only matched
optimization diagnostic. It does not load checkpoint payloads, generate
samples, replace the matched 50K/100K quality evaluations, or authorize a later
training or release stage.

## Exact source and execution

- Output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1`
- Training revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- Training branch: `scale/generation-stability-quality-bridge-100k`
- Dataset: `imagenet_256`
- Dataset identity:
  `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
- Runtime identity:
  `d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`
- Cutoff: step `28,000`
- Images seen per method: `1,792,000`
- Effective batch: `64`
- CoFiTok parameters: `62,834,083`
- Dense parameters: `62,824,707`
- Relative parameter gap: `+0.0149240648%`

The generation-pair contract is valid with no mismatched shared model,
data/diffusion/runtime/optimization, or shared stabilization-loss fields.

The report builder ran from the clean isolated audit checkout at:

```text
branch: analysis/generation-resume-aware-matched-trajectory-v1
revision: 40ee55202545ea01e48fb6642d1b5a51b5264663
builder SHA256: 2477345f81a99895f73b5c9a93844c8005ca959a2ed74ab3fa5a077c4753c21a
tracked dirty: false
```

The builder's filesystem SHA256, committed Git blob SHA256, and report-bound
SHA256 are identical.

## Exact-resume evidence

The CoFiTok run manifest binds:

```text
resume checkpoint: checkpoint_step_00020000.pt
reconciliation status: reconciled
resume step: 20,000
retained rows: 401
orphaned rows: 4
orphan archive SHA256: 6b3c4bca8ffa99e814ee77559d140f3a0db6cf7b78a8d78e58ef1b716e2f1cec
first post-resume step: 20,001
```

The physical orphan archive exists and independently rehashes to the bound
digest. The frozen canonical prefix has exactly one permitted off-grid row,
step `20,001`; rows `20,000`, `20,001`, and `20,050` are ordered correctly.
The retained-row count is independently verified. Therefore the extra CoFiTok
row is exact-resume evidence, not a duplicate or unmatched validation event.

At the 28K cutoff, CoFiTok has `562` canonical metric rows and dense has `561`.
Both have exactly `28` fixed-validation events at steps 1K through 28K, finite
numeric metrics, exact sample accounting, and complete matched validation
provenance.

## Matched fixed-validation result

All 28 events are paired by optimizer step, validation event and batch indices,
image count, and fixed noise seed.

| statistic | CoFiTok | dense identity |
|---|---:|---:|
| mean epsilon MSE | `0.0314285629` | `0.0314151143` |
| lower-MSE event count | `16/28` | `12/28` |

The ratio-of-means relative delta is `+0.0428094%`, so CoFiTok is marginally
higher on the shared primary validation metric. The exact 28K endpoint delta is
`+0.0390607%`. The largest absolute per-event relative difference is
`1.088895%`.

During the rollout-consistency full-scale interval from 10K through 28K,
CoFiTok is lower on `12/19` events, while its mean is `+0.0188655%` above dense.
EMA-teacher consistency is still inactive before its predeclared 30K start, so
this snapshot makes no EMA-teacher effect claim.

The supported interpretation is close matched optimization with no visible
CoFiTok-specific training collapse. It does not show a training-space advantage
and cannot establish endpoint sample quality, distribution support, requested
class fidelity, or a CoFiTok generation-quality win.

## Frozen evidence

Local evidence directory:

```text
artifacts/reports/generation/matched_training_trajectory_step_00028000_2026-08-19
```

Identities:

- `trajectory_report.json`: `34,593` bytes, SHA256
  `bb83c6c38e5f1a0cce03a9d44590e338d5e33a5f6ea95c0a6c1872eb51c70250`
- CoFiTok metrics through 28K: `526,231` bytes, `562` rows, SHA256
  `1045778fef2f775f15f175d35fc33bbb3f2beaacf5696e08b9380c2ed7e941d8`
- Dense metrics through 28K: `488,750` bytes, `561` rows, SHA256
  `6308000a818553919dfd7f85eac33df164693e45eb6b9067178aa4d4d47d30df`
- CoFiTok manifest SHA256:
  `c81bc1f1b75b1e09f257ee8177b144e12832f305ccdefda8c0715f8e145a061b`
- Dense manifest SHA256:
  `bf0b8854c7c7dae8983507d797677372b1dd100aad8726b3acf9dab9c9791dd8`

An independent rebuild from the frozen 28K snapshots produced a byte-exact
report with the same SHA256. The focused Linux CPU-only suite passed `21/21`;
the builder passed `py_compile`, `git diff --check`, and tracked-clean checks.
The incremental deployment bundle was `4,336` bytes with SHA256
`8d61e04147ce3020d0165d25b38249f0981318cce6b5d0d983baccd41952c3bb`.

The formal remote checkout remained unchanged at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`.
The only GPU compute process remained the active dense quality-bridge trainer.

## Claim boundary and next evidence

This diagnostic supports only:

- a matched resolved training contract through step 28K;
- an exact-resume-aware, source-bound fixed-validation trajectory;
- absence of an obvious method-specific optimization collapse in the shared
  primary loss.

It is not a sample-quality metric, significance claim, promotion gate,
full-training authorization, or release evidence. The next decisive quality
evidence remains matched dense 50K completion and same-protocol CoFiTok/dense
EMA sampling. A broad generation advantage still requires the terminal matched
100K formal sampling, FID/precision/recall, class fidelity, uncertainty,
runtime fairness, visual review, and claim guards.
