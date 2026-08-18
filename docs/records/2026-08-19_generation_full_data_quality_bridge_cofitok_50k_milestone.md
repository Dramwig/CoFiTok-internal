# Full-data quality bridge CoFiTok 50K milestone audit

Date: 2026-08-19

## Purpose

Freeze and audit the first completed sampling milestone of the active matched
full-ImageNet-256 100K quality bridge without competing with the dense trainer.
This work copied only small JSON evidence and one requested-class contact
sheet. It did not load checkpoint payloads, use the GPU, signal any process,
modify the active checkout, or alter the running experiment.

## Authoritative source

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
cofitok_rgbtail3_rollout_x0_u2_ema_teacher/milestones/step_00050000
```

Training identity:

- revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- branch: `scale/generation-stability-quality-bridge-100k`
- tracked state: clean
- dataset: `imagenet_256`
- dataset identity:
  `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
- step: `50,000 / 100,000`
- images seen: `3,200,000`
- parameter count: `62,834,083`

Checkpoint identity:

- filename: `checkpoint_step_00050000.pt`
- bytes: `1,010,937,514`
- SHA256:
  `d7100a6e8f67adb2b1630d36c17d0f2bc94fb93ed99d3e270a0867979c3343b4`
- integrity sidecar, `latest.json`, dataset identity, runtime identity, step and
  Git identity all agree.

## Sampling and generated quality

The completed exploratory milestone used EMA, 2,048 samples, DDIM-50,
CFG 1.5, guidance rescale 0, eta 0, balanced-modulo class requests, bf16,
seed 0, and CoFiTok prefix budget 8. The sampling report binds sample-set
SHA256 `2d1efc1c15d057ee6c916f9685145d324b9c4deb08f28a27eaea31401f5dfe61`.

Results:

| metric | result |
|---|---:|
| FID | `221.0507338` |
| IS mean | `4.4841922` |
| IS std | `0.4050556` |
| top-1 class fidelity | `3/2048 = 0.146484%` |
| top-5 class fidelity | `14/2048 = 0.683594%` |
| predicted-class coverage | `213/1000 = 21.3%` |
| mean target probability | `0.00105494` |
| target NLL | `7.51485` |

Precision/recall were not enabled at this small milestone. The sixteen-class
contact sheet is dominated by high-frequency residual texture. Several images
contain coarse object-like silhouettes, but requested-class identity is not
reliable. The visual result therefore agrees with the near-random class
fidelity rather than supporting a semantic generation claim.

## Mechanism result

The 256-image checkpoint evaluation at `t=500` reports:

- ordered rank `1/6` by prefix path MSE AUC;
- ordered path AUC `0.0929926814`;
- endpoint clean MSE `0.0156310825`;
- exact zero-token maximum absolute output `0.0`;
- shuffled-to-ordered endpoint ratio `130.3483`.

This is positive evidence that the ordered, restricted synthesis mechanism
remains intact at full-data step 50K. It does not rescue the weak generated
sample quality.

## Live matched-pair state at audit time

At `2026-08-19T07:32:40+08:00`:

- the pair monitor was `running / dense_identity_training`;
- dense had reached step `19,600`, or `1,254,400` images;
- the sole GPU workload was the expected dense trainer;
- `quality_bridge_result.json` did not exist;
- `followup_experiment_decision.json` did not exist;
- all conditioning-ranking stages remained source-bound waiters with no child
  GPU process.

Pair-monitor snapshot identity at that instant:

- bytes: `41,538`
- SHA256:
  `ba3ee937d1c819730cdfd329a51152a3039d6eef6744736d2391a23a25a54ea0`

## Frozen local evidence

```text
artifacts/reports/generation/
full_data_quality_bridge_cofitok_step_00050000_2026-08-19

artifacts/figures/generation/
full_data_quality_bridge_cofitok_step_00050000_2026-08-19
```

`milestone_summary.json` binds every copied source by byte count and SHA256.
All local copies were compared with the authoritative remote SHA256 values and
matched exactly. The checkpoint payload and all 2,048 individual samples were
intentionally not copied.

## Scientific decision

The milestone supports two simultaneous conclusions:

1. the scoped CoFiTok ordering/restricted-synthesis mechanism remains healthy;
2. the current 50K CoFiTok generator is not yet semantically or visually
   usable under this exploratory sampling protocol.

No direct CoFiTok-vs-dense quality comparison is allowed until dense reaches
the exact same milestone and protocol. The FID must not be directly compared
with the frozen 10% DDIM-100 result because training data, sample count and
sampling protocol differ. The required evidence remains matched step-50K,
then exact 100K completion and the terminal matched 10K DDIM-100 evaluation.
