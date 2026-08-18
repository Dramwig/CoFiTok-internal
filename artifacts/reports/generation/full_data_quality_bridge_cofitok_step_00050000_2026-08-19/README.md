# Full-data quality bridge: CoFiTok step-50K milestone

This directory freezes the completed CoFiTok half-milestone from the active
full-ImageNet-256 matched 100K quality bridge. The copied reports and contact
sheet are byte-identical to the remote sources. The 1.01 GB checkpoint itself
remains only in the authoritative remote generation directory; its integrity
sidecar and `latest.json` binding are included here.

## Bound training state

- Revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- Branch: `scale/generation-stability-quality-bridge-100k`
- Dataset: `imagenet_256`
- Dataset identity:
  `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
- Step: `50,000 / 100,000`
- Images seen: `3,200,000`
- Parameters: `62,834,083`
- Checkpoint SHA256:
  `d7100a6e8f67adb2b1630d36c17d0f2bc94fb93ed99d3e270a0867979c3343b4`

The checkpoint integrity sidecar, `latest.json`, preflight, sampling manifest,
sampling progress, sampling report, sample-set digest, evaluator reports, Git
identity, dataset identity, and runtime identity all agree.

## Exploratory DDIM-50 quality

Protocol: EMA, 2,048 samples, DDIM-50, CFG 1.5, balanced-modulo classes,
bf16, seed 0, prefix budget 8.

| metric | result |
|---|---:|
| FID | `221.0507` |
| Inception Score | `4.4842 +/- 0.4051` |
| sampling throughput | `0.9315 images/s` |
| class top-1 | `3/2048 = 0.1465%` |
| class top-5 | `14/2048 = 0.6836%` |
| predicted classes | `213/1000` |
| mean requested-class probability | `0.001055` |

Precision/recall were intentionally disabled for this 2,048-sample milestone.
The requested-class contact sheet visibly retains severe high-frequency
residual structure. Some coarse silhouettes occur, but requested-class identity
is not dependable. This is consistent with the quantitative class-fidelity
failure.

## Mechanism result

The single-checkpoint mechanism diagnostic remains healthy at `t=500`:

- learned order ranks `1/6` by prefix path AUC;
- exact `S(0)=0` holds;
- shuffled endpoint MSE is `130.35x` the ordered endpoint MSE;
- the ordered endpoint clean MSE is `0.0156311`;
- the ordered prefix path MSE AUC is `0.0929927`.

Thus the ordered restricted factorization survives full-data scaling to 50K,
while the generated samples still fail absolute and semantic-quality criteria.

## Claim boundary

This is a source-bound **single-method exploratory milestone**, not a matched
quality result. Dense identity had not reached step 50K when this archive was
created. It cannot establish a CoFiTok-vs-dense FID advantage, cannot replace
the terminal 100K gate, and cannot be directly compared with the frozen 10%
10K-sample DDIM-100 gate because the training data, sample count, and sampler
protocol differ.

The authoritative machine-readable interpretation and all source identities
are in `milestone_summary.json`.
