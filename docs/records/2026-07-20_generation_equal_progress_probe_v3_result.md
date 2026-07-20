# Generation equal-progress probe v3 result

Date: 2026-07-20

Role: non-formal mechanism and directional-quality probe. This run cannot
authorize the 10K promotion gate, a fresh matched 50K pair, or full 300K
training.

## Immutable inputs

- Git revision: `a37b2e410aa558e831688009b13012a6a7da616b`
- Dataset: `imagenet_256_10pct`
- Checkpoint SHA256:
  `b03e3bee66dfb368330f056df3afc2bac451ed2a52a8c90ddae292825d1ef83c`
- Layout: channels `[4,4,8,8,8,8,1,2]`, spatial strides
  `[16,16,8,8,4,4,1,1]`
- Training: 5,000 updates, effective batch 64, seed 2027
- Objective: equal denoising progress, prefix/component weights `0.15/0.3`,
  batch-aggregated uniform energy weight `0.5`
- Mechanism eval: 512 validation images, timestep 500, ordered/reverse plus 16
  fixed random orders
- Sampling: 512 images, deterministic DDIM-50, CFG 1.5, EMA weights

The training report, integrity sidecar, sampling report, two mechanism reports,
two aggregate reports, and read-only monitor all completed. The monitor ended
with `status=pass` and `stage=complete`.

## Result

The stronger equal-progress and energy terms did not recover ordered prefixes
and reduced directional sample quality:

| candidate | endpoint MSE | ordered path AUC | rank / 18 | FID-512 | IS-512 |
|---|---:|---:|---:|---:|---:|
| v2 denoise path | 0.0199395 | 0.1995274 | 6 | 275.1347 | 2.2749 |
| v2 epsilon band | 0.0200607 | 0.2939404 | 18 | 281.8623 | 2.1310 |
| v3 equal progress | 0.0224811 | 0.2858201 | 17 | 375.8400 | 1.4768 |

The v3 EMA endpoint is 12.7% worse than the v2 denoise-path endpoint. Its
ordered path AUC is 43.3% higher, where lower is better, and the ordered path
is far from a numerical tie: the best tested random order has path AUC
`0.1062590`. The aggregate selection therefore remains
`revise_architecture_or_objective`, with automatic 50K/300K launch disabled.

Raw weights do not change this conclusion:

| weights | endpoint MSE | ordered path AUC | rank / 18 |
|---|---:|---:|---:|
| EMA | 0.0224811 | 0.2858201 | 17 |
| raw model | 0.0203134 | 0.2794341 | 17 |

EMA lag is not the ordering failure.

## Energy-scope diagnosis

The final training batch reported energy-budget MSE `0.0235302`, but fixed
`t=500` mechanism evaluation remained tail-heavy. The normalized EMA component
energy ratios were:

```text
[0.0049, 0.0037, 0.0105, 0.0018, 0.0162, 0.0167, 0.3316, 0.6146]
```

Tokens 7 and 8 still carry 94.6% of aggregate component energy. The fixed-step
uniform-target MSE is `0.04543`, close to the collapsed v2 value `0.05231` and
not to the much lower mixed-timestep training batches.

The implemented energy loss first averages component energy over the whole
batch and only then normalizes across tokens. Because each training batch mixes
timesteps, the model can reduce this loss while assigning different token roles
to different samples or noise levels. It does not enforce balanced token use
for each sample at the fixed timestep used by the mechanism gate.

This is a concrete objective loophole, not evidence that more training steps
will recover ordering.

## Visual audit

Two inspected sample indices show the same pattern. Prefix 1 is near pure
noise, prefix 2 adds broad low-frequency color, and prefix 4 produces local
texture and contour structure. Prefix 8 becomes high-frequency and noisy
instead of resolving into a stable class-conditional image. The visual result
agrees with rank 17/18 and FID 375.84: v3 is not a usable progressive generator.

## Next action

No formal configuration or recipe contract is changed by this result. Before
another 5K training run:

1. extend the evaluator with theoretical denoise-path component energies and
   per-sample model energy statistics;
2. evaluate the immutable v3 checkpoint at timesteps
   `50/250/500/750/950` under the same fixed-order protocol;
3. preserve legacy batch energy behavior, while adding an explicit per-sample
   scope for new probes;
4. choose between per-sample uniform energy and per-sample target-energy
   matching from those reports.

The tracked diagnostic runbook is
`artifacts/runbooks/generation_rank_recovery_timestep_diagnostic_2026-07-20.sh`.
It refuses to run while training, sampling, the v3 runbook, or another
checkpoint evaluator is active.

Authoritative aggregate sources are mirrored under
`artifacts/reports/generation/rank_recovery_probe_2026-07-20_v3/` as an
untracked local archive. Checkpoints and complete sample sets remain on the
server.
