# Frozen existing-sample support audit

Date: 2026-08-05

## Purpose

The frozen stability gate remains `fail/hold` with CoFiTok FID `138.2970` and
dense FID `151.4477`. More importantly, formal recall is approximately `0.009`
for both methods. Before requesting any new GPU sampling or training, this audit
uses the already frozen 10K sample sets to distinguish simple duplicate collapse
from broader support, class-conditioning, or high-frequency defects.

This is a CPU-only, supplemental, non-authorizing audit. It cannot authorize
sampling, training, the 100K bridge, or full 300K. It does not replace formal
FID, Inception Score, precision, recall, class fidelity, or visual review.

## Immutable inputs

- Frozen promotion gate SHA256:
  `2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`.
- ImageNet-256 manifest: `405,484,553` bytes, SHA256
  `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`.
- Physical 50K validation tree SHA256:
  `19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f`.
- CoFiTok frozen sample-set SHA256:
  `de17e26507c57f6ab473aa20fe5749e19dcd22941391dd7709b3d700297731da`.
- Dense frozen sample-set SHA256:
  `97c21492f4146ea1df889b639687e836dab54e46d5a4aff3c6ec4a59f2394e83`.

The real reference subset selects ten evenly spaced, sorted validation images
per each of the 1,000 sorted WNID directories. Its pass-major ordering matches
the formal `balanced_modulo` generated class schedule. The report also rehashes
the complete 50K real tree and both complete 10K generated sets.

## Implementation and execution identity

The final code candidate is:

- branch `scale/generation-stability-frozen-sample-support-audit-v1`;
- revision `dcf003afa2c6c3c390684af8419ae4ad557aafaa`;
- tree `3ba846fc407f2b093115ad2ca454fdf0344f8cf0`.

The first run at parent revision `65fa30a` failed before reading images because
the shell entrypoint used `PYTHONPATH=src` while importing a sibling `scripts`
module. It wrote no report and used no GPU. Revision `dcf003a` changes the
entrypoint to `PYTHONPATH=.:src` and adds a subprocess regression test that
executes the script exactly as the runbook does.

Validation evidence:

- parent exact full repository suite: `1140 passed, 6 skipped`;
- final revision local targeted suite: `10 passed`;
- canonical parent-layout tests: `4 passed`;
- final Linux targeted suite: `10 passed`;
- Linux runbook `bash -n`: pass;
- direct Linux entrypoint `--help`: pass;
- runtime with `CUDA_VISIBLE_DEVICES=""`: CUDA unavailable;
- final full physical replay: exact report equality.

The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`, tracked clean. Its full porcelain byte-stream SHA256
was unchanged at
`774cfad9f88f6e4fa953556e466b02b16212184930ca9fcbf4e8c233ab1cb79c`.
FieldScope PID `433140` remained untouched at approximately `15,412 MiB` GPU
memory. The audit itself hid CUDA and ran under `nice -n 15 ionice -c3`.

## Authoritative result

The physical report is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/frozen_existing_sample_support_audit/support_audit.json
bytes: 43144
sha256: 6b3acfa62780b951b0adb45f817335b8086bafbed9dc7dcb90ba0afa07d0d9eb
```

The report records `device.type=cpu`, `CUDA_VISIBLE_DEVICES=""`, exact clean
audit Git identity, and all non-authorizing flags. A second full physical replay
rehashes and decodes the same image sets, recreates the complete report, and
returns `FULL_PHYSICAL_REPLAY_EXACT`; the report SHA remains unchanged.

## Findings

### Not duplicate collapse

All three matched 10K sets have zero decoded-pixel duplicates and zero dHash
duplicates. The formal recall collapse is therefore not explained by repeated
identical or trivially perceptually identical PNGs.

### Shared excessive local variation

Mean absolute neighboring-pixel gradients are:

| set | gradient | ratio to real |
|---|---:|---:|
| real matched subset | 0.04065 | 1.000 |
| CoFiTok | 0.09476 | 2.331 |
| dense identity | 0.12138 | 2.986 |

Dense is more affected than CoFiTok. Hard endpoint saturation does not explain
the contrast: the real subset has a larger exact `0/255` pixel fraction than
CoFiTok, while the generated sets still have much larger local gradients.

### Weak coarse class support

For 8x8 RGB descriptors, same-class distance should be smaller than a matched
adjacent-class reference. The same/adjacent ratio is `0.9577` for real images,
but `0.9945` for CoFiTok and `0.9911` for dense. Between-class variance fractions
are `0.1890` real, `0.1038` CoFiTok, and `0.1121` dense. Generated class-support
fractions are only `54.9%` and `59.3%` of the real reference respectively.

This is descriptive evidence of weak coarse class-conditioned structure. It is
not a replacement for the frozen classifier-fidelity evaluator and has no
promotion threshold.

### Reduced low-resolution support

8x8 RGB entropy effective ranks are `13.98` real, `11.10` CoFiTok, and `7.84`
dense. Components needed for 90% variance are `20`, `13`, and `8`. CoFiTok
retains more coarse support than dense, consistent with its better FID, but
neither set recovers real support.

### Same seed dominates the matched pair

CoFiTok and dense images at the same global index use the same class ID and
initial-noise seed. Their paired 8x8 RMS distance averages `0.0670`, only `30.1%`
of CoFiTok's within-class cross-seed distance and `24.5%` of dense's. Paired
dHash Hamming distance averages `3.61`, whereas within-set same-class distances
are approximately `31.9`.

This strongly shows a shared seed-conditioned visual trajectory across the two
matched systems. It does not, by itself, prove whether the cause is guidance,
the DDIM policy, or shared undertraining.

## Sampling versus training decision

The evidence rejects simple duplicate collapse and does not support a
CoFiTok-specific factorization failure: the severe signatures are shared with
the matched dense baseline, and CoFiTok is generally less affected. Frozen
single-policy samples cannot causally separate sampler/CFG effects from 50K
undertraining.

The next discriminating action remains the prepared, non-authorizing matched
512-sample sampling-recovery diagnostic at the exact approved code candidate.
It changes only shared sampling settings and therefore tests the cheaper
hypothesis before committing to a 100K training bridge. Only if a non-baseline
case improves both methods should an independently authorized 10K confirmation
run. If the sweep does not recover both methods, the evidence shifts toward
training scale/data/recipe rather than inference configuration.

No sampling approval sentinel was created. No recovery, confirmation, 100K, or
300K process was launched. FieldScope is still active, and any future GPU action
requires exact user authorization after GPU availability is independently
rechecked.
