# Generation stability true coarse-energy gate

Date: 2026-07-31

## Problem

The existing formal generation gate defined coarse utilization as the sum of
the first `K-2` component-energy ratios. That definition is correct for the
immutable v3 layout, whose final two tokens are the full-resolution
rank-complete tail. It is incorrect for the stability `rgbtail3` layout, whose
final three tokens are full-resolution one-channel tails.

Applying the old rule to `rgbtail3` would count token 6 as coarse and could pass
the 5% utilization threshold even when energy remained concentrated entirely in
the full-resolution tail. This directly weakens the diagnostic intended to
detect tail-token energy concentration.

The old source-report contract also hard-coded the v3 training and evaluation
directories. A stability post-evaluation could therefore either fail path
validation or, worse, be redirected toward old evidence.

## Implementation

Commit `2c2c1f5166b73d4f28df93b276901671ac1a7836` adds:

- A self-describing coarse/tail partition in generation-gate evidence.
- `token_spatial_stride_suffix_v1`, which requires a positive-stride prefix
  followed by a contiguous stride-1 suffix.
- For `rgbtail3`, coarse tokens are exactly 1-5 and the full-resolution tail is
  exactly tokens 6-8.
- Backward-compatible `legacy_last_two_tokens` validation for immutable v3 gate
  reports that predate stride-partition evidence.
- A fail-closed stability policy: missing, malformed, non-integral, or
  non-suffix stride metadata cannot fall back to `K-2`.
- A distinct `stability_scaling` source profile bound to the new 50K run and
  evaluation directories.
- Explicit source-profile/scientific-stage compatibility: stability and v3
  scaling profiles may feed only a scaling gate; full sources may feed only a
  full gate.
- Dormant post-evaluation runbook
  `artifacts/runbooks/generation_stability_ema_teacher_50k_posteval_after_training.sh`.

The post-evaluation runbook requires the final 5K decision SHA, exact clean
target revision/branch, a rebuilt decision validation, recipe-v4 config
validation, and a rebuilt completed 50K pair summary before any CUDA work. It
then performs:

```text
checkpoint mechanism evaluation: 1,024 images/method at t=500
distribution sampling:            10,000 EMA images/method
sampler:                          deterministic DDIM-100
CFG:                              1.5, batched
precision:                        bf16
prefix visualization:             64 paired samples at 1/2/4/8
gate coarse partition:             stride-derived tokens 1-5 vs tail 6-8
```

The gate keeps the established scaling thresholds: FID at most 100, no more
than 5% FID or endpoint-MSE regression against matched dense, ordered rank 1,
at least 5% true coarse-token energy, exact zero-token synthesis, and shuffled
token mismatch. A scientific hold is preserved as a valid result. A pass is
fully revalidated, but the runbook contains no training entry point and cannot
start full 300K.

## Verification

Local verification:

```text
targeted gate/runbook tests: pass
complete pytest:             pass, 3 existing skips
git diff --check:            pass
```

Incremental bundle:

```text
bytes:  15,953
sha256: 2c702accc08ade80bbe1be1e0ea9441ddad2d6a2f2f8c06796ecfac0dfccf9a6
base:   c8e3def25d179ffe325b329a2d9685302be62f1f
head:   2c2c1f5166b73d4f28df93b276901671ac1a7836
```

Remote `git bundle verify` passed. The detached Linux rehearsal at
`/tmp/cofitok-stability-gate-2c2c1f5` passed 62 targeted tests and all 90
tracked runbooks passed `bash -n`; tracked status remained clean. The official
repository stayed at `1ebcc15210e63a776a2ba448481cbd8bb94a4066`, and the
active 5K checkout stayed at
`59db142fc45d69dc92bb0333be5ac2d0162d9dc4`.

No stability 50K training, post-evaluation, or full 300K process was launched.

## Active 5K milestone

During this change, the active CoFiTok member reached protected step 2,500:

```text
images seen:               160,000
checkpoint bytes:          1,006,321,770
checkpoint SHA256:         bc9e7942f9fac143064a901a78f33ddf7bd638c8df4d2409481e1305a794d6df
total loss:                0.04665259178727865
epsilon loss:              0.0251476657576859
rollout scale:             1.0
EMA-teacher scale:         0.0
validation epsilon MSE:    0.03686438500881195
```

The strict provenance observer verified both protected checkpoints, both
sidecars, the step-2,500 `latest.json`, clean revision, formal dataset identity,
runtime environment, and all 101 logged float32 rollout/teacher schedule rows.
It reported `running`, stage `cofitok_training`, with no issues. Teacher scale
remaining zero is correct because activation begins at step 3,000. The next
scientifically important protected milestone is 3,750, after teacher activation
and partial warmup.

The small milestone evidence was copied to:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/milestones/step_00002500/
```

`SOURCE_MANIFEST.json` binds the copied observer, integrity sidecar,
`latest.json` snapshot, and run-manifest snapshot by byte count and SHA256. The
observer snapshot was taken at training step 2,675 while its latest protected
checkpoint remained step 2,500. The checkpoint payload itself was not copied.

## EMA-teacher activation

The strict provenance observer independently captured the first logged
nonzero teacher event at step 3,025:

```text
images seen:                    193,600
EMA-teacher scale:              0.02500000037252903
expected float32 scale:         25 / 1000
EMA-teacher consistency loss:   0.0055482672760263085
total loss:                     0.04847301635891199
epsilon loss:                   0.02906544366851449
rollout consistency loss:       0.01372776145581156
rollout scale:                  1.0
gradient norm, before clipping: 0.08743015676736832
```

Step 3,000 correctly had zero teacher scale; the next logged interval at step
3,025 therefore establishes activation without an off-by-one schedule error.
The expected float32 value and the logged value agree, all 122 rollout and
teacher schedule rows passed, the run manifest and training revision remained
verified, and the observer reported `running` with no issues. This is an event
snapshot, not a checkpoint milestone: the latest protected checkpoint remained
step 2,500 and no checkpoint payload was copied.

The small evidence is stored at:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/events/
  teacher_activation_step_00003025/
```

Its `SOURCE_MANIFEST.json` binds the observer snapshot by byte count and
SHA256. The next protected scientific comparison remains step 3,750, where the
teacher scale should be 0.75 and validation can be compared with steps 1,250
and 2,500. This event does not authorize stability 50K.

## Protected step 3,750

The first protected checkpoint after teacher activation and partial warmup
passed integrity and provenance verification:

```text
images seen:                    240,000
checkpoint bytes:               1,006,321,770
checkpoint SHA256:              4906500378f77a8ba1af22836e7fd97bf30e89aa6bb908db58fdc120f2219a0c
total loss:                     0.07099776808172464
epsilon loss:                   0.042367526330053806
rollout consistency loss:       0.07733985711820424
rollout scale:                  1.0
EMA-teacher consistency loss:   0.002891819181968458
EMA-teacher scale:              0.75
gradient norm, before clipping: 0.7962989807128906
validation epsilon MSE:         0.022965285927057266
validation images/noise seed:   16 / 102030
```

The observer verified the step-3,750 sidecar and `latest.json` binding, all
three protected checkpoints, clean training revision, dataset and runtime
identities, run manifest, and all 152 logged rollout/teacher schedule rows. It
reported `running` with no issues. Relative to the same validation protocol,
step 3,750 MSE was 30.5261% lower than step 1,250 and 37.7033% lower than step
2,500. This is positive evidence that partial EMA-teacher warmup mitigated the
earlier validation regression; it is not yet a final mechanism or generation
quality result.

The small evidence was copied to:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/milestones/step_00003750/
```

The source manifest binds the observer, sidecar, latest and run-manifest
snapshots, plus a 153-row metrics snapshot that directly contains the step
3,750 validation event. The metrics snapshot ends at step 3,800, while the
observer snapshot ends at step 3,775; both still bind the protected step-3,750
checkpoint. The checkpoint payload itself was not copied. CoFiTok must still
finish step 5,000, the matched dense member must finish, and raw n8/n64
mechanism gates must pass before stability 50K can be authorized.

## CoFiTok member completion

The CoFiTok member completed all 5,000 steps and the locked runbook
automatically advanced to matched dense training:

```text
training complete:               true
images seen:                     320,000
parameter count:                 62,834,083
elapsed seconds:                 14,222.479915857315
peak VRAM bytes:                 15,238,400,000
checkpoint bytes:                1,006,321,770
checkpoint SHA256:               cb432c75ccbc4eba00dab878e43dd0e45740ebdd9a6b95e0cd014ce97b6d6450
total loss:                      0.0521534513682127
epsilon loss:                    0.0316624497063458
rollout consistency loss:        0.013888129265978932
EMA-teacher consistency loss:    0.0019236642983742058
rollout/teacher schedule scales: 1.0 / 1.0
gradient norm, before clipping:  0.06653366982936859
validation epsilon MSE:          0.018617089837789536
```

The final validation MSE was 43.6801% lower than step 1,250, 49.4984% lower
than step 2,500, and 18.9338% lower than step 3,750 under the same 16-image,
fixed-noise protocol. The final report binds clean revision `59db142`, exact
dataset/runtime identities, 320,000 images seen, and the final checkpoint
sidecar. The strict observer verified all four checkpoints and switched to
`dense_identity_training` with no issues; dense had reached step 25 when the
snapshot was taken.

The small completion evidence was copied to:

```text
artifacts/reports/generation/stability_probe_2026-07-29/
  pair5k_rollout_x0_u2_ema_teacher/milestones/step_00005000_cofitok/
```

The source manifest binds six files by bytes and SHA256, including the complete
201-row metrics history and authoritative training report. The checkpoint
payload itself was not copied. CoFiTok completion is necessary but not
sufficient for scaling: the dense member and raw n8/n64 mechanism gates remain
mandatory, and stability 50K is still unauthorized.
