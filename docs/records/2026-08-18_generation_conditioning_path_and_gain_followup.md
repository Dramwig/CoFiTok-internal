# Conditioning-path activation and inference-gain follow-up

Date: 2026-08-18

## Purpose and boundary

This follow-up separates two possible causes of the frozen ImageNet-256 10%
pair's weak requested-class fidelity:

1. the class signal is lost or disconnected inside the shared U-Net predictor;
2. the class signal remains active, but training has not aligned it with the
   correct denoising target.

Both diagnostics are permanently CPU-only and non-authorizing. They did not
generate diffusion samples, alter checkpoint bytes, move the formal checkout,
or signal any trainer, controller, monitor, or waiter. Class and time
conditioning remain confined to `T_k`; neither diagnostic changes or
conditions the restricted `S_k` synthesis operator.

## Reproducible implementation lineage

The work used the existing isolated branch and checkout:

```text
branch: analysis/generation-conditioning-sensitivity-diagnostic-v1
remote checkout: /tmp/cofitok-conditioning-sensitivity-a5c6478
path-activation commit: c7c880594ec2988be60721310ac46aa25a2e9374
gain-sweep evaluator commit: 6a2b00edfb2e9f2f104afaea821597d929d5537f
matched-comparison commit: 3db734143bcbfb8e846c7c23decd8e3c86f1d4fd
matched-comparison tree: 1a9388b42d0c9ce8b65c044ac07b96dbffb6dab9
```

Entry points:

```text
scripts/audit_generation_conditioning_path.py
scripts/evaluate_generation_conditioning_gain_sweep.py
scripts/build_generation_conditioning_gain_sweep_comparison.py
```

The gain evaluator verifies the adjacent checkpoint-integrity sidecar before
deserialization, binds the selected validation images and label map, restores
the null embedding exactly after every gain, and atomically publishes a report.
The comparison builder reopens and recomputes both source summaries, requires
the same evaluator Git, images, labels, noise, timestep, weights, and gain grid,
and emits a permanently non-authorizing matched interpretation.

Execution controls were:

```text
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
nice -n 19
ionice -c 3
```

The only GPU compute process observed during execution was the active formal
`dense_identity` quality-bridge trainer.

## Frozen sources and matched protocol

Both diagnostics use the same frozen matched 50K EMA checkpoints:

| method | checkpoint SHA256 |
|---|---|
| CoFiTok K8 | `ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a` |
| dense identity | `325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716` |

Shared examples are the first deterministic validation image for labels 0--7.
The wrong label is `(correct + 500) mod 1000`, the null label is 1000, and
noise seed is `102030 + sample_index`.

## Conditioning-path activation

The path audit uses timesteps 100, 500, and 900 and hooks all 18 conditioned
ResBlocks: eight encoder blocks, two middle blocks, and eight decoder blocks.
It records the class/time embedding norm ratio and the correct/wrong/null
modulation difference at every block.

Mean class-embedding norm relative to the time embedding:

| method | t=100 | t=500 | t=900 |
|---|---:|---:|---:|
| CoFiTok K8 | 15.16% | 20.49% | 17.98% |
| dense identity | 13.21% | 18.13% | 17.79% |

Mean correct-vs-null modulation difference across all conditioned blocks:

| method | t=100 | t=500 | t=900 |
|---|---:|---:|---:|
| CoFiTok K8 | 8.70% | 10.77% | 11.41% |
| dense identity | 9.73% | 12.35% | 15.48% |

Mean correct-vs-wrong modulation difference is also nonzero throughout:
`10.60% / 13.71% / 14.02%` for CoFiTok and
`12.28% / 15.97% / 18.40%` for dense at the three timesteps.

There is no encoder, middle, or decoder block at which the class signal
abruptly disappears. Nevertheless, the final epsilon correct-vs-null
difference is only about `0.24%--1.12%` for CoFiTok and `0.24%--1.07%` for
dense, and the earlier matched sensitivity audit did not show a consistent
correct-label denoising advantage. The failure is therefore not an input
wiring defect or a CoFiTok `S_k` bottleneck.

## Frozen inference-only gain sweep

For each checkpoint, the learned class-embedding displacement from the
classifier-free null embedding was multiplied by gains
`0, 0.5, 1, 2, 4`. Images, forward noise, `t=500`, EMA weights, and every
other parameter were held fixed. Gain zero is an exact control: correct,
wrong, and null class embeddings become identical and all paired output
deltas are exactly zero.

Correct-label wins out of eight examples:

| method | gain | vs wrong | vs null | one-sided p vs wrong | one-sided p vs null |
|---|---:|---:|---:|---:|---:|
| CoFiTok K8 | 0.5 | 5/8 | 4/8 | 0.3633 | 0.6367 |
| CoFiTok K8 | 1 | 5/8 | 4/8 | 0.3633 | 0.6367 |
| CoFiTok K8 | 2 | 6/8 | 0/8 | 0.1445 | 1.0000 |
| CoFiTok K8 | 4 | 7/8 | 0/8 | 0.0352 | 1.0000 |
| dense identity | 0.5 | 2/8 | 2/8 | 0.9648 | 0.9648 |
| dense identity | 1 | 5/8 | 1/8 | 0.3633 | 0.9961 |
| dense identity | 2 | 5/8 | 0/8 | 0.3633 | 1.0000 |
| dense identity | 4 | 5/8 | 0/8 | 0.3633 | 1.0000 |

At gain 4, correct-vs-wrong output difference grows from `0.00658` to
`0.21209` relative RMS for CoFiTok and from `0.00584` to `0.48423` for dense.
Thus the conditioning path is responsive to scale. However, correct-vs-null
denoising becomes worse for every example in both methods. Mean relative MSE
advantage versus null falls to `-7.5356` for CoFiTok and `-29.1271` for dense.
Increasing inference gain therefore amplifies a learned class-dependent
direction without making that direction the correct semantic denoising
direction.

The source-bound comparison reports:

```text
shared_inference_gain_recovery_supported: false
gain_only_amplifies_without_semantic_recovery: true
recommended_next_action:
  develop_matched_training_time_label_ranking_or_contrastive_denoising_loss
```

## Evidence identities

Remote authoritative roots:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/conditioning_path_activation_c7c8805/
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/conditioning_gain_sweep_6a2b00e/
```

Local evidence copies:

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/conditioning_path_activation_2026-08-18/
artifacts/reports/generation/stability_scaling_50k_ema_teacher/conditioning_gain_sweep_2026-08-18/
```

| artifact | bytes | SHA256 |
|---|---:|---|
| path activation, CoFiTok | 236,073 | `168ded6fada60f4c84717b5f94fa516354b42f4665b4644ef7306527b7e41166` |
| path activation, dense | 236,110 | `3511d5beb43e313517cc17c8eb47f65a1688a3f7d7e538a90302843c49231655` |
| gain sweep, CoFiTok | 80,582 | `1f1f72443901dcca68f802412907cb646e4261d49bdef86eefa48f062438826a` |
| gain sweep, dense | 80,519 | `ee4027272b40644b1a7bb71db8c8b4fcc0c0e6cea479e73e06781060455bac9f` |
| matched gain comparison | 13,080 | `15537338114718e42a8b74acca0768dbbca42385cf0e6526677e460d69191294` |

Incremental deployment bundles:

```text
6a2b00e bundle: 4,982 bytes
SHA256: 2d1ff88dcff7440ba692e83ff5581e9408fe0bc53007a8980418808eeafe3e54

3db7341 bundle: 4,907 bytes
SHA256: 13afb3425dd634de2c8756a0920ac05c3df9acaca358933b5d61a1bc9331e4a4
```

## Validation and scientific decision

- Local targeted suite at the comparison commit: `12 passed`.
- Linux targeted suite at the comparison commit: `12 passed`.
- Both gain reports passed exact `--resume` revalidation against their
  checkpoint, request, dataset, and Git bindings.
- The matched comparison passed exact `--resume` revalidation.
- All five synchronized local files match their remote SHA256 exactly.
- The remote isolated checkout is at `3db7341`, tracked clean; the formal
  checkout was not moved.

The next recipe change should be a shared, matched training-time semantic
alignment objective on the final epsilon prediction, such as a stop-gradient
label-ranking or contrastive denoising loss. It must be enabled identically for
CoFiTok and `dense_identity`, must not inject class information into `S_k`, and
must first pass CPU unit tests and a separately authorized short matched GPU
probe. Merely increasing inference-time class gain or continuing CFG/rescale
sweeps is not supported by the evidence.
