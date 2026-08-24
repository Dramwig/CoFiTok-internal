# Existing-Checkpoint Exposure Semantic Trajectory

## Purpose

The 1K class-ranking and semantic residual-alignment objectives both failed for
CoFiTok and dense identity. The residual-alignment candidate also regressed the
held-out correct-label epsilon MSE for both methods. This diagnostic therefore
does not add another semantic auxiliary loss. It asks a narrower question:
whether semantic alignment changes as ordinary matched training exposure grows.

The diagnostic reuses the completed no-semantic-auxiliary 5K pair at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher
```

The source training revision is
`59db142fc45d69dc92bb0333be5ac2d0162d9dc4`. Both methods used effective batch
64, completed 5,000 optimizer steps and 320,000 images, and share dataset and
runtime identities. The preparation physically rehashes the selected 1,250,
2,500, and 5,000 checkpoint payloads and validates their integrity sidecars,
training reports, complete metrics JSONL, run manifests, and final latest
pointers.

## Evaluation contract

All six checkpoint evaluations use the same 16 held-out validation images,
labels, wrong-label offset, timesteps, and per-image noise:

```text
weights=ema
labels=128..143
wrong_label_offset=250
timesteps=100,500,700,900
noise_seed=314159 + sample_index
device=cpu
```

Timestep 100 is descriptive. Timesteps 500, 700, and 900 are averaged within
each image before inference. Exact one-sided sign tests therefore use 16 images,
not 48 correlated image-timestep rows.

For each method, the report separately tests:

- reduction in correct-label epsilon MSE from step 1,250 to 5,000;
- positive correct-label advantage over wrong and null labels at step 5,000;
- improvement in those two semantic advantages from step 1,250 to 5,000.

The canonical decision is one of:

```text
shared_exposure_semantic_recovery
denoising_improves_but_semantic_alignment_does_not
method_asymmetry
no_exposure_recovery
```

## Authorization boundary

The execution is permanently CPU-only and non-authorizing. It loads existing
checkpoint payloads but launches no training, diffusion sampling, GPU work,
promotion, full-scale run, export, release, or process signal. Its result does
not replace the 100K terminal quality gate and always records:

```text
generation_advantage_proven=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
```
