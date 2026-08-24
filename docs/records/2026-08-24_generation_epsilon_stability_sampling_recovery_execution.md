# Matched 100K epsilon-stability 1K sampling-recovery execution

Date: 2026-08-24

## Scope

This execution path is limited to the separately approved, permanently
non-authorizing matched diagnostic:

> Approve the non-authorizing matched 1000-sample sampling-recovery diagnostic only.

It runs the immutable CoFiTok K8 and dense-identity 100K EMA checkpoints through
the same eight sampling cases, with 1,000 balanced-modulo ImageNet-256 samples
per method and case. The cases share one fresh per-index random stream and differ
only in the source-bound epsilon-to-x0 sampling controls in the canonical design.

The execution does not authorize an independent 10K confirmation, training,
300K scaling, checkpoint promotion, export, release, or process signals. Its
result always keeps `generation_advantage_proven=false`.

## Controller and evidence flow

The single serial controller is:

```text
scripts/run_generation_epsilon_stability_sampling_recovery.py
```

The source-bound Linux entrypoint is:

```text
artifacts/runbooks/generation_epsilon_stability_sampling_recovery_v1.sh
```

Before creating the output root, the controller:

1. requires the exact clean revision/tree/branch and the terminal-compatible
   runtime environment;
2. physically replays the preparation and execution authorization, including
   both approximately 1 GB checkpoint payloads, sidecars, latest files,
   training reports, dataset manifest, real-set source, evaluator manifest,
   classifier weights/report, and terminal route sources;
3. rejects any duplicate controller, existing output/lock, or active GPU
   compute process;
4. observes three consecutive idle-GPU polls without signaling any process.

It then runs exactly one child at a time. For each of the 16 method/case arms it
creates the native schema-v6 sampling report, torch-fidelity FID/IS report,
fixed ResNet-50 class-fidelity report, CPU artifact-statistics report, and a
source-replayed case observation. The first 1,000 images of the bound real set
are decoded as RGB and losslessly materialized as numbered PNG files, with both
source-byte identities and decoded-pixel digests recorded for replay. This
supports the authoritative ImageNet validation tree's JPEG storage without
changing any source pixels. The final observation matrix and result are
physically replayed before the controller reports completion.

The canonical remote output is versioned and does not overlap prior evidence:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_epsilon_stability_sampling_recovery_v1
```

Partial output or a controller failure remains fail-closed. No automatic
confirmation or follow-up experiment is launched from the screening result.
