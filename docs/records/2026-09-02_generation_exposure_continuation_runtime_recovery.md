# Exposure continuation runtime recovery (2026-09-02)

## Scope

This record documents the source-compatible recovery of the bounded
ImageNet-256 exposure continuation from the locked matched 100K checkpoints to
110K. It does not change the training objective, architecture, token layout,
dataset, sampler, source checkpoint, execution revision, or scientific gate.
It does not authorize 250M/300K training, promotion, export, release, or a broad
generation-quality claim.

## Bound identities

- source checkout: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree
  `6cef27723196fd363379bca2e7b85b1678ebd777`, branch
  `scale/generation-stability-quality-bridge-100k`;
- execution checkout: `5c23141a24a3385101ed6081c1ed656aad955a99`, tree
  `ffe66d2d2fc666884a21856bbc84a23fe667f88b`, branch
  `analysis/generation-exposure-capacity-source-compatible-20260901`;
- execution authorization SHA256:
  `c4548f6ba12e031850f99da4e3ab1a9ad2ae60700d09533c969527e15caca8b8`;
- exact-stage authorization SHA256:
  `a05e5a3342417cb36af3ea4f002c62e13d02b39b9af38a35158b60daead69194`;
- exposure gate SHA256:
  `171512fa8ca3d18208f5a5dc4be1213661fb45735b38b62ef0a5cd2d065ab16c`;
- preparation SHA256:
  `e9652b12912c399d9e462475f3777e40cb27b16d0221734f11ce11b032aa524c`;
- standing authorization SHA256:
  `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.

The versioned output root is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure_capacity_disambiguation_v1/exposure
```

## Initial failure

The first controller reached `training_cofitok` but failed before the first
optimizer step with:

```text
Checkpoint runtime environment mismatch at: runtime_environment.torch.num_threads
```

The locked 100K checkpoint records PyTorch intra-op and inter-op thread counts
of `104/104`. The active shell inherited `OMP_NUM_THREADS=22` and
`MKL_NUM_THREADS=22`, producing an actual intra-op count of 22. A CPU-only
preflight with `OMP_NUM_THREADS=104`, `MKL_NUM_THREADS=104`, and the training
script's standard `PYTORCH_ALLOC_CONF=expandable_segments:True` reproduced the
source runtime environment exactly:

```text
actual runtime SHA256:   d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
expected runtime SHA256: d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
mismatch paths: []
```

The first failed trainer had created an empty method directory but no
`latest.json`, metrics, checkpoint, report, or other file. Recovery therefore
failed closed instead of implicitly reconstructing state. The exact directory
was rechecked as empty and removed with non-recursive `rmdir`; controller logs
and status evidence were preserved.

## Successful exact recovery

The same authorization and output root were reused with
`EXPOSURE_CONTINUATION_RESUME=true` and the source-compatible thread settings.
The controller PID is `824960`; the CoFiTok trainer PID is `825065`.

The continuation passed exact checkpoint/config/runtime/Git validation and
published strictly increasing finite metrics:

| step | samples seen | epsilon loss | total loss | gradient norm | LR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100001 | 6,400,064 | 0.02161609 | 0.04334975 | 0.58576930 | 1e-5 |
| 100050 | 6,403,200 | 0.02812170 | 0.04969284 | 0.04433239 | 1e-5 |
| 100100 | 6,406,400 | 0.02897207 | 0.04878422 | 0.07754358 | 1e-5 |

At the step-100100 snapshot the RTX PRO 6000 reported 81,321 MiB used and
100% utilization. Its only compute application was trainer PID `825065` with
81,308 MiB. The run manifest binds the source checkpoint payload SHA256
`b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e`,
the source and target configs, the ImageNet-256 dataset identity, the clean
execution Git identity, and `preserve_source_scheduler_horizon`. The restored
scheduler remains at the source 100K terminal floor (`1e-5`) for the bounded
additional exposure.

## Current boundary

The controller is responsible for the serial sequence:

```text
CoFiTok 110K
-> dense identity 110K
-> matched EMA DDIM-100 sampling (10K per method)
-> FID/IS/precision/recall and class fidelity
-> checkpoint mechanism and rollout-stability diagnostics
-> exposure_capacity_result.json
```

Until that result is complete and passes its source-bound qualification, the
terminal scientific state remains `hold`, `generation_advantage_proven=false`,
and no 250M/300K training or release step is authorized.
