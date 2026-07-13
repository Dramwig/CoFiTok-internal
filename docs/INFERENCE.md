# CoFiTok generation inference

## Stable API

All checkpoint inference uses `cofitok.generation.GenerationSession`. A session
verifies checkpoint integrity before deserialization, applies EMA or model
weights once, constructs the diffusion schedule once, and can serve repeated
immutable `GenerationRequest` values without reloading the checkpoint.

Each request explicitly contains:

- per-image seeds and class IDs;
- DDIM steps, eta, precision, and clipping policy;
- CoFiTok prefix budget;
- CFG scale/rescale and batched/sequential execution mode.

The session validates class and token ranges before sampling, uses one generator
per seed, runs under inference mode/autocast, rejects non-finite or malformed
outputs, returns CPU tensors, and attaches checkpoint SHA/step/integrity plus the
resolved sampling protocol. EMA deployment artifacts additionally propagate the
source training checkpoint SHA, runtime-environment SHA, and Git identity through
session metadata and CLI reports.

## CLI

Formal experiments remain server-only. Once a completed checkpoint exists:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
export PYTHONPATH=src

python scripts/infer_generation.py \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_full_cofitok_k8_300k/checkpoint_step_00300000.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/inference/example \
  --class-ids 207 \
  --seeds 11,12,13,14 \
  --prefix-budgets 1,2,4,8 \
  --batch-size 4 --sample-steps 250 \
  --guidance-scale 1.5 --weights ema --precision bf16
```

One class ID is broadcast to every seed; otherwise provide one class ID per
seed. Filenames encode seed, class, and prefix. PNG publication is atomic and
`inference_report.json` records the checkpoint identity, request, elapsed time,
and SHA256 for every output. Existing reports/images are not overwritten unless
`--overwrite` is explicit.

## EMA-only deployment artifact

After final completion, prefer the smaller verified artifact for routine
inference:

```bash
python scripts/infer_generation.py \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/generation/exports/imagenet256_full_300k/cofitok_k8_ema_inference.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/inference/deployed \
  --class-ids 207 --seeds 101,102 --prefix-budgets 8 \
  --sample-steps 250 --guidance-scale 1.5 --weights ema --precision bf16
```

The artifact contains EMA-applied weights only. Keep `--weights ema`; requesting
`model` is rejected. The original full training checkpoint remains mandatory for
exact resume and scientific provenance. Artifact schema v2 is self-describing:
its payload and sidecar both bind the source training revision and environment.

## Formal sampling

`generate_samples.py` uses the same `GenerationSession`; its immutable sampling
manifest identifies API version 1 and protocol `cofitok_ddim_sampling_v1`.
Sampling manifest schema v3 and completed report schema v6 also bind the exact
sampler, train/sample timestep counts and resolved timestep list, CFG scale and
rescale, CFG batching, eta, `x0` clipping, precision, random-stream policy, and
the actual Python/PyTorch/CUDA/GPU/project-lock
environment. Exact resume rejects any environment drift. Formal metrics, the
matched-batch selector, quality gates, and the completion audit recompute and
cross-check that fingerprint, and reject 50K evidence that bypasses this API or
uses different CoFiTok/dense environments. Batch sampling additionally provides
atomic progress, exact resume, numbered PNG validation, and sample-set SHA256.

The scaling protocol is EMA DDIM-100; the final full-data protocol is EMA
DDIM-250. Both use CFG 1.5, zero guidance rescale, batched CFG, `eta=0`,
`clip_x0=true`, bf16, seed zero, balanced-modulo classes, and per-index random
streams invariant to batch size and resume boundaries. These are formal
evidence requirements, not only CLI defaults.
