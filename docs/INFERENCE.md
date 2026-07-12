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
resolved sampling protocol.

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

## Formal sampling

`generate_samples.py` uses the same `GenerationSession`; its immutable sampling
manifest identifies API version 1. Formal metrics and the completion audit reject
50K evidence that bypasses this API. Batch sampling additionally provides atomic
progress, exact resume, numbered PNG validation, and sample-set SHA256.
