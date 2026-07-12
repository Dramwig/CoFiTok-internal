# Streamed Generated Quality Protocol, 8192/8192

Date: 2026-07-08

Purpose: extend generated-sample evidence from the prior 2048 generated vs
8192 real DDIM50 protocol to 8192 generated vs 8192 real images without storing
full generated sample directories.

## Command

Remote runbook:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
bash artifacts/runbooks/generated_quality_stream_8192_2026-07-08.sh
```

The runbook calls:

```bash
PYTHONPATH=src python scripts/evaluate_generated_samples_stream.py \
  --checkpoint <main 20k checkpoint> \
  --config <matching config> \
  --output-dir <checkpoint_root/generated_quality_stream_*_8192_ddim50_2026-07-08> \
  --sample-count 8192 \
  --real-count 8192 \
  --sample-steps 50 \
  --batch-size 64 \
  --prefix-budget 8 \
  --enable-inception-fid
```

The streamed evaluator accumulates generated low-res and Inception features in
batches and writes only `generated_quality_report.json`. If `TORCH_HOME` is not
set and the checkpoint lives under the project `checkpoints` directory, the
script uses `/root/autodl-tmp/CoFiTok/checkpoints/torch_cache` for torchvision
weights.

## Results

| dataset | variant | train steps | generated | real | low-res Frechet proxy | Inception Frechet |
|---|---|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only | 20k | 8192 | 8192 | 7.5761 | 272.9955 |
| Tiny ImageNet-200 | K8 light | 20k | 8192 | 8192 | 7.3264 | 271.0260 |
| ImageNet-64 HF | epsilon-only | 20k | 8192 | 8192 | 6.5490 | 240.8919 |
| ImageNet-64 HF | K8 light | 20k | 8192 | 8192 | 6.6993 | 269.2785 |

Generated report directories:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_tiny_epsilononly_20k_seed2_8192_ddim50_2026-07-08
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_tiny_k8_light_20k_seed2_8192_ddim50_2026-07-08
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_imagenet_hf_epsilononly_20k_seed2_8192_ddim50_2026-07-08
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_imagenet_hf_k8_light_20k_seed2_8192_ddim50_2026-07-08
```

## Interpretation

This is stronger generated-sample evidence than the 2048/8192 smoke protocol,
but it remains a small-model DDIM/Inception-Frechet protocol rather than an
official 50k FID evaluation. The result is mixed:

- Tiny ImageNet-200 slightly favors K8 light over epsilon-only.
- ImageNet-64 HF still favors epsilon-only, especially on Inception Frechet.

The stable paper claim should remain prefix-controllable denoising at comparable
endpoint quality, not a robust unconditional generated-sample quality win.
