# Streamed Generated Quality Protocol, HF 50k/50k

Date: 2026-07-08

Purpose: extend the generated-sample quality evidence from the 8192/8192
streamed DDIM50 protocol to an ImageNet-64-family 50000 generated vs 50000 real
validation protocol. This targets the strict generation-quality gap without
storing generated images in the repository or checkpoint tree.

Runbook:

```bash
bash artifacts/runbooks/generated_quality_stream_hf_50000_2026-07-08.sh
```

Default parameters:

```text
dataset: imagenet_1k_64x64_hf
variants: epsilon_only, light_denoise_path
train steps: 20000
sample_count: 50000
real_count: 50000
sample_steps: 50
batch_size: 64
prefix_budget: 8
metric: streamed low-res Frechet proxy plus torchvision Inception-Frechet
```

Expected remote output directories:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_imagenet_hf_epsilononly_20k_seed2_50000_ddim50_2026-07-08
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_imagenet_hf_k8_light_20k_seed2_50000_ddim50_2026-07-08
```

Status: completed on `pro6000`.

Results:

| dataset | variant | train steps | generated | real | low-res Frechet proxy | Inception Frechet | elapsed seconds | report dir |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ImageNet-64 HF | epsilon-only | 20000 | 50000 | 50000 | 6.4496 | 238.0329 | 161.68 | `generated_quality_stream_imagenet_hf_epsilononly_20k_seed2_50000_ddim50_2026-07-08` |
| ImageNet-64 HF | K8 light | 20000 | 50000 | 50000 | 6.6096 | 266.8045 | 161.39 | `generated_quality_stream_imagenet_hf_k8_light_20k_seed2_50000_ddim50_2026-07-08` |

Interpretation rule: even if both rows complete, this is 50k-scale streamed
Inception-Frechet evidence, not by itself a robust unconditional
generation-quality win across datasets. The strict completion gap remains open
unless CoFiTok also establishes a stable quality advantage under a formal
benchmark protocol.

Observed interpretation: the 50k-scale HF protocol still favors epsilon-only by
both low-res proxy and Inception-Frechet. This strengthens the negative
generation-quality conclusion while improving the scale of the evidence.
