# Quality Metrics: Torchvision Inception Frechet

Date: 2026-07-08

Purpose:

Add reproducible FID-style and LPIPS distribution/perceptual metrics for CoFiTok
prefix outputs.

Environment:

```text
server: pro6000
conda env: pf-vlm
torch: 2.7.1+cu128
torchvision: 0.22.1+cu128
cuda available: true
lpips: 0.1.4, installed with pip --no-deps from https://pypi.org/simple
scipy: 1.15.3, installed with pip --no-deps from https://pypi.org/simple for lpips import
torchmetrics: unavailable
torch_fidelity: unavailable
```

Project dependency declaration:

```text
pyproject.toml optional extra:
  cofitok[quality]
entries:
  lpips==0.1.4
  scipy>=1.15
```

Implementation:

```text
script:
  CoFiTok-internal/scripts/evaluate_quality.py
flag:
  --enable-inception-fid
feature extractor:
  torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
feature:
  final fc replaced with Identity, yielding 2048-dim pool features
input preprocessing:
  CoFiTok image range [-1, 1] -> [0, 1]
  bilinear resize to 299x299
  ImageNet mean/std normalization
distance:
  Frechet distance between clean-image and prefix-output feature distributions
```

Weight cache:

```text
TORCH_HOME:
  /root/autodl-tmp/CoFiTok/checkpoints/torch_cache
weight:
  /root/autodl-tmp/CoFiTok/checkpoints/torch_cache/hub/checkpoints/inception_v3_google-0cc3c7bd.pth
size:
  104M
sha256:
  0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef
symlink check:
  find /root/autodl-tmp/CoFiTok/checkpoints/torch_cache -type l -print
  no output
```

Scope:

The current reports use the same 256 validation images, fixed timestep 500, and
default prefix budgets. This is a small validation-slice FID-style metric, not a
full-dataset publication FID. It is stronger than the low-res proxy but should
still be labelled by sample count and feature extractor.

Reports:

```text
quality_cifar10_epsilononly_256_t500_inception_2026-07-08
quality_cifar10_k8_light_256_t500_inception_2026-07-08
quality_tiny_epsilononly_256_t500_inception_2026-07-08
quality_tiny_k8_light_256_t500_inception_2026-07-08
quality_imagenet_hf_epsilononly_256_t500_inception_2026-07-08
quality_imagenet_hf_k8_light_256_t500_inception_2026-07-08
```

LPIPS setup:

```text
package:
  lpips==0.1.4
package location:
  /root/autodl-tmp/conda/envs/pf-vlm/lib/python3.10/site-packages/lpips
model:
  lpips.LPIPS(net='alex')
lpips linear weights:
  /root/autodl-tmp/conda/envs/pf-vlm/lib/python3.10/site-packages/lpips/weights/v0.1/alex.pth
torchvision AlexNet trunk weight:
  /root/autodl-tmp/CoFiTok/checkpoints/torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth
size:
  234M
sha256:
  7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
```

LPIPS+Inception reports:

```text
quality_cifar10_epsilononly_256_t500_lpips_inception_2026-07-08
quality_cifar10_k8_light_256_t500_lpips_inception_2026-07-08
quality_tiny_epsilononly_256_t500_lpips_inception_2026-07-08
quality_tiny_k8_light_256_t500_lpips_inception_2026-07-08
quality_imagenet_hf_epsilononly_256_t500_lpips_inception_2026-07-08
quality_imagenet_hf_k8_light_256_t500_lpips_inception_2026-07-08
quality_tiny_epsilononly_20k_256_t500_lpips_inception_2026-07-08
quality_imagenet_hf_epsilononly_20k_256_t500_lpips_inception_2026-07-08
```

Generated sample quality smoke:

```text
script:
  CoFiTok-internal/scripts/evaluate_generated_samples.py
sample source:
  sample_checkpoint.py --save-images
protocol:
  64 generated samples vs 256 real validation images for first smoke
  256 generated samples vs 1024 real validation images for 20k Tiny/HF follow-up
metrics:
  low-res Frechet proxy
  torchvision Inception Frechet
reports:
  generated_quality_cifar10_k8_light_64_ddim20_2026-07-08
  generated_quality_tiny_imagenet_k8_light_64_ddim20_2026-07-08
  generated_quality_imagenet_hf_k8_light_64_ddim20_2026-07-08
  generated_quality_tiny_imagenet_k8_light_20k_64_ddim20_2026-07-08
  generated_quality_imagenet_hf_k8_light_20k_64_ddim20_2026-07-08
  generated_quality_tiny_imagenet_k8_light_20k_256_ddim20_2026-07-08
  generated_quality_imagenet_hf_k8_light_20k_256_ddim20_2026-07-08
  generated_quality_tiny_epsilononly_20k_256_ddim20_2026-07-08
  generated_quality_imagenet_hf_epsilononly_20k_256_ddim20_2026-07-08
scope:
  smoke metric only, not publication-scale sample FID
```
