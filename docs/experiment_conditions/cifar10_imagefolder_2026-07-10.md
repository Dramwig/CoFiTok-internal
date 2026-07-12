# CIFAR10 ImageFolder Derivative

Date: 2026-07-10

Alias:

```text
cifar10_imagefolder
```

Source:

```text
/root/autodl-tmp/CoFiTok/datasets/cifar10/extracted/cifar-10-batches-py
```

Derived directory:

```text
/root/autodl-tmp/CoFiTok/datasets/cifar10/derived/imagefolder
```

Reason: external pixel-diffusion baselines read image-folder style datasets,
while the canonical CIFAR10 source in this project is the original Python batch
format. The derivative preserves the official train/test split and writes RGB
PNG files for loader compatibility.

Expected counts:

```text
train=50000
test=10000
image_size=32x32
channels=3
```

Use only as input to external baseline adapters. CoFiTok configs should keep
using the canonical `cifar10` dataset alias.
