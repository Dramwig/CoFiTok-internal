# 2026-07-07 Dataset Inventory and Loader Smoke

Purpose: verify the current server-side CoFiTok datasets and make the first
real-data dataloaders usable before any training run.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok
```

## Dataset Inventory

Available under `/root/autodl-tmp/CoFiTok/datasets`:

- `cifar10`
  - Raw archive: `raw/cifar-10-python.tar.gz`
  - Extracted data: `extracted/cifar-10-batches-py`
  - Train/test: 50,000 / 10,000 images
  - Classes: 10
  - Resolution: 32x32 RGB
  - SHA256: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- `tiny_imagenet_200`
  - Raw archive: `raw/tiny-imagenet-200.zip`
  - Extracted data: `extracted/tiny-imagenet-200`
  - Train/val/test: 100,000 / 10,000 / 10,000 images
  - Classes: 200
  - Train class dirs: 200
  - Train images per class: 500 min / 500 max
  - Resolution: 64x64 RGB
  - SHA256: `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`

Not downloaded:

- `downsampled_imagenet_64`, intentionally deferred until CIFAR-10 and Tiny
  ImageNet MVP runs are working.

Symlink check:

```text
find /root/autodl-tmp/CoFiTok -path /root/autodl-tmp/CoFiTok/checkpoints/hf_cache -prune -o -type l -print
```

Result: no output.

## Loader Fix

Real-data loader verification found two issues in the scaffold:

- CIFAR-10 data was complete, but the loader pointed torchvision at `raw/`
  instead of `extracted/`.
- Tiny ImageNet validation images are stored under `val/images` with labels in
  `val_annotations.txt`; `ImageFolder` treated validation as one class.

Fix:

- Point CIFAR-10 to `/root/autodl-tmp/CoFiTok/datasets/cifar10/extracted`.
- Add `TinyImageNetDataset` that reads `wnids.txt` and `val_annotations.txt`.
- Add `configs/smoke_cifar10_cuda.json`.
- Add `configs/smoke_tiny_imagenet_cuda.json`.
- Add data registry tests.

## Verification

Remote tests:

```text
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python -m pytest -q
7 passed
```

Real dataset loader check:

```text
cifar10 train len=50000 shape=(3, 32, 32)
cifar10 val len=10000 shape=(3, 32, 32)
tiny_imagenet_200 train len=100000 shape=(3, 64, 64)
tiny_imagenet_200 val len=10000 shape=(3, 64, 64)
```

CUDA smoke reports:

```text
/root/autodl-tmp/CoFiTok/checkpoints/smoke/cifar10_cuda_smoke.json
/root/autodl-tmp/CoFiTok/checkpoints/smoke/tiny_imagenet_cuda_smoke.json
```

Local report copies:

```text
artifacts/reports/smoke/cifar10_cuda_smoke.json
artifacts/reports/smoke/tiny_imagenet_cuda_smoke.json
```

Key results:

```text
cifar10_cuda_smoke:
  actual_device: cuda
  epsilon_shape: [16, 3, 32, 32]
  zero_token_component_energy: 0.0

tiny_imagenet_cuda_smoke:
  actual_device: cuda
  epsilon_shape: [8, 3, 64, 64]
  zero_token_component_energy: 0.0
```

## Decision

Datasets P0 and P1 are complete enough for the first MVP loop. Do not download
`downsampled_imagenet_64` yet.

Next step: implement a short CIFAR-10 training entrypoint with checkpoint and
prefix visualization outputs, then run a small `K=4` smoke training job before
moving to Tiny ImageNet.
