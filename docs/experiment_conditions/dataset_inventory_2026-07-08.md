# Dataset Inventory: MVP Required Data

Date: 2026-07-08

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok
```

Dataset root:

```text
/root/autodl-tmp/CoFiTok/datasets
```

## Completed Required Datasets

### P0: cifar10

Status: completed and verified.

Paths:

```text
/root/autodl-tmp/CoFiTok/datasets/cifar10/raw/cifar-10-python.tar.gz
/root/autodl-tmp/CoFiTok/datasets/cifar10/extracted/cifar-10-batches-py
/root/autodl-tmp/CoFiTok/datasets/cifar10/metadata
```

Summary:

- Train: 50,000 images.
- Test: 10,000 images.
- Classes: 10.
- Resolution: 32x32 RGB.
- SHA256: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
- Metadata: `/root/autodl-tmp/CoFiTok/datasets/cifar10/metadata/dataset_summary.json`.

### P1: tiny_imagenet_200

Status: completed and verified.

Paths:

```text
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/raw/tiny-imagenet-200.zip
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/extracted/tiny-imagenet-200
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/metadata
```

Summary:

- Train: 100,000 images.
- Validation: 10,000 images.
- Test: 10,000 images.
- Classes: 200.
- Resolution: 64x64 RGB in sampled verification.
- SHA256: `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`.
- Metadata: `/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/metadata/dataset_summary.json`.

### P2: downsampled_imagenet_64

Status: completed and verified from the strict Academic Torrents source on 2026-07-09.

Paths:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/train_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/valid_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/train_64x64
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/valid_64x64
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/metadata
```

Source:

```text
Academic Torrents details: https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b
Info hash: 96816a530ee002254d29bf7a61c0c158d3dedc3b
```

Summary:

- Train: 1,281,149 PNG images.
- Validation: 49,999 PNG images.
- Resolution: 64x64 PNG in sampled verification.
- Labels: no class-label files were present in the torrent payload; use as unconditional image data unless labels are supplied separately.
- SHA256:
  - `train_64x64.tar`: `e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7`.
  - `valid_64x64.tar`: `eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd`.
- Metadata: `/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/metadata/dataset_summary.json`.
- Detailed record: `CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_2026-07-09.md`.

### P2 fallback: imagenet_1k_64x64_hf

Status: completed, exported, and verified as an ImageNet-family 64x64 fallback.

Paths:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/raw/data/*.parquet
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/raw/SOURCE_HF_STREAMING.md
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/metadata
```

Local raw hub:

```text
D:/datasets_raw_hub/registry/imagenet_1k_64x64_hf/raw/data/*.parquet
D:/datasets_raw_hub/registry/imagenet_1k_64x64_hf/raw/checksums.sha256
```

Source:

```text
Hugging Face repo: benjamin-paine/imagenet-1k-64x64
URL: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64
```

Summary:

- Train: 1,281,167 images.
- Validation: 50,000 images.
- Classes: 1,000.
- Resolution: 64x64 RGB PNG export.
- Raw package: Hugging Face parquet shards stored under both local raw hub and server `raw/data/`.
- Per-image SHA256 values are stored in the train/validation manifests.
- Raw parquet SHA256 values are stored in `raw/checksums.sha256`.
- Metadata: `/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/metadata/dataset_summary.json`.

Important distinction:

`imagenet_1k_64x64_hf` is not the exact `downsampled_imagenet_64` torrent payload. Keep `downsampled_imagenet_64` and `imagenet_1k_64x64_hf` distinct in configs, tables, and paper text.

## Optional Image Datasets

Additional small-to-medium image datasets were staged after the MVP-required set.
See:

```text
CoFiTok-internal/docs/experiment_conditions/optional_image_datasets_2026-07-08.md
```

Completed optional aliases:

```text
cifar100
flowers102
pets
dtd
caltech101
eurosat
resisc45
svhn
gtsrb
cub_200_2011
stanford_dogs
fgvc_aircraft
```

## Main-Paper Additions

The project scope was expanded on 2026-07-09 to include non-classification diffusion datasets for formal main-paper experiments.

Detailed record:

```text
CoFiTok-internal/docs/experiment_conditions/ffhq_64_afhqv2_64_2026-07-09.md
```

Completed aliases:

```text
ffhq_64
afhqv2_64
```

Summary:

| Alias | Server path | Local raw hub | Images | Resolution | Role |
|---|---|---|---:|---|---|
| `ffhq_64` | `/root/autodl-tmp/CoFiTok/datasets/ffhq_64` | `D:/datasets_raw_hub/registry/ffhq_64/raw` | 70,000 | 64x64 | P0 non-classification face diffusion data |
| `afhqv2_64` | `/root/autodl-tmp/CoFiTok/datasets/afhqv2_64` | `D:/datasets_raw_hub/registry/afhqv2_64/raw` | 15,803 | 64x64 | P0/P1 animal texture/detail data |

## ImageNet-256 Scaling Data

Detailed record:

```text
CoFiTok-internal/docs/experiment_conditions/imagenet_256_10pct_2026-07-09.md
```

Completed server aliases:

```text
imagenet_1k_256x256_hf
imagenet_256
imagenet_256_10pct
```

Summary:

| Alias | Server path | Images / rows | Resolution | Role |
|---|---|---:|---|---|
| `imagenet_1k_256x256_hf` | `/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_256x256_hf` | 1,431,167 parquet rows | 256x256 | HF full raw source for ImageNet-256 |
| `imagenet_256` | `/root/autodl-tmp/CoFiTok/datasets/imagenet_256` | 1,331,167 JPEG images | 256x256 | P1 full ImageNet-256 train + val |
| `imagenet_256_10pct` | `/root/autodl-tmp/CoFiTok/datasets/imagenet_256_10pct` | 178,161 JPEG images | 256x256 | P1 10% train + full val scaling validation |

Local raw source:

```text
D:/datasets_raw_hub/registry/imagenet1k/raw
```

The local official ImageNet raw is the canonical local reusable source. The current server `imagenet_256_10pct` experiment path was derived from the HF 256x256 parquet raw, so do not assume local and server derived JPEG bytes are identical.

## Explicitly Not Downloaded For MVP

These local raw hub datasets exist or may be accessible, but are outside the current MVP dataset boundary:

- Full ImageNet / ILSVRC original images.
- ImageNet-A, ImageNet-R, ImageNet-Sketch, ImageNet-V2.
- Stanford Cars.
- COCO, LAION, CelebA-HQ, ImageNet-256, LSUN-bedroom-256, LSUN-church-256, FFHQ-256.

They should not be used for first-round CoFiTok conclusions unless the project scope is deliberately expanded and a new experiment-condition record is written.
