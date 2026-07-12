# ImageNet-256 Feasibility Record

Date: 2026-07-09

Purpose: evaluate the P1 ImageNet-256 route for formal main-paper scaling/baseline alignment.

## Status

Not yet staged as `/root/autodl-tmp/CoFiTok/datasets/imagenet_256`.

The official ImageNet-1K raw files are already present in the local raw hub and have been verified against official MD5 values. This means ImageNet-256 should be treated as a derived dataset from local official raw, not as a new download unless a separate preprocessed source is intentionally chosen and documented.

## Local Raw Source

```text
D:/datasets_raw_hub/registry/imagenet1k/raw/ILSVRC2012_devkit_t12.tar.gz
D:/datasets_raw_hub/registry/imagenet1k/raw/ILSVRC2012_img_train.tar
D:/datasets_raw_hub/registry/imagenet1k/raw/ILSVRC2012_img_val.tar
```

Verified MD5:

```text
ILSVRC2012_img_train.tar          1d675b47d978889d74fa0da5fadfb00e
ILSVRC2012_img_val.tar            29b22e2961454d5413ddabcf34fc5622
ILSVRC2012_devkit_t12.tar.gz      fa75699e90414af021442c21a62c3abf
```

The train and val hashes match the official ImageNet-1K README values stored in:

```text
D:/datasets_raw_hub/registry/imagenet1k/README.md
```

## Current Storage Constraint

Server free space after staging FFHQ-64 and AFHQv2-64:

```text
/root/autodl-tmp: 193G free
```

Local D drive free space:

```text
about 5.0T free
```

The local official raw payload is approximately:

```text
train tar: 147,897,477,120 bytes
val tar:     6,744,924,160 bytes
devkit:          2,568,145 bytes
```

Directly uploading the official raw to `pro6000` and also keeping a full 256x256 derived dataset is not a good default under the current 193G server free-space budget.

## Recommended Route

Recommended default:

1. Keep official ImageNet-1K raw canonical in local raw hub.
2. Generate `imagenet_256` as a derived dataset from local raw, preferably first as a 100-class or 10% subset for scaling validation.
3. Upload only the derived ImageNet-256 subset/full export plus a metadata pointer to the local raw source unless the server has enough space for both raw and derived copies.
4. If full ImageNet-256 is required on server, first free or add at least 250G extra space so raw, derived images, metadata, and temporary files can coexist safely.

Suggested server alias when staged:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_256
```

Suggested layout:

```text
imagenet_256/
|-- raw/
|   `-- SOURCE_LOCAL_IMAGENET1K.md
|-- extracted/
|   |-- train/
|   `-- val/
`-- metadata/
    |-- image_manifest.jsonl
    `-- export_summary.json
```

## Paper Use

Use ImageNet-256 for P1 baseline alignment with DiT, MAR, D-AR, FlexTok, TiTok-family settings. Do not block the 64x64 main table on ImageNet-256 completion; it is a scaling/baseline-alignment dataset, not the first evidence source for CoFiTok prefix denoising.

