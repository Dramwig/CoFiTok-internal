# ImageNet-256 Full Dataset Record

Date: 2026-07-09

Server:

```text
pro6000
```

## Status

Completed and verified on `pro6000`.

This is the full ImageNet-256 train+validation ImageFolder export for P1 baseline alignment and scaling experiments.

## Paths

Full HF raw source:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_256x256_hf
```

Full derived export:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_256
```

## Source

```text
Hugging Face repo: benjamin-paine/imagenet-1k-256x256
URL: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256
Snapshot: 1bd0400450249a7fe90c0aece37d0d03e7ea956a
```

The raw parquet images are already 256x256 JPEGs. The export copied JPEG bytes directly into an ImageFolder-style layout; no resize, crop, normalization, filtering, or JPEG recoding was applied on the server.

## Split And Content

| Split | Images | Class dirs | Resolution |
|---|---:|---:|---|
| train | 1,281,167 | 1,000 | 256x256 RGB JPEG |
| val | 50,000 | 1,000 | 256x256 RGB JPEG |

The HF raw `test` split is retained under `imagenet_1k_256x256_hf/raw` but was not exported for current training/evaluation because labels are not part of the standard ImageNet validation protocol used here.

## Verification

Full raw:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_256x256_hf/raw/checksums.sha256
sha256sum -c checksums.sha256: OK for 46 files
```

Full export:

```text
file count: 1,331,167
split count: train 1,281,167 / val 50,000
size check: 1,331,167 images are 256x256
mode check: 1,331,167 images are RGB
manifest sha256: 9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0
```

Server sizes:

```text
imagenet_1k_256x256_hf: 19G
imagenet_256:           21G
```

## Local Raw Hub

The canonical local reusable raw source remains the official ImageNet-1K raw archive set:

```text
D:/datasets_raw_hub/registry/imagenet1k/raw
```

The full server export is HF-parquet-derived and is authoritative for experiments using:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_256
```

Do not silently substitute local official-raw-derived images or the 10% subset for this full server export in configs, tables, or paper text.

Attempted local mirrors of the 19G HF parquet raw via `scp` and local `curl` were stopped because transfer speed was too low for this pass. Partial local files were removed. The local reusable source for ImageNet-scale work remains the official ImageNet-1K raw archive set above.
