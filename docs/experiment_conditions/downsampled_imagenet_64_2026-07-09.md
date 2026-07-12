# downsampled_imagenet_64 Dataset Record

Date: 2026-07-09

Server:

```text
pro6000
```

Dataset root:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64
```

## Status

Completed and verified from the strict Academic Torrents source.

This is distinct from the previously staged Hugging Face fallback:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf
```

Do not silently substitute one alias for the other in experiment configs, tables, or paper text.

## Source

Academic Torrents:

```text
https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b
```

Info hash:

```text
96816a530ee002254d29bf7a61c0c158d3dedc3b
```

Download route used:

```text
aria2c magnet+DHT on pro6000
```

Notes:

- The ImageNet official small archive URLs returned HTTP 404 on 2026-07-09.
- The Academic Torrents web/download endpoint timed out from the available local and remote environments.
- The magnet route succeeded; aria2 obtained metadata via DHT/peers and completed BT piece verification.

## Paths

Raw:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/train_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/valid_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/checksums.sha256
```

Extracted:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/train_64x64
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/valid_64x64
```

Metadata:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/metadata/dataset_summary.json
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/metadata/download_logs/downsampled_imagenet_64_aria2_20260709_102244.log
```

## Checksums

```text
1104077da8c88db149d74ffb6d1f9fe17b32abf535880966ee1684559820cdfa  96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent
e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7  small/train_64x64.tar
eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd  small/valid_64x64.tar
```

## Split And Content

| Split | Path | PNG files |
|---|---|---:|
| train | `extracted/train_64x64` | 1,281,149 |
| valid | `extracted/valid_64x64` | 49,999 |

Sampled files were verified as 64x64 PNG images.

The torrent payload is a flat image split and does not include class-label files. Use it as an unconditional image dataset unless labels are supplied separately. This is acceptable for the first CoFiTok diffusion runs because `S_k` must not receive class labels in any case.

## Size

```text
raw:       12G
extracted: 14G
```

## Preprocessing

No crop, resize, normalization, filtering, or relabeling was applied. The raw torrent payload was extracted only, and extracted file ownership/permissions were normalized to `root:root`, directories `755`, files `644`.
