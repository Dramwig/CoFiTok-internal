# ybforever Deployment Conditions

Date: 2026-08-17

## Workspace

- Host: `ybforever` (`yubohuang@100.88.240.40:2222`)
- Project root: `/home/yubohuang/zixi/CoFiTok`
- Code root: `/home/yubohuang/zixi/CoFiTok/CoFiTok-internal`
- Data root: `/home/yubohuang/zixi/CoFiTok/datasets`
- Checkpoint root: `/home/yubohuang/zixi/CoFiTok/checkpoints`
- Git revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- Branch: `scale/generative-system`

## Runtime

- GPUs: 2 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97,887 MiB each)
- Driver: 572.83
- Project environment: `.venv` under the code root
- PyTorch: `2.11.0+cu128`, CUDA `12.8`
- Config portability: set `COFITOK_DATA_ROOT=/home/yubohuang/zixi/CoFiTok/datasets` when running configs that retain the legacy data-root literal.

## Downloaded Datasets

### cifar10

- Source: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
- Archive: `datasets/cifar10/raw/cifar-10-python.tar.gz`
- SHA256: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- Extracted: `datasets/cifar10/extracted/cifar-10-batches-py`

### tiny_imagenet_200

- Source: `https://cs231n.stanford.edu/tiny-imagenet-200.zip`
- Archive: `datasets/tiny_imagenet_200/raw/tiny-imagenet-200.zip`
- SHA256: `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`
- Extracted: `datasets/tiny_imagenet_200/extracted/tiny-imagenet-200`

### imagenet_1k_64x64_hf

- Source loader: `benjamin-paine/imagenet-1k-64x64`
- Endpoint: `https://hf-mirror.com`
- Cache: `checkpoints/hf_cache`
- Output: `datasets/imagenet_1k_64x64_hf/extracted`
- Status: completed and manifest verified on 2026-08-17.
- Splits: 1,281,167 train images and 50,000 validation images.

### imagenet_1k_256x256_hf and imagenet_256

- Source loader: `benjamin-paine/imagenet-1k-256x256`
- Endpoint: `https://hf-mirror.com`
- Cache: `checkpoints/hf_cache`
- Raw snapshot: `datasets/imagenet_1k_256x256_hf/raw` (45 parquet shards; about 19 GiB)
- Derived output: `datasets/imagenet_256/extracted`
- Export: `scripts/export_imagenet_256_hf_subset.py --train-stride 1`
- Preprocessing: source 256x256 RGB JPEG bytes were copied into an ImageFolder layout without resizing or JPEG recoding.
- Splits: 1,281,167 train images and 50,000 validation images.
- Export manifest SHA256: `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`.
- Label-to-WNID mapping SHA256: `3e8c34f680433258998fdd30cf4c15442e2ce3cbd35c27507becc0f690b19b91`.

### imagenet_256_10pct

- Source: the deployed `imagenet_1k_256x256_hf/raw` snapshot above.
- Derived output: `datasets/imagenet_256_10pct/extracted`
- Export: `scripts/export_imagenet_256_hf_subset.py --train-stride 10 --train-offset 0`
- Sampling: retain each class-local train index satisfying `index % 10 == 0`; retain the complete validation split.
- Preprocessing: source 256x256 RGB JPEG bytes were copied without resizing or JPEG recoding.
- Splits: 128,161 train images and 50,000 validation images across 1,000 classes per split.
- Export manifest SHA256: `dcdd622564941ad418fffa960051f14f0ad41852c681ed9a8329c5e6f561cca7`.

### downsampled_imagenet_64

- Strict source: Academic Torrents `96816a530ee002254d29bf7a61c0c158d3dedc3b`.
- Download method: the verified torrent metadata was copied from `pro6000`, then the data itself was obtained through the public BT swarm on `ybforever` with user-local `aria2`.
- Torrent SHA256: `1104077da8c88db149d74ffb6d1f9fe17b32abf535880966ee1684559820cdfa`.
- Raw archive SHA256: `e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7` for `small/train_64x64.tar`; `eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd` for `small/valid_64x64.tar`.
- Output: `datasets/downsampled_imagenet_64/extracted/{train_64x64,valid_64x64}`.
- Splits: 1,281,149 train PNG files and 49,999 validation PNG files.
- Preprocessing: extraction only, with no crop, resize, filtering, or relabeling. Sampled files are 64x64 RGB PNG images.

### ffhq_64

- Reused local host source: `/home/yubohuang/zixi/datasets/ffhq_64/ffhq-64x64.zip`.
- Source: `Dmini/FFHQ-64x64` (`ffhq-64x64.zip`).
- Raw SHA256: `76af01fc2b10b6631e08713365c7ae2a9d8ee99a7a62ac834ade684a913094ff`.
- Project output: `datasets/ffhq_64/extracted/images`.
- Preprocessing: ZIP extraction only; source 64x64 RGB PNG bytes were retained without crop, resize, or recoding.
- Split: 70,000 images, represented as the unsplit training set required by the project data registry.
- Deployment manifest SHA256: `70a5aed816c8803d5c13a65e778fe5e1e095cbe39afedc498af8e6c1449f9e4c`.

### afhqv2_64

- Reused local host source: `/home/yubohuang/zixi/datasets/afhqv2_64/data` and `dataset_infos.json`.
- Source: `huggan/AFHQv2` (13 training parquet shards).
- All 13 source parquet SHA256 values and `dataset_infos.json` SHA256 matched the 2026-07-09 CoFiTok record before copying into the project data root.
- Project output: `datasets/afhqv2_64/extracted/train/{cat,dog,wild}`.
- Preprocessing: source images were converted to RGB and resized from 512x512 to 64x64 with PIL LANCZOS.
- Split: 15,803 images: 5,558 cat, 5,169 dog, and 5,076 wild.
- Deployment manifest SHA256: `b3e0deb87b91c630a98a3fa7a17d5a2f96dcf99397227add51e32eb788f0e7b9`.

The FFHQ and AFHQ deployment manifests are new provenance files because their conversion was regenerated from verified raw assets on `ybforever`; they do not replace the locked manifest hashes recorded for `pro6000`.

Both aliases were verified with their one-step CUDA smoke configurations on GPU 1, using `COFITOK_DATA_ROOT=/home/yubohuang/zixi/CoFiTok/datasets`. The reports are under `checkpoints/smoke/ybforever_ffhq64_cuda_smoke_20260817` and `checkpoints/smoke/ybforever_afhqv2_64_cuda_smoke_20260817`.

## Reused Weights

Real copies were placed under `checkpoints/` from the pre-existing host cache:

- `baselines/d_ar`
- `baselines/ml_flextok`
- `baselines/titok_1d_tokenizer`
- `evaluators`

The pre-existing `MAR-B` directory contained only small log files and was not treated as a usable checkpoint. The official MAR assets were instead downloaded from `jadechoghari/mar` through `https://hf-mirror.com` at revision `773c9bdcd98740c7b876ce0d66f684a0e2468990`:

- `baselines/mar/official_imagenet256_eval_only/official_pth/checkpoint-last.pth`: 1,663,614,946 bytes, SHA256 `7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c`.
- `baselines/mar/official_imagenet256_eval_only/official_pth/kl16.ckpt`: 265,900,046 bytes, SHA256 `34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f`.

These are the pinned PTH `model_ema` and KL-16 VAE assets for the secondary MAR ImageNet-256 eval-only protocol. They must not be merged into the matched-training CoFiTok/dense comparison.

## Optional Legacy Classification Datasets

The following historical optional datasets were recreated directly from public
sources and verified against the SHA256 values recorded on `pro6000` before
extraction. No preprocessing or split generation was applied.

| Alias | Retrieval | Status |
| --- | --- | --- |
| `cifar100` | Toronto | complete |
| `flowers102` | Oxford VGG | complete |
| `pets` | Oxford VGG | complete |
| `dtd` | Oxford VGG | complete |
| `caltech101` | CaltechDATA | complete; extracted nested `101_ObjectCategories.tar.gz` |
| `cub_200_2011` | CaltechDATA | complete |
| `svhn` | Stanford | complete |
| `stanford_dogs` | Stanford | complete |
| `fgvc_aircraft` | Oxford VGG | complete |

`EuroSAT_RGB.zip` could not be resolved because the target host maps
`zenodo.org` to a blocked address. The original `taesiri/NWPU-RESISC45` archive
is no longer publicly accessible through Hugging Face, and the historical
test-only GTSRB archive endpoint returns HTTP 404 despite the published index.
The exact `eurosat`, `resisc45`, and `gtsrb` directories are therefore queued
for a checksum-preserving fallback transfer from `pro6000` after the generation
archive has completed its own verification.

The public retrieval helpers are retained in
`scripts/deployment/download_optional_image_datasets_2026-08-17.sh`,
`scripts/deployment/download_optional_image_datasets_remaining_2026-08-17.sh`,
and
`scripts/deployment/download_optional_image_datasets_public_recovery_2026-08-17.sh`.

## Historical Generation Archive

The full source tree `checkpoints/generation` has an apparent size of
125,271,009,651 bytes across 131,026 regular files. It is being copied with
`rsync --partial --append-verify` from `pro6000` through a loopback-only reverse
SSH tunnel to this host. The transfer is split into two measured streams: one
for `stability_probe_2026-07-29` and `stability_scaling_50k_ema_teacher`, and
one for all remaining paths. This preserves recoverability while avoiding the
single-stream small-file latency bottleneck.

On success the source runner performs a metadata-only `rsync --dry-run
--itemize-changes`; an empty report is required before the temporary tunnel key
and authorization entry may be removed. The three legacy dataset fallback
directories run only after that archive check passes.

After both of those checks pass, the deployment also synchronizes the
first-party historical checkpoint runs that are outside `generation/`. It
excludes the public-retrievable `baselines/` assets and the target-specific
`hf_cache`, `torch_cache`, and `evaluators` directories. This follow-up uses
the same resumable transport and requires an empty metadata-only dry run.

If the target SSH endpoint temporarily stops completing new handshakes, the
deployment uses a single-stream recovery supervisor with bounded connection
timeouts and retry intervals. The later dataset and historical-run stages
wait for verified markers rather than a particular initial rsync PID, so they
survive an interrupted initial session.
