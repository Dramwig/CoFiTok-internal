# Optional Image Datasets Staged on pro6000 (2026-07-08)

This record covers additional small-to-medium image datasets staged after the core
CoFiTok MVP datasets (`cifar10`, `tiny_imagenet_200`, and the ImageNet-64
fallback). These datasets are not the first-line paper evidence, but they are useful
for smoke tests, robustness checks, fine-grained object domains, and texture/scene
diagnostics.

Remote dataset root:

```text
/root/autodl-tmp/CoFiTok/datasets
```

All entries below follow the project layout:

```text
/root/autodl-tmp/CoFiTok/datasets/<alias>/raw
/root/autodl-tmp/CoFiTok/datasets/<alias>/extracted
/root/autodl-tmp/CoFiTok/datasets/<alias>/metadata/dataset_summary.json
```

No center crop, resize, normalization, filtering, or generated train/val split was
applied during staging. Class labels may be used by experiments, but `S_k` must not
receive labels, timestep, prompt, previous tokens, or other conditioning.

## Completed

| Alias | Source / transfer route | Raw artifact(s) | Extracted data summary | Notes |
|---|---|---|---|---|
| `cifar100` | Local raw upload from `D:/datasets_raw_hub/registry/cifar100/raw`; Toronto official source was stopped after very low throughput | `cifar-100-python.tar.gz` | 50,000 train / 10,000 test, 100 classes, 32x32 RGB | Torchvision-compatible python archive |
| `flowers102` | Oxford VGG direct download | `102flowers.tgz`, `imagelabels.mat`, `setid.mat` | 8,189 flower images, 102 classes | Official split is in `setid.mat` |
| `pets` | Oxford VGG direct download | `images.tar.gz`, `annotations.tar.gz` | 7,390 original pet images, 37 classes | `annotations/` also contains trimap PNG files; use `images/` for image count |
| `dtd` | Oxford VGG direct download | `dtd-r1.0.1.tar.gz` | 5,640 texture images, 47 classes | Official train/val/test split files included |
| `caltech101` | Caltech Data direct download | `caltech-101.zip` | 9,144 images under `101_ObjectCategories` | No fixed official split in raw package |
| `eurosat` | Zenodo direct download | `EuroSAT_RGB.zip` | 27,000 images, 10 classes, 64x64 RGB | Create experiment split in config |
| `resisc45` | Hugging Face hosted ZIP direct download | `NWPU-RESISC45.zip` | 31,500 images, 45 classes, 256x256 RGB | Create experiment split in config |
| `svhn` | Stanford direct download | `train_32x32.mat`, `test_32x32.mat`, `extra_32x32.mat` | 73,257 train / 26,032 test / 531,131 extra, 10 classes, 32x32 RGB | Stored as `.mat`; copied into `extracted/` unchanged |
| `gtsrb` | Local raw upload from `D:/datasets_raw_hub/registry/gtsrb/raw` | `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip` | 12,630 test images, 43 classes | Current local raw is test-only; train split is not staged |
| `cub_200_2011` | Caltech Data direct download | `CUB_200_2011.tgz` | 11,788 bird images, 200 classes | Official train/test split file included |
| `stanford_dogs` | Stanford direct download | `images.tar`, `annotation.tar`, `lists.tar` | 20,580 dog images, 120 classes | Official train/test lists included |
| `fgvc_aircraft` | Oxford VGG direct download | `fgvc-aircraft-2013b.tar.gz` | 10,000 aircraft images, 100 variants | Official train/val/test split files included |

## Verification Snapshot

Server disk after staging:

```text
/dev/md0  500G total, 271G used, 230G available, 55% used
```

Remote dataset sizes:

```text
223M  /root/autodl-tmp/CoFiTok/datasets/gtsrb
225M  /root/autodl-tmp/CoFiTok/datasets/eurosat
339M  /root/autodl-tmp/CoFiTok/datasets/cifar100
455M  /root/autodl-tmp/CoFiTok/datasets/caltech101
677M  /root/autodl-tmp/CoFiTok/datasets/flowers102
877M  /root/autodl-tmp/CoFiTok/datasets/resisc45
1.2G  /root/autodl-tmp/CoFiTok/datasets/dtd
1.6G  /root/autodl-tmp/CoFiTok/datasets/pets
1.7G  /root/autodl-tmp/CoFiTok/datasets/stanford_dogs
2.3G  /root/autodl-tmp/CoFiTok/datasets/cub_200_2011
3.0G  /root/autodl-tmp/CoFiTok/datasets/svhn
5.2G  /root/autodl-tmp/CoFiTok/datasets/fgvc_aircraft
```

Symlink audit:

```text
find /root/autodl-tmp/CoFiTok -path /root/autodl-tmp/CoFiTok/checkpoints/hf_cache -prune -o -type l -print
```

No project symlinks were reported.

## Not Staged in This Pass

- `stanford_cars`: local raw exists as `D:/datasets_raw_hub/registry/stanford_cars/raw/stanford-cars-dataset.zip`, but the original Stanford URL is no longer available online according to Torchvision documentation, common remote alternatives require Kaggle credentials, and local `scp` throughput is too slow for a 1.8GB upload in this pass.
- `imagenet_a`, `imagenet_r`, `imagenet_v2`, `imagenet_sketch`, `country211`: useful as robustness or geography benchmarks, but not necessary for the CoFiTok MVP and larger than the current experimental need.
- Full ImageNet, COCO, LAION, and text-image/VQA corpora remain intentionally excluded by the project boundary for the first round.
