# tiny_imagenet_200 EDM RGB Train Derivative

Date: 2026-07-09

Alias:

```text
tiny_imagenet_200_edm_rgb_train
```

Source train directory:

```text
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/extracted/tiny-imagenet-200/train
```

Derived train directory:

```text
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/derived/edm_rgb_train
```

Reason: EDM's `ImageFolderDataset` asserts a uniform RGB image shape and does
not convert grayscale JPEGs to RGB. CoFiTok's own loader and the evaluation
script convert images to RGB, so this derivative keeps the same semantic train
split while making EDM's loader protocol-compatible.

Counts:

```text
source_total=100000
rgb_hardlinked=98179
grayscale_converted=1821
copied_fallback=0
```

Conversion rule:

- RGB 64x64 JPEG files were hardlinked into the derived directory.
- Grayscale 64x64 JPEG files were converted to RGB PNG files.
- No symlinks were created.

Manifest:

```text
/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200/metadata/edm_rgb_train_manifest_2026-07-09.json
```

Use only for EDM Tiny ImageNet training. Other CoFiTok and external baseline
evaluations should continue to reference `tiny_imagenet_200` through their
normal dataset alias/config unless their loader has the same RGB-shape issue.
