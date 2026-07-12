# Downsampled ImageNet 64 Preparation Plan

Date: 2026-07-08

Purpose: prepare the P2 validation dataset after CIFAR-10 and Tiny ImageNet
showed stable CoFiTok MVP behavior.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok
```

Storage check:

```text
/root/autodl-tmp total: 500G
used: 237G
available: 264G
current dataset sizes:
  cifar10: 341M
  tiny_imagenet_200: 717M
project symlinks outside hf_cache: none observed
```

Candidate source:

```text
alias: downsampled_imagenet_64
preferred public loader: TensorFlow Datasets downsampled_imagenet/64x64
source URL: https://www.tensorflow.org/datasets/catalog/downsampled_imagenet
reported download size: 11.73 GiB
reported dataset size: 10.80 GiB
reported splits:
  train: 1,281,149 examples
  validation: 49,999 examples
reported features:
  image only, uint8 RGB
supervised keys:
  None
```

Remote environment check:

```text
tensorflow_datasets: not installed in pf-vlm
tensorflow: not installed in pf-vlm
huggingface datasets: installed
webdataset: not installed
```

Export environment:

```text
path: /root/autodl-tmp/CoFiTok/envs/tfds-export
size after install: 313M
created with: /root/miniconda3/bin/conda run -n pf-vlm python -m venv
final state:
  removed after TFDS source failed, to keep the project tree symlink-clean;
  recreate from the freeze archive if the TFDS source becomes available again.
pip source note:
  default Aliyun index did not expose tensorflow-datasets;
  install succeeded with -i https://pypi.org/simple
key packages:
  tensorflow-datasets==4.9.10
  array_record==0.8.1
  numpy==2.2.6
  pillow==12.3.0
  pyarrow==24.0.0
  tqdm==4.68.4
freeze archive:
  docs/experiment_conditions/downsampled_imagenet_64_tfds_export_freeze_2026-07-08.txt
```

TFDS inspect-only check:

```text
command:
  /root/autodl-tmp/CoFiTok/envs/tfds-export/bin/python \
    scripts/prepare_downsampled_imagenet64_tfds.py \
    --inspect-only \
    --summary-output /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64_tfds_inspect.json
result:
  wrote metadata JSON without calling download_and_prepare()
local archive:
  docs/experiment_conditions/downsampled_imagenet_64_tfds_inspect_2026-07-08.json
observed from builder before download:
  tfds_version: 4.9.10
  builder_version: 2.0.0
  features: image only
  supervised_keys: None
  split sizes/download sizes remain unknown until full download metadata is prepared
```

Decision:

The prepared-image loader path is ready, but the preferred TFDS source is not
currently usable from `pro6000`.

Original strategy:

1. Install and record `tensorflow-datasets` in an isolated server environment,
   then prepare `downsampled_imagenet/64x64` into:

```text
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64
```

2. If TFDS installation is too invasive for `pf-vlm`, create a separate
   environment and document the full environment. This was done as
   `/root/autodl-tmp/CoFiTok/envs/tfds-export`.

3. Only use a Hugging Face mirror/upload if the source, format, labels, and
   license/derivation are recorded clearly. Prefer the TFDS path for the first
   paper-facing P2 validation.

Full TFDS attempt:

```text
process log:
  /root/autodl-tmp/CoFiTok/logs/prepare_downsampled_imagenet64_2026-07-08.log
result:
  failed before downloading data
error:
  tensorflow_datasets.core.download.util.DownloadError
  Failed to get url https://image-net.org/small/train_64x64.tar. HTTP code: 404.
space consumed:
  8K metadata under /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64
```

Academic Torrents fallback attempt:

```text
source:
  https://academictorrents.com/download/96816a530ee002254d29bf7a61c0c158d3dedc3b
direct curl:
  connection timed out
with /etc/network_turbo:
  Squid 503 HTML, not a torrent file
evidence:
  /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64_academic_503.html
```

Current fallback:

```text
alias:
  imagenet_1k_64x64_hf
record:
  docs/experiment_conditions/imagenet_1k_64x64_hf_plan_2026-07-08.md
note:
  This fallback is ImageNet-1K 64x64 repack/resized data, not the exact TFDS
  downsampled_imagenet/64x64 dataset. Keep the distinction explicit.
```

Implemented code path:

```text
dataset alias: downsampled_imagenet_64
training loader expects:
  /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/train/<class>/**/*.{png,jpg,jpeg}
  /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/extracted/validation/<class>/**/*.{png,jpg,jpeg}
fallback validation split name:
  val
unlabeled class folder:
  unlabeled
default export sharding:
  10,000 images per nested shard directory, e.g. unlabeled/shard_00000/
```

Export script:

```text
CoFiTok-internal/scripts/prepare_downsampled_imagenet64_tfds.py
```

The export script imports `tensorflow_datasets` only when it is run. It writes:

```text
extracted/train_manifest.jsonl
extracted/validation_manifest.jsonl
extracted/manifest_summary.json
```

Each manifest row includes split, relative path, class name, label if present,
image width/height, and sha256 of the exported image bytes.

The training loader checks for `<split>_manifest.jsonl` first and uses it when
available. If no manifest exists, it falls back to recursive image discovery
under the split directory.

Added configs:

```text
configs/smoke_downsampled_imagenet64_cuda.json
configs/train_downsampled_imagenet64_k8_channelmask_p150eval_5k_cuda.json
configs/train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda.json
configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json
```

Loader validation:

```text
remote pytest: 26 passed
script py_compile: passed
prepare script --help: passed
temporary loader smoke report:
  /root/autodl-tmp/CoFiTok/tmp/downsampled_imagenet64_loader_smoke/smoke_report.json
local archive:
  artifacts/reports/smoke/downsampled_imagenet64_loader_smoke.json
```

The loader smoke used 8 copied Tiny ImageNet images in a temporary prepared
`downsampled_imagenet_64` layout. It did not create or modify the formal P2
dataset target directory. Smoke result:

```text
dataset: downsampled_imagenet_64
actual_device: cuda
epsilon output shape: [8, 3, 64, 64]
component shapes: 4 x [8, 3, 64, 64]
zero_token_component_energy_ratio: 0.0
```

Required record after download:

- source URL and loader version;
- download date;
- raw/cache paths and extracted/exported paths;
- train/validation split sizes;
- sha256/manifest for exported files if converted;
- image resolution and transform;
- label availability and whether labels are used;
- config names and checkpoint output directories.
