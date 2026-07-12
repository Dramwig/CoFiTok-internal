# ImageNet-1K 64x64 HF Fallback Plan

Date: 2026-07-08

Purpose: provide a public 64x64 ImageNet-family P2 validation dataset after the
preferred TFDS `downsampled_imagenet/64x64` source failed.

Reason for fallback:

```text
TFDS downsampled_imagenet/64x64 attempted source:
  https://image-net.org/small/train_64x64.tar
observed failure:
  HTTP 404 via tensorflow_datasets download_and_prepare()

Academic Torrents mirror attempted:
  https://academictorrents.com/download/96816a530ee002254d29bf7a61c0c158d3dedc3b
observed failure:
  direct connection timed out
  AutoDL network_turbo returned Squid 503 HTML, not a torrent file
evidence:
  /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64_academic_503.html
```

Fallback source:

```text
dataset alias: imagenet_1k_64x64_hf
Hugging Face repo: benjamin-paine/imagenet-1k-64x64
repo URL: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64
description: ILSVRC/ImageNet-1K repack resized to 64x64 in Parquet format
license tag: other
columns: image, label
class labels: ImageNet-1K labels
```

Hugging Face Dataset Viewer probe:

```text
splits:
  train
  validation
  test
size endpoint:
  total rows: 1,431,167
  train rows: 1,281,167
  validation rows: 50,000
  test rows: 100,000
  parquet bytes total: 1,939,600,412
  train parquet bytes: 1,735,154,498
  validation parquet bytes: 68,599,878
  test parquet bytes: 135,846,036
first rows endpoint: passed for validation
```

Archived API probe files:

```text
docs/experiment_conditions/hf_probe/benjamin_splits.json
docs/experiment_conditions/hf_probe/benjamin_size.json
docs/experiment_conditions/hf_probe/benjamin_first_rows.json
```

Export script:

```text
CoFiTok-internal/scripts/prepare_hf_image_dataset.py
```

Training loader:

```text
dataset alias: imagenet_1k_64x64_hf
expected prepared root:
  /root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted
split layout:
  train/<class>/shard_00000/*.png
  validation/<class>/shard_00000/*.png
manifest-first loading:
  train_manifest.jsonl
  validation_manifest.jsonl
```

Temporary export probe:

```text
path:
  /root/autodl-tmp/CoFiTok/tmp/imagenet_1k_64x64_hf_probe
rows:
  train: 8
  validation: 8
command shape:
  source /etc/network_turbo
  PYTHONPATH=src conda run -n pf-vlm python scripts/prepare_hf_image_dataset.py \
    --dataset-id benjamin-paine/imagenet-1k-64x64 \
    --config default \
    --splits train validation \
    --output-root <tmp>/datasets/imagenet_1k_64x64_hf/extracted \
    --format png \
    --shard-size 4 \
    --max-examples-per-split 8
```

Temporary loader smoke:

```text
report:
  artifacts/reports/smoke/imagenet_1k_64x64_hf_probe/smoke_report.json
dataset: imagenet_1k_64x64_hf
actual_device: cuda
epsilon output shape: [8, 3, 64, 64]
zero_token_component_energy_ratio: 0.0
```

Decision:

Use `imagenet_1k_64x64_hf` as the P2 public 64x64 ImageNet-family validation
dataset if the preferred TFDS downsampled source remains unavailable. Keep its source and preprocessing distinct from `downsampled_imagenet_64` in all tables and paper text.

Formal export:

```text
launcher:
  CoFiTok-internal/scripts/run_prepare_imagenet_1k_64x64_hf.sh
log:
  /root/autodl-tmp/CoFiTok/logs/prepare_imagenet_1k_64x64_hf_2026-07-08.log
pid file:
  /root/autodl-tmp/CoFiTok/logs/prepare_imagenet_1k_64x64_hf_2026-07-08.pid
output:
  /root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted
cache:
  /root/autodl-tmp/CoFiTok/checkpoints/hf_cache
final status:
  completed
  train_manifest rows: 1,281,167
  validation_manifest rows: 50,000
  manifest_summary.json: written
  file count under extracted/: 1,331,170
  dataset dir size: 14G
  /root/autodl-tmp available after export: 250G
formal smoke:
  /root/autodl-tmp/CoFiTok/checkpoints/smoke/imagenet_1k_64x64_hf_smoke.json
local smoke archive:
  artifacts/reports/smoke/imagenet_1k_64x64_hf_smoke.json
formal smoke result:
  actual_device: cuda
  epsilon output shape: [8, 3, 64, 64]
  zero_token_component_energy_ratio: 0.0
manifest summary archive:
  docs/experiment_conditions/imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json
export log archive:
  docs/experiment_conditions/imagenet_1k_64x64_hf_export_log_2026-07-08.txt
```

5k P2 runs:

```text
channel-mask:
  final clean MSE 0.1230
  path AUC 4.8844
  effective tokens 1.111
epsilon-only:
  final clean MSE 0.0998
  path AUC 1.1000
  effective tokens 5.339
full denoise-path:
  final clean MSE 0.1074
  path AUC 0.0417
  effective tokens 6.758
light denoise-path:
  final clean MSE 0.1018
  path AUC 0.0620
  effective tokens 6.787
```

Conclusion:

The fallback dataset reproduces the CIFAR-10/Tiny ImageNet pattern. Epsilon-only
has the best endpoint, full denoise-path has the best path AUC, and light
denoise-path remains the strongest endpoint/path tradeoff.

Seed-2 repeat and order eval:

```text
epsilon-only seed2:
  final clean MSE 0.0940
  path AUC 0.8837
  effective tokens 5.400
light denoise-path seed2:
  final clean MSE 0.0963
  path AUC 0.0641
  effective tokens 6.768
two-seed means:
  epsilon-only final/path: 0.0969 / 0.9919
  light denoise-path final/path: 0.0990 / 0.0631
order eval mean path AUC:
  ordered 0.0641
  random0 0.2523
  reverse 0.7177
```

Updated conclusion:

The light denoise-path p1.5 objective remains the current MVP candidate on the
ImageNet-family 64x64 fallback: it gives endpoint quality close to epsilon-only,
much better path alignment, high effective token use, and clear order
sensitivity across two seeds.

Simultaneous predictor ablation:

```text
config:
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_cuda.json
change:
  predictor_use_feedback = false
result:
  final clean MSE 0.0995
  path AUC 0.0600
  effective tokens 6.845
order eval path AUC:
  ordered 0.0611
  random0 0.2242
  reverse 0.6505
```

Interpretation:

The current tiny model does not require explicit token feedback to learn useful
ordered denoising components. The evidence supports ordered component
factorization and restricted `S_k`; the final paper claim should not overstate
feedback/AR dependence.

Deep-`S_k` ablation:

```text
config:
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda.json
change:
  synthesis_mode = deep_decoder
  deep_synthesis_hidden_channels = 32
  deep_synthesis_depth = 4
validation:
  pytest 35 passed
result:
  final clean MSE 0.0977
  path AUC 0.0366
  effective tokens 6.792
  zero-token component energy ratio 0.0522
order eval path AUC:
  ordered 0.0382
  random0 0.1361
  reverse 0.6431
```

Interpretation:

The deep synthesis ablation gives better endpoint/path metrics, but the
zero-token diagnostic is no longer zero. This confirms the central design
boundary: deep or biased `S_k` can become a decoder-like component prior and is
not acceptable as the default CoFiTok operator.

Token count scaling:

```text
configs:
  train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_cuda.json
  train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json
  train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_cuda.json
train summary:
  K4:  final/path/effective = 0.0988 / 0.0665 / 3.318
  K8:  final/path/effective = 0.1018 / 0.0620 / 6.787
  K16: final/path/effective = 0.0992 / 0.0674 / 13.560
order eval path AUC:
  K4:  ordered 0.0674, random0 0.0727, reverse 0.7516
  K8:  ordered 0.0635, random0 0.2441, reverse 0.6947
  K16: ordered 0.0680, random0 0.4680, reverse 0.7020
zero-token ratio:
  K4/K8/K16 all 0.0
```

Interpretation:

K8 is the current restricted-MVP sweet spot. K4 is a good cheap baseline but has
weak random-order separation; K16 uses many tokens but does not improve path AUC
in this 5k budget.

Cross-batch quality metrics:

```text
script:
  scripts/evaluate_quality.py
validation:
  pytest 38 passed
setup:
  validation images 256
  fixed timestep 500
  feature size 8
optional perceptual packages:
  lpips unavailable
  torchmetrics unavailable
metric caveat:
  low-res Fréchet proxy is not formal Inception FID
```

Final-prefix quality:

```text
channel-mask:
  final MSE / PSNR / proxy = 0.1749 / 13.593 / 3.9121
epsilon-only:
  final MSE / PSNR / proxy = 0.1262 / 15.011 / 1.1015
K4 light:
  final MSE / PSNR / proxy = 0.1267 / 14.994 / 1.1506
K8 light:
  final MSE / PSNR / proxy = 0.1289 / 14.917 / 1.0864
K16 light:
  final MSE / PSNR / proxy = 0.1265 / 14.999 / 1.1003
deep-S_k:
  final MSE / PSNR / proxy = 0.1255 / 15.034 / 1.0278
```

Interpretation:

The quality layer is consistent with previous evidence: channel-mask is weak;
epsilon-only has good endpoint quality but weak prefixes; K8 remains the best
restricted distribution-proxy endpoint among light variants; deep-`S_k` gives
better endpoint quality but fails the zero-token constraint and remains an
ablation only.
