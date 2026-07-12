# 2026-07-08 Denoise-Path Objective

Purpose: replace the v0 clean-prefix objective and direct epsilon-band objective
with a schedule-aware partial denoising path. The goal is to keep restricted
condition-free `S_k`, activate more ordered tokens, and retain endpoint quality.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Validation:

```text
pytest: 24 passed
python -m py_compile scripts/train_short.py scripts/evaluate_checkpoint.py: passed
json.tool configs/train_cifar10_k8_denoisepath_3k_cuda.json: passed
json.tool configs/train_tiny_imagenet_k8_denoisepath_2k_cuda.json: passed
```

## Code Change

- Added loss config fields:
  - `epsilon_band_prefix_weight`
  - `epsilon_band_component_weight`
  - `denoise_path_prefix_weight`
  - `denoise_path_component_weight`
  - `denoise_path_progress_power`
- Added direct epsilon-band loss over coarse-to-fine low-pass targets of the
  true epsilon noise.
- Added denoise-path loss. Prefix targets now interpolate from the zero-epsilon
  x0 estimate toward coarse-to-fine clean targets:

```text
target_x0_m = lerp(x0_from_zero_epsilon, coarse_to_fine_clean_m, progress_m)
progress_m = (m / K) ^ denoise_path_progress_power
```

- Added `prefix_mse_to_denoise_path` to `scripts/train_short.py` reports.
- Added report scalars:
  - `prefix_mse_to_clean_auc`
  - `prefix_mse_to_denoise_path_auc`
  - `energy_entropy`
  - `energy_entropy_normalized`
  - `energy_effective_token_count`
- Added normalized synthesis diagnostics:
  - `zero_token_component_energy_ratio`
  - `random_token_component_energy_ratio`
  - `shuffled_component_relative_mse`
- Added `scripts/evaluate_checkpoint.py` for no-retraining ordered / reverse /
  random component-order evaluation from a saved checkpoint.
- Added configs:
  - `configs/train_cifar10_k8_epsilonband_3k_cuda.json`
  - `configs/train_cifar10_k8_epsilonband_hybrid_3k_cuda.json`
  - `configs/train_cifar10_k8_denoisepath_3k_cuda.json`
  - `configs/train_tiny_imagenet_k8_epsilonband_2k_cuda.json`
  - `configs/train_tiny_imagenet_k8_denoisepath_2k_cuda.json`

## Commands

CIFAR direct epsilon-band:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_epsilonband_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilonband_3k_2026-07-08
```

CIFAR epsilon-band hybrid:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_epsilonband_hybrid_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilonband_hybrid_3k_2026-07-08
```

CIFAR denoise-path:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_3k_2026-07-08
```

Tiny ImageNet denoise-path:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_2k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_2k_2026-07-08
```

## Artifacts

Remote:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilonband_3k_2026-07-08/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilonband_hybrid_3k_2026-07-08/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_3k_2026-07-08/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_2k_2026-07-08/report.json
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_epsilonband_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_epsilonband_hybrid_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_denoisepath_3k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_2k_2026-07-08/report.json
```

## Results

CIFAR-10 K8:

| run | epsilon | final clean MSE | prefix 4 clean MSE | prefix 7 clean MSE | tail ratio | late-half ratio | active tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel-mask 3k | 0.1449 | 0.1960 | 0.2086 | 0.2039 | 0.0228 | 0.0000 | 1 |
| direct epsilon-band 3k | 0.0491 | 0.1592 | 11.6394 | 9.3035 | 0.9999 | 0.9988 | 3 |
| epsilon-band hybrid 3k | 0.0734 | 0.3536 | 4.4491 | 4.3837 | 0.7280 | 0.7146 | 1 |
| denoise-path 3k | 0.0530 | 0.1709 | 3.1827 | 0.3883 | 0.9055 | 0.4825 | 7 |

CIFAR denoise-path target MSE:

```text
prefix_mse_to_denoise_path:
0.0828, 0.0621, 0.0588, 0.0624, 0.0917, 0.1065, 0.1455, 0.1709
component_energy_ratio:
0.0945, 0.1363, 0.1504, 0.1363, 0.1336, 0.1138, 0.1293, 0.1058
zero-token energy: 0.0
random-token component energy: 0.1791
shuffled-token MSE: 0.0397
```

Tiny ImageNet K8:

| run | epsilon | final clean MSE | prefix 4 clean MSE | prefix 7 clean MSE | tail ratio | late-half ratio | active tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel-mask 2k | 0.1427 | 0.3007 | 0.3338 | 0.3183 | 0.0246 | 0.0009 | 1 |
| channel-mask 5k seed2 | 0.1293 | 0.2641 | 0.2817 | 0.2756 | 0.0322 | 0.0003 | 1 |
| denoise-path 2k | 0.0537 | 0.2263 | 3.3017 | 0.4319 | 0.8678 | 0.4938 | 7 |

Tiny denoise-path target MSE:

```text
prefix_mse_to_denoise_path:
0.0680, 0.0943, 0.0907, 0.1009, 0.1065, 0.1336, 0.1716, 0.2263
component_energy_ratio:
0.1322, 0.0776, 0.1628, 0.1336, 0.1282, 0.1450, 0.1169, 0.1037
zero-token energy: 0.0
random-token component energy: 0.1743
shuffled-token MSE: 0.0420
```

## Interpretation

- Direct epsilon-band is endpoint-positive on CIFAR but unusable as ordered
  denoising: almost all energy goes into the last groups and early prefixes
  are far from clean.
- Adding a small clean-prefix/monotonic hybrid to epsilon-band improves energy
  distribution but destroys endpoint quality. This is not a useful branch.
- Denoise-path is the first objective that gives both:
  - competitive endpoint quality;
  - broad token utilization under restricted `S_k`.
- Clean-MSE is no longer the right metric for early prefixes under denoise-path,
  because early prefixes intentionally represent partial denoising rather than
  immediate clean reconstruction. Use `prefix_mse_to_denoise_path` alongside
  final clean MSE.
- The zero-token diagnostic remains exact, supporting `S_k(0)=0`. Random-token
  energy is nonzero but does not by itself show semantic decoding; shuffled MSE
  remains a required diagnostic for later larger runs.

## Decision

Promote `denoise-path` to the current main objective for the next MVP round.
Do not continue tuning direct epsilon-band or scalar tail-stage schedules until
denoise-path has controlled repeats.

Next steps:

- run seed repeats for CIFAR-10 and Tiny ImageNet denoise-path;
- compare against same-step channel-mask controls under the new
  `prefix_mse_to_denoise_path` metric;
- add a denoise-path progress-power sweep, likely `0.75`, `1.0`, and `1.5`;
- if repeats hold, run a longer Tiny ImageNet diagnostic before considering
  downsampled ImageNet-64.

## Follow-Up: Seed Repeat

Purpose: test whether the denoise-path result is stable across a second seed
before tuning progress schedules.

Configuration:

```text
CIFAR-10 seed2:
  config: train_cifar10_k8_denoisepath_3k_seed2_cuda.json
  seed: 149
  steps: 3000

Tiny ImageNet seed2:
  config: train_tiny_imagenet_k8_denoisepath_2k_seed2_cuda.json
  seed: 151
  steps: 2000
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_3k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_3k_seed2_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_2k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_2k_seed2_2026-07-08
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_denoisepath_3k_seed2_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_denoisepath_3k_seed2_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_denoisepath_2k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_2k_seed2_2026-07-08/prefix_final.png
```

| dataset | seed | epsilon | final clean MSE | prefix 4 path MSE | prefix 7 path MSE | tail ratio | late-half ratio | active tail | shuffled MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | seed1 | 0.0530 | 0.1709 | 0.0624 | 0.1455 | 0.9055 | 0.4825 | 7 | 0.0397 |
| CIFAR-10 | seed2 | 0.0300 | 0.1616 | 0.0584 | 0.1238 | 0.8745 | 0.4861 | 7 | 0.0378 |
| Tiny ImageNet | seed1 | 0.0537 | 0.2263 | 0.1009 | 0.1716 | 0.8678 | 0.4938 | 7 | 0.0420 |
| Tiny ImageNet | seed2 | 0.0542 | 0.2165 | 0.0852 | 0.1717 | 0.9045 | 0.4939 | 7 | 0.0412 |

Two-seed means:

| dataset | mean epsilon | mean final clean MSE | mean late-half ratio | mean active tail |
|---|---:|---:|---:|---:|
| CIFAR-10 | 0.0415 | 0.1663 | 0.4843 | 7.0 |
| Tiny ImageNet | 0.0540 | 0.2214 | 0.4938 | 7.0 |

Interpretation:

- Denoise-path is stable across this two-seed smoke on both datasets.
- Unlike v0 tail-stage training, late-token utilization is not a fragile
  afterthought: all seven tail tokens are active in all four runs.
- Endpoint quality remains better than the earlier channel-mask controls while
  energy is broadly distributed.

Updated decision:

Proceed to a small progress-power sweep. The most useful next comparison is
Tiny ImageNet seed1 with `denoise_path_progress_power` values:

```text
0.75: more aggressive early progress
1.00: current default
1.50: slower early progress, stronger late refinement
```

Success criterion: keep final clean MSE near or below 0.23 while improving the
shape of `prefix_mse_to_denoise_path` or visual prefix progression.

## Follow-Up: Tiny Progress-Power Sweep

Purpose: test whether the denoise path should move quickly early or leave more
progress to later tokens.

Configuration:

```text
dataset: tiny_imagenet_200
seed: 139
steps: 2000
base loss: epsilon 1.0, denoise_path_prefix 0.5, denoise_path_component 1.0
progress powers: 0.75, 1.00, 1.50
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p075_2k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p075_2k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_2k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_2k_2026-07-08
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p075_2k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p075_2k_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_2k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_2k_2026-07-08/prefix_final.png
```

| progress power | epsilon | final clean MSE | prefix 4 path MSE | prefix 6 path MSE | prefix 7 path MSE | tail ratio | late-half ratio | active tail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.0546 | 0.2428 | 0.1307 | 0.1572 | 0.1905 | 0.7087 | 0.3264 | 7 |
| 1.00 | 0.0537 | 0.2263 | 0.1009 | 0.1336 | 0.1716 | 0.8678 | 0.4938 | 7 |
| 1.50 | 0.0530 | 0.2162 | 0.0662 | 0.1016 | 0.1502 | 0.9754 | 0.7275 | 7 |

Path-MSE curves:

```text
p=0.75:
0.1271, 0.1439, 0.1254, 0.1307, 0.1339, 0.1572, 0.1905, 0.2428

p=1.00:
0.0680, 0.0943, 0.0907, 0.1009, 0.1065, 0.1336, 0.1716, 0.2263

p=1.50:
0.0211, 0.0418, 0.0570, 0.0662, 0.0767, 0.1016, 0.1502, 0.2162
```

Interpretation:

- `progress_power=1.5` is best in this sweep: endpoint quality improves and the
  denoise-path MSE curve is lower at every prefix.
- The tradeoff is stronger late emphasis: late-half energy rises to 72.75%.
  This is acceptable for the current hypothesis because all tail tokens remain
  active and the visual prefix sequence still progresses gradually.
- `progress_power=0.75` is the wrong direction for Tiny: it pushes too much
  progress early and hurts endpoint quality.

Updated decision:

Use `denoise_path_progress_power=1.5` as the current Tiny ImageNet candidate.
Next step is a seed repeat of `p=1.5` on Tiny and a matching CIFAR run. If this
holds, update the main configs to make `p=1.5` the default MVP setting.

## Follow-Up: Progress-Power 1.5 Seed Check

Purpose: confirm whether `progress_power=1.5` remains better after a second
seed on Tiny and after matching CIFAR runs.

Configuration:

```text
CIFAR-10:
  train_cifar10_k8_denoisepath_p150_3k_cuda.json, seed 137
  train_cifar10_k8_denoisepath_p150_3k_seed2_cuda.json, seed 149

Tiny ImageNet:
  train_tiny_imagenet_k8_denoisepath_p150_2k_cuda.json, seed 139
  train_tiny_imagenet_k8_denoisepath_p150_2k_seed2_cuda.json, seed 151
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_p150_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_3k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_p150_3k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_3k_seed2_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_2k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_2k_seed2_2026-07-08
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_denoisepath_p150_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_denoisepath_p150_3k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_2k_seed2_2026-07-08/report.json
```

Per-run comparison:

| dataset | power | seed | epsilon | final clean MSE | prefix 4 path MSE | prefix 7 path MSE | late-half ratio | active tail |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 1.0 | seed1 | 0.0530 | 0.1709 | 0.0624 | 0.1455 | 0.4825 | 7 |
| CIFAR-10 | 1.0 | seed2 | 0.0300 | 0.1616 | 0.0584 | 0.1238 | 0.4861 | 7 |
| CIFAR-10 | 1.5 | seed1 | 0.0518 | 0.1664 | 0.0463 | 0.1398 | 0.7219 | 7 |
| CIFAR-10 | 1.5 | seed2 | 0.0299 | 0.1574 | 0.0442 | 0.1171 | 0.7252 | 7 |
| Tiny ImageNet | 1.0 | seed1 | 0.0537 | 0.2263 | 0.1009 | 0.1716 | 0.4938 | 7 |
| Tiny ImageNet | 1.0 | seed2 | 0.0542 | 0.2165 | 0.0852 | 0.1717 | 0.4939 | 7 |
| Tiny ImageNet | 1.5 | seed1 | 0.0530 | 0.2162 | 0.0662 | 0.1502 | 0.7275 | 7 |
| Tiny ImageNet | 1.5 | seed2 | 0.0520 | 0.2096 | 0.0612 | 0.1530 | 0.7262 | 7 |

Two-seed means:

| dataset | power | mean epsilon | mean final clean MSE | mean prefix 4 path MSE | mean prefix 7 path MSE | mean late-half ratio |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 1.0 | 0.0415 | 0.1663 | 0.0604 | 0.1346 | 0.4843 |
| CIFAR-10 | 1.5 | 0.0409 | 0.1619 | 0.0452 | 0.1285 | 0.7236 |
| Tiny ImageNet | 1.0 | 0.0540 | 0.2214 | 0.0930 | 0.1717 | 0.4938 |
| Tiny ImageNet | 1.5 | 0.0525 | 0.2129 | 0.0637 | 0.1516 | 0.7269 |

Interpretation:

- `progress_power=1.5` improves mean endpoint quality and path-MSE on both
  CIFAR-10 and Tiny ImageNet.
- It consistently shifts more work to later tokens. This is a useful bias for
  CoFiTok because the tail remains active rather than collapsing to one token.
- Compared with the earlier v0 tail-stage objective, this is a cleaner result:
  late-token utilization and endpoint quality move in the same direction.

Updated decision:

Promote `denoise_path_progress_power=1.5` to the current MVP default. The next
substantial experiment should be a longer Tiny ImageNet run with this objective
and same-step channel-mask control, not another scalar loss sweep.

## Follow-Up: Tiny 5k Same-Step Control

Purpose: test the current MVP candidate against the old channel-mask objective
at the same seed and step count on the first main dataset.

Configuration:

```text
dataset: tiny_imagenet_200
seed: 139
steps: 5000

control:
  train_tiny_imagenet_k8_channelmask_5k_cuda.json

candidate:
  train_tiny_imagenet_k8_denoisepath_p150_5k_cuda.json
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_5k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_5k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_2026-07-08
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_5k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_5k_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_5k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_5k_2026-07-08/prefix_final.png
```

| run | epsilon | final clean MSE | prefix 4 clean MSE | prefix 7 clean MSE | prefix 4 path MSE | prefix 7 path MSE | tail ratio | late-half ratio | active tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| channel-mask 2k | 0.1427 | 0.3007 | 0.3338 | 0.3183 | n/a | n/a | 0.0246 | 0.0009 | 1 |
| denoise-path p1.5 2k | 0.0530 | 0.2162 | 5.2656 | 0.6407 | 0.0662 | 0.1502 | 0.9754 | 0.7275 | 7 |
| channel-mask 5k | 0.0655 | 0.2528 | 0.3018 | 0.2801 | 3.2767 | 0.4423 | 0.0204 | 0.0005 | 1 |
| denoise-path p1.5 5k | 0.0416 | 0.1723 | 5.1275 | 0.6079 | 0.0287 | 0.0948 | 0.9782 | 0.7295 | 7 |

Component energy ratios:

```text
channel-mask 5k:
0.9796, 0.0193, 0.0004, 0.0001, 0.0001, 0.0001, 0.0001, 0.0003

denoise-path p1.5 5k:
0.0218, 0.0373, 0.0982, 0.1131, 0.1463, 0.1744, 0.1785, 0.2304
```

Interpretation:

- This is the strongest Tiny ImageNet result so far. At equal 5k steps,
  denoise-path p1.5 improves final clean MSE from 0.2528 to 0.1723.
- Denoise-path also keeps all tail tokens active and distributes energy across
  the sequence; channel-mask still collapses almost entirely into token 1.
- The denoise-path prefix images remain intentionally noisy before late tokens,
  so early clean-MSE is not the right measure. The path-MSE curve improves
  substantially from 2k to 5k.
- Visual inspection agrees with the metrics: channel-mask looks smoother early
  but does not recover structure as well; denoise-path reveals structure through
  later prefixes and has a stronger final row.

Updated decision:

Treat `denoise_path_progress_power=1.5` as the current CoFiTok MVP objective.
The next paper-facing diagnostic should add:

- repeat Tiny 5k p1.5 and channel-mask 5k on seed2;
- run CIFAR p1.5 5k only if needed for a matched small-data table;
- add quantitative prefix-path AUC and energy-entropy metrics to reports so
  ordered utilization is summarized by one scalar rather than only raw arrays.

## Follow-Up: Tiny 5k Seed2 With AUC/Entropy Reports

Purpose: repeat the 5k Tiny ImageNet comparison with a second seed and add
compact scalar summaries for prefix-path fit and energy spread.

Code change:

- Added `src/cofitok/metrics.py`.
- `scripts/train_short.py` now writes:
  - `prefix_mse_to_clean_auc`
  - `prefix_mse_to_denoise_path_auc`
  - `energy_entropy`
  - `energy_entropy_normalized`
  - `energy_effective_token_count`

Validation:

```text
pytest: 22 passed
python -m py_compile scripts/train_short.py: passed
```

Configuration:

```text
dataset: tiny_imagenet_200
steps: 5000

seed2 control:
  train_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_cuda.json
  seed: 103
  objective: channel-mask clean-prefix baseline
  report path target: progress_power 1.5

seed2 candidate:
  train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_cuda.json
  seed: 103
  objective: denoise-path p1.5
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/prefix_final.png
```

Per-run comparison:

| run | epsilon | final clean MSE | path AUC | clean AUC | late-half ratio | active tail | energy entropy norm | effective tokens | zero energy | shuffled MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed1 channel-mask 5k | 0.0655 | 0.2528 | 3.3561 | 0.3116 | 0.0005 | 1 | 0.0507 | 1.111 | 0.0 | 0.2626 |
| seed1 denoise-path p1.5 5k | 0.0416 | 0.1723 | 0.0511 | 4.6713 | 0.7295 | 7 | 0.9195 | 6.767 | 0.0 | 0.0386 |
| seed2 channel-mask 5k | 0.1302 | 0.2736 | 4.9461 | 0.3185 | 0.0003 | 1 | 0.0730 | 1.164 | 0.0 | 0.2597 |
| seed2 denoise-path p1.5 5k | 0.0723 | 0.1781 | 0.0558 | 4.5968 | 0.7353 | 7 | 0.9217 | 6.798 | 0.0 | 0.0389 |

Two-seed means:

| method | mean epsilon | mean final clean MSE | mean path AUC | mean late-half ratio | mean energy entropy norm | mean effective tokens |
|---|---:|---:|---:|---:|---:|---:|
| channel-mask 5k | 0.0979 | 0.2632 | 4.1511 | 0.0004 | 0.0619 | 1.138 |
| denoise-path p1.5 5k | 0.0570 | 0.1752 | 0.0535 | 0.7324 | 0.9206 | 6.783 |

Interpretation:

- The 5k Tiny advantage survives the second seed. Denoise-path p1.5 improves
  final clean MSE by about 33% relative to channel-mask in the two-seed mean.
- Path AUC is orders of magnitude lower for denoise-path, which means prefixes
  track the intended partial denoising path rather than merely looking smooth.
- Energy entropy confirms the qualitative observation: channel-mask uses about
  one effective token, while denoise-path uses nearly seven effective tokens.
- Zero-token remains exact. The shuffled-token MSE is much smaller for
  denoise-path because its components are high-energy and aligned to the
  denoise path; this diagnostic should be complemented with visual shuffled
  grids or a normalized mismatch score before being used as a paper claim.

Updated decision:

This is now the first paper-facing diagnostic candidate:

```text
Tiny ImageNet, K=8, restricted S_k, 5k steps, two seeds:
denoise-path p1.5 beats channel-mask on endpoint quality and ordered token
utilization while preserving S_k(0)=0.
```

Next concrete step:

- add a normalized shuffle diagnostic and visual shuffled grid;
- run reverse-order / random-order evaluation from the trained p1.5 checkpoint
  without retraining first;
- then decide whether a full retrained reverse/random/simultaneous ablation is
  worth the GPU time.

## Follow-Up: Checkpoint Order Evaluation

Purpose: evaluate whether the trained component order matters without retraining
a separate ablation model. This is not a replacement for full retrained
reverse/random baselines, but it tests whether the learned components are
aligned to the intended prefix path.

Checkpoint:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/checkpoint_final.pt
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/checkpoint_final.pt \
  --component-order ordered \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_ordered_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/checkpoint_final.pt \
  --component-order reverse \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_reverse_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_seed2_2026-07-08/checkpoint_final.pt \
  --component-order random \
  --random-order-seed 0 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_random0_2026-07-08
```

Local copies:

```text
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_ordered_2026-07-08/report.json
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_ordered_2026-07-08/prefix_final.png
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_ordered_2026-07-08/prefix_final_shuffled.png
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_reverse_2026-07-08/report.json
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_seed2_random0_2026-07-08/report.json
```

| component order | indices | final clean MSE | prefix 2 path MSE | prefix 4 path MSE | prefix 6 path MSE | prefix 7 path MSE | path AUC | clean AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ordered | 0,1,2,3,4,5,6,7 | 0.1685 | 0.0192 | 0.0363 | 0.0663 | 0.0958 | 0.0543 | 4.5782 |
| reverse | 7,6,5,4,3,2,1,0 | 0.1685 | 0.6169 | 1.0842 | 0.7013 | 0.2932 | 0.6744 | 2.2045 |
| random0 | 4,0,7,3,2,5,1,6 | 0.1685 | 0.0973 | 0.3163 | 0.2529 | 0.1100 | 0.2141 | 3.3125 |

Normalized shuffle diagnostics from the ordered eval:

```text
shuffled_component_relative_mse: 1.9952
shuffled_final_clean_mse: 23.6559
shuffled_final_mse_ratio: 140.4135
zero_token_component_energy: 0.0
energy_entropy_normalized: 0.9221
energy_effective_token_count: 6.804
```

Interpretation:

- The final full-prefix MSE is identical across ordered/reverse/random because
  all components are summed at `m=K`.
- Prefix behavior is strongly order-dependent. Ordered accumulation gives the
  best path AUC; reverse is much worse, and random is between them.
- The shuffled visual grid does not recover the original sample structure, and
  normalized shuffle metrics show severe mismatch. This strengthens the claim
  that sample-specific information is in the tokens rather than in a fixed
  synthesis prior.

Updated decision:

Use the checkpoint order evaluation as a cheap diagnostic figure/table, but do
not overclaim it as the full order ablation. The next full ablation should train
at least one no-order/simultaneous baseline or add a trainer option that removes
the path-order supervision while keeping the same `S_k`.

## Follow-Up: Epsilon-Only No-Path Ablation

Purpose: test whether endpoint quality and ordered prefix controllability emerge
from final epsilon prediction alone. This keeps the same K8 model and restricted
`S_k`, but removes clean-prefix, monotonic, and denoise-path supervision.

Configuration:

```text
dataset: tiny_imagenet_200
steps: 5000
model: same K8 channel-mask restricted S_k
report path target: progress_power 1.5

seed1:
  train_tiny_imagenet_k8_epsilononly_p150eval_5k_cuda.json
  seed: 139

seed2:
  train_tiny_imagenet_k8_epsilononly_p150eval_5k_seed2_cuda.json
  seed: 103
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_epsilononly_p150eval_5k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_epsilononly_p150eval_5k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_epsilononly_p150eval_5k_seed2_2026-07-08
```

Fair p1.5 eval reports for historical seed1 controls:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_tiny_imagenet_k8_channelmask_p150eval_5k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_5k_2026-07-08/checkpoint_final.pt \
  --component-order ordered \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_tiny_imagenet_k8_channelmask_p150eval_5k_ordered_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_5k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_5k_2026-07-08/checkpoint_final.pt \
  --component-order ordered \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_tiny_imagenet_k8_denoisepath_p150_5k_ordered_2026-07-08
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_epsilononly_p150eval_5k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_epsilononly_p150eval_5k_seed2_2026-07-08/prefix_final.png
artifacts/reports/eval_tiny_imagenet_k8_channelmask_p150eval_5k_ordered_2026-07-08/report.json
artifacts/reports/eval_tiny_imagenet_k8_denoisepath_p150_5k_ordered_2026-07-08/report.json
artifacts/reports/eval_tiny_imagenet_k8_channelmask_p150eval_5k_seed2_ordered_2026-07-08/report.json
```

Per-run comparison:

| method | seed | final clean MSE | path AUC | clean AUC | late-half ratio | active tail | entropy norm | effective tokens | shuffle relative MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| channel-mask | seed1 | 0.2550 | 4.8189 | 0.3141 | 0.0005 | 1 | 0.0518 | 1.114 | 1.9883 |
| channel-mask | seed2 | 0.2626 | 4.9558 | 0.3086 | 0.0003 | 1 | 0.0721 | 1.162 | 1.9974 |
| epsilon-only | seed1 | 0.1629 | 1.1171 | 6.9911 | 0.8196 | 7 | 0.8257 | 5.568 | 1.7190 |
| epsilon-only | seed2 | 0.1693 | 0.8459 | 6.4949 | 0.8818 | 7 | 0.8090 | 5.377 | 1.8147 |
| denoise-path p1.5 | seed1 | 0.1770 | 0.0524 | 4.6986 | 0.7297 | 7 | 0.9193 | 6.765 | 1.9841 |
| denoise-path p1.5 | seed2 | 0.1685 | 0.0543 | 4.5782 | 0.7348 | 7 | 0.9221 | 6.804 | 1.9952 |

Two-seed means:

| method | mean final clean MSE | mean path AUC | mean clean AUC | mean late-half ratio | mean entropy norm | mean effective tokens | mean shuffle relative MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel-mask | 0.2588 | 4.8873 | 0.3114 | 0.0004 | 0.0619 | 1.138 | 1.9929 |
| epsilon-only | 0.1661 | 0.9815 | 6.7430 | 0.8507 | 0.8174 | 5.472 | 1.7669 |
| denoise-path p1.5 | 0.1727 | 0.0534 | 4.6384 | 0.7323 | 0.9207 | 6.784 | 1.9896 |

Interpretation:

- Epsilon-only is a strong endpoint baseline. It slightly beats denoise-path
  p1.5 on final clean MSE in this 5k Tiny setup.
- Endpoint quality alone does not yield the intended ordered partial denoising:
  epsilon-only path AUC is about 18x worse than denoise-path p1.5.
- Epsilon-only does activate tail tokens, but less uniformly than denoise-path
  and with a much weaker match to the prescribed denoising path.
- This clarifies the tradeoff: denoise-path p1.5 is not simply a better final
  epsilon predictor; it is the objective that makes prefixes meaningful and
  keeps token utilization broad.

Updated decision:

For paper framing, separate the claims:

```text
Quality: epsilon-only and denoise-path both beat clean-prefix channel-mask.
Prefix controllability/order: denoise-path is decisively better than
epsilon-only.
```

Next step:

- add an endpoint-preserving hybrid using epsilon-only strength plus a lighter
  denoise-path term, e.g. `denoise_path_prefix_weight=0.15` and
  `denoise_path_component_weight=0.3`, with `progress_power=1.5`;
- evaluate whether it keeps epsilon-only endpoint quality while recovering a
  low path AUC.

## Follow-Up: Light Denoise-Path Hybrid

Purpose: recover more of epsilon-only endpoint quality while retaining the path
alignment and broad token utilization of full denoise-path.

Configuration:

```text
dataset: tiny_imagenet_200
steps: 5000
progress_power: 1.5
epsilon_weight: 1.0
denoise_path_prefix_weight: 0.15
denoise_path_component_weight: 0.3

seed1:
  train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json
  seed: 139

seed2:
  train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_cuda.json
  seed: 103
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_2026-07-08
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_light_5k_2026-07-08/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_2026-07-08/report.json
artifacts/reports/train_tiny_imagenet_k8_denoisepath_p150_light_5k_seed2_2026-07-08/prefix_final.png
```

Per-run comparison:

| method | seed | final clean MSE | path AUC | clean AUC | late-half ratio | entropy norm | effective tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| channel-mask | seed1 | 0.2550 | 4.8189 | 0.3141 | 0.0005 | 0.0518 | 1.114 |
| channel-mask | seed2 | 0.2626 | 4.9558 | 0.3086 | 0.0003 | 0.0721 | 1.162 |
| epsilon-only | seed1 | 0.1629 | 1.1171 | 6.9911 | 0.8196 | 0.8257 | 5.568 |
| epsilon-only | seed2 | 0.1693 | 0.8459 | 6.4949 | 0.8818 | 0.8090 | 5.377 |
| full denoise-path | seed1 | 0.1770 | 0.0524 | 4.6986 | 0.7297 | 0.9193 | 6.765 |
| full denoise-path | seed2 | 0.1685 | 0.0543 | 4.5782 | 0.7348 | 0.9221 | 6.804 |
| light denoise-path | seed1 | 0.1664 | 0.0759 | 4.6937 | 0.7267 | 0.9221 | 6.804 |
| light denoise-path | seed2 | 0.1746 | 0.0763 | 4.7182 | 0.7332 | 0.9249 | 6.843 |

Two-seed means:

| method | mean final clean MSE | mean path AUC | mean late-half ratio | mean entropy norm | mean effective tokens |
|---|---:|---:|---:|---:|---:|
| channel-mask | 0.2588 | 4.8873 | 0.0004 | 0.0619 | 1.138 |
| epsilon-only | 0.1661 | 0.9815 | 0.8507 | 0.8174 | 5.472 |
| full denoise-path | 0.1727 | 0.0534 | 0.7323 | 0.9207 | 6.784 |
| light denoise-path | 0.1705 | 0.0761 | 0.7300 | 0.9235 | 6.824 |

Interpretation:

- Light denoise-path is the best tradeoff so far. It nearly matches
  epsilon-only endpoint quality while retaining a low path AUC and high
  effective token count.
- Full denoise-path still has the best path AUC, but the light version gives
  a slightly better endpoint with only a small path-AUC cost.
- Epsilon-only remains important as a quality baseline: it shows that final
  denoising quality can be strong without ordered prefixes, so the CoFiTok
  claim should focus on the combination of endpoint quality and prefix
  controllability.

Updated decision:

Use `light denoise-path p1.5` as the current Tiny ImageNet MVP candidate:

```text
epsilon_weight = 1.0
denoise_path_prefix_weight = 0.15
denoise_path_component_weight = 0.3
denoise_path_progress_power = 1.5
```

Next step:

- run the same light objective on CIFAR-10 for a matched small-data table;
- then decide whether to move to `downsampled_imagenet_64` or first add
  retrained random/reverse/no-path baselines.

## Follow-Up: CIFAR-10 Matched Table

Purpose: verify whether the Tiny ImageNet light-denoise-path tradeoff transfers
to a second common dataset under matched seeds and metrics.

Configuration:

```text
dataset: cifar10
steps: 3000
seeds: 137, 149
model: same K8 channel-mask restricted S_k
report path target: progress_power 1.5
```

Methods:

```text
channel-mask:
  clean-prefix + monotonic baseline
  train_cifar10_k8_channelmask_p150eval_3k_cuda.json
  train_cifar10_k8_channelmask_p150eval_3k_seed2_cuda.json

epsilon-only:
  final epsilon loss only
  train_cifar10_k8_epsilononly_p150eval_3k_cuda.json
  train_cifar10_k8_epsilononly_p150eval_3k_seed2_cuda.json

full denoise-path:
  denoise_path_prefix_weight = 0.5
  denoise_path_component_weight = 1.0
  train_cifar10_k8_denoisepath_p150_3k_cuda.json
  train_cifar10_k8_denoisepath_p150_3k_seed2_cuda.json

light denoise-path:
  denoise_path_prefix_weight = 0.15
  denoise_path_component_weight = 0.3
  train_cifar10_k8_denoisepath_p150_light_3k_cuda.json
  train_cifar10_k8_denoisepath_p150_light_3k_seed2_cuda.json
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_channelmask_p150eval_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_p150eval_3k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_channelmask_p150eval_3k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_p150eval_3k_seed2_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_epsilononly_p150eval_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilononly_p150eval_3k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_epsilononly_p150eval_3k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_epsilononly_p150eval_3k_seed2_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_denoisepath_p150_light_3k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_3k_seed2_2026-07-08
```

Full denoise-path fair eval:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_cifar10_k8_denoisepath_p150_3k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_3k_2026-07-08/checkpoint_final.pt \
  --component-order ordered \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_cifar10_k8_denoisepath_p150_3k_ordered_2026-07-08

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/evaluate_checkpoint.py \
  --config configs/train_cifar10_k8_denoisepath_p150_3k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_3k_seed2_2026-07-08/checkpoint_final.pt \
  --component-order ordered \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/eval_cifar10_k8_denoisepath_p150_3k_seed2_ordered_2026-07-08
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_channelmask_p150eval_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_channelmask_p150eval_3k_seed2_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_epsilononly_p150eval_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_epsilononly_p150eval_3k_seed2_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08/report.json
artifacts/reports/train_cifar10_k8_denoisepath_p150_light_3k_seed2_2026-07-08/report.json
artifacts/reports/eval_cifar10_k8_denoisepath_p150_3k_ordered_2026-07-08/report.json
artifacts/reports/eval_cifar10_k8_denoisepath_p150_3k_seed2_ordered_2026-07-08/report.json
```

Per-run comparison:

| method | seed | final clean MSE | path AUC | clean AUC | late-half ratio | active tail | entropy norm | effective tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| channel-mask | seed1 | 0.2016 | 4.9586 | 0.2279 | 0.0003 | 1 | 0.0398 | 1.086 |
| channel-mask | seed2 | 0.2202 | 5.0198 | 0.2587 | 0.0006 | 1 | 0.0442 | 1.096 |
| epsilon-only | seed1 | 0.1566 | 0.8030 | 6.5123 | 0.8528 | 7 | 0.7965 | 5.240 |
| epsilon-only | seed2 | 0.1417 | 1.3022 | 7.2044 | 0.8830 | 6 | 0.7890 | 5.159 |
| full denoise-path | seed1 | 0.1613 | 0.0707 | 4.6302 | 0.7218 | 7 | 0.9237 | 6.826 |
| full denoise-path | seed2 | 0.1581 | 0.0699 | 4.7250 | 0.7261 | 7 | 0.9205 | 6.782 |
| light denoise-path | seed1 | 0.1580 | 0.1003 | 4.7045 | 0.7175 | 7 | 0.9242 | 6.834 |
| light denoise-path | seed2 | 0.1488 | 0.0960 | 4.7936 | 0.7254 | 7 | 0.9252 | 6.847 |

Two-seed means:

| method | mean final clean MSE | mean path AUC | mean late-half ratio | mean entropy norm | mean effective tokens |
|---|---:|---:|---:|---:|---:|
| channel-mask | 0.2109 | 4.9892 | 0.0004 | 0.0420 | 1.091 |
| epsilon-only | 0.1492 | 1.0526 | 0.8679 | 0.7928 | 5.200 |
| full denoise-path | 0.1597 | 0.0703 | 0.7240 | 0.9221 | 6.804 |
| light denoise-path | 0.1534 | 0.0982 | 0.7215 | 0.9247 | 6.841 |

Interpretation:

- CIFAR-10 matches the Tiny ImageNet pattern.
- Epsilon-only remains the strongest endpoint-only baseline but has poor path
  alignment compared with either denoise-path objective.
- Full denoise-path gives the best path AUC.
- Light denoise-path is again the best endpoint/path tradeoff: it nearly
  matches epsilon-only final quality while retaining low path AUC and broad
  effective token usage.

Updated decision:

`light denoise-path p1.5` is now the current cross-dataset MVP candidate across
CIFAR-10 and Tiny ImageNet. The next milestone should either:

- run a 64x64 public validation dataset (`downsampled_imagenet_64`) if storage
  and loader setup are ready; or
- add retrained reverse/random/no-path baselines on Tiny only, to strengthen the
  ordering claim before scaling data.

## P2 Downsampled ImageNet 64 Loader Preparation

Purpose:

Prepare the ImageNet-64 validation path without starting a large download before
the export environment is recorded.

Implementation:

```text
src/cofitok/data/registry.py
  Added PreparedImageDataset.
  Added dataset alias downsampled_imagenet_64.
  Loader prefers split manifests when present and falls back to recursive scan.
  Expected split layout:
    <data_root>/downsampled_imagenet_64/extracted/train/<class>/*
    <data_root>/downsampled_imagenet_64/extracted/validation/<class>/*

scripts/prepare_downsampled_imagenet64_tfds.py
  Exports TFDS downsampled_imagenet/64x64 into the prepared layout.
  Writes per-split manifest JSONL and manifest_summary.json.
  Treats missing labels as class folder unlabeled.
  Defaults to 10k-image nested shards, e.g. unlabeled/shard_00000/.

configs/smoke_downsampled_imagenet64_cuda.json
configs/train_downsampled_imagenet64_k8_channelmask_p150eval_5k_cuda.json
configs/train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda.json
configs/train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json
```

Validation:

```text
remote pytest: 26 passed
py_compile:
  scripts/prepare_downsampled_imagenet64_tfds.py
  scripts/train_short.py
  scripts/evaluate_checkpoint.py
prepare script --help: passed
```

Temporary loader smoke:

```text
root:
  /root/autodl-tmp/CoFiTok/tmp/downsampled_imagenet64_loader_smoke
method:
  copied 8 Tiny ImageNet images into a fake prepared downsampled_imagenet_64
  train/validation/unlabeled layout
command:
  PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/smoke_forward.py \
    --config /root/autodl-tmp/CoFiTok/tmp/downsampled_imagenet64_loader_smoke/smoke_config.json \
    --output /root/autodl-tmp/CoFiTok/tmp/downsampled_imagenet64_loader_smoke/smoke_report.json
local archive:
  artifacts/reports/smoke/downsampled_imagenet64_loader_smoke.json
result:
  actual_device: cuda
  epsilon output shape: [8, 3, 64, 64]
  component shapes: 4 x [8, 3, 64, 64]
  zero_token_component_energy_ratio: 0.0
```

Export environment:

```text
path:
  /root/autodl-tmp/CoFiTok/envs/tfds-export
size:
  313M
key package:
  tensorflow-datasets==4.9.10
inspect-only metadata:
  /root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64_tfds_inspect.json
freeze archive:
  docs/experiment_conditions/downsampled_imagenet_64_tfds_export_freeze_2026-07-08.txt
```

Decision:

The P2 training code path is ready. Do not start full P2 training until the
actual TFDS export has been completed and recorded in
`docs/experiment_conditions/downsampled_imagenet_64_plan_2026-07-08.md`.

## P2 Source Fallback

TFDS `downsampled_imagenet/64x64` source status:

```text
full attempt:
  scripts/prepare_downsampled_imagenet64_tfds.py
log:
  /root/autodl-tmp/CoFiTok/logs/prepare_downsampled_imagenet64_2026-07-08.log
result:
  failed before data download
error:
  https://image-net.org/small/train_64x64.tar returned HTTP 404
```

Academic Torrents status:

```text
source:
  https://academictorrents.com/download/96816a530ee002254d29bf7a61c0c158d3dedc3b
direct:
  timed out
network_turbo:
  returned Squid 503 HTML
```

HF fallback implemented:

```text
alias:
  imagenet_1k_64x64_hf
source:
  benjamin-paine/imagenet-1k-64x64
script:
  scripts/prepare_hf_image_dataset.py
configs:
  configs/smoke_imagenet_1k_64x64_hf_cuda.json
  configs/train_imagenet_1k_64x64_hf_k8_channelmask_p150eval_5k_cuda.json
  configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_cuda.json
  configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json
record:
  docs/experiment_conditions/imagenet_1k_64x64_hf_plan_2026-07-08.md
```

HF API probe:

```text
splits:
  train / validation / test
rows:
  train 1,281,167
  validation 50,000
  test 100,000
parquet bytes:
  total 1,939,600,412
temporary export:
  8 train + 8 validation rows
temporary smoke:
  artifacts/reports/smoke/imagenet_1k_64x64_hf_probe/smoke_report.json
result:
  actual_device cuda
  epsilon shape [8, 3, 64, 64]
  zero_token_component_energy_ratio 0.0
```

Formal HF export:

```text
launcher:
  scripts/run_prepare_imagenet_1k_64x64_hf.sh
output:
  /root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf/extracted
rows:
  train 1,281,167
  validation 50,000
size:
  14G
file count under extracted:
  1,331,170
formal smoke:
  /root/autodl-tmp/CoFiTok/checkpoints/smoke/imagenet_1k_64x64_hf_smoke.json
formal smoke result:
  actual_device cuda
  epsilon shape [8, 3, 64, 64]
  zero_token_component_energy_ratio 0.0
```

Decision:

Proceed with `imagenet_1k_64x64_hf` as an ImageNet-family 64x64 P2 fallback
only with explicit naming. Do not present it as the exact TFDS
`downsampled_imagenet_64` dataset.

## HF ImageNet-1K 64x64 P2 Table

Dataset:

```text
alias: imagenet_1k_64x64_hf
source: benjamin-paine/imagenet-1k-64x64
train rows: 1,281,167
validation rows: 50,000
prepared size: 14G
```

Commands used the matching 5k configs:

```text
train_imagenet_1k_64x64_hf_k8_channelmask_p150eval_5k_cuda.json
train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_cuda.json
train_imagenet_1k_64x64_hf_k8_denoisepath_p150_5k_cuda.json
train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json
```

Local report archives:

```text
artifacts/reports/train_imagenet_1k_64x64_hf_k8_channelmask_p150eval_5k_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_5k_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_2026-07-08/report.json
```

Single-seed comparison:

| method | final clean MSE | path AUC | clean AUC | late-half ratio | active tail | entropy norm | effective tokens | shuffled final ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| channel-mask | 0.1230 | 4.8844 | 0.1546 | 0.0004 | 1 | 0.0508 | 1.111 | 200.1 |
| epsilon-only | 0.0998 | 1.1000 | 6.9691 | 0.8439 | 7 | 0.8056 | 5.339 | 238.0 |
| full denoise-path | 0.1074 | 0.0417 | 4.6454 | 0.7323 | 7 | 0.9189 | 6.758 | 219.7 |
| light denoise-path | 0.1018 | 0.0620 | 4.6673 | 0.7322 | 7 | 0.9209 | 6.787 | 233.5 |

Interpretation:

- The CIFAR-10/Tiny ImageNet pattern transfers to the ImageNet-family 64x64
  fallback.
- Epsilon-only is the best endpoint-only baseline, but its denoise-path AUC is
  much worse.
- Full denoise-path gives the best path AUC.
- Light denoise-path is again the best endpoint/path tradeoff: final MSE is
  close to epsilon-only while keeping low path AUC and broad token usage.

## HF ImageNet-1K 64x64 Seed-2 and Order Eval

Purpose:

Strengthen the P2 fallback evidence by repeating the two most important methods
and checking component order sensitivity for the current MVP candidate.

Seed-2 configs:

```text
configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_seed2_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_cuda.json
```

Seed-2 local archives:

```text
artifacts/reports/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_seed2_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_2026-07-08/report.json
```

Two-seed comparison for the endpoint/path tradeoff:

| method | seed | final clean MSE | path AUC | clean AUC | late-half ratio | effective tokens | shuffled final ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| epsilon-only | seed1 | 0.0998 | 1.1000 | 6.9691 | 0.8439 | 5.339 | 238.0 |
| epsilon-only | seed2 | 0.0940 | 0.8837 | 6.5713 | 0.8803 | 5.400 | 250.2 |
| light denoise-path | seed1 | 0.1018 | 0.0620 | 4.6673 | 0.7322 | 6.787 | 233.5 |
| light denoise-path | seed2 | 0.0963 | 0.0641 | 4.7108 | 0.7431 | 6.768 | 244.3 |

Two-seed means:

| method | mean final clean MSE | mean path AUC | mean late-half ratio | mean effective tokens | mean shuffled final ratio |
|---|---:|---:|---:|---:|---:|
| epsilon-only | 0.0969 | 0.9919 | 0.8621 | 5.369 | 244.1 |
| light denoise-path | 0.0990 | 0.0631 | 0.7376 | 6.777 | 238.9 |

Order-eval commands used `scripts/evaluate_checkpoint.py` on the light
denoise-path checkpoints with `component-order = ordered / reverse / random`.
The random order used `random_order_seed = 0`.

Order-eval local archives:

```text
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_ordered_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_reverse_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_random0_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_reverse_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_random0_2026-07-08/report.json
```

Order-eval per seed:

| seed | order | final clean MSE | path AUC | clean AUC | shuffled final ratio |
|---|---|---:|---:|---:|---:|
| seed1 | ordered | 0.1041 | 0.0635 | 4.6891 | 225.2 |
| seed1 | reverse | 0.1041 | 0.6947 | 2.1694 | 225.2 |
| seed1 | random0 | 0.1041 | 0.2441 | 3.3695 | 225.2 |
| seed2 | ordered | 0.0986 | 0.0647 | 4.6969 | 239.9 |
| seed2 | reverse | 0.0986 | 0.7406 | 2.0789 | 239.9 |
| seed2 | random0 | 0.0986 | 0.2605 | 3.2878 | 239.9 |

Order-eval means:

| order | mean final clean MSE | mean path AUC | mean clean AUC | mean shuffled final ratio |
|---|---:|---:|---:|---:|
| ordered | 0.1013 | 0.0641 | 4.6930 | 232.5 |
| random0 | 0.1013 | 0.2523 | 3.3286 | 232.5 |
| reverse | 0.1013 | 0.7177 | 2.1242 | 232.5 |

Interpretation:

- The endpoint/path tradeoff is stable across seeds.
- Ordered prefixes are substantially better aligned to the denoise path than
  random or reverse accumulation, while the full endpoint is unchanged because
  all components are still summed.
- Shuffled-token diagnostics remain strongly non-matching, and zero-token
  diagnostics remain zero.

## Simultaneous Predictor Ablation

Purpose:

Test whether the current evidence depends on explicit token-to-token feedback in
the tiny predictor, or whether ordered component heads plus denoise-path
supervision are sufficient.

Implementation:

```text
ModelConfig.predictor_use_feedback: bool = True by default
TinyTokenPredictor(use_feedback=False):
  disables token feedback layers
  all token heads read the same hidden state
  S_k and all synthesis restrictions remain unchanged
```

Config:

```text
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_cuda.json
```

Validation:

```text
pytest: 29 passed
```

Run:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_2026-07-08
```

Local archive:

```text
artifacts/reports/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_2026-07-08/report.json
```

Matched comparison against feedback light denoise-path seed1:

| variant | final clean MSE | path AUC | clean AUC | late-half ratio | effective tokens | shuffled final ratio | zero ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| light feedback | 0.1018 | 0.0620 | 4.6673 | 0.7322 | 6.787 | 233.5 | 0.0 |
| light simultaneous | 0.0995 | 0.0600 | 4.6128 | 0.7301 | 6.845 | 237.4 | 0.0 |

Order eval for the simultaneous checkpoint:

| order | final clean MSE | path AUC | clean AUC | shuffled final ratio | effective tokens |
|---|---:|---:|---:|---:|---:|
| ordered | 0.1025 | 0.0611 | 4.6329 | 227.3 | 6.846 |
| random0 | 0.1025 | 0.2242 | 3.4013 | 227.3 | 6.846 |
| reverse | 0.1025 | 0.6505 | 2.2069 | 227.3 | 6.846 |

Interpretation:

- In the current tiny predictor, explicit token feedback is not necessary for
  the denoise-path/light objective to produce ordered useful components.
- The safer paper claim should emphasize ordered dense-noise component
  factorization with restricted `S_k`, not that autoregressive token feedback is
  the only source of ordering.
- The simultaneous checkpoint still shows strong order sensitivity under
  reverse/random prefix accumulation, so the learned component order remains
  meaningful.

## Deep-`S_k` Degeneration Ablation

Purpose:

Test the risk called out in the project boundary: if `S_k` is allowed to become
a biased nonlinear decoder, endpoint/path metrics may improve while the
zero-token diagnostic fails.

Implementation:

```text
ModelConfig.synthesis_mode: "restricted" by default
ModelConfig.synthesis_mode: "deep_decoder" for ablation only
DeepSynthesis:
  input: current token only
  layers: biased nonlinear convolutions
  hidden channels: 32 in the HF ImageNet-64 fallback run
  depth: 4 in the HF ImageNet-64 fallback run
```

Validation:

```text
remote pytest: 35 passed
remote py_compile: passed for configs.py, cofitok.py, synthesis.py
```

Config:

```text
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda.json
```

Local archives:

```text
artifacts/reports/imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_ordered_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_random0_2026-07-08/report.json
artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_reverse_2026-07-08/report.json
```

Matched seed-1 comparison against restricted light denoise-path:

| variant | final clean MSE | path AUC | clean AUC | late-half ratio | effective tokens | zero ratio | random ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| restricted light | 0.1018 | 0.0620 | 4.6673 | 0.7322 | 6.787 | 0.0000 | 7.3871 |
| deep-`S_k` | 0.0977 | 0.0366 | 4.6510 | 0.7161 | 6.792 | 0.0522 | 1.3905 |

Order eval for deep-`S_k`:

| order | final clean MSE | path AUC | zero ratio | random ratio |
|---|---:|---:|---:|---:|
| ordered | 0.1022 | 0.0382 | 0.0521 | 1.3806 |
| random0 | 0.1022 | 0.1361 | 0.0521 | 1.3806 |
| reverse | 0.1022 | 0.6431 | 0.0521 | 1.3806 |

Interpretation:

- The strong synthesis ablation improves endpoint/path numbers, but it violates
  the hard interpretability constraint: zero tokens produce nonzero components.
- This is useful negative evidence. It shows why the main CoFiTok claim should
  keep `S_k` restricted, bias-free, and `S_k(0)=0` by construction.
- Order sensitivity still holds for deep-`S_k`, but that does not rescue it as a
  valid main operator because token-independent component priors are already
  present.

## Token Count Scaling on HF ImageNet-64 Fallback

Purpose:

Complete the first-stage `K = 4, 8, 16` comparison using the current best
restricted light denoise-path objective.

Configs:

```text
configs/train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json
configs/train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_cuda.json
```

Local archives:

```text
artifacts/reports/train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_2026-07-08/report.json
artifacts/reports/train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_2026-07-08/report.json
```

Training summary:

| K | final clean MSE | path AUC | clean AUC | effective tokens | late-half ratio | active tail | zero ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0988 | 0.0665 | 3.7977 | 3.318 | 0.7394 | 3 | 0.0 |
| 8 | 0.1018 | 0.0620 | 4.6673 | 6.787 | 0.7322 | 7 | 0.0 |
| 16 | 0.0992 | 0.0674 | 5.1033 | 13.560 | 0.7039 | 15 | 0.0 |

Order eval:

| K | ordered path AUC | random0 path AUC | reverse path AUC | eval final clean MSE |
|---:|---:|---:|---:|---:|
| 4 | 0.0674 | 0.0727 | 0.7516 | 0.1029 |
| 8 | 0.0635 | 0.2441 | 0.6947 | 0.1041 |
| 16 | 0.0680 | 0.4680 | 0.7020 | 0.1021 |

Interpretation:

- All three token counts preserve `S_k(0)=0`.
- K4 is a strong cost baseline and has the best train-batch endpoint, but random
  component order is only mildly worse than ordered; it has less room to express
  a rich prefix curriculum.
- K8 remains the current MVP sweet spot: best path AUC among restricted runs,
  broad token use, and clear random/reverse order sensitivity.
- K16 uses many tokens effectively, but does not improve path AUC after 5k
  steps. It likely needs longer training or stronger tail/objective shaping
  before it is worth promoting.

## Cross-Batch Quality Metrics

Purpose:

Add a validation quality layer beyond single-batch MSE/path diagnostics.

Implementation:

```text
script:
  scripts/evaluate_quality.py
metrics:
  prefix MSE
  prefix PSNR in normalized image range [-1, 1]
  low-res Fréchet proxy over deterministic 8x8 RGB/color features
optional:
  LPIPS if package is installed
environment:
  lpips not installed
  torchmetrics not installed
validation:
  pytest 38 passed
```

Important caveat:

The low-res Fréchet proxy is a small-scale relative distribution metric. It is
not Inception FID and should be reported only as a quick rFID-like proxy until a
formal FID/LPIPS stack is installed and recorded.

Evaluation setup:

```text
dataset: imagenet_1k_64x64_hf validation
images: 256
batches: 8
fixed timestep: 500
feature size: 8
prefix budgets: default 1,2,4,8,K
```

Local archives:

```text
artifacts/reports/quality_imagenet_hf_channelmask_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_epsilononly_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_k4_light_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_k8_light_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_k16_light_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_deepsk_256_t500_2026-07-08/quality_report.json
```

Final-prefix quality:

| variant | final MSE | final PSNR | final low-res Fréchet proxy | MSE AUC | PSNR AUC | Fréchet proxy AUC |
|---|---:|---:|---:|---:|---:|---:|
| channel-mask | 0.1749 | 13.593 | 3.9121 | 0.2382 | 12.485 | 5.0606 |
| epsilon-only | 0.1262 | 15.011 | 1.1015 | 8.2977 | -0.741 | 5.5060 |
| K4 light | 0.1267 | 14.994 | 1.1506 | 4.9529 | 2.293 | 4.5490 |
| K8 light | 0.1289 | 14.917 | 1.0864 | 6.7020 | 0.142 | 5.3458 |
| K16 light | 0.1265 | 14.999 | 1.1003 | 7.9052 | -1.013 | 5.7661 |
| deep-`S_k` | 0.1255 | 15.034 | 1.0278 | 6.7630 | 0.158 | 5.3838 |

Interpretation:

- Channel-mask remains clearly poor on final quality and distribution proxy.
- Epsilon-only has strong final PSNR, but its prefix curve remains poor because
  early prefixes are not trained to be meaningful.
- Among restricted light variants, K8 has the best final low-res Fréchet proxy
  on this 256-image validation slice, while K4 has the best proxy AUC because it
  has fewer prefix stages and less room for bad early prefixes.
- Deep-`S_k` again gives the best final quality numbers, but this must be read
  together with its failed zero-token diagnostic; better quality alone is not a
  valid CoFiTok result if `S_k(0)=0` is violated.

## Quality Metrics Across P0/P1/P2 Datasets

Purpose:

Apply the same 256-image quality protocol to the three current public datasets:
CIFAR-10, Tiny ImageNet-200, and the HF ImageNet-1K 64x64 fallback.

Local archives:

```text
artifacts/reports/quality_cifar10_channelmask_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_cifar10_epsilononly_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_cifar10_k8_light_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_tiny_channelmask_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_tiny_epsilononly_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_tiny_k8_light_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_channelmask_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_epsilononly_256_t500_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_k8_light_256_t500_2026-07-08/quality_report.json
```

Final-prefix quality and proxy-curve AUC:

| dataset | variant | final MSE | final PSNR | final proxy | proxy AUC | MSE AUC |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | channel-mask | 0.2034 | 12.938 | 4.3115 | 5.3941 | 0.2516 |
| CIFAR-10 | epsilon-only | 0.1456 | 14.390 | 1.5634 | 6.0239 | 8.1429 |
| CIFAR-10 | K8 light | 0.1531 | 14.170 | 1.8491 | 5.5970 | 6.7221 |
| Tiny ImageNet | channel-mask | 0.2026 | 12.955 | 4.3987 | 5.6925 | 0.2700 |
| Tiny ImageNet | epsilon-only | 0.1461 | 14.375 | 1.1641 | 5.3155 | 8.3620 |
| Tiny ImageNet | K8 light | 0.1487 | 14.297 | 1.1861 | 5.2576 | 6.7288 |
| ImageNet-64 HF | channel-mask | 0.1749 | 13.593 | 3.9121 | 5.0606 | 0.2382 |
| ImageNet-64 HF | epsilon-only | 0.1262 | 15.011 | 1.1015 | 5.5060 | 8.2977 |
| ImageNet-64 HF | K8 light | 0.1289 | 14.917 | 1.0864 | 5.3458 | 6.7020 |

Interpretation:

- The same broad tradeoff appears on all three datasets: epsilon-only is a
  strong endpoint baseline, while K8 light lowers prefix-curve error compared
  with epsilon-only.
- Channel-mask is consistently poor as a final-quality baseline.
- Tiny ImageNet and ImageNet-64 HF are the most relevant quality settings. On
  both, K8 light is close to epsilon-only on final quality and better on
  prefix-curve MSE AUC. On ImageNet-64 HF it also slightly improves the final
  low-res Fréchet proxy.
- CIFAR-10 is best treated as smoke/diagnostic evidence; it is less informative
  for visual distribution quality.

## Reproducible Experiment Summary Tables

Purpose:

Avoid hand-copying values from dozens of JSON reports. `scripts/summarize_experiments.py`
collects train, order-eval, quality, sampling, and generated-sample quality
reports into JSON/CSV/Markdown tables.

Validation:

```text
remote pytest: 47 passed
summary counts:
  train reports: 85
  order-eval reports: 26
  quality reports: 29
  sampling reports: 12
  generated-quality reports: 9
```

Artifacts:

```text
artifacts/reports/summary_2026-07-08/experiment_summary.json
artifacts/reports/summary_2026-07-08/train_summary.csv
artifacts/reports/summary_2026-07-08/order_eval_summary.csv
artifacts/reports/summary_2026-07-08/quality_summary.csv
artifacts/reports/summary_2026-07-08/sample_summary.csv
artifacts/reports/summary_2026-07-08/generated_quality_summary.csv
artifacts/reports/summary_2026-07-08/core_summary.md
```

Core tables included:

```text
ImageNet-64 HF Training Matrix:
  channel-mask, epsilon-only, full denoise-path, light denoise-path,
  simultaneous predictor, deep-S_k ablation
ImageNet-64 HF Order Matrix:
  K=4/8/16 ordered/random/reverse
Cross-Dataset Quality Matrix:
  CIFAR-10, Tiny ImageNet-200, ImageNet-64 HF
  channel-mask, epsilon-only, K8 light
Prefix-Aware Sampling Smoke Matrix:
  CIFAR-10, Tiny ImageNet-200, ImageNet-64 HF
  K8 light, DDIM-20, prefix budgets 1/4/8
Generated Sample Quality Smoke Matrix:
  CIFAR-10, Tiny ImageNet-200, ImageNet-64 HF
  K8 light, DDIM-20, 64 generated samples vs 256 real images
```

Interpretation:

The summary artifacts are now the preferred source for paper tables and future
comparisons. Individual report JSON files remain the source of truth; the
summary script is deterministic and can be rerun after new experiments.

Update after sampling and generated-sample quality:

```text
remote pytest: 47 passed
summary counts:
  train reports: 85
  order-eval reports: 26
  quality reports: 29
  sampling reports: 12
  generated-quality reports: 9
```

## Torchvision Inception Fréchet Metrics

Purpose:

Add a stronger FID-style distribution metric without installing new packages.
`evaluate_quality.py --enable-inception-fid` uses torchvision Inception-V3
ImageNet weights and computes Fréchet distance over 2048-dim pool features.

Environment record:

```text
docs/experiment_conditions/quality_metrics_inception_2026-07-08.md
```

Weight cache:

```text
TORCH_HOME=/root/autodl-tmp/CoFiTok/checkpoints/torch_cache
weight: inception_v3_google-0cc3c7bd.pth
size: 104M
sha256: 0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef
```

256-image validation results:

| dataset | variant | final MSE | final PSNR | low-res proxy | Inception Fréchet | Inception AUC |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | epsilon-only | 0.1456 | 14.390 | 1.5634 | 334.069 | 493.358 |
| CIFAR-10 | K8 light | 0.1531 | 14.170 | 1.8491 | 327.889 | 493.234 |
| Tiny ImageNet | epsilon-only | 0.1461 | 14.375 | 1.1641 | 346.726 | 383.444 |
| Tiny ImageNet | K8 light | 0.1487 | 14.297 | 1.1861 | 343.602 | 384.562 |
| ImageNet-64 HF | epsilon-only | 0.1262 | 15.011 | 1.1015 | 331.304 | 372.463 |
| ImageNet-64 HF | K8 light | 0.1289 | 14.917 | 1.0864 | 321.768 | 372.130 |

Interpretation:

- K8 light has slightly worse final MSE/PSNR than epsilon-only, as before.
- K8 light has lower final Inception Fréchet on all three datasets in this
  256-image fixed-timestep evaluation.
- The result supports the existing tradeoff claim: CoFiTok's light objective
  preserves endpoint distribution quality while greatly improving prefix/path
  controllability.

## Official LPIPS Metrics

Purpose:

Add the standard LPIPS Alex metric to the same 256-image quality protocol.

Environment changes:

```text
installed:
  lpips==0.1.4
  scipy==1.15.3
install mode:
  pip install --no-deps --index-url https://pypi.org/simple
record:
  docs/experiment_conditions/quality_metrics_inception_2026-07-08.md
```

LPIPS+Inception reports:

```text
artifacts/reports/quality_cifar10_epsilononly_256_t500_lpips_inception_2026-07-08/quality_report.json
artifacts/reports/quality_cifar10_k8_light_256_t500_lpips_inception_2026-07-08/quality_report.json
artifacts/reports/quality_tiny_epsilononly_256_t500_lpips_inception_2026-07-08/quality_report.json
artifacts/reports/quality_tiny_k8_light_256_t500_lpips_inception_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_epsilononly_256_t500_lpips_inception_2026-07-08/quality_report.json
artifacts/reports/quality_imagenet_hf_k8_light_256_t500_lpips_inception_2026-07-08/quality_report.json
```

Final-prefix metrics:

| dataset | variant | final PSNR | final LPIPS | LPIPS AUC | final Inception | Inception AUC |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | epsilon-only | 14.390 | 0.1639 | 0.3585 | 334.069 | 493.358 |
| CIFAR-10 | K8 light | 14.170 | 0.1674 | 0.3685 | 327.889 | 493.234 |
| Tiny ImageNet | epsilon-only | 14.375 | 0.5689 | 0.9264 | 346.726 | 383.444 |
| Tiny ImageNet | K8 light | 14.297 | 0.5652 | 0.9252 | 343.602 | 384.562 |
| ImageNet-64 HF | epsilon-only | 15.011 | 0.5354 | 0.9784 | 331.304 | 372.463 |
| ImageNet-64 HF | K8 light | 14.917 | 0.5495 | 0.9807 | 321.768 | 372.130 |

Interpretation:

- LPIPS gives a mixed endpoint picture: K8 light is slightly better on Tiny
  ImageNet, slightly worse on CIFAR-10 and ImageNet-64 HF.
- Inception Fréchet remains better for K8 light on all three datasets.
- Taken with path AUC, order sensitivity, zero-token diagnostics, and effective
  token usage, the current evidence supports K8 light as the MVP method rather
  than an endpoint-only MSE winner.

## Prefix-Aware Reverse Sampling Smoke

Purpose:

Complete the method implementation path from pure Gaussian noise to generated
images, not only single-step denoising from a noised validation image.

Implementation:

```text
script:
  scripts/sample_checkpoint.py
sampler:
  DDIM-style reverse update
default eta:
  0.0
prefix support:
  use output.prefix_epsilons[m - 1] at every reverse step
validation:
  pytest 43 passed
```

Sampling smoke runs:

```text
samples_cifar10_k8_light_16_ddim20_2026-07-08:
  dataset cifar10
  samples 16
  DDIM steps 20
  prefix budgets 1,4,8
samples_tiny_k8_light_8_ddim20_2026-07-08:
  dataset tiny_imagenet_200
  samples 8
  DDIM steps 20
  prefix budgets 1,4,8
samples_imagenet_hf_k8_light_8_ddim20_2026-07-08:
  dataset imagenet_1k_64x64_hf
  samples 8
  DDIM steps 20
  prefix budgets 1,4,8
```

Local archives:

```text
artifacts/reports/samples_cifar10_k8_light_16_ddim20_2026-07-08/sample_report.json
artifacts/reports/samples_cifar10_k8_light_16_ddim20_2026-07-08/samples_prefix_8.png
artifacts/reports/samples_tiny_k8_light_8_ddim20_2026-07-08/sample_report.json
artifacts/reports/samples_tiny_k8_light_8_ddim20_2026-07-08/samples_prefix_8.png
artifacts/reports/samples_imagenet_hf_k8_light_8_ddim20_2026-07-08/sample_report.json
artifacts/reports/samples_imagenet_hf_k8_light_8_ddim20_2026-07-08/samples_prefix_8.png
```

Interpretation:

This closes an implementation gap: CoFiTok checkpoints now have a reproducible
reverse-sampling entrypoint that respects token prefixes. These short-run
checkpoints are still MVP diagnostics, not final unconditional generation
quality claims.

## Generated Sample Distribution Smoke

Purpose:

Add a small sample-vs-real distribution check for images generated from pure
Gaussian noise, rather than only evaluating denoised validation images at a
fixed timestep.

Implementation:

```text
scripts/sample_checkpoint.py:
  added --save-images
  writes samples_prefix_<m>/sample_00000.png style directories
scripts/evaluate_generated_samples.py:
  compares generated PNGs against a dataset split
  metrics: low-res Frechet proxy and optional torchvision Inception Frechet
validation:
  remote pytest 47 passed
```

Generated-sample reports:

```text
artifacts/reports/generated_quality_cifar10_k8_light_64_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_tiny_imagenet_k8_light_64_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_imagenet_hf_k8_light_64_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_tiny_imagenet_k8_light_20k_64_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_imagenet_hf_k8_light_20k_64_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_tiny_imagenet_k8_light_20k_256_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_imagenet_hf_k8_light_20k_256_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_tiny_epsilononly_20k_256_ddim20_2026-07-08/generated_quality_report.json
artifacts/reports/generated_quality_imagenet_hf_epsilononly_20k_256_ddim20_2026-07-08/generated_quality_report.json
```

Smoke metrics:

| dataset | variant | train steps | generated | real | low-res Frechet proxy | Inception Frechet |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | K8 light | 3k | 64 | 256 | 7.2663 | 313.8881 |
| Tiny ImageNet-200 | K8 light | 5k | 64 | 256 | 8.2197 | 373.2350 |
| Tiny ImageNet-200 | K8 light | 20k | 64 | 256 | 7.9563 | 326.7515 |
| Tiny ImageNet-200 | epsilon-only | 20k | 256 | 1024 | 5.7540 | 263.9912 |
| Tiny ImageNet-200 | K8 light | 20k | 256 | 1024 | 6.9828 | 281.2646 |
| ImageNet-64 HF | K8 light | 5k | 64 | 256 | 8.5337 | 374.3523 |
| ImageNet-64 HF | K8 light | 20k | 64 | 256 | 7.9744 | 321.2626 |
| ImageNet-64 HF | epsilon-only | 20k | 256 | 1024 | 5.2657 | 262.0618 |
| ImageNet-64 HF | K8 light | 20k | 256 | 1024 | 6.6183 | 276.0005 |

Interpretation:

This closes another method-completeness gap: generated samples can now be
saved as individual images and evaluated reproducibly. The numbers should be
read only as smoke metrics because the sample count is still below a
publication-scale FID setup.

## Follow-Up: 20k K8 Light Scaling

Purpose:

Check whether the current K8 light objective continues improving beyond the
short 5k Tiny/ImageNet-64 HF runs without breaking restricted `S_k`.

Configs:

```text
configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_cuda.json
```

Training metrics:

| dataset | steps | final clean MSE | path AUC | effective K | zero ratio | shuffle ratio |
|---|---:|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | 5k | 0.1664 | 0.0759 | 6.804 | 0.0000 | 143.52 |
| Tiny ImageNet-200 | 20k | 0.1427 | 0.0388 | 6.780 | 0.0000 | 165.26 |
| ImageNet-64 HF | 5k | 0.1018 | 0.0620 | 6.787 | 0.0000 | 233.50 |
| ImageNet-64 HF | 20k | 0.0852 | 0.0305 | 6.758 | 0.0000 | 275.65 |

256-image fixed-timestep quality:

| dataset | steps | final PSNR | final LPIPS | final Inception | low-res proxy |
|---|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | 5k | 14.297 | 0.5652 | 343.602 | 1.1861 |
| Tiny ImageNet-200 | 20k | 14.865 | 0.5505 | 346.083 | 0.9674 |
| ImageNet-64 HF | 5k | 14.917 | 0.5495 | 321.768 | 1.0864 |
| ImageNet-64 HF | 20k | 15.558 | 0.5150 | 360.795 | 0.8874 |

Interpretation:

- Longer K8 light training improves final clean MSE, path AUC, PSNR, LPIPS, and
  low-res proxy on Tiny ImageNet-200 and ImageNet-64 HF.
- The restricted synthesis diagnostic remains clean: zero-token component
  energy ratio is still 0.0.
- Generated-sample Inception Frechet improves strongly at 20k for both datasets.
- Increasing the generated-sample smoke from 64 to 256 samples improves the
  measured Inception Frechet again, suggesting the 20k gain is not a one-batch
  artifact.
- A fair 20k epsilon-only baseline still has better generated-sample Frechet
  than K8 light. CoFiTok should not claim unconditional generation quality wins
  over endpoint-only training at the current small-model scale.
- Fixed-timestep reconstruction Inception Frechet is not monotonic: it worsens
  at 20k despite better MSE/LPIPS and better generated-sample Inception. Treat
  this as a metric-sensitivity flag and keep both protocols labelled.

## Follow-Up: 20k Epsilon-Only Fair Baseline

Purpose:

Make the quality comparison fair by training epsilon-only for the same 20k step
budget as K8 light on Tiny ImageNet-200 and ImageNet-64 HF.

Configs:

```text
configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_cuda.json
```

Training diagnostics:

| dataset | variant | final clean MSE | path AUC | effective K | zero ratio | shuffle ratio |
|---|---|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only 20k | 0.1413 | 1.1121 | 5.336 | 0.0000 | 166.07 |
| Tiny ImageNet-200 | K8 light 20k | 0.1427 | 0.0388 | 6.780 | 0.0000 | 165.26 |
| ImageNet-64 HF | epsilon-only 20k | 0.0842 | 1.0746 | 5.240 | 0.0000 | 277.98 |
| ImageNet-64 HF | K8 light 20k | 0.0852 | 0.0305 | 6.758 | 0.0000 | 275.65 |

Fixed-timestep endpoint quality:

| dataset | variant | final PSNR | final LPIPS | final Inception | low-res proxy |
|---|---|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only 20k | 14.976 | 0.5479 | 344.162 | 0.8355 |
| Tiny ImageNet-200 | K8 light 20k | 14.865 | 0.5505 | 346.083 | 0.9674 |
| ImageNet-64 HF | epsilon-only 20k | 15.743 | 0.5204 | 330.643 | 0.7160 |
| ImageNet-64 HF | K8 light 20k | 15.558 | 0.5150 | 360.795 | 0.8874 |

Generated-sample smoke, 256 generated vs 1024 real:

| dataset | variant | low-res Frechet proxy | Inception Frechet |
|---|---|---:|---:|
| Tiny ImageNet-200 | epsilon-only 20k | 5.7540 | 263.991 |
| Tiny ImageNet-200 | K8 light 20k | 6.9828 | 281.265 |
| ImageNet-64 HF | epsilon-only 20k | 5.2657 | 262.062 |
| ImageNet-64 HF | K8 light 20k | 6.6183 | 276.001 |

Generated-sample medium smoke, 1024 generated vs 4096 real:

```text
sampling seed: 2601024
DDIM steps: 20
prefix budget: 8
batch size: 64
```

| dataset | variant | low-res Frechet proxy | Inception Frechet |
|---|---|---:|---:|
| Tiny ImageNet-200 | epsilon-only 20k | 5.5385 | 227.281 |
| Tiny ImageNet-200 | K8 light 20k | 6.7311 | 245.262 |
| ImageNet-64 HF | epsilon-only 20k | 5.0077 | 230.279 |
| ImageNet-64 HF | K8 light 20k | 6.3373 | 241.988 |

Interpretation:

- Epsilon-only remains the stronger endpoint/generation-quality baseline at the
  same 20k budget. This remains true after increasing the generated-sample
  check from 256-vs-1024 to 1024-vs-4096.
- K8 light nearly matches final clean MSE but has dramatically better path AUC
  and broader effective token usage.
- The current CoFiTok claim should focus on ordered, controllable prefix
  denoising at comparable endpoint quality, not beating epsilon-only on
  unconditional sample Frechet.

## Follow-Up: 1024 Generated-Sample Evaluation

Purpose:

Reduce variance in the generated-sample distribution smoke by evaluating the
20k Tiny ImageNet-200 and ImageNet-64 HF fair baselines with 1024 generated
samples against 4096 validation images.

Remote artifacts:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generated_tiny_epsilononly_20k_1024_ddim20_2026-07-08/sample_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_tiny_k8_light_20k_1024_ddim20_2026-07-08/sample_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_imagenet_hf_epsilononly_20k_1024_ddim20_2026-07-08/sample_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_imagenet_hf_k8_light_20k_1024_ddim20_2026-07-08/sample_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_tiny_epsilononly_20k_1024_ddim20_2026-07-08/generated_quality_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_tiny_k8_light_20k_1024_ddim20_2026-07-08/generated_quality_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_imagenet_hf_epsilononly_20k_1024_ddim20_2026-07-08/generated_quality_report.json
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_imagenet_hf_k8_light_20k_1024_ddim20_2026-07-08/generated_quality_report.json
```

Updated summary counts:

```text
train reports: 85
order-eval reports: 26
quality reports: 29
sampling reports: 16
generated-quality reports: 13
```

Interpretation:

The larger smoke confirms the previous conclusion instead of reversing it:
epsilon-only remains better on sample Frechet, while K8 light remains valuable
for ordered prefix denoising rather than pure unconditional sample quality.

## Report Figure Package

Purpose:

Create a reusable draft figure package from the deterministic experiment
summary, so paper/report graphics are regenerated from source tables rather
than hand-copied numbers.

Implementation:

```text
script:
  scripts/make_report_figures.py
input:
  artifacts/reports/summary_2026-07-08/experiment_summary.json
output:
  artifacts/figures/summary_2026-07-08/
validation:
  local py_compile: passed
  local figure manifest/image verification: 6 figures
  remote pytest: 49 passed
  remote figure verification: 6 PNGs, each 1260x760
```

Generated figures:

```text
artifacts/figures/summary_2026-07-08/path_auc_20k.png
artifacts/figures/summary_2026-07-08/final_clean_mse_20k.png
artifacts/figures/summary_2026-07-08/effective_tokens_20k.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_256.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_1024.png
artifacts/figures/summary_2026-07-08/imagenet_hf_order_path_auc.png
artifacts/figures/summary_2026-07-08/figure_manifest.json
```

Interpretation:

- These figures are now the preferred quick visual evidence package for the MVP
  report.
- The strongest current figure is `path_auc_20k.png`: it shows the main
  CoFiTok advantage, large prefix-path improvement at nearly matched endpoint
  MSE.
- The generated Inception figures must remain labelled as smoke metrics, not
  final FID results.

## Paper-Facing Visual Panels

Purpose:

Create reproducible visual evidence panels from existing PNG report artifacts,
so prefix progression, order sensitivity, `S_k` degeneration diagnostics, and
generated-sample smoke can be inspected without manually arranging figures.

Implementation:

```text
script:
  scripts/make_visual_panels.py
input:
  artifacts/reports/*/prefix_final.png
  artifacts/reports/*/prefix_final_shuffled.png
  artifacts/reports/*/samples_prefix_8.png
output:
  artifacts/figures/visual_panels_2026-07-08/
validation:
  local py_compile: passed
  local visual inspection: prefix, S_k diagnostic, sampling panels
  remote pytest: 50 passed
  remote panel verification: 4 PNGs
```

Generated panels:

```text
artifacts/figures/visual_panels_2026-07-08/prefix_comparison_20k.png
artifacts/figures/visual_panels_2026-07-08/order_ablation_prefix_panel.png
artifacts/figures/visual_panels_2026-07-08/sk_diagnostic_panel.png
artifacts/figures/visual_panels_2026-07-08/sampling_comparison_20k.png
artifacts/figures/visual_panels_2026-07-08/visual_panel_manifest.json
```

Interpretation:

- `prefix_comparison_20k.png` is the main visual companion to the path-AUC
  table: K8 light visibly makes late prefixes meaningful while keeping the same
  final endpoint target.
- `order_ablation_prefix_panel.png` is the visual companion to the
  random/reverse order table.
- `sk_diagnostic_panel.png` shows restricted/shuffled and deep-`S_k` ablation
  panels together; deep `S_k` remains ablation-only evidence.
- `sampling_comparison_20k.png` is qualitative smoke evidence only. Use the
  generated-sample Frechet tables for quantitative sample-vs-real comparison.
