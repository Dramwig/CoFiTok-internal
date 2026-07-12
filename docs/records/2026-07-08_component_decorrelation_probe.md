# Component Decorrelation Probe

Date: 2026-07-08

Purpose: validate whether the newly implemented
`component_decorrelation_weight` is useful on real data, rather than only being
covered by unit tests and tensor smoke checks.

## Setup

Baseline:

```text
configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_3k_2026-07-08
```

Decor probe:

```text
configs/train_cifar10_k8_denoisepath_p150_light_decor_3k_cuda.json
component_decorrelation_weight = 0.05
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_denoisepath_p150_light_decor_3k_2026-07-08
```

Both runs use CIFAR-10, K=8, seed 137, 3000 training steps, and the light
denoise-path objective.

Local lightweight artifact mirror:

```text
artifacts/reports/component_decorrelation_probe_2026-07-08/
artifacts/reports/component_decorrelation_probe_2026-07-08/component_decorrelation_probe_summary.json
```

## Code Changes

- Added `configs/train_cifar10_k8_denoisepath_p150_light_decor_3k_cuda.json`.
- `scripts/summarize_experiments.py` now classifies
  `component_decorrelation_ablation` variants and preserves
  `final_component_decorrelation_loss`.
- `scripts/evaluate_quality.py` now reports post-hoc component correlation:
  mean and max absolute cosine over component pairs and evaluated images.

Validation:

```text
local targeted pytest: pass, 19 tests
remote targeted pytest: pass, 19 tests
```

## Training Metrics

| variant | final MSE | path AUC | clean AUC | effective K | late-half energy | active tail | shuffle ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.157950 | 0.100325 | 4.704517 | 6.833542 | 0.717460 | 7 | 144.485 |
| decor 0.05 | 0.154395 | 0.242932 | 5.501738 | 5.988543 | 0.711498 | 6 | 148.496 |

The decor run slightly improves endpoint clean MSE but worsens denoise-path AUC
and clean-prefix AUC, and it reduces effective token usage.

## Order Diagnostics

| variant | order | final MSE | path AUC | clean AUC | effective K |
|---|---|---:|---:|---:|---:|
| baseline | ordered | 0.154554 | 0.099357 | 4.660819 | 6.834470 |
| baseline | random | 0.154554 | 0.256283 | 3.628121 | 6.834470 |
| baseline | reverse | 0.154554 | 0.674152 | 2.287383 | 6.834470 |
| decor 0.05 | ordered | 0.150142 | 0.240602 | 5.457052 | 5.983868 |
| decor 0.05 | random | 0.150142 | 0.579886 | 3.699328 | 5.983868 |
| decor 0.05 | reverse | 0.150142 | 1.039994 | 2.175934 | 5.983868 |

Order sensitivity is still present after decor training: random and reverse
orders remain worse than ordered on denoise-path AUC. However, the ordered decor
path AUC is already worse than the baseline ordered path AUC, so this weight is
not a safe default.

## Quality And Correlation Probe

Both rows evaluate 256 validation images at timestep 500.

| variant | final MSE | PSNR | low-res Frechet proxy | MSE AUC | proxy AUC | mean abs cosine | max abs cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.153115 | 14.170 | 1.849053 | 6.722058 | 5.597029 | 0.634461 | 0.773306 |
| decor 0.05 | 0.149438 | 14.276 | 1.804085 | 8.070447 | 5.923978 | 0.204991 | 0.649108 |

The decor loss does what it is supposed to do locally: it reduces post-hoc mean
absolute component cosine from 0.634 to 0.205. It also slightly improves the
endpoint reconstruction metrics in this small probe. The cost is worse prefix
curve AUC and weaker effective token utilization.

## Decision

`component_decorrelation_weight=0.05` should remain a second-stage ablation, not
the new default. It is a real lever for component separation, but at this
weight it fights the denoise-path ordering objective.

Next recommended probes:

```text
component_decorrelation_weight in {0.005, 0.01}
optionally enable decor only after the prefix/order losses have stabilized
repeat on Tiny ImageNet only if CIFAR lower-weight probes improve path AUC
```

