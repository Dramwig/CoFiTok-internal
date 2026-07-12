# CoFiTok MVP Technical Report

Date: 2026-07-08

Status: internal MVP report, not a final paper draft.

## Claim

CoFiTok factorizes pixel-space diffusion noise prediction into an ordered
sequence of compressed denoising tokens. Each token is expanded by a restricted,
condition-free synthesis operator into a dense negative-noise component, and
cumulative prefixes yield controllable partial denoising results.

The current MVP evidence supports a scoped claim:

> A tiny CoFiTok predictor with restricted `S_k` can learn ordered denoising
> components on CIFAR-10, Tiny ImageNet-200, and an ImageNet-1K 64x64 fallback.
> Compared with endpoint-only epsilon training, the light denoise-path objective
> preserves endpoint quality while greatly improving prefix/path alignment,
> token utilization, and order sensitivity.

## Method

For a noised image

```text
x_t = alpha_t x_0 + sigma_t epsilon
```

CoFiTok predicts

```text
epsilon_hat = sum_k S_k(z_k)
```

where `z_k` is a compressed denoising token and `S_k` maps only the current
token to one dense component. Prefixes are valid predictions:

```text
epsilon_hat_1:m = sum_{k=1}^m S_k(z_k)
x0_hat_1:m = (x_t - sigma_t * epsilon_hat_1:m) / alpha_t
```

Current default synthesis:

```text
synthesis_mode = restricted
S_k input = z_k only
layers = bias-free 1x1 projection + bias-free local convolution
S_k(0) = 0 by construction
```

The strong synthesis ablation:

```text
synthesis_mode = deep_decoder
biased nonlinear convolutions
token-only input is preserved
not valid as the default CoFiTok operator
```

## Objective

The current MVP method uses the light denoise-path objective:

```text
L = L_epsilon
  + 0.15 * L_denoise_path_prefix
  + 0.30 * L_denoise_path_component
```

The denoise path interpolates from the no-update `x_t` reconstruction toward
coarse-to-fine spatial targets, using progress power `1.5`.

## Datasets

| alias | role | status |
|---|---|---|
| CIFAR-10 | P0 smoke and diagnostics | available, trained/evaluated |
| Tiny ImageNet-200 | P1 main MVP dataset | available, trained/evaluated |
| ImageNet-1K 64x64 HF fallback | P2 ImageNet-family validation | available, trained/evaluated |
| downsampled_imagenet_64 | strict P2 source | available and loader-ready; not used in current result tables |

The strict `downsampled_imagenet_64` Academic Torrents payload was completed on
2026-07-09. Existing experiment tables still use the explicitly named
`imagenet_1k_64x64_hf` fallback and should not be reported as exact
`downsampled_imagenet_64` runs unless rerun on that source.

## Core Results

Generated source:

```text
artifacts/reports/summary_2026-07-08/core_summary.md
```

Method overview figure:

```text
artifacts/figures/method_overview_2026-07-08/method_overview.png
artifacts/figures/method_overview_2026-07-08/method_overview.svg
artifacts/figures/method_overview_2026-07-08/method_overview_manifest.json
```

The method figure is rendered by `scripts/make_method_figure.py`. It records
the default `S_k` restrictions directly in the figure manifest: token-only
input, no image/timestep/label/prompt/previous-token/skip/constant access,
bias-free shallow local synthesis, `S_k(0)=0`, and no VAE-style final decoder.

Draft report figures:

```text
artifacts/figures/summary_2026-07-08/path_auc_20k.png
artifacts/figures/summary_2026-07-08/final_clean_mse_20k.png
artifacts/figures/summary_2026-07-08/effective_tokens_20k.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_256.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_1024.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_8192.png
artifacts/figures/summary_2026-07-08/generated_inception_20k_hf_50000.png
artifacts/figures/summary_2026-07-08/imagenet_hf_order_path_auc.png
artifacts/figures/summary_2026-07-08/figure_manifest.json
```

These figures are rendered from `experiment_summary.json` by
`scripts/make_report_figures.py`. The generated-sample Frechet figures are
smoke-to-formal-scale internal metrics. The main Tiny/HF table reaches 8192
generated images against 8192 real images, and the HF ImageNet-64 rows now also
include a streamed 50000 generated vs 50000 real DDIM50 protocol. These are
internal torchvision Inception-Frechet measurements; the separate official FID
matrix below records the external `pytorch-fid` image-directory protocol.

Draft visual panels:

```text
artifacts/figures/visual_panels_2026-07-08/prefix_comparison_20k.png
artifacts/figures/visual_panels_2026-07-08/order_ablation_prefix_panel.png
artifacts/figures/visual_panels_2026-07-08/sk_diagnostic_panel.png
artifacts/figures/visual_panels_2026-07-08/sampling_comparison_20k.png
artifacts/figures/visual_panels_2026-07-08/visual_panel_manifest.json
```

These panels are rendered from existing PNG report artifacts by
`scripts/make_visual_panels.py`. They are paper-facing draft panels for visual
inspection; the generated-sample panel remains qualitative smoke evidence.

Seed-103 ImageNet-64 HF quality slice:

```text
docs/records/2026-07-08_quality_seed_slice.md
```

This record adds fixed-timestep 256-image PSNR, low-res Frechet proxy,
torchvision Inception Frechet, and LPIPS Alex for seed-103 epsilon-only, K4/K8/K16
light CoFiTok, simultaneous predictor, and deep `S_k` ablation runs.

### ImageNet-64 HF Training Matrix

| variant | seed | final MSE | path AUC | effective K | zero ratio | shuffle ratio |
|---|---:|---:|---:|---:|---:|---:|
| channel-mask | 139 | 0.1230 | 4.8844 | 1.111 | 0.0000 | 200.1 |
| epsilon-only | 139 | 0.0998 | 1.1000 | 5.339 | 0.0000 | 238.0 |
| epsilon-only | 103 | 0.0940 | 0.8837 | 5.400 | 0.0000 | 250.2 |
| full denoise-path | 139 | 0.1074 | 0.0417 | 6.758 | 0.0000 | 219.7 |
| light denoise-path | 139 | 0.1018 | 0.0620 | 6.787 | 0.0000 | 233.5 |
| light denoise-path | 103 | 0.0963 | 0.0641 | 6.768 | 0.0000 | 244.3 |
| simultaneous predictor | 139 | 0.0995 | 0.0600 | 6.845 | 0.0000 | 237.4 |
| simultaneous predictor | 103 | 0.1024 | 0.0723 | 6.801 | 0.0000 | 233.1 |
| deep `S_k` ablation | 139 | 0.0977 | 0.0366 | 6.792 | 0.0522 | 242.0 |
| deep `S_k` ablation | 103 | 0.1327 | 0.0433 | 6.748 | 0.0450 | 177.3 |

Interpretation:

- Epsilon-only is the strongest endpoint baseline.
- Light denoise-path keeps endpoint quality close while improving path AUC by
  roughly an order of magnitude.
- Deep `S_k` improves endpoint/path numbers but fails `S_k(0)=0`, so it is a
  useful degeneration ablation rather than a valid main method.
- The deep `S_k` and simultaneous-predictor ablations now have seed-103 repeats
  recorded in `docs/records/2026-07-08_ablation_seed_repeats.md`.

### Order Sensitivity

| K | seed | order | final MSE | path AUC | zero ratio |
|---:|---:|---|---:|---:|---:|
| 4 | 139 | ordered | 0.1029 | 0.0674 | 0.0000 |
| 4 | 139 | random | 0.1029 | 0.0727 | 0.0000 |
| 4 | 139 | reverse | 0.1029 | 0.7516 | 0.0000 |
| 4 | 103 | ordered | 0.0980 | 0.0575 | 0.0000 |
| 4 | 103 | random | 0.0980 | 0.0652 | 0.0000 |
| 4 | 103 | reverse | 0.0980 | 0.6975 | 0.0000 |
| 8 | 139 | ordered | 0.1041 | 0.0635 | 0.0000 |
| 8 | 139 | random | 0.1041 | 0.2441 | 0.0000 |
| 8 | 139 | reverse | 0.1041 | 0.6947 | 0.0000 |
| 8 | 103 | ordered | 0.0986 | 0.0647 | 0.0000 |
| 8 | 103 | random | 0.0986 | 0.2605 | 0.0000 |
| 8 | 103 | reverse | 0.0986 | 0.7406 | 0.0000 |
| 16 | 139 | ordered | 0.1021 | 0.0680 | 0.0000 |
| 16 | 139 | random | 0.1021 | 0.4680 | 0.0000 |
| 16 | 139 | reverse | 0.1021 | 0.7020 | 0.0000 |
| 16 | 103 | ordered | 0.1056 | 0.0701 | 0.0000 |
| 16 | 103 | random | 0.1056 | 0.5294 | 0.0000 |
| 16 | 103 | reverse | 0.1056 | 0.7032 | 0.0000 |

Interpretation:

- Full endpoint is unchanged by component order because all components are still
  summed.
- Prefix path quality strongly depends on order, especially for K8/K16.
- K4/K16 now have seed-103 repeats recorded in
  `docs/records/2026-07-08_token_scaling_seed_repeats.md`.
- K8 is the current MVP sweet spot: strong path AUC, broad token use, and clear
  random/reverse separation. K16 uses more components effectively but does not
  improve ordered path AUC at 5k steps.

### Component Separation Pressure

This is an ablation, not the headline objective. The default CoFiTok rows above
use the no-decorrelation light denoise-path objective because it has the cleanest
prefix-order interpretation. A conservative scheduled decorrelation term tests
whether mild explicit component separation can reduce component correlation
without breaking ordered prefixes.

Source:

```text
docs/records/2026-07-08_component_decorrelation_tiny_seed_confirmation.md
artifacts/reports/component_decorrelation_tiny_conservative_seed_confirm_2026-07-08/component_decorrelation_tiny_conservative_seed_confirm_summary.json
```

Schedule:

```text
component_decorrelation_weight = 0.003
component_decorrelation_start_step = 3000
component_decorrelation_warmup_steps = 1500
full weight step = 4500 of 5000
```

| seed | variant | ordered path AUC | mean abs cosine | final MSE | effective K |
|---:|---|---:|---:|---:|---:|
| 139 | baseline | 0.0766 | 0.7223 | 0.1487 | 6.8043 |
| 139 | scheduled decor 0.003 | 0.0804 | 0.6890 | 0.1486 | 6.8316 |
| 103 | baseline | 0.0740 | 0.6967 | 0.1503 | 6.8430 |
| 103 | scheduled decor 0.003 | 0.0776 | 0.6644 | 0.1501 | 6.8829 |

Mean delta versus the no-decor baseline across seeds:

| metric | mean delta |
|---|---:|
| ordered path AUC | +0.0036 |
| mean abs component cosine | -0.0328 |
| final MSE | -0.0002 |
| effective token count | +0.0336 |

Interpretation:

- Scheduled decor 0.003 consistently reduces component cosine across both seeds.
- The ordered path-AUC penalty stays small relative to stronger 0.005 decor runs.
- Random/reverse prefixes remain worse than ordered prefixes, so this pressure
  does not collapse the ordering signal.
- Use scheduled decor 0.003 as the component-separation ablation table; keep the
  no-decor light denoise-path objective as the main CoFiTok result.

### Stronger Multiscale Backbone

Source:

```text
docs/records/2026-07-08_multiscale_20k_backbone_validation.md
docs/records/2026-07-08_multiscale_fair_generation_protocol.md
```

The stronger `multiscale_unet` predictor was extended from the completed 10k
pilots to matched 20k train-scale runs on both main datasets while keeping the
restricted `S_k` operator unchanged.

| dataset | steps | final clean MSE | path AUC | effective K | zero ratio |
|---|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | 10k | 0.125013 | 0.043255 | 6.810 | 0.0000 |
| Tiny ImageNet-200 | 20k | 0.123434 | 0.034593 | 6.761 | 0.0000 |
| ImageNet-64 HF | 10k | 0.077563 | 0.036447 | 6.749 | 0.0000 |
| ImageNet-64 HF | 20k | 0.065420 | 0.025294 | 6.721 | 0.0000 |

Interpretation:

- The 20k multiscale rows improve over their 10k pilots in endpoint clean MSE
  and path AUC on both datasets.
- The zero-token ratio remains 0.0, so the stronger predictor does not relax or
  hide a violation of the restricted synthesis contract.
- These are train-scale backbone validation rows. The final paper should only
  add 20k multiscale quality/sampling tables if it wants to make backbone
  scaling a central claim.
- The matched multiscale generated-quality protocol was also run for
  epsilon-only and K8 light at 20k. It improves absolute sample metrics versus
  the older tiny-conv sampler rows, but epsilon-only remains slightly better
  than K8 light on generated-sample Frechet metrics.

### Cross-Dataset Quality

| dataset | variant | steps | final PSNR | final Inception | final LPIPS |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | epsilon-only | 3k | 14.390 | 334.069 | 0.1639 |
| CIFAR-10 | K8 light | 3k | 14.170 | 327.889 | 0.1674 |
| Tiny ImageNet | epsilon-only | 20k | 14.976 | 344.162 | 0.5479 |
| Tiny ImageNet | K8 light | 20k | 14.865 | 346.083 | 0.5505 |
| ImageNet-64 HF | epsilon-only | 20k | 15.743 | 330.643 | 0.5204 |
| ImageNet-64 HF | K8 light | 20k | 15.558 | 360.795 | 0.5150 |

Interpretation:

- At the same 20k budget, epsilon-only is the stronger endpoint-quality
  baseline on MSE/PSNR/proxy and most fixed-timestep Frechet metrics.
- K8 light nearly matches final clean MSE but improves path AUC by roughly
  28-35x and uses more tokens effectively.
- The main value of CoFiTok is not endpoint MSE alone; it is endpoint quality
  plus prefix controllability and non-degenerate ordered components.

### Full-Validation Reconstruction/Prefix Sweep

Source:

```text
docs/records/2026-07-08_full_validation_quality_sweep.md
```

This sweep evaluates the full validation split for the two main seed-103 20k
rows on Tiny ImageNet-200 and ImageNet-1K 64x64 HF. It uses deterministic
fixed-timestep reconstruction/prefix metrics only: no LPIPS or Inception is
computed for these full-validation rows.

| dataset | variant | val images | final MSE | MSE AUC | low-res Frechet proxy AUC |
|---|---|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only | 10000 | 0.126565 | 7.955221 | 5.366480 |
| Tiny ImageNet-200 | K8 light | 10000 | 0.131721 | 6.584363 | 4.946370 |
| ImageNet-64 HF | epsilon-only | 50000 | 0.107975 | 8.069069 | 5.225075 |
| ImageNet-64 HF | K8 light | 50000 | 0.114178 | 6.604900 | 4.873849 |

Interpretation:

- Endpoint MSE still slightly favors epsilon-only on both full validation sets.
- K8 light improves prefix/path aggregate metrics on both full validation sets:
  MSE AUC and low-res Frechet proxy AUC both decrease.
- This closes the deterministic full-validation reconstruction/prefix evidence
  gap, while the formal generated-sample FID gap remains open.

### ImageNet-64 HF Seed-103 Quality Slice

| variant | K | final MSE | final PSNR | final Inception | final LPIPS | MSE AUC |
|---|---:|---:|---:|---:|---:|---:|
| epsilon-only | 8 | 0.1227 | 15.131 | 325.567 | 0.5541 | 8.0990 |
| K4 light | 4 | 0.1243 | 15.075 | 337.129 | 0.5405 | 4.9383 |
| K8 light | 8 | 0.1247 | 15.062 | 341.226 | 0.5463 | 6.7878 |
| K16 light | 16 | 0.1298 | 14.887 | 346.699 | 0.5384 | 7.7460 |
| simultaneous predictor | 8 | 0.1293 | 14.904 | 318.749 | 0.5757 | 6.6234 |
| deep `S_k` | 8 | 0.1676 | 13.779 | 298.040 | 0.6153 | 6.6139 |

Interpretation:

- This seed-103 slice reduces the selected-quality-metric seed-coverage gap for
  ImageNet-64 HF 5k runs.
- K4/K8/K16 light remain close to epsilon-only endpoint MSE while improving MSE
  AUC, but the metrics are fixed-timestep reconstruction metrics, not sample
  FID.
- Deep `S_k` quality must be interpreted together with its failed zero-token
  diagnostic and is not a valid default method.

## Diagnostics

The restricted `S_k` runs satisfy:

```text
zero-token component energy ratio: 0.0
shuffled-token final ratios: large, non-matching
random-token ratios: reported per run
```

Deep `S_k` ablation:

```text
final clean MSE: 0.0977
path AUC: 0.0366
zero-token component energy ratio: 0.0522
```

This directly supports the design boundary: stronger `S_k` can improve apparent
quality while carrying token-independent priors.

## Sampling

`scripts/sample_checkpoint.py` implements prefix-aware DDIM sampling.

Sampling smoke:

| dataset | variant | train steps | samples | DDIM steps | prefix budgets | eta |
|---|---|---:|---:|---:|---|---:|
| CIFAR-10 | K8 light | 3k | 16 | 20 | 1,4,8 | 0.0 |
| CIFAR-10 | K8 light | 3k | 64 | 20 | 8 | 0.0 |
| Tiny ImageNet-200 | K8 light | 5k | 8 | 20 | 1,4,8 | 0.0 |
| Tiny ImageNet-200 | K8 light | 5k | 64 | 20 | 8 | 0.0 |
| Tiny ImageNet-200 | K8 light | 20k | 64 | 20 | 8 | 0.0 |
| Tiny ImageNet-200 | epsilon-only | 20k | 256 | 20 | 8 | 0.0 |
| Tiny ImageNet-200 | K8 light | 20k | 256 | 20 | 8 | 0.0 |
| ImageNet-64 HF | K8 light | 5k | 8 | 20 | 1,4,8 | 0.0 |
| ImageNet-64 HF | K8 light | 5k | 64 | 20 | 8 | 0.0 |
| ImageNet-64 HF | K8 light | 20k | 64 | 20 | 8 | 0.0 |
| ImageNet-64 HF | epsilon-only | 20k | 256 | 20 | 8 | 0.0 |
| ImageNet-64 HF | K8 light | 20k | 256 | 20 | 8 | 0.0 |

Generated sample distribution smoke-to-50k protocol:

| dataset | variant | train steps | generated | real | low-res Frechet proxy | Inception Frechet |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | K8 light | 3k | 64 | 256 | 7.2663 | 313.8881 |
| Tiny ImageNet-200 | K8 light | 5k | 64 | 256 | 8.2197 | 373.2350 |
| Tiny ImageNet-200 | K8 light | 20k | 64 | 256 | 7.9563 | 326.7515 |
| Tiny ImageNet-200 | epsilon-only | 20k | 256 | 1024 | 5.7540 | 263.9912 |
| Tiny ImageNet-200 | K8 light | 20k | 256 | 1024 | 6.9828 | 281.2646 |
| Tiny ImageNet-200 | epsilon-only | 20k | 1024 | 4096 | 5.5385 | 227.2814 |
| Tiny ImageNet-200 | K8 light | 20k | 1024 | 4096 | 6.7311 | 245.2619 |
| Tiny ImageNet-200 | epsilon-only | 20k | 2048 | 8192 | 7.5941 | 276.9332 |
| Tiny ImageNet-200 | K8 light | 20k | 2048 | 8192 | 7.3397 | 274.7529 |
| Tiny ImageNet-200 | epsilon-only | 20k | 8192 | 8192 | 7.5761 | 272.9955 |
| Tiny ImageNet-200 | K8 light | 20k | 8192 | 8192 | 7.3264 | 271.0260 |
| ImageNet-64 HF | K8 light | 5k | 64 | 256 | 8.5337 | 374.3523 |
| ImageNet-64 HF | K8 light | 20k | 64 | 256 | 7.9744 | 321.2626 |
| ImageNet-64 HF | epsilon-only | 20k | 256 | 1024 | 5.2657 | 262.0618 |
| ImageNet-64 HF | K8 light | 20k | 256 | 1024 | 6.6183 | 276.0005 |
| ImageNet-64 HF | epsilon-only | 20k | 1024 | 4096 | 5.0077 | 230.2794 |
| ImageNet-64 HF | K8 light | 20k | 1024 | 4096 | 6.3373 | 241.9877 |
| ImageNet-64 HF | epsilon-only | 20k | 2048 | 8192 | 6.5585 | 246.0469 |
| ImageNet-64 HF | K8 light | 20k | 2048 | 8192 | 6.6856 | 274.0251 |
| ImageNet-64 HF | epsilon-only | 20k | 8192 | 8192 | 6.5490 | 240.8919 |
| ImageNet-64 HF | K8 light | 20k | 8192 | 8192 | 6.6993 | 269.2785 |
| ImageNet-64 HF | epsilon-only | 20k | 50000 | 50000 | 6.4496 | 238.0329 |
| ImageNet-64 HF | K8 light | 20k | 50000 | 50000 | 6.6096 | 266.8045 |
| Tiny ImageNet-200 | epsilon-only multiscale | 20k | 10000 | 10000 | 5.2057 | 136.7132 |
| Tiny ImageNet-200 | K8 light multiscale | 20k | 10000 | 10000 | 5.7001 | 141.4577 |
| ImageNet-64 HF | epsilon-only multiscale | 20k | 50000 | 50000 | 5.5593 | 134.2443 |
| ImageNet-64 HF | K8 light multiscale | 20k | 50000 | 50000 | 6.2099 | 136.8376 |

The 50000/50000 HF rows close the sample-count scale gap for the ImageNet-family
fallback, but they do not close the quality-claim gap. The older tiny-conv
8192/8192 Tiny rows slightly favor K8 light, while the matched multiscale 20k
rows favor epsilon-only on both Tiny ImageNet-200 and ImageNet-64 HF. The stable
current claim is therefore prefix controllability at comparable endpoint
quality, not a robust generated-sample quality win.

Official `pytorch-fid` image-directory protocol:

| dataset | variant | train steps | generated | real | official FID |
|---|---|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only multiscale | 20k | 10000 | 10000 | 148.8414 |
| Tiny ImageNet-200 | K8 light multiscale | 20k | 10000 | 10000 | 152.2474 |
| ImageNet-64 HF | epsilon-only multiscale | 20k | 50000 | 50000 | 140.7213 |
| ImageNet-64 HF | K8 light multiscale | 20k | 50000 | 50000 | 134.0818 |

The official FID evidence is mixed: HF ImageNet-family favors K8 light, while
Tiny ImageNet-200 favors epsilon-only. This closes the official-FID protocol
gap, but it still does not support a broad unconditional generation-quality win.

## Reproducibility

Key scripts:

```text
scripts/train_short.py
scripts/evaluate_checkpoint.py
scripts/evaluate_quality.py
scripts/sample_checkpoint.py
scripts/evaluate_generated_samples.py
scripts/summarize_experiments.py
scripts/validate_summary_consistency.py
scripts/validate_mvp_evidence.py
scripts/make_remote_recovery_bundle.py
scripts/make_method_figure.py
scripts/make_report_figures.py
scripts/make_visual_panels.py
```

Current validation:

```text
latest completed remote pytest: pass
local and remote summary consistency: ok
local and remote idea-requirement readiness: ok
local and remote synthesis-contract validation: ok
summary artifacts:
  artifacts/reports/summary_2026-07-08/experiment_summary.json
  artifacts/reports/summary_2026-07-08/core_summary.md
  artifacts/reports/summary_2026-07-08/train_summary.csv
  artifacts/reports/summary_2026-07-08/order_eval_summary.csv
  artifacts/reports/summary_2026-07-08/quality_summary.csv
  artifacts/reports/summary_2026-07-08/sample_summary.csv
  artifacts/reports/summary_2026-07-08/generated_quality_summary.csv
  artifacts/reports/summary_2026-07-08/official_fid_summary.csv
  artifacts/reports/official_fid_protocol_2026-07-09/
  artifacts/figures/method_overview_2026-07-08/method_overview_manifest.json
  artifacts/figures/summary_2026-07-08/figure_manifest.json
  artifacts/figures/visual_panels_2026-07-08/visual_panel_manifest.json
summary counts:
  train: 111
  order_eval: 95
  quality: 58
  sampling: 22
  generated_quality: 29
  official_fid: 4
```

The consistency check was run with:

```bash
PYTHONPATH=src python scripts/validate_summary_consistency.py \
  --summary-dir artifacts/reports/summary_2026-07-08 \
  --doc docs/reports/cofitok_mvp_report_2026-07-08.md \
  --doc docs/records/2026-07-08_completion_audit.md
```

The MVP evidence coverage check was run locally with:

```bash
PYTHONPATH=src python scripts/validate_mvp_evidence.py \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json
```

Paper draft artifacts:

```text
../paper/README.md
../paper/draft.md
../paper/figures_and_tables.md
../paper/citation_notes.md
../paper/citation_verification.md
../paper/citation_claim_audit.md
../paper/full_pdf_claim_audit.md
../paper/full_pdf_claim_audit_sources.json
../paper/references.bib
```

The current paper draft is available as venue-neutral Markdown and LaTeX.
Current citations have verified arXiv BibTeX entries, provenance notes, an
abstract-level claim audit, and a full-PDF claim audit for the current
related-work wording.

Remote sync note:

```text
docs/records/2026-07-08_remote_io_incident.md
```

The follow-up queue completed on `pro6000` after SSH recovered. Final remote
and local gates now pass, including `validate_publication_readiness.py
--require-ready`; see `docs/records/2026-07-08_remote_queue_launch.md` and
`docs/records/2026-07-08_publication_readiness_gap.md`.

## Limitations

- Models are tiny and trained for short MVP budgets.
- Inception/LPIPS quality metrics include 1024-image fixed-timestep validation
  slices for the main publication-readiness rows, but they are not full
  validation-set reconstruction FID.
- Generated sample distribution is evaluated up to 8192 generated samples vs
  8192 real images for the main Tiny/HF variants, plus 50000/50000 on the HF
  ImageNet-64 fallback. External `pytorch-fid` reports now exist for the matched
  multiscale 20k rows, but the result is mixed across datasets and does not show
  a robust CoFiTok sample-quality win.
- Exact `downsampled_imagenet_64` is now staged and loader-ready; current
  result tables still use the explicitly named HF fallback.
- Current paper text, figures, and visual panels are draft paper-facing
  artifacts. An AAAI-27 anonymous-submission build now exists under
  `paper/venues/aaai27/`, while final checklist handling, author metadata,
  proceedings-metadata decisions, and camera-ready layout still need polishing.

## Next Steps

1. Use `paper/venues/aaai27/` as the current submission-format build unless a
   different target venue is chosen.
2. Decide whether to replace arXiv `@misc` entries with venue proceedings
   metadata.
3. If generation quality becomes central, train stronger/longer models before
   making any sample-quality claim.
4. Add a larger or longer K8 light run only if the goal shifts toward closing
   the remaining endpoint/generation gap to epsilon-only.
