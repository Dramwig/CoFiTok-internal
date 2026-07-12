# CoFiTok ImageNet-64 HF Seed-2 Quality Slice

Date: 2026-07-08

Purpose: add fixed-timestep quality evidence for seed-103 ImageNet-64 HF
runs. This complements the training and order-diagnostic seed repeats with
PSNR, low-res Frechet proxy, torchvision Inception Frechet, and LPIPS Alex.

Protocol:

```text
split = val
image_count = 256
fixed_timestep = 500
prefix budgets = default per K
metrics = MSE, PSNR, low-res Frechet proxy, Inception Frechet, LPIPS Alex
```

All six reports used `--enable-lpips --enable-inception-fid`, and both optional
metric backends were available.

## Final-Prefix Quality

| variant | K | steps | final MSE | final PSNR | final Inception | final LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| epsilon-only | 8 | 5000 | 0.1227 | 15.131 | 325.567 | 0.5541 |
| K4 light | 4 | 5000 | 0.1243 | 15.075 | 337.129 | 0.5405 |
| K8 light | 8 | 5000 | 0.1247 | 15.062 | 341.226 | 0.5463 |
| K16 light | 16 | 5000 | 0.1298 | 14.887 | 346.699 | 0.5384 |
| simultaneous predictor | 8 | 5000 | 0.1293 | 14.904 | 318.749 | 0.5757 |
| deep `S_k` | 8 | 5000 | 0.1676 | 13.779 | 298.040 | 0.6153 |

## Prefix-Curve AUC

| variant | K | MSE AUC | Inception AUC | LPIPS AUC |
|---|---:|---:|---:|---:|
| epsilon-only | 8 | 8.0990 | 376.821 | 0.9989 |
| K4 light | 4 | 4.9383 | 360.385 | 0.9298 |
| K8 light | 8 | 6.7878 | 373.592 | 0.9909 |
| K16 light | 16 | 7.7460 | 377.651 | 1.0062 |
| simultaneous predictor | 8 | 6.6234 | 372.982 | 0.9892 |
| deep `S_k` | 8 | 6.6139 | 367.586 | 0.9961 |

## Interpretation

- At seed 103 and 5k steps, K4/K8/K16 light CoFiTok remain close to the
  epsilon-only endpoint MSE while preserving much better prefix MSE AUC than
  epsilon-only.
- K4 has the best MSE AUC in this fixed-timestep slice, consistent with a
  compact but less expressive token budget; K16 uses many tokens but does not
  improve endpoint or prefix AUC at 5k.
- Deep `S_k` is not a valid default operator despite some Inception-style
  numbers, because the same seed also fails the zero-token diagnostic. This
  quality slice should be read together with
  `docs/records/2026-07-08_ablation_seed_repeats.md`.
- These results reduce the selected-quality-metric seed-coverage gap, but they
  are still 256-image fixed-timestep reconstruction metrics rather than
  full-dataset sample FID.
