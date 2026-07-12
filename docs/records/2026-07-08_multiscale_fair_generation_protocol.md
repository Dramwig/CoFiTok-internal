# 2026-07-08 Multiscale Fair Generation Protocol

## Purpose

The previous 20k multiscale backbone runs covered CoFiTok light denoise-path training, but the generated-quality comparison still used the older epsilon-only backbone. This protocol adds an exact multiscale epsilon-only control and evaluates both variants with the same dataset, training length, predictor family, DDIM sampler, sample count, and Inception-Frechet feature path.

## Configs

- `configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json`
- `configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json`
- Existing CoFiTok controls:
  - `configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json`
  - `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json`

Both new controls keep restricted, token-only, bias-free `S_k`; only the predictor backbone is matched to the multiscale runs and the loss is changed to epsilon-only.

## Remote command

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
nohup bash artifacts/runbooks/multiscale_fair_generation_2026-07-08.sh \
  > artifacts/reports/multiscale_fair_generation_2026-07-08.log 2>&1 &
```

## Evaluation plan

| Dataset | Variant | Train steps | Samples | Real images | Sampler |
|---|---:|---:|---:|---:|---|
| tiny_imagenet_200 | epsilon-only multiscale | 20000 | 10000 | 10000 | DDIM-50 |
| tiny_imagenet_200 | CoFiTok light multiscale | 20000 | 10000 | 10000 | DDIM-50 |
| imagenet_1k_64x64_hf | epsilon-only multiscale | 20000 | 50000 | 50000 | DDIM-50 |
| imagenet_1k_64x64_hf | CoFiTok light multiscale | 20000 | 50000 | 50000 | DDIM-50 |

## Interpretation rule

The claim can only be strengthened if CoFiTok improves or closely matches epsilon-only under this matched multiscale setting while preserving prefix controllability and restricted-synthesis diagnostics. If epsilon-only remains better, the record should state that CoFiTok currently provides controllable denoising-factorization evidence but not a generation-quality win at this scale.

## Results

Remote run:

```text
artifacts/reports/multiscale_fair_generation_2026-07-08.log
```

Summary counts after this run:

```text
train: 111
order_eval: 95
quality: 58
sampling: 22
generated_quality: 29
```

Matched multiscale generated-quality rows:

| Dataset | Variant | Backbone | Train steps | Samples | Real images | Low-res Frechet proxy | Inception Frechet | Elapsed seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only | multiscale_unet | 20000 | 10000 | 10000 | 5.2057 | 136.7132 | 83.70 |
| Tiny ImageNet-200 | CoFiTok K8 light | multiscale_unet | 20000 | 10000 | 10000 | 5.7001 | 141.4577 | 83.60 |
| ImageNet-64 HF | epsilon-only | multiscale_unet | 20000 | 50000 | 50000 | 5.5593 | 134.2443 | 400.79 |
| ImageNet-64 HF | CoFiTok K8 light | multiscale_unet | 20000 | 50000 | 50000 | 6.2099 | 136.8376 | 398.98 |

Matched multiscale train rows:

| Dataset | Variant | Final clean MSE | Path AUC | Effective K | Zero-token ratio |
|---|---|---:|---:|---:|---:|
| Tiny ImageNet-200 | epsilon-only | 0.117273 | 0.899751 | 6.039 | 0.0000 |
| Tiny ImageNet-200 | CoFiTok K8 light | 0.123434 | 0.034593 | 6.761 | 0.0000 |
| ImageNet-64 HF | epsilon-only | 0.062801 | 0.914623 | 6.035 | 0.0000 |
| ImageNet-64 HF | CoFiTok K8 light | 0.065420 | 0.025294 | 6.721 | 0.0000 |

Interpretation:

- The multiscale predictor materially improves absolute generated-sample metrics versus the older tiny-conv 20k sampler rows on both variants.
- Under the matched multiscale 20k protocol, epsilon-only remains better than CoFiTok K8 light on generated-sample Frechet metrics for both datasets.
- CoFiTok K8 light still preserves the intended controllability evidence: much lower path AUC, broader effective token use, and zero-token ratio 0.0 with restricted `S_k`.
- The current paper claim should remain: prefix-controllable ordered denoising factorization at comparable endpoint quality, not unconditional generation-quality superiority.
