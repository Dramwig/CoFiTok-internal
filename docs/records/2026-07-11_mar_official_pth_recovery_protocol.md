# MAR official PTH recovery protocol (2026-07-11)

## Why this recovery is required

The completed 50K run used `jadechoghari/mar` community-converted
`mar-base.safetensors` and `kl16.safetensors`. Its ADM metrics (FID 5.5610,
IS 52.0020) differ substantially from the LTH14 MAR-B README values (FID 2.31,
IS 281.7). The result is mechanically complete, but it must be labeled as a
community-conversion audit rather than an official-checkpoint reproduction.

## Recovered assets

The Hugging Face repository history retains the original pickle checkpoints at
revision `773c9bdcd98740c7b876ce0d66f684a0e2468990`:

| asset | bytes | sha256 |
|---|---:|---|
| `checkpoint-last.pth` | 1,663,614,946 | `7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c` |
| `kl16.ckpt` | 265,900,046 | `34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f` |

Remote destination:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/official_pth
```

The model must be loaded from `checkpoint["model_ema"]`; the VAE must be loaded
from `checkpoint["model"]`. Sampling uses the pinned LTH14 repository commit
`c6d53f7fa6427634b5850ebed771b7c2d19ea21f` and the README MAR-B parameters:

```text
num_iter=256, num_sampling_steps=100, cfg=2.9,
cfg_schedule=linear, temperature=1.0, 50 images/class
```

## Gates before formal 50K

1. Verify both hashes and checkpoint dictionary keys.
2. Strict-load `model_ema` into the official MAR-B architecture.
3. Compare community safetensors against both `model` and `model_ema` to identify
   which state was converted.
4. Pass a small official-PTH generation smoke test with valid RGB 256x256 PNGs.
5. Run the official-PTH 50K into a separate sample directory and evaluate with
   the same ADM reference/evaluator used by D-AR and ReTok.

Both MAR rows remain secondary ImageNet-256 eval-only evidence and must not enter
the P0 same-budget all-dataset table.

## Audit result and launch

The conversion audit found:

```text
mar-base.safetensors vs checkpoint['model']: 364/364 tensors exact
mar-base.safetensors vs checkpoint['model_ema']: 0/364 tensors exact
EMA RMSE: 0.0208652; max absolute difference: 0.5549803
kl16.safetensors vs kl16.ckpt['model']: 312/312 tensors exact
```

Source comparison of `models/mar.py` and `models/diffloss.py` against the HF
conversion found only package-relative imports and a fixed ImageNet
`class_num=1000`; the sampling implementation is otherwise unchanged. This
isolates the failed 50K result to the non-EMA state rather than an algorithmic
fork in the adapter source.

An 8-image EMA smoke passed strict loading and RGB 256x256 validation. The
official PTH EMA 50K run is active under:

```text
artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.sh
```

The completed non-EMA run remains at FID 5.5610 / IS 52.0020 as an audit-only
artifact. It must not populate the MAR paper row.
