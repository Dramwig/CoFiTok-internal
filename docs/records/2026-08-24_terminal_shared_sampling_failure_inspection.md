# Terminal shared-sampling failure inspection (2026-08-24)

## Scope

This is a CPU-only, permanently non-authorizing inspection of already generated
100K terminal samples. It did not load a checkpoint, launch sampling, alter the
running Dense terminal process, or authorize any training, promotion, export,
or release.

The inspected source directories are the matched `DDIM-100`, CFG `1.5`, hard
`x0`-clip terminal outputs under:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
```

The fixed subset contains indices `000000` through `000023` for CoFiTok prefix
8 and dense prefix 1. The subset digests using the repository's framed
`sample_set_sha256` implementation are:

- CoFiTok: `0d97199fa3e04f33fd18c840d9270022bf4466f59ac508aed0ebc4bde9e0d789`
- dense identity: `bcf4742d892752bbd519676a5a3405b2bfdcf47dd04111992d76a26b0d71a529`

## Observation

The two methods reproduce nearly the same scene layout, color field, and
high-frequency impulse artifacts for matched seeds. Across the 24 pairs:

| diagnostic | value |
|---|---:|
| paired pixel correlation, mean | 0.937659 |
| paired pixel correlation, median | 0.949378 |
| paired PSNR, mean | 21.3948 dB |
| paired pixel MSE, mean | 0.00842078 |

The per-method artifact summaries were:

| diagnostic | CoFiTok | dense identity |
|---|---:|---:|
| channel saturation fraction | 0.004346 | 0.013565 |
| total variation | 0.123634 | 0.116175 |
| mean absolute residual from 3x3 median | 0.038130 | 0.035102 |
| median residual above 0.2 | 0.033143 | 0.039836 |

These values do not prove generation quality, but they contradict a simple
factorization-only failure explanation. The dominant visual defects are shared
by the matched CoFiTok and dense systems, while dense is more saturated and has
more large local impulse outliers in this fixed subset.

## Sampling-system implication

The locked cosine schedule starts the formal sampler at timestep `999`, where
`alpha_bar=2.42796e-9` and `sqrt(alpha_bar)=4.92743e-5`. Converting epsilon to
`x0` at that point amplifies epsilon error by approximately twenty thousand
before hard clipping. Candidate nonterminal starts materially reduce that
amplification while leaving the starting marginal close to unit Gaussian:

| start timestep | sqrt(alpha_bar) | schedule sigma |
|---:|---:|---:|
| 999 | 0.0000493 | 1.000000 |
| 995 | 0.0062325 | 0.999981 |
| 990 | 0.0140229 | 0.999902 |
| 975 | 0.0373868 | 0.999301 |

The completed immutable sampling-recovery confirmation changed guidance
rescale to `1.0` but retained timestep `999` and hard `x0` clipping. Its hold
therefore does not test nonterminal initialization, dynamic thresholding, or
Min-SNR training.

## Engineering consequence

The quality-repair branch adds those controls without changing CoFiTok's
epsilon-token semantics. The existing formal milestone/scaling/full contracts
continue to reject them, so they cannot silently replace locked evidence. Any
future GPU diagnostic must be created after the terminal route receipt, use new
versioned roots, remain matched across CoFiTok and dense, and have a separate
non-authorizing execution receipt.
