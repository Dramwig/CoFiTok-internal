# CoFiTok Paper Claim Pack

## Decision

The current evidence conditionally supports a top-tier-style method submission
under the scoped claim of ordered, restricted dense-noise factorization with
prefix-controllable denoising. It does not support a broad generation-quality
or visual-tokenizer-superiority claim.

## Supported Evidence

- Broad short-budget coverage: CoFiTok path AUC is lower than the same-backbone
  dense predictor on 8/8 datasets.
- Matched 20k repeats: path AUC is lower in
  4/4 paired dataset-seed runs, with a mean relative
  reduction of 96.31%.
- The repeated structural gain has an explicit endpoint cost: mean final-MSE
  change is +2.89%, with endpoint wins in
  0/4 pairs.
- CoFiTok exceeds the channel-mask control by
  1.076 dB PSNR on average.
- Restricted CoFiTok keeps zero-token ratio at zero; deep-$S_k$ has nonzero
  leakage on 8/8 datasets.
- Related-method state: completed official 50K: D-AR, ReTok; running/eval pending: MAR. All rows remain secondary and outside P0 fair training.

## Claims To Avoid

- CoFiTok is not the best lowres Frechet method on the current broad table
  (0/8 wins).
- CoFiTok is best on only 1/8 Inception-style rows.
- Do not claim superiority over EDM, dense epsilon, FlexTok, TiTok, D-AR, MAR,
  or ReTok outside their matched task and protocol.
- Do not count official pretrained eval-only rows as same-budget retraining.

## Reviewer-Risk Boundary

The positive result is mechanistic and interface-level: meaningful denoising
prefixes arise from factorizing dense noise prediction through restricted
token-only synthesis. The main residual risks are small-model/short-budget
scaling, mixed sample quality, and the absence of same-budget all-dataset
training for architecture-incompatible nearest methods. These must remain
limitations rather than being hidden by the matrix completion counts.
