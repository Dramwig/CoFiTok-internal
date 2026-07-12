# CoFiTok Paper Claim Pack

## Decision

The current evidence clears the core empirical gates for a scoped method submission about ordered, restricted dense-noise factorization with prefix-controllable denoising. It does not support a broad generation-quality or
visual-tokenizer-superiority claim.

## Supported Evidence

- Broad short-budget coverage: CoFiTok path AUC is lower than the
  endpoint-only factorized control on 8/8 datasets.
- Matched 20k repeats: path AUC is lower in
  4/4 paired dataset-seed runs, with a mean relative
  reduction of 96.18%.
- The repeated structural gain has an explicit endpoint cost: mean endpoint
  x0-MSE change at t=500 is +4.20%, with endpoint wins in
  0/4 pairs.
- ImageNet-256 scaling state: ImageNet-256 K4 20k confirmatory gates: 7/7 passed; overall PASS; mean endpoint-MSE change -6.73%; ordered ranks among all 24 permutations 1/1.
- CoFiTok exceeds the channel-mask control by
  1.066 dB PSNR on average.
- Restricted CoFiTok keeps zero-token ratio at zero; deep-$S_k$ has nonzero
  leakage on 8/8 datasets.
- Related-method state: completed official 50K: D-AR, MAR, ReTok. All rows remain secondary and outside P0 matched-dataset/step training.

## Claims To Avoid

- CoFiTok is not the best lowres Frechet method on the current broad table
  (0/8 wins).
- CoFiTok is best on only 1/8 Inception-style rows.
- Do not claim superiority over EDM, direct dense epsilon, FlexTok, TiTok, D-AR, MAR,
  or ReTok outside their matched task and protocol.
- Do not count official pretrained eval-only rows as matched-dataset/step retraining.

## Reviewer-Risk Boundary

The positive result is mechanistic and interface-level: meaningful denoising
prefixes arise from factorizing dense noise prediction through restricted
token-only synthesis. The main residual risks are small-model/short-budget
scaling, mixed sample quality, and the absence of matched-dataset/step all-dataset
training for architecture-incompatible nearest methods. These must remain
limitations rather than being hidden by the matrix completion counts.
