# CoFiTok Paper Claim Pack

Source: `paper_evidence_report.md`

## Recommended Main Claim

CoFiTok factorizes pixel-space diffusion noise prediction into an ordered sequence of restricted denoising components. The experiments support that this factorization produces prefix-controllable partial denoising and that the restricted synthesis operator is important for avoiding decoder-like shortcuts.

## Claims That Current Evidence Supports

- CoFiTok gives a measurable prefix-control signal: path AUC is lower than same-backbone dense prediction on 8/8 datasets.
- CoFiTok is not just a channel-mask decomposition: the channel-mask ablation is worse by 1.076 dB PSNR on average.
- A stronger synthesis operator changes the behavior in the wrong direction: deep-`S_k` has nonzero zero-token ratio on 8/8 datasets, while CoFiTok remains zero in current rows.
- Random-token diagnostic evidence is available both numerically and as an appendix panel; the ImageNet-64 K=8 panel has ordered final MSE 0.0991 versus random-token final MSE 27.3458.
- The method works across CIFAR-10, Tiny-ImageNet, ImageNet-64 variants, FFHQ-64, AFHQv2-64, ImageNet-256 10%, and ImageNet-256 full under the current 3k/5k-step protocol.
- FlexTok and TiTok reconstruction rows are useful as related tokenizer baselines, but only as eval-only reconstruction evidence.
- D-AR official ImageNet-256 50K eval-only is complete as a secondary related-method row, not as a same-budget P0 training baseline.
- MAR and ReTok now have official-checkpoint protocol evidence: MAR HF safetensors generation smoke; ReTok official GPT+VQ ImageNet-256 50K eval-only metrics are complete. ReTok remains secondary-only because it is not same-budget retraining.

## Claims To Avoid

- Do not claim generation-quality SOTA.
- Do not claim CoFiTok beats EDM as a pixel diffusion generator.
- Do not claim CoFiTok is generally better than dense epsilon prediction on all quality metrics.
- Do not claim superiority over FlexTok/TiTok as visual tokenizers; their task is decoder-based reconstruction, not dense noise factorization.
- Do not count D-AR/MAR/ReTok as completed fair baselines; D-AR and ReTok 50K rows are secondary eval-only, and MAR smoke is protocol evidence.

## Main Result Table Recommendation

Use a two-part table:

1. **P0 generation and denoising table**: CoFiTok, Dense epsilon, Improved DDPM, EDM, plus D-AR as protocol-blocked. Include lowres/Inception Frechet for generation and PSNR/path AUC/effective tokens for CoFiTok/dense diagnostics.
2. **P1 related tokenizer table**: FlexTok and TiTok eval-only reconstruction at 256x256. Label explicitly as "official pretrained reconstruction, not retrained generation".
3. **Secondary related-method table**: D-AR and ReTok official ImageNet-256 50K eval-only rows are complete; MAR has official-checkpoint smoke evidence only.

Do not merge P1 reconstruction rows into the P0 generation table.

## Suggested Result Paragraph

Across eight datasets, CoFiTok does not dominate monolithic pixel diffusion on Frechet-style generation metrics: EDM is strongest on most low-resolution Frechet rows and dense epsilon is strongest on ImageNet-256 Inception-style rows. The advantage of CoFiTok is instead diagnostic and structural. CoFiTok achieves lower prefix path AUC than the same-backbone dense predictor on all eight datasets, improves over the channel-mask decomposition by 1.076 dB PSNR on average, and keeps the zero-token diagnostic at zero where a stronger deep synthesis operator exhibits nonzero zero-token leakage on all datasets. These results support the scoped claim that ordered restricted denoising-token factorization yields controllable partial denoising, not the broader claim of unconditional generation-quality superiority.

## Reviewer Risk Assessment

| risk | current answer | remaining weakness |
|---|---|---|
| "This is just another tokenizer." | CoFiTok tokens expand to dense negative-noise components through restricted condition-free `S_k`, not image latents decoded by a VAE-like decoder. | Need figures that show pixel-space prefix denoising, not just tables. |
| "Why not a dense predictor?" | Same-backbone dense has better or similar PSNR/generation metrics, but much worse path AUC on 8/8 datasets. | Need explain path AUC clearly and visually. |
| "Why not EDM?" | EDM is a stronger generator; CoFiTok's claim is controllable dense-noise factorization, not raw generation quality. | Main paper must not frame EDM as defeated. |
| "Is `S_k` secretly a decoder?" | Zero-token ratio stays zero for CoFiTok; deep-`S_k` ablation shows leakage; an appendix random-token panel is available. | Decide whether to include the appendix random-token panel in supplemental material. |
| "Closest AR/diffusion baselines missing?" | D-AR and ReTok official ImageNet-256 50K eval-only rows are complete as secondary rows; MAR has HF safetensors smoke. FlexTok/TiTok are eval-only tokenizer reconstruction baselines. | Fair same-budget D-AR/MAR/ReTok training is still absent; MAR 50K is not runnable yet, and ReTok 50K is complete as secondary eval-only evidence. |

## Venue Stance

Current evidence is plausible for a top-tier submission only under a narrow method-and-diagnostics framing. It is not strong enough for a broad generation-performance paper. The paper should sell:

- new factorization target: dense diffusion noise prediction;
- restricted synthesis as a necessary constraint;
- prefix denoising as the user-visible property;
- diagnostics and ablations as the core empirical contribution.

The highest-value remaining paper work is figure quality, claim discipline, and implementing MAR HF-safetensors 50K only if a stronger related-method appendix is required.
