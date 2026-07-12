# P1 Tokenizer Reconstruction Eval256 Completion - 2026-07-10

## Result

Status: `completed_eval_only`

FlexTok and TiTok / 1D Tokenizer were evaluated on all eight project datasets
with 256 images per dataset, resized to 256 before tokenizer reconstruction.
This is a secondary tokenizer reconstruction table, not a P0 diffusion
generation table.

## Evidence

Table:

```text
CoFiTok-internal/artifacts/reports/baselines/p1_tokenizer_reconstruction_eval256_2026-07-10/p1_tokenizer_reconstruction_table.md
```

Run status:

```text
CoFiTok-internal/artifacts/runbooks/p1_tokenizer_reconstruction_eval256_2026-07-10.status
```

Remote run:

```text
MAX_IMAGES=256 TAG=eval256_2026-07-10 bash artifacts/runbooks/p1_tokenizer_reconstruction_eval_2026-07-10.sh
status=finished rc=0 finished_at=2026-07-10T08:47:47+08:00
```

## Summary

| dataset | FlexTok PSNR | TiTok PSNR | FlexTok lowres F | TiTok lowres F |
| --- | ---: | ---: | ---: | ---: |
| cifar10 | 26.7679 | 19.3461 | 0.0373 | 0.1678 |
| tiny_imagenet_200 | 21.8110 | 17.3008 | 0.0442 | 0.1600 |
| imagenet_1k_64x64_hf | 23.5966 | 18.0363 | 0.0390 | 0.1817 |
| downsampled_imagenet_64 | 22.5683 | 16.7253 | 0.0418 | 0.1956 |
| ffhq_64 | 24.0572 | 17.3911 | 0.0422 | 0.2307 |
| afhqv2_64 | 24.0421 | 17.2868 | 0.0358 | 0.2168 |
| imagenet_256_10pct | 19.1230 | 15.5439 | 0.0438 | 0.1959 |
| imagenet_256 | 19.1226 | 15.5439 | 0.0438 | 0.1959 |

## Interpretation

FlexTok/TiTok address visual-tokenizer related-work pressure, but they are
decoder-based image reconstruction baselines. They do not test the same
dense-noise factorization claim as CoFiTok and must not be counted as completed
P0 generation baselines.
