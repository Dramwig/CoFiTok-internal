# Generation preflight evidence (2026-07-12)

All GPU reports were produced on `pro6000` with the RTX PRO 6000 Blackwell and
were copied back as small immutable evidence. Checkpoints and sample images stay
under `/root/autodl-tmp/CoFiTok/checkpoints/generation/`.

Key results:

- Matched config audit passed: CoFiTok 62,950,800 parameters vs. dense
  62,824,707, a +0.20% gap with identical backbone/data/runtime/optimizer.
- CUDA bf16 smoke and EMA CFG/DDIM sample export completed.
- cuDNN benchmark search caused an unsafe 84+ GiB allocation and was disabled.
- CoFiTok microbatch sweep at effective batch 64: 8/8 = 8.14 GiB,
  16/4 = 15.05 GiB, 32/2 = 28.88 GiB. Batch 16 / accum 4 was fastest.
- Real-data CoFiTok and dense batch-16 preflights completed with finite train
  and validation losses. Dense peak memory was 13.87 GiB.

These are engineering gates, not generation-quality evidence. The next gate is
the matched 50K-step `imagenet_256_10pct` training and 10K+ EMA-sample FID.
