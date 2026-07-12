# CoFiTok Generation System

This branch upgrades CoFiTok from short-budget mechanism validation to a
class-conditional ImageNet-256 generation system. Locked paper evidence remains
on `paper-evidence-locked`; generation work lives on `scale/generative-system`.

## System boundary

- `ScalableUNetTokenPredictor` is the expressive `T_k`: ADM-style residual
  U-Net, spatial attention, timestep conditioning, ImageNet class conditioning,
  CFG dropout, and optional activation checkpointing.
- `RestrictedSynthesisBank` remains the default `S_k`. Every `S_k` receives only
  its current token, is bias-free and linear/local, and preserves `S_k(0)=0`.
- The dense control uses the identical U-Net and training protocol with one
  direct `dense_identity` epsilon head.
- Production training uses bf16, gradient accumulation, gradient clipping,
  cosine LR, EMA, isolated DataLoader RNG, atomic checkpoints, retention, and
  exact model/optimizer/scheduler/RNG/sampler recovery.
- Zero-weight objectives are not materialized in the production graph. The
  structural `S_k(0)=0` invariant is enforced by architecture and tested
  separately instead of paying for a gradient-free zero-token term every step.
- Production inference loads EMA by default and supports deterministic DDIM,
  classifier-free guidance, guidance rescaling, prefix budgets, and resumable
  numbered PNG export.
- Every global sample index owns an independent RNG stream. The stream is
  invariant to batch size and resume boundaries, and the same numbered sample
  uses the same stream at every prefix budget. Prefix comparisons are therefore
  paired rather than comparisons between unrelated initial noises.
- Sampling writes an immutable checkpoint-and-protocol manifest before the
  first image. `--resume` accepts only an exact manifest match, skips completed
  numbered images, and regenerates missing images from their original streams.
- Generation evaluation uses `torch-fidelity==0.4.x` with generated samples as
  input 1 and the recursive 50K ImageNet validation directory as input 2. One
  report records FID, Inception Score, precision, recall, exact image counts,
  package version, seed, cache name, and runtime.
- Formal metric evaluation requires the corresponding `sampling_report.json`,
  an exact zero-based numbered image set, and a valid checkpoint SHA256. The
  promotion gate cross-checks that sample metrics and mechanism diagnostics use
  the same checkpoint bytes and matched sampling protocol.

## Server paths

```text
code:        /root/autodl-tmp/CoFiTok/CoFiTok-internal
datasets:    /root/autodl-tmp/CoFiTok/datasets
checkpoints: /root/autodl-tmp/CoFiTok/checkpoints/generation
```

## Promotion gates

1. Code gate: full tests, CPU exact-resume smoke, CUDA bf16 smoke, zero-token
   contract, and matched parameter audit pass.
2. Data gate: 10 real ImageNet-256 optimizer steps complete with finite losses,
   stable gradients, recorded throughput, and a restart from checkpoint.
3. Scaling gate: matched 50K-step CoFiTok and dense runs on
   `imagenet_256_10pct`; generate at least 10K EMA samples per method and compute
   FID under one real-image directory and evaluator version.
4. Full gate: matched 300K-step runs on full `imagenet_256`, 50K EMA samples,
   official FID plus IS/precision/recall, prefix diagnostics, and checkpoint
   hashes. Promote only if CoFiTok keeps its prefix-control advantage without a
   material endpoint generation regression against dense.

The 10% gate is an engineering and architecture decision point. It is not a
replacement for the full-data result and must not overwrite locked paper tables.

## Commands

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
export PYTHONPATH=src

python scripts/validate_generation_configs.py \
  --cofitok-config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_dense_50k.json \
  --output artifacts/reports/generation/config_pair_10pct.json

python scripts/train_generation.py \
  --config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_cofitok_k8_50k

python scripts/train_generation.py \
  --config configs/generation/imagenet256_10pct_cofitok_k8_50k.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_cofitok_k8_50k \
  --resume auto

python scripts/generate_samples.py \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/generation/<run>/checkpoint_step_00050000.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/<run>/samples_50k_cfg15 \
  --num-samples 50000 --batch-size 32 --sample-steps 250 \
  --guidance-scale 1.5 --weights ema

python scripts/evaluate_generation_metrics.py \
  --real-dir /root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val \
  --generated-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/<run>/samples_50k_cfg15/prefix_8 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/<run>/samples_50k_cfg15/metrics \
  --cache-root /root/autodl-tmp/CoFiTok/checkpoints/generation/eval_cache/torch_fidelity
```

The 10% post-training gate is encoded in
`artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh`. It refuses partial
or dirty-worktree training reports, generates matched 10K EMA samples at DDIM
100 / CFG 1.5 with manifest-checked resume, evaluates both methods with the same cached real features, and
exports a 64-image CoFiTok prefix diagnostic at budgets 1/2/4/8.

Full-scale execution is deliberately gated. The runbook
`artifacts/runbooks/generation_full_matched_300k_after_gate.sh` refuses to start
unless the 10% promotion report passes. After both full-data 300K runs finish,
`artifacts/runbooks/generation_full_posteval_50k.sh` produces matched 50K EMA
samples at DDIM-250 / CFG 1.5, full generation metrics, checkpoint mechanism
diagnostics, and the final large-scale gate report.
