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
- Every new checkpoint has an atomic integrity sidecar containing byte count,
  SHA256, step, and payload format. `latest.json` is bound to the same metadata;
  automatic resume verifies the pointer, sidecar, file bytes, payload step, and
  payload format before restoring state. Full readiness additionally requires
  the final training hash to match the checkpoint used for sampling.
- The active 10% matched queue predates integrity sidecars. Its post-evaluation
  runbook therefore performs a one-time legacy migration: load and validate the
  final payload's exact-resume fields, hash the immutable checkpoint bytes, add
  the sidecar, and atomically bind `latest.json` plus `training_report.json` to
  that hash. It never rewrites the checkpoint payload.
- Full 300K runs checkpoint every 5K optimizer steps and retain the latest
  three states, matching the 10% gate cadence while bounding recovery loss on
  the multi-day full-data queue.
- Zero-weight objectives are not materialized in the production graph. The
  structural `S_k(0)=0` invariant is enforced by architecture and tested
  separately instead of paying for a gradient-free zero-token term every step.
- Production inference loads EMA by default and supports deterministic DDIM,
  classifier-free guidance, guidance rescaling, prefix budgets, and resumable
  numbered PNG export.
- Formal post-evaluation runbooks first execute one real model forward with the
  requested EMA/model weights, precision, prefix budget, batch size, and CFG
  batching mode. The preflight records checkpoint SHA256, output finiteness,
  latency, and CUDA baseline/peak memory, and fails before a sampling manifest
  or partial image directory is created when the target inference shape OOMs.
- Classifier-free guidance can evaluate conditional and unconditional branches
  in one batch, with a sequential fallback. The selected execution mode is part
  of the immutable sampling manifest and must match across compared methods.
- Every global sample index owns an independent RNG stream. The stream is
  invariant to batch size and resume boundaries, and the same numbered sample
  uses the same stream at every prefix budget. Prefix comparisons are therefore
  paired rather than comparisons between unrelated initial noises.
- Sampling writes an immutable checkpoint-and-protocol manifest before the
  first image. `--resume` accepts only an exact manifest match, skips completed
  numbered images, and regenerates missing images from their original streams.
- Each PNG is encoded to a same-directory partial file and atomically published
  only after encoding succeeds. JSON manifests and reports use the same atomic
  replacement rule, so interruption cannot turn a partial file into apparent
  completion or destroy the last valid report.
- Resume and completion checks verify PNG decoding, CRC, dimensions, and color
  mode rather than file existence alone. The expected `[C, H, W]` is immutable
  sampling provenance; formal metrics revalidate every generated image and the
  promotion gate requires the recorded shape to match each training config.
- Completed sampling reports commit to each exact numbered sample set with a
  canonical SHA256 over filename and file bytes. Metric evaluation recomputes
  this digest before FID, and the gate requires both matched methods to carry a
  valid sample-set digest alongside their checkpoint hashes.
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
   hashes. Declare the system ready only if CoFiTok keeps its prefix-control
   advantage without a material endpoint generation regression against dense,
   all distribution metrics are finite, and CoFiTok FID is at most 20.0.

The 10% gate is an engineering and architecture decision point. It is not a
replacement for the full-data result and must not overwrite locked paper tables.

The full queue alternates CoFiTok and dense at 50K, 100K, 200K, and 300K
milestones. At each matched point it produces 2,048 fixed-protocol EMA samples
at DDIM-50 / CFG 1.5, FID/IS trend metrics, and a 256-image mechanism audit.
These milestone reports are explicitly non-claim diagnostics: they expose
severe quality or ordering regressions before another long segment consumes GPU
time, but they never replace the final 50K DDIM-250 evaluation.

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
unless the 10% promotion report passes. It uses exact-resume training segments
and the read-only `generation_full_milestone_eval.sh` quality protocol at each
matched milestone. After both full-data 300K runs finish,
`artifacts/runbooks/generation_full_posteval_50k.sh` produces matched 50K EMA
samples at DDIM-250 / CFG 1.5, full generation metrics, checkpoint mechanism
diagnostics, the final large-scale gate report, and a two-tier strong-baseline
comparison. CoFiTok versus `dense_identity` is the direct matched-training
panel. D-AR, MAR, and ReTok remain a separate official-pretrained contextual
panel because their training budgets and ADM evaluator differ; machine-readable
policy explicitly forbids cross-panel numeric ranking.

Live long-run health can be audited without loading the model or competing for
GPU time using `scripts/audit_generation_training_progress.py`. It verifies
strictly increasing finite metrics, resume-aware timing segments, checkpoint
cadence and `latest.json`, then records recent loss/gradient summaries and ETA.
Logged gradient norms are explicitly treated as pre-clipping total norms.
