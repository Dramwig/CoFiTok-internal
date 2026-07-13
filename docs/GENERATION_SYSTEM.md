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
- Exact resume also requires the fully resolved current training config to
  equal the config embedded in the checkpoint before any model, EMA, optimizer,
  scheduler, scaler, or RNG state is restored. Changes such as micro-batch or
  gradient accumulation fail with named config paths instead of silently
  changing the training trajectory.
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
  inference-code Git revision/branch/tracked-dirty state, latency, and CUDA
  baseline/peak memory, and fails before a sampling manifest or partial image
  directory is created when the target inference shape OOMs. Runtime selection
  never reuses a preflight from another revision or a dirty worktree.
- Classifier-free guidance can evaluate conditional and unconditional branches
  in one batch, with a sequential fallback. The selected execution mode is part
  of the immutable sampling manifest and must match across compared methods.
- Every global sample index owns an independent RNG stream. The stream is
  invariant to batch size and resume boundaries, and the same numbered sample
  uses the same stream at every prefix budget. Prefix comparisons are therefore
  paired rather than comparisons between unrelated initial noises.
- Sampling writes an immutable checkpoint-and-protocol manifest before the
  first image, including revision/branch/tracked-dirty code provenance.
  `--resume` accepts only an exact manifest match, skips completed numbered
  images, and regenerates missing images from their original streams. Metrics,
  promotion/final gates, and completion audit preserve and validate that code
  provenance; formal post-evaluation runbooks also fail early on tracked dirt.
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
   all distribution metrics are finite and inside their mathematical ranges,
   CoFiTok FID is at most 20.0, precision and recall are each at least 0.30,
   and neither precision nor recall is more than 0.05 below matched dense.

The 10% gate is an engineering and architecture decision point. It is not a
replacement for the full-data result and must not overwrite locked paper tables.
The precision/recall floors are conservative non-collapse readiness checks, not
a generation-SOTA claim. The final completion audit binds FID/precision/recall
back to the formal generation reports and rejects a gate with weaker thresholds.

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
policy explicitly forbids cross-panel numeric ranking. The comparison binds the
tracked official table by SHA256 and preserves alias, method identity, eval-only
status, protocol, source metrics path, and table role. The completion audit
recomputes direct training/sampling cost fields and cross-checks every contextual
row against that source table. See
`docs/records/2026-07-12_generation_baseline_comparison_provenance.md`.

Full-training milestone checkpoints are reproducible assets, not disposable
rolling saves. Both 300K configs protect 50K, 100K, 200K, and 300K while also
retaining the newest three recovery checkpoints. The final full-training
auditor fails if any reached protected checkpoint is absent, and integrity
sidecars follow the same retention decision. See
`docs/records/2026-07-12_generation_milestone_checkpoint_retention.md`.

The matched direct panel also reports compute instead of assuming equal cost
from equal steps. Checkpoints carry cumulative elapsed time and peak VRAM across
segmented resumes. `cofitok.generation_cost.training_cost_summary` validates
effective batch and exact images seen, then exposes training hours,
images/second, and peak memory in the final JSON/Markdown/CSV comparison. The
full paired-config preflight confirms 62,950,800 vs 62,824,707 parameters
(+0.200706%). See
`docs/records/2026-07-12_generation_matched_compute_accounting.md`.
`cofitok.generation_pair.generation_pair_contract` additionally compares every
shared resolved model field, all data/diffusion/runtime/optimization fields,
the positive primary epsilon loss, and exact factorized-versus-dense identities.
Only token/synthesis/feedback fields and CoFiTok auxiliary losses may differ;
the dense baseline must keep every auxiliary loss weight at zero. Both the
training-pair validator and promotion/final gates use this contract. See
`docs/records/2026-07-12_generation_pair_contract.md`.

The revision transition after the active 10% pair is also gated. Run
`scripts/deploy_generation_posttraining_pipeline.ps1` locally only after both
50K training reports are complete. It verifies the pinned remote revision,
clean tracked files, completed reports, and absent training processes before a
fast-forward-only bundle deployment. Before merge it rejects any remote
untracked path that would become tracked by the target revision. Remote tests
and shell syntax checks must pass, then an atomic receipt binds the bundle
SHA256, validated 10% pair, pinned source revision, exact target revision, and
clean tracked state before it launches
`artifacts/runbooks/generation_complete_pipeline_after_10pct.sh`. That pipeline
uses an exclusive lock and atomic stage status, runs the 10K promotion gate,
the alternating full 300K queue, the formal 50K-sample evaluation, and both
gate decisions in order. A held gate or interrupted stage is recorded as a
failure rather than reported as generation readiness.
The final completion audit requires the deployment receipt, so changing remote
HEAD alone cannot prove a controlled revision transition.
The local bundle is prerequisite-aware: it advertises only the upgrade HEAD
and excludes history reachable from the pinned training commit. The local
deployer verifies the single advertised head before transfer, while the remote
`git bundle verify` proves that the pinned prerequisite exists before fetch.
This preserves the same fast-forward trust boundary without retransmitting the
repository's full historical object graph.
Before moving HEAD, the remote helper fetches only the verified bundle objects
and extracts the target revision's training-pair validator and complete
`src/cofitok/` package with `git archive` into a temporary isolated Python
path. Pre-deployment validation therefore runs the exact target code
even though the worktree is still pinned to the legacy training revision. The
temporary tree is removed on exit, and a failed validation never merges the
target revision.

While the pinned legacy 10% pair runs, a separate read-only monitor may be
launched from `/tmp` without changing the training revision. It atomically
records pair stage, steps, metric age, checkpoints, process presence, disk, and
GPU state; it fails closed on a 30-minute metric stall or an incomplete queue
with no process after a transition grace period. It never loads checkpoints,
uses the GPU, restarts training, or modifies either run directory. See
`docs/records/2026-07-12_generation_10pct_readonly_monitor.md`.

Both promotion decisions are provenance gates, not only metric thresholds.
They require the CoFiTok and dense training reports to share the same
40-character `git.revision` on `scale/generative-system`, use the same real
ImageNet-256 directory and evaluator, and contain exactly the requested 10K or
50K generated samples. Both metrics reports must also bind the same clean
evaluator revision/branch; at full scale it must equal the full training and
sampling revision. Ordered-prefix/zero/shuffle checkpoint diagnostics carry an
independent mechanism-evaluator Git state under the same matched clean/full
revision rules. Formal class-conditional sampling must start at index zero, use
balanced modulo labels and EMA weights, preserve per-sample random streams
across batch-size/resume changes, and bind both checkpoint and sample bytes by
SHA256. The transition-field correction and negative tests are
recorded in
`docs/records/2026-07-12_generation_gate_provenance_hardening.md`.

Exact training recovery also covers the append-only metrics history. On
`--resume`, `cofitok.training.metrics.reconcile_metrics_for_resume` keeps the
latest row for each step at or before the durable checkpoint, archives
checkpoint-ahead or superseded rows with a content-addressed SHA256, and
atomically rewrites canonical JSONL before training continues. The result is
bound into the resumed run manifest. A fresh invocation refuses any output
directory containing prior training state. Train/eval iterators are created
after RNG and sampler restoration, and the deterministic validation iterator
is advanced to the batch implied by the restored optimizer step. A CPU
trajectory test requires uninterrupted and segmented-resume states to match
exactly. See
`docs/records/2026-07-12_generation_resume_metrics_reconciliation.md`.

Inference has a strict checkpoint trust boundary. The shared generation loader
must verify the adjacent integrity sidecar's filename, byte size, SHA256,
format version, and step before deserializing or applying EMA/model weights.
The resolved sidecar path is propagated through preflight, sampling, metrics,
and gate provenance. Legacy 10% weights therefore require the migration stage;
full checkpoints satisfy this contract at creation. See
`docs/records/2026-07-12_generation_checkpoint_trust_boundary.md`.

Long sampling runs expose atomic, resumable progress. Each completed batch
updates `sampling_progress.json`, which is bound to the immutable sampling
manifest SHA and carries cumulative elapsed time, throughput, ETA, PID/host,
completed count, and failure/completion state. Formal metrics recompute the
manifest SHA and require completed progress with exact counts/budgets/sample
digests; the promotion/final gate enforces the same boundary for both methods.
The gate also requires finite positive cumulative sampling time, and the final
comparison publishes sampling elapsed time and throughput.
See `docs/records/2026-07-12_generation_sampling_progress.md`.

Large-scale completion is fail-closed. Before the completion pipeline can
publish `pass`, `scripts/audit_large_scale_generation_completion.py` must verify
the pinned 10% pair and promotion gate, full matched 300K pair, training audits,
all four milestones, formal paired 50K sampling, final gate, and final comparison.
Missing evidence is `in_progress`, contradictory evidence is `failed`, and only
the full chain is `complete`. See
`docs/records/2026-07-12_large_scale_generation_completion_audit.md`.

Before full 300K training starts, both methods run the same checkpoint-free
training-runtime candidates `16x4`, `32x2`, and `64x1`. Selection preserves
effective batch 64, requires both methods to fit below 90% VRAM, and minimizes
the slower method's synchronized optimizer-step time. The selected microbatch
and accumulation are applied to every full-training segment and verified again
by the completion audit. See
`docs/records/2026-07-12_generation_training_runtime_selection.md`.

Formal 10K and 50K sampling also selects one shared batch from
`16,32,64,128`. Both checkpoints run repeated synchronized EMA/CFG forwards;
eligible candidates must pass for both methods below 90% VRAM, and selection
maximizes the slower method's output-images/second. Per-index random streams
keep generated samples invariant to the selected batch. The final comparison
reports batch, elapsed time, and realized throughput. See
`docs/records/2026-07-12_generation_sampling_batch_selection.md`.

Stable inference is exposed through `cofitok.generation.GenerationSession` and
immutable `GenerationRequest` objects. The class/seed/prefix CLI and formal
sampler share this implementation; both attach checkpoint integrity and exact
protocol provenance. Final 50K evidence must declare inference API version 1.
See `docs/INFERENCE.md` and
`docs/records/2026-07-12_stable_generation_session.md`.
The formal sampler also has an end-to-end CPU checkpoint-to-PNG subprocess
test and derives actual DDIM timesteps from `GenerationSession.schedule`, so
the exact 10K/50K CLI entry point is covered beyond lower-level sampler tests.
See `docs/records/2026-07-12_formal_sampling_cli_integration.md`.
All Python entry points referenced by the post-training runbooks are also
source-to-source contract tested: the test extracts every multiline command,
starts the corresponding CLI through `--help`, and verifies every option used
by the runbook exists in that parser. See
`docs/records/2026-07-12_generation_runbook_cli_contracts.md`.

Promotion and final post-evaluation also create deterministic fixed-index
CoFiTok/dense endpoint panels plus a separate `1/2/4/8` prefix-path panel.
Their report binds every panel and source image to checkpoint/sample-set SHA,
rejects exact duplicates in the fixed endpoint selections, and remains explicitly
non-quantitative. It also records the builder Git revision, branch, and tracked
worktree state; the completion audit accepts only the clean full-training
revision. This visual evidence is required in addition to formal metrics. See
`docs/records/2026-07-12_generation_visual_quality_audit.md`.

The multi-week completion run is wrapped by a bounded supervisor. Only
recoverable execution stages (`posteval_10pct`, `full_training`,
`full_posteval`, `inference_export`) receive up to four exponential-backoff retries; scientific
gates, preconditions, unknown states, and completion-audit failures stop
immediately. Existing gates and fully paired milestones are evidence-driven
skip points, with protected checkpoint/sidecar checks before milestone reuse.
See `docs/records/2026-07-12_generation_completion_supervisor.md`.

After the final gate, both methods export separate EMA-only deployment
artifacts. Their type-specific sidecars are verified before deserialization;
source training checkpoint SHA, step, artifact SHA/bytes, real-forward preflight,
and short DDIM smoke PNGs are required by completion. These artifacts are smaller
inference copies and never replace exact-resume training checkpoints. See
`docs/records/2026-07-12_deployable_ema_inference_artifact.md`.

Live long-run health can be audited without loading the model or competing for
GPU time using `scripts/audit_generation_training_progress.py`. It verifies
strictly increasing finite metrics, resume-aware timing segments, checkpoint
cadence and `latest.json`, then records recent loss/gradient summaries and ETA.
For the pinned legacy 10% queue, `legacy_compute` records the newest checkpoint
byte count and SHA256 without modifying it; the post-training migration later
binds those bytes to a sidecar. New full runs use `--integrity-policy required`,
which rejects a missing or mismatched sidecar and `latest.json` integrity binding.
Logged gradient norms are explicitly treated as pre-clipping total norms.
