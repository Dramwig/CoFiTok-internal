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
- Authoritative scaling/full CoFiTok uses true variable-shape token fields:
  spatial strides `[16,16,8,8,4,4,1,1]` and channel counts
  `[4,4,8,8,8,8,1,2]`. Token scalar capacities increase from `1,024` to
  `131,072`, while a dense RGB epsilon field has `196,608` scalars. Each token
  is therefore strictly compressed. `T_k` pools before each variable-channel
  head, feeds back only that emitted token, and `S_k` performs fixed bilinear
  upsampling before its shallow local linear convolution.
- A restricted linear synthesis layout must allocate at least three aggregate
  full-resolution token channels for RGB epsilon prediction. Otherwise the
  highest-frequency output is rank-deficient regardless of model size or
  training time. The formal config and recipe contract reject such layouts;
  the final two rank-complete tokens use one and two full-resolution channels
  while preserving the previous aggregate token-scalar budget exactly.
- The dense control uses the identical U-Net and training protocol with one
  direct `dense_identity` epsilon head.
- Formal scaling and full-data training additionally pass
  `cofitok_generation_training_recipe_v1`. This contract prevents a matched
  pair from passing fairness checks after both methods are identically weakened:
  it locks the ImageNet stage, diffusion horizon/target, U-Net capacity,
  class-dropout CFG training, bf16/TF32 runtime, effective batch 64, optimizer,
  EMA, checkpoint cadence, and CoFiTok denoise-path objective. Runtime selection
  may change micro-batch and accumulation only while their product remains 64.
- Production training uses bf16, gradient accumulation, gradient clipping,
  cosine LR, EMA, isolated DataLoader RNG, atomic checkpoints, retention, and
  exact model/optimizer/scheduler/RNG/sampler recovery.
- Every upgrade-branch scaling/full training command is owned by a monitor-aware
  watchdog. The watchdog requires a fresh report from the expected pair monitor,
  fails closed when that monitor reports `failed`/`stalled`, stops publishing, or
  disappears, and terminates the complete POSIX training process group before
  returning a distinct operational exit code. Its atomic status JSON preserves
  command, monitor summary, child exit, timing, and bounded invocation history.
- Formal data identity is verified before model construction by rehashing the
  authoritative ImageNet metadata manifest and checking its byte count plus
  train/validation sizes against the recorded dataset condition. The resulting
  identity SHA is embedded in the run manifest, runtime benchmark, checkpoint
  payload, integrity sidecar, latest pointer, and training report. Full 300K
  resume and matched-pair validation fail before weight deserialization when
  this identity changes. The pinned legacy 10% reports may omit it only while
  authorizing the code transition; they cannot authorize full-data training.
  The authoritative compressed 10% rerun requires bound provenance.
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
- The active pinned 10% matched queue predates integrity sidecars and the true
  compressed-token layout. It is retained as immutable legacy evidence and a
  deployment prerequisite only. After deployment, a fresh same-revision
  compressed CoFiTok/dense 50K pair creates native integrity sidecars and is the
  only 10% pair eligible for promotion sampling and the scaling gate.
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
  baseline/peak memory, plus a canonical Python/PyTorch/CUDA/GPU/project-lock
  runtime-environment fingerprint. It fails before a sampling manifest or partial
  image directory is created when the target inference shape OOMs. Runtime
  selection never reuses a preflight from another revision or a dirty worktree.
- Classifier-free guidance can evaluate conditional and unconditional branches
  in one batch, with a sequential fallback. The selected execution mode is part
  of the immutable sampling manifest and must match across compared methods.
- Every global sample index owns an independent RNG stream. The stream is
  invariant to batch size and resume boundaries, and the same numbered sample
  uses the same stream at every prefix budget. Prefix comparisons are therefore
  paired rather than comparisons between unrelated initial noises.
- Sampling writes an immutable checkpoint-and-protocol manifest before the
  first image, including revision/branch/tracked-dirty code provenance and the
  actual sampling process runtime-environment fingerprint. Manifest schema v3
  and completed report schema v6 carry the full canonical environment and SHA.
  `--resume` accepts only an exact manifest match, skips completed numbered
  images, and regenerates missing images from their original streams. Metrics,
  promotion/final gates, and completion audit preserve and validate code and
  environment provenance; formal post-evaluation runbooks also fail early on
  tracked dirt. Environment drift therefore cannot silently resume an old set.
- Formal sampling is identified by `cofitok_ddim_sampling_v1`. The scaling gate
  requires deterministic DDIM-100 and the full gate requires deterministic
  DDIM-250; both require EMA, CFG 1.5 with zero guidance rescale, batched CFG,
  `eta=0`, `clip_x0=true`, bf16, seed zero, balanced-modulo classes, and the
  batch/resume-invariant per-index random stream. The report must reproduce the
  immutable manifest's complete sampling dictionary, and the resolved timestep
  list must equal the shared scheduler's exact selection. Both methods being
  identically misconfigured is therefore a named gate failure, not a match.
- The DDIM recurrence also has a closed-form oracle regression test. An epsilon
  predictor constructed from a fixed known `x0` must recover that image after a
  strided trajectory for both deterministic DDIM and nonzero `eta`; this checks
  the update equation independently of a learned model or report schema.
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
  package version, seed, cache name, and runtime. Before evaluation, every real
  image byte and root-relative path is hashed with
  `cofitok_image_tree_sha256_v1`. The digest is embedded in metrics schema v2
  and in the effective torch-fidelity cache key, so changed real data cannot
  silently reuse stale cached features.
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

The failed revision-`04a738c` 10% pair keeps its original `compressed_*` run
directories as immutable evidence. The next authoritative scaling attempt uses
fresh identities resolved by `cofitok.generation_paths`:

```text
CoFiTok: imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2
dense:   imagenet256_10pct_rankcomplete_dense_50k_v2
report:  imagenet256_10pct_rankcomplete_matched_50k_v2
```

Active runbooks import these paths through
`scripts/print_generation_workspace_paths.py`; gate source validation and the
terminal completion audit use the same Python contract. A completed run may be
skipped only when its resolved config, Git identity, step identity,
`latest.json`, checkpoint, and integrity sidecar all match the current request.

After the probe winner is committed and deployed, run
`generation_attest_deployed_revision.sh` before starting the fresh formal 50K
pair. It creates a consolidated bundle from the locked source revision through
the selected target, reruns the complete remote test and runbook-syntax suites,
and writes target-SHA-versioned evidence under `checkpoints/generation/deployment/`.
The completion audit reads only that target-specific receipt and its bound
sources. Earlier transition and recovery receipts remain immutable historical
evidence instead of being overwritten by a mutable canonical filename.

## Promotion gates

1. Code gate: full tests, CPU exact-resume smoke, CUDA bf16 smoke, zero-token
   contract, matched parameter audit, and formal training-recipe contract pass.
2. Data gate: 10 real ImageNet-256 optimizer steps complete with finite losses,
   stable gradients, recorded throughput, and a restart from checkpoint.
3. Scaling gate: matched 50K-step CoFiTok and dense runs on
   `imagenet_256_10pct`; generate at least 10K EMA samples per method and compute
   FID under one real-image directory and evaluator version. Promotion requires
   CoFiTok FID at most 100.0, no more than 5% FID or endpoint-MSE regression
   against dense, ordered-prefix rank 1, exact zero-token synthesis, and a
   shuffled-token mismatch.
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
Before either authorization is consumed, `validate_generation_gate_report.py`
also requires the complete named gate set, rejects duplicate or failed checks,
recomputes the core inequalities from the report summary, and cross-checks the
FID, endpoint, ordering, zero-token, shuffle, and full precision/recall evidence.
The 300K runbook and completion audit call this same contract, so editing only a
gate's status or decision cannot authorize an expensive downstream stage.
Formal gate files also bind all six authoritative source reports: CoFiTok/dense
training, distribution metrics, and checkpoint-mechanism evaluation. Each source
is recorded by authoritative path, byte count, and SHA256. The shared core
verifier rehashes them before the CLI may reuse a gate, before every formal
300K training start or resume, and before final EMA export consumes its release
gate; the
completion pipeline reruns a recoverable post-evaluation stage when source-only
verification fails, while a source-valid scientific `hold` remains a
non-retryable quality decision. The terminal audit independently requires the
same verified source set. See
`docs/records/2026-07-14_generation_gate_source_provenance.md` and
`docs/records/2026-07-14_generation_authorization_source_freshness.md`.
The full 300K trainer also consumes the exact scaling gate through
`--authorization-gate`. Its canonical gate identity, file SHA256, byte count,
stage, and decision are copied into the run manifest, every checkpoint payload,
the pre-deserialization integrity sidecar, `latest.json`, and the completed
training report. Every resumed milestone must present the same gate bytes.
The full matched-pair validator and terminal completion audit recompute the
identity from the actual scaling report and require both methods and both real
step-300K sidecars to bind it. A gate created after training, a replaced gate,
or CoFiTok/dense runs authorized by different gates therefore cannot complete.
Formal sampling and EMA export also compare the deserialized checkpoint payload
with this sidecar before applying model or EMA weights, covering the final 300K
checkpoint even when it is sampled without a subsequent training resume.

A failed scientific scaling gate never triggers full training. After a
representation-level failure, the recovery path first runs
`generation_rank_recovery_probe_2026-07-19.sh`: two same-backbone 5K candidates
compare denoise-path and epsilon-band ordering objectives under the rank-complete
layout. Each candidate receives a 512-image checkpoint audit and a deterministic
DDIM-50 sample probe. The aggregate explicitly marks those FID/IS values as
non-formal, requires manual visual review, and cannot launch either a 50K or
300K run. The selected objective must then complete a fresh same-revision 10%
matched 50K pair and pass the unchanged 10K promotion gate.

The full queue alternates CoFiTok and dense at 50K, 100K, 200K, and 300K
milestones. At each matched point it produces 2,048 fixed-protocol EMA samples
at DDIM-50 / CFG 1.5, FID/IS trend metrics, and a 256-image mechanism audit.
These milestone reports are explicitly non-claim diagnostics: they expose
severe quality or ordering regressions before another long segment consumes GPU
time, but they never replace the final 50K DDIM-250 evaluation.

Milestone report schema v2 binds the CoFiTok/dense generation-metrics and
checkpoint-evaluation source reports by authoritative method/step path, byte
count, and SHA256. A milestone is skipped on resume only after those four files
are rehashed and the EMA DDIM-50/CFG 1.5/bf16/fixed-stream contract is
revalidated. The final completion audit repeats the source and protocol checks;
a stale aggregate JSON cannot prove that a milestone completed.

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

bash artifacts/runbooks/generation_rank_recovery_probe_2026-07-19.sh
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
Before any checkpoint evaluation or sampling, the full post-evaluation runbook
also rehashes the scaling gate's six source reports and reruns the complete full
matched-training-pair validator against the deployed revision, recipe, dataset,
parameter gap, checkpoint pointers, and scaling authorization.

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
full paired-config preflight confirms 62,837,576 vs 62,824,707 parameters
(+0.020484%). See
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
untracked path that would become tracked by the target revision. The collision
guard derives only paths added between pinned and target revisions, queries
untracked state with those bounded pathspecs, and checks untracked file/symlink
parent blockers; historical artifact trees are never enumerated wholesale.
Remote tests and shell syntax checks must pass, then an atomic schema-v2 receipt
binds the validated legacy 10% pair, pinned source revision, exact target revision, and
clean tracked state before it launches
`artifacts/runbooks/generation_complete_pipeline_after_10pct.sh`. That pipeline
uses an exclusive lock and atomic stage status, first trains and validates a
fresh compressed 10% matched pair on the deployed revision, then runs its 10K
promotion gate, the alternating full 300K queue, the formal 50K-sample
evaluation, and both gate decisions in order. A held gate or interrupted stage
is recorded as a failure rather than reported as generation readiness.
The transition bundle is atomically archived under
`checkpoints/generation/deployment/` instead of relying on `/tmp`. The receipt
binds its bytes, SHA256, advertised target head, and exact pinned prerequisite.
The helper parses the bundle header before fetch and requires exactly one
prerequisite equal to the pinned 10% training revision; receipt generation and
terminal audit independently repeat that check on the archived bytes. The
receipt also binds three durable source reports: the bounded pre-merge conflict
JSON, a JUnit XML from the full
remote pytest suite, and a JSON report that enumerates every target-tracked
shell runbook with `git ls-files` and checks each with `bash -n`. The final
completion audit reopens and rehashes all four source files, revalidates their
content, and rejects a changed byte, omitted runbook, summary mismatch, or
missing archived bundle. Changing remote HEAD or editing the receipt alone
therefore cannot prove a controlled revision transition. See
`docs/records/2026-07-13_generation_deployment_evidence_provenance.md`.
The local bundle is prerequisite-aware: it advertises only the upgrade HEAD
and excludes history reachable from the pinned training commit. The local
deployer verifies the single advertised head before transfer, while the remote
`git bundle verify` proves that the pinned prerequisite exists before fetch.
This preserves the same fast-forward trust boundary without retransmitting the
repository's full historical object graph.
Before moving HEAD, the remote helper fetches only the verified bundle objects
and extracts the target revision's training-pair validator and complete
`src/cofitok/` package plus the bounded collision checker with `git archive`
into a temporary isolated Python
path. Pre-deployment validation therefore runs the exact target code
even though the worktree is still pinned to the legacy training revision. The
temporary tree is removed on exit, and a failed validation never merges the
target revision.
See `docs/records/2026-07-13_bounded_deployment_conflict_scan.md`.

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
 evaluator revision/branch, canonical evaluator runtime environment, and exact
 real-set content digest; at full scale the revision must equal full training and
 sampling. Ordered-prefix/zero/shuffle checkpoint diagnostics carry an
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
Formal full-training checkpoints additionally bind their scaling authorization
before deserialization. Capturing that authorization on every segment reopens
the gate and verifies all six bound source reports before checkpoint loading;
see
`docs/records/2026-07-13_generation_training_authorization_binding.md`.

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
The formal pair is accepted only when its real-set tree digest, content-addressed
cache key, and evaluator environment are identical and bound through the gate
and comparison schema v5, including the exact formal sampling protocol fields.
The comparison binds both training reports, both 50K metrics reports, and the
final gate by authoritative path, byte count, and SHA256; the completion audit
rereads those files before accepting any displayed metric or cost field.
Missing evidence is `in_progress`, contradictory evidence is `failed`, and only
the full chain is `complete`. See
`docs/records/2026-07-12_large_scale_generation_completion_audit.md`.

Before full 300K training starts, both methods run the same checkpoint-free
training-runtime candidates `16x4`, `32x2`, and `64x1`. Selection preserves
effective batch 64, requires both methods to fit below 90% VRAM, and minimizes
the slower method's synchronized optimizer-step time. The selected microbatch
and accumulation are applied to every full-training segment. Every completed
benchmark records the canonical Python/PyTorch/CUDA/GPU/project-lock runtime
environment; all candidates, both methods, and the subsequent full training
reports must share one exact environment SHA. Cached benchmark reuse and the
completion audit both reject environment or tracked-Git drift. The selector is
also state-aware: before either formal run contains training state it may build
or refresh the shared selection, but once either run directory is non-empty it
only validates and reuses the existing report. That frozen report binds both
run paths, all three candidates, the 300K horizon, benchmark horizon, config
SHA256 values, clean branch/revision, dataset identity, and runtime environment.
A missing or drifted report fails before any benchmark subprocess or model load.
See
`docs/records/2026-07-12_generation_training_runtime_selection.md`.

The full ImageNet-256 pair also uses the same per-image random horizontal flip
probability of `0.5`. Augmentation happens on the normalized device batch before
noise and timestep sampling, using the training process PyTorch RNG that is
already captured by exact-resume checkpoints. Validation remains deterministic
and unaugmented. The pinned 10% legacy pair keeps its resolved default of `0.0`,
so this upgrade does not alter the active training protocol. See
`docs/records/2026-07-13_generation_exact_resume_horizontal_flip.md`.

Production scalable U-Net token heads are zero initialized. Both the K-token
CoFiTok predictor and the one-head dense control therefore begin with an exact
zero epsilon prediction instead of method-dependent random output variance.
Restricted synthesis weights remain nonzero, so the first loss backward pass
reaches every token head and training leaves the zero state immediately. This
initialization affects only fresh training; loading an existing checkpoint
strictly replaces all initialized state. See
`docs/records/2026-07-13_generation_zero_initialized_output_heads.md`.

Formal 10K and 50K sampling also selects one shared batch from
`16,32,64,128`. Both checkpoints run repeated synchronized EMA/CFG forwards;
eligible candidates must pass for both methods below 90% VRAM, and selection
maximizes the slower method's output-images/second. Both methods must also expose
the same canonical sampling environment; the selected fingerprint is bound to
the formal generation reports. Per-index random streams keep generated samples
invariant to the selected batch. Once either matched formal output directory
contains sampling state, the selector becomes read-only: it must reproduce and
reuse the original selection rather than rerun preflights or rewrite the report.
The lock binds both output paths, all candidates, both checkpoint identities,
the full EMA/bf16/CFG measurement protocol, clean Git state, and benchmark root.
Successful preflights from different runtime environments cannot participate in
one ranking. The final comparison reports batch, elapsed time, and realized
throughput. See
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

Before 10K promotion evaluation, full 300K training, and formal 50K evaluation,
the runbooks also perform a structured storage-capacity preflight. It reserves
space for conservative checkpoint copies, planned PNGs, evaluator caches, and a
safety margin; insufficient capacity exits non-retryably before large artifacts
are created. See
`docs/records/2026-07-13_generation_storage_capacity_preflight.md`.

The full 300K matched run also publishes a generic read-only operational monitor
with optimizer-step freshness, finite-metric checks, checkpoint cadence, process,
disk, and GPU state. Milestone boundaries force synchronous health snapshots;
the final completion audit requires a clean-revision `pass` report with both
exact 300K runs and all protected milestone checkpoint stats. See
`docs/records/2026-07-13_full_generation_operational_monitor.md`.

The compressed 10% and full 300K runbooks launch that monitor before training
and wrap every training segment with
`scripts/run_generation_training_watchdog.py`. Exit codes `86`, `87`, `88`, and
`89` respectively mean terminal monitor failure, no fresh startup report,
monitor-report silence, and monitor-process disappearance; a training-process
failure otherwise preserves the child's exit code. This converts monitor
evidence into active fail-closed process ownership instead of leaving an
unhealthy GPU job running until a human notices. The pinned legacy pair remains
untouched because its code revision cannot change mid-pair. See
`docs/records/2026-07-14_generation_training_watchdog.md`.

Every upgrade-branch training checkpoint also binds a canonical runtime
environment fingerprint across payload, integrity sidecar, latest pointer, and
training report. Exact resume rejects Python, package, PyTorch/CUDA/cuDNN/driver,
GPU, backend, environment-variable, or project-lock drift before deserializing
state. Git revision, branch, and tracked state are bound at the same checkpoint
trust boundary. The final audit requires identical CoFiTok/dense environment SHA
values and checkpoint pointers from the clean deployed revision.
See `docs/records/2026-07-13_generation_runtime_environment_fingerprint.md`.

The same pre-deserialization boundary binds formal dataset provenance. The
full ImageNet-256 manifest must remain exactly `405,484,553` bytes with SHA256
`9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`,
and loaders must resolve `1,281,167` train plus `50,000` validation images. The
10% identity is retained for the next rerun, while the active pinned queue is
reported as an explicit legacy exception rather than retroactively claiming
evidence it did not record. See
`docs/records/2026-07-13_generation_dataset_provenance.md`.

After the final gate, both methods export separate EMA-only deployment
artifacts. Their type-specific sidecars are verified before deserialization;
source training checkpoint SHA, runtime-environment SHA, Git identity, step,
scaling-gate training authorization, final quality release authorization,
artifact SHA/bytes, real-forward preflight, and short DDIM smoke PNGs are required
by completion. Artifact schema v4 propagates the source identity, training
authorization, and exact passing full gate through the payload, sidecar, export
report, loader, session, preflight, and inference report. Release authorization
capture rehashes the final gate's six source reports before `torch.load`, so a
stale quality decision cannot authorize an artifact. These artifacts are smaller
inference copies and never replace exact-resume training checkpoints. See
`docs/records/2026-07-12_deployable_ema_inference_artifact.md` and
`docs/records/2026-07-13_inference_artifact_final_release_authorization.md`.

The terminal completion audit does not trust those JSON reports alone. It
rehashes both physical step-300K exact-resume checkpoints through their
integrity sidecars and separately rehashes both exported EMA artifacts.
Missing, truncated, or replaced model bytes fail the named checkpoint or
deployment-artifact gate even when an older report still claims completion.
Formal sampling separately requires one canonical runtime-environment SHA shared
by the selected CoFiTok/dense preflights, immutable manifests, metrics reports,
and final quality gate.

Live long-run health can be audited without loading the model or competing for
GPU time using `scripts/audit_generation_training_progress.py`. It verifies
strictly increasing finite metrics, resume-aware timing segments, checkpoint
cadence and `latest.json`, then records recent loss/gradient summaries and ETA.
For the pinned legacy 10% queue, `legacy_compute` records the newest checkpoint
byte count and SHA256 without modifying it; the post-training migration later
binds those bytes to a sidecar. New full runs use `--integrity-policy required`,
which rejects a missing or mismatched sidecar and `latest.json` integrity binding.
Logged gradient norms are explicitly treated as pre-clipping total norms.
