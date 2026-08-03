# CoFiTok Generation System

This branch upgrades CoFiTok from short-budget mechanism validation to a
class-conditional ImageNet-256 generation system. Locked paper evidence remains
on `paper-evidence-locked`. The original large-generation implementation and
its immutable runs remain on `scale/generative-system`; the rollout-stability
correction is developed independently on `scale/generation-stability`.

## System boundary

- `ScalableUNetTokenPredictor` is the expressive `T_k`: ADM-style residual
  U-Net, spatial attention, timestep conditioning, ImageNet class conditioning,
  CFG dropout, and optional activation checkpointing.
- Formal scaling/full runs use `FixedBasisSynthesisBank` as `S_k`. Every
  operator receives only its current token and applies a deterministic,
  parameter-free, bias-free RGB channel projection, preserving `S_k(0)=0`
  exactly. `RestrictedSynthesisBank` remains available for historical evidence
  and ablation only.
- The immutable v3 scaling/full evidence uses spatial strides
  `[16,16,8,8,4,4,1,1]` and channel counts `[4,4,8,8,8,8,1,2]`. The stability
  correction uses the separately versioned `rgbtail3` layout with strides
  `[16,16,8,8,4,1,1,1]` and channels `[4,4,8,8,8,1,1,1]`. Every individual
  token remains smaller than a dense RGB epsilon field, while three independent
  full-resolution one-channel tail tokens provide rank-complete RGB synthesis
  without forcing the final token to carry two channels. `T_k` pools before
  each variable-channel head, feeds back only that emitted token, and `S_k`
  performs fixed bilinear upsampling before its fixed channel projection.
- A restricted linear synthesis layout must allocate at least three aggregate
  full-resolution token channels for RGB epsilon prediction. Otherwise the
  highest-frequency output is rank-deficient regardless of model size or
  training time. The immutable v3 contract satisfies this with final one- and
  two-channel tokens; the stability contract instead requires three separate
  one-channel tail tokens. Both contracts reject rank-deficient layouts.
- The dense control uses the identical U-Net and training protocol with one
  direct `dense_identity` epsilon head.
- Generation recipe schema
  `cofitok_generation_training_recipe_v4` preserves the historical
  `legacy_scaling`, `scaling`, and `full` contracts and adds
  `stability_scaling` and `stability_full`. The stability stages lock the
  `rgbtail3` layout, Hellinger-stable capacity objective, low-SNR
  high-frequency term, two-step clipped-`x0` rollout consistency, and late EMA
  teacher consistency for both matched methods. Runtime selection may change
  micro-batch and accumulation only while their product remains 64. The
  qualified `stability_scaling` stage remains at 128 base channels;
  the dormant `stability_full` stage is a separate 256-channel capacity
  tier and cannot alter or reinterpret the 50K evidence.
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
- Formal pair monitors use `checkpoint_integrity_policy=required`. Once a
  checkpoint passes its bounded write-grace window, the monitor verifies the
  checkpoint filename/byte count/step and declared SHA256 against the adjacent
  sidecar, then verifies that `latest.json` binds the same metadata. This live
  check is explicitly metadata-only: it neither loads nor rehashes a checkpoint
  payload while training is active. Resume, sampling, promotion, and release
  remain responsible for recomputing the payload SHA256 at their trust
  boundaries.
- Earlier 10% matched queues are retained as immutable legacy evidence only.
  The fresh same-revision fixed-basis v3 CoFiTok/dense 50K pair creates native
  integrity sidecars and is the only 10% pair eligible for new promotion
  sampling and the scaling gate.
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

The official server repository remains the immutable deployed generation
revision until an explicitly authorized transition. Stability probes and Linux
rehearsals use detached worktrees under `/tmp`; they must not move the official
repository or mutate an active training checkout.

## Stability recovery path

- The current recovery candidate is
  `rgbtail3 + clipped-x0 two-step rollout + late EMA teacher`. The matched 5K
  qualification run is pinned to
  `59db142fc45d69dc92bb0333be5ac2d0162d9dc4`; its final raw n=8 and two-seed
  n=64 decision is the only input allowed to authorize 50K preparation.
- The formal 10% successor configs are
  `imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json`
  and
  `imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json`.
  Both use 50,000 optimizer steps and effective batch 64. Rollout consistency
  uses weight `0.1`, start `0`, warmup `10,000`, timestep delta `10`, two
  unrolls, batch fraction `0.125`, and clipped-`x0` mode. EMA teacher
  consistency uses weight `0.25`, start `30,000`, warmup `10,000`, and batch
  fraction `0.0625`.
- The matched parameter counts are CoFiTok `62,834,083` and dense
  `62,824,707`, a relative gap of `+0.014924%`. The shared U-Net, diffusion,
  data, optimizer, runtime, rollout, and teacher fields are exact pair-contract
  matches; dense factorization-only auxiliary losses remain zero.
- `validate_generation_stability_scaling_decision.py` verifies the expected
  decision SHA, rehashes every screening and robust source report, rebuilds the
  decision from those sources, and requires exact equality. Editing a decision
  and supplying its new SHA therefore cannot bypass the qualification evidence.
- `generation_stability_ema_teacher_matched_50k_after_gate.sh` is a dormant
  successor runbook. Before any CUDA benchmark or training it requires the
  passing 5K decision, exact clean target revision and branch, recipe-v4 config
  validation, storage headroom, and an idle GPU. It then selects one matched
  `16x4`, `32x2`, or `64x1` runtime, requires integrity-aware monitoring, and
  permits exact resume only at the same target revision.
- Completing this matched 50K pair still does not authorize full 300K.
  `build_generation_stability_50k_summary.py` binds both 50K training reports,
  decision/config validation, exact 3.2M images per method, effective batch 64,
  branch/revision, and pair contract, while explicitly setting
  `formal_300k_authorization_allowed=false`. A formal EMA sampling and mechanism
  gate is still required.
- The dormant `stability_full` pair uses `base_channels=256` with
  CoFiTok `250,153,763` parameters and dense identity `250,135,043`,
  a relative gap of `+0.007484%`. This is an approximately 250M-parameter
  matched capacity tier; the 62.8M parameter counts above remain the exact 50K
  qualification architecture. Full training is still source-bound to a passing
  50K promotion gate and is not authorized by this configuration change alone.
- The 250M CUDA qualification and the 300K launch are two separate operations.
  `generation_stability_ema_teacher_full_readiness_after_gate.sh` runs only
  after the passing source-bound 50K gate, requires an idle GPU and absent
  formal training state, validates the exact 250M pair, captures the 4x storage
  reserve, benchmarks all five runtime candidates, and writes immutable
  `full_training_readiness.json` before exiting. It never starts the monitor or
  trainer. The full 300K runbook requires the readiness SHA256, replays every
  bound source and the current CUDA environment, reads the selected runtime
  from that artifact, performs a fresh launch-time storage check, writes an
  immutable `full_training_launch_receipt.json`, and only then enters the
  resumable milestone loop. The receipt binds the readiness, promotion gate,
  isolated deployment receipt, exact training configs, runtime selection,
  launch-time storage report, and both run paths. A resumed launch must provide
  the receipt SHA256 and replay it before updating only
  `storage_capacity_current.json`; it cannot overwrite the original launch
  storage evidence. The runbook never calls the runtime selector.
- A separate fail-closed readiness waiter closes the handoff between the long
  50K post-evaluation and the 250M CUDA qualification. It runs from the exact
  receipt-bound large-capacity checkout, requires the post-evaluation waiter to
  finish successfully, verifies the passing `stability_scaling` gate including
  all source-report hashes and exact training/evaluation revisions, waits for
  an idle GPU, and then invokes only
  `generation_stability_ema_teacher_full_readiness_after_gate.sh`. It can replay
  an existing readiness artifact after restart. Its status always records
  `full_training_launch_allowed=false`; it has no reference to the 300K training
  runbook and cannot launch a trainer.- Deployment is a third, earlier stage and is CPU-only.
  `generation_deploy_large_capacity_readiness_checkout.sh` verifies the
  persistent bundle against the still-pinned formal repository, clones an
  isolated checkout under `checkpoints/generation/deployment/large_capacity`,
  stores the receipt, JUnit XML, and runbook syntax report under
  `deployment/large_capacity/deployments/<full-target-sha>/`, runs the complete
  CPU test suite and every tracked shell runbook syntax check, and atomically
  writes a deterministic deployment receipt. Prior target evidence is never
  overwritten; the historical fixed receipt path remains compatibility-only.
  It neither
  moves the formal repository nor invokes readiness or training. Readiness must
  execute from the exact receipt-bound checkout and binds the receipt as a
  seventh immutable source. The deployment receipt explicitly authorizes only
  readiness execution; it sets full-training launch authorization to false.
- Stability mechanism evaluation derives the coarse/tail partition from
  `token_spatial_strides`: the `rgbtail3` layout measures coarse utilization
  over tokens 1-5 and treats tokens 6-8 as the full-resolution tail. The
  stability source profile requires this stride metadata and fails closed
  instead of falling back to the legacy v3 `K-2` partition.
- Dormant runbook
  `generation_stability_ema_teacher_50k_posteval_after_training.sh` reruns the
  5K decision, recipe-v4 pair validation, and completed 50K pair summary before
  any GPU work. It then uses the formal matched EMA DDIM-100 protocol, 10,000
  samples per method, 1,024-image timestep-500 mechanism evaluation, and an
  independent `stability_scaling` source-path profile. It can write and verify
  a scaling gate but cannot invoke full training.
- Schema-v3 stability gates add a matched EMA rollout-stability contract to
  the formal distribution-quality gate. Both the 50K scaling post-evaluation
  and the dormant full post-evaluation run 64 fixed validation images with the
  same DDIM step count, CFG `1.5`, clipped `x0`, bf16 precision, and final
  checkpoint bytes as their FID sampling protocol. The resulting qualification
  reuses the established tail-energy, single-token concentration, ordered-rank,
  endpoint/validation regression, free-rollout high-frequency, reconstruction,
  zero-token, and shuffle-mismatch checks. A supplied qualification is a
  blocking gate row and is bound by absolute path, byte count, and SHA256;
  schema-v3 `stability_scaling`/`stability_full` gates cannot omit it. Historical
  schema-v2 gates remain replayable so locked evidence is not retroactively
  invalidated.
- Schema-v4 `stability_scaling` gates also make distribution support a blocking
  decision input instead of merely reporting it. CoFiTok precision and recall
  must each be at least `0.10`, and neither may be more than `0.05` below the
  matched dense member. These are deliberately low non-collapse floors for the
  10% scaling decision; they do not replace the stricter full-stage `0.30`
  floors and do not constitute an ImageNet generation-quality claim. Historical
  schema-v2/v3 gates remain replayable, but every newly built stability-scaling
  gate must contain the named `scaling_precision_recall_quality` evidence row.
- Schema-v5 `stability_scaling` and `stability_full` gates add a source-bound
  class-conditional fidelity qualification over the exact formal EMA sample
  sets already used for distribution metrics. The evaluator is fixed to
  torchvision ResNet-50 ImageNet-1K V2 weights (`102,540,417` bytes, SHA256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`),
  its category ordering and preprocessing are pinned, and requested labels are
  reconstructed from the zero-based PNG index under the balanced-modulo class
  schedule. Scaling requires both matched methods to reach top-1 `0.01`, top-5
  `0.05`, predicted-class coverage `0.25`, and normalized predicted-class
  entropy `0.50`; full requires `0.10`, `0.25`, `0.50`, and `0.70`. CoFiTok
  top-1 and top-5 may each trail dense by at most `0.05` absolute. The gate
  binds both raw classifier reports and their paired qualification by path,
  bytes, and SHA256, then recomputes every absolute and relative check. This
  evidence detects ignored, collapsed, or permuted class conditioning; it
  complements but cannot replace FID/IS/precision/recall, EMA rollout
  stability, visual review, or any full-training/release authorization.
- Frozen schema-v2/v3 stability gates can be audited without rewriting their
  immutable bytes by
  `scripts/build_generation_stability_distribution_support.py`. The CPU-only
  supplemental builder first rehashes every source bound by the original gate,
  then rereads the exact CoFiTok/dense 10K metrics bytes and independently
  checks the formal EMA DDIM-100 protocol, matched real set/evaluator/code/runtime,
  finite metric domains, `0.10` precision/recall floors, and `0.05` matched
  retention. Its deterministic report binds the original gate, both metrics
  files, and a clean builder Git identity. It is explicitly non-authorizing:
  it cannot replace the original gate or rollout qualification and always sets
  `full_training_launch_allowed=false`.
- The current frozen schema-v2 post-evaluation also has a separate, resumable
  follow-up runbook:
  `generation_stability_frozen_50k_supplemental_after_posteval.sh`. It first
  verifies the exact successful v4 post-evaluation waiter and original gate,
  then refuses a busy GPU. From one clean supplemental checkout it reruns the
  matched 1,024-image EMA timestep-500 checkpoint evaluations and adds matched
  64-image EMA DDIM-100 free-rollout evaluations. Rerunning the checkpoint
  diagnostics is intentional: the checkpoint and rollout reports must share
  the same supplemental evaluation Git identity, so the frozen checkout's
  older checkpoint reports cannot be mixed with later rollout reports. Five
  high-cost/source-bound stages use immutable receipts and exact replay; a
  sixth CPU-only step reconstructs the distribution-support qualification from
  the original gate. The final combined report is `pass` only when the base
  gate, distribution-support check, and EMA rollout qualification all pass.
  It remains diagnostic-only: it does not replace the original gate or
  readiness receipt, does not invoke a trainer, and fixes
  `full_training_launch_allowed=false`.
- `generation_stability_frozen_50k_supplemental_waiter.sh` makes that quality
  follow-up operational without racing the already queued large-capacity CUDA
  readiness stage. It validates the exact post-evaluation and readiness waiter
  identities and freshness, waits for readiness to become terminal and for an
  idle GPU, and then launches only the supplemental runbook. New GPU contention
  or a held supplemental lock returns to the wait loop. A scientific
  supplemental `hold` is preserved as a completed diagnostic rather than
  promoted; a pass is independently replayed from all direct and nested source
  bytes. The waiter's process lock and every status record remain explicitly
  non-authorizing. See
  `docs/records/2026-08-03_generation_frozen_supplemental_waiter.md`.
- The waiter now has a separately attested, dormant Linux evaluation checkout
  at `/tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal`,
  fixed to `c212b9e2b64d1b302b17a9d4e30a296d773d4215` on
  `scale/generation-large-capacity`. From an empty `CUDA_VISIBLE_DEVICES` it
  passed `70` focused tests and syntax checks for all `104/104` runbooks. The
  source checkout and target remained tracked-clean, the temporary incremental
  bundle was removed, and no waiter or GPU evaluation was started. The exact
  bundle, entrypoint, JUnit, and syntax hashes are bound by
  `artifacts/reports/generation/stability_frozen_supplemental_checkout_2026-08-03/checkout_attestation_receipt.json`;
  see also
  `docs/records/2026-08-03_generation_frozen_supplemental_checkout_attestation.md`.
- After that attestation, exactly one detached CPU coordination waiter was
  restored as PID `861117`. It holds the status-output lock, is bound to the
  clean `c212b9e` checkout, and remained healthy through a full poll at
  `waiting_for_frozen_postevaluation` with no child. Its authoritative status
  is
  `stability_scaling_50k_ema_teacher/reports/frozen_posteval_supplemental/supplemental_waiter.json`.
  The waiter can launch only after frozen post-evaluation completes, readiness
  is terminal, and the GPU is idle; it remains
  `supplemental_non_authorizing=true` and
  `full_training_launch_allowed=false`. Exact launch/status evidence is in
  `artifacts/reports/generation/stability_frozen_supplemental_waiter_launch_2026-08-03/`;
  see
  `docs/records/2026-08-03_generation_frozen_supplemental_waiter_launch.md`.
- The supplemental is nevertheless a mandatory **quality prerequisite** for a
  later 300K launch. Schema-v3 `full_training_launch_receipt.json` replays the
  supplemental from its four direct reports, rehashes every nested source,
  requires all three quality checks to pass, and binds its bytes as an
  eleventh launch source. This is intentionally asymmetric: a failed or absent
  supplemental blocks launch, while a passing supplemental still cannot
  authorize launch by itself. Readiness, its revision bridge, fresh storage,
  deployment identity, and separate human launch authority remain required.
  The post-training supervisor and terminal completion audit independently
  rehash the same supplemental binding.
  Launch-time verification lives in a standalone control script, so a future
  compatible control target can consume it without changing the immutable
  trainer package. The current branch HEAD is not directly bridge-compatible
  with the active `5dd3488...` readiness source; it must not be used as a
  launch target without either a dedicated compatible target or fresh target
  readiness.
  The dedicated compatible target is now prepared on branch
  `scale/generation-stability-full-control-quality-27ed` at
  `9019dd3f0f504e799c03496ec41a653b61deaa02`: all 66 training-critical blobs
  and the full-training execution suffix remain identical to `5dd3488...`.
  A prerequisite-aware 64,538-byte incremental bundle was verified locally and
  against the clean remote `5dd3488...` checkout, then deleted without fetch,
  merge, or checkout mutation. The receipt is
  `artifacts/reports/generation/readiness_compatible_quality_control_target_2026-08-03/incremental_bundle_rehearsal_receipt.json`.
  The target remains un-fetched and un-deployed; its existence and rehearsal do
  not create a bridge, deployment receipt, launch receipt, or full-training
  authorization.
- Stability post-evaluation uses separate, explicit training and evaluation
  Git identities. This prevents the gate builder's legacy
  `scale/generative-system` default from rejecting the intentionally isolated
  stability branches, while preserving that default for existing generation
  paths. The source pair remains bound to its original training revision;
  sampling, distribution metrics, and checkpoint evaluation must share the
  exact clean post-evaluation revision.
- `generation_stability_ema_teacher_50k_posteval_waiter.sh` waits only for the
  exact training monitor and a complete source-bound pair summary, fails on
  stale/failed/mismatched state, and launches formal EMA post-evaluation only
  after the GPU is idle. Its successful exit is operational evidence, not a
  scientific scaling decision: `promotion_gate.json` must independently pass
  complete validation. The waiter cannot launch full 300K training.

The failed learned-synthesis v2 pair keeps its original run and report
directories as immutable evidence. The historical fixed-basis v3 attempt uses
the following identities resolved by `cofitok.generation_paths`; the stability
path does not reuse or overwrite them:

```text
CoFiTok: imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3
dense:   imagenet256_10pct_fixed_basis_dense_50k_v3
report:  imagenet256_10pct_fixed_basis_matched_50k_v3
```

Active runbooks import these paths through
`scripts/print_generation_workspace_paths.py`; gate source validation and the
terminal completion audit use the same Python contract. A completed run may be
skipped only when its resolved config, Git identity, step identity,
`latest.json`, checkpoint, and integrity sidecar all match the current request.

After the fixed-basis v3 recipe is committed and deployed, run
`generation_attest_deployed_revision.sh` before starting the fresh formal 50K
pair. It creates a consolidated bundle from the locked source revision through
the selected target, reruns the complete remote test and runbook-syntax suites,
and writes target-SHA-versioned evidence under `checkpoints/generation/deployment/`.
The completion audit reads only that target-specific receipt and its bound
sources. Earlier transition and recovery receipts remain immutable historical
evidence instead of being overwritten by a mutable canonical filename.

The 5K rank-recovery probe also records raw-model and EMA mechanism evaluations
under the same checkpoint/config/Git identity. This detects short-horizon EMA
lag without changing the formal contract: visual probe samples, promotion
sampling, full milestones, final 50K sampling, and exported inference artifacts
all continue to require EMA weights.

The fresh versioned 10% matched 50K pair uses the same random horizontal-flip
probability (`0.5`) as the full 300K pair for both CoFiTok and dense identity.
Legacy failed evidence and the fixed objective probes retain `0.0` and remain
separately labeled.

## Promotion gates

1. Code gate: full tests, CPU exact-resume smoke, CUDA bf16 smoke, zero-token
   contract, matched parameter audit, and formal training-recipe contract pass.
2. Data gate: 10 real ImageNet-256 optimizer steps complete with finite losses,
   stable gradients, recorded throughput, and a restart from checkpoint.
3. Scaling gate: matched 50K-step CoFiTok and dense runs on
   `imagenet_256_10pct`; generate at least 10K EMA samples per method and compute
   FID under one real-image directory and evaluator version. Promotion requires
   CoFiTok FID at most 100.0, no more than 5% FID or endpoint-MSE regression
   against dense, ordered-prefix rank 1, at least 5% of normalized component
   energy in compressed tokens 1-6, exact zero-token synthesis, and a
   shuffled-token mismatch. Newly built schema-v4 `stability_scaling` gates
   additionally require CoFiTok precision and recall each at least 0.10 and no
   more than 0.05 below matched dense, so a superficially acceptable FID cannot
   hide precision or recall collapse. Schema-v5 additionally requires the
   paired fixed-classifier qualification described above: both methods must
   meet the scaling top-1/top-5/coverage/entropy floors and CoFiTok may trail
   dense top-1 or top-5 by at most 0.05 absolute.
4. Full gate: matched 300K-step runs on full `imagenet_256`, 50K EMA samples,
   official FID plus IS/precision/recall, prefix diagnostics, and checkpoint
   hashes. Declare the system ready only if CoFiTok keeps its prefix-control
   advantage without a material endpoint generation regression against dense,
   all distribution metrics are finite and inside their mathematical ranges,
   CoFiTok FID is at most 20.0, precision and recall are each at least 0.30,
   and neither precision nor recall is more than 0.05 below matched dense. The
   full checkpoint must again keep at least 5% of normalized component energy
   in tokens 1-6. Schema-v5 also requires both methods to meet the full
   class-fidelity top-1/top-5/coverage/entropy floors, with the same maximum
   0.05 absolute CoFiTok top-1/top-5 regression against dense.

The 10% gate is an engineering and architecture decision point. It is not a
replacement for the full-data result and must not overwrite locked paper tables.
The precision/recall floors are conservative non-collapse readiness checks, not
a generation-SOTA claim: stability scaling uses 0.10 while the full gate keeps
the stronger 0.30 floor. The final completion audit binds FID/precision/recall
back to the formal generation reports and rejects a gate with weaker thresholds.
Before either authorization is consumed, `validate_generation_gate_report.py`
also requires the complete named gate set, rejects duplicate or failed checks,
recomputes the core inequalities from the report summary, and cross-checks the
FID, endpoint, ordering, zero-token, shuffle, scaling distribution-support, and
full precision/recall evidence. For schema-v5 stability profiles it also
reopens both raw class-fidelity reports, verifies the fixed classifier and
formal sampling identities, and recomputes the paired qualification.
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

The subsequent equal-progress v3 and exact target-energy v4 probes were also
rejected. V4 reached a finite endpoint but ranked 17/18 at timestep 500 and
kept about 90% of learned component energy in the final two tokens. The
diagnosis is a target/layout mismatch: equal full-resolution progress cannot
be represented by the early spatially compressed synthesis subspaces. The v5
capacity-path probe therefore derives each progress increment from the square
root of the token's restricted synthesis rank proxy and derives the spatial
target from its actual token resolution. This mode is opt-in; historical
`power` targets retain their exact behavior.

V5 recovered rank 1 but its heavy objective damaged directional quality. V6
combined the light v2 weights with capacity progress and recovered both the v2
endpoint trajectory and rank 1, but assigned only about 0.56% of component
energy to tokens 1-6 and remained worse than v2 on the 512-image directional
FID. V7 kept the v6 protocol fixed and opted into squared Hellinger distance for
the capacity-target energy distribution. It reached rank 1 at all five audited
timesteps, raised token-1-6 energy to 7.48% at `t=500`, kept endpoint MSE within
3.7% of v6, and changed the 512-image directional FID by only +0.8%. Fixed-index
visual review also showed low-frequency structure by prefix 4 instead of the
near-identical early noise fields seen in v6.

The learned local synthesis in formal v2 nevertheless failed the 10K promotion
gate: CoFiTok FID was `191.2347` versus dense `115.0052`. A bounded sampling
recovery sweep ruled out EMA choice, CFG scale, and guidance rescale as the
root cause. V8 then retained the v7 layout, objective, data, backbone, seed, and
5K budget while replacing each learned synthesis operator with a deterministic
fixed basis. It ranked first at all five audited timesteps, reduced `t=500` EMA
endpoint MSE from `0.02072` to `0.01763`, reduced path AUC by 8.17%, and raised
token-1-6 energy from 7.48% to 8.11%. Its directional FID-512 improved from
`296.38` to `260.31`, with IS increasing from `1.668` to `3.175`; fixed-index
visuals also showed materially stronger endpoint structure. V8 is therefore
the selected formal synthesis and v3 requires `fixed_basis`, kernel size 1,
and `fixed_one` gamma. Both scaling and full gates continue to reject
token-1-6 energy below 5%.

V5-v8 remain non-formal 5K evidence and cannot themselves authorize 50K or
300K. The selected v8 recipe must complete the fresh v3 same-revision 10%
matched 50K pair and pass the unchanged 10K promotion gate.
See `docs/records/2026-07-20_generation_target_energy_probe_v4_result.md`,
`docs/records/2026-07-20_generation_capacity_path_probe_v5_result.md`,
`docs/records/2026-07-21_generation_capacity_path_light_probe_v6_result.md`, and
`docs/records/2026-07-21_generation_capacity_path_hellinger_probe_v7_result.md`,
and `docs/records/2026-07-25_generation_fixed_basis_probe_v8_result.md`.

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
  --cofitok-config configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  --dense-config configs/generation/imagenet256_10pct_fixed_basis_dense_50k.json \
  --output artifacts/reports/generation/config_pair_10pct.json

python scripts/train_generation.py \
  --config configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3

python scripts/train_generation.py \
  --config configs/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3 \
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
images/second, and peak memory in the final JSON/Markdown/CSV comparison. Pair
monitor schema now accumulates exact GPU compute identities across every poll,
including PID/start ticks/argv/cwd, process memory, observation gaps, and any
unrelated process. Comparison schema v8 binds that terminal monitor as a sixth
source and, for `stability_full`, binds the class-fidelity qualification as a
seventh source. Training wall time
and img/s remain published as raw observations, but
they are eligible for a direct efficiency ranking only when observation coverage
starts before training, remains continuous through pair completion, and never
sees unrelated GPU compute. Contended or incomplete runs retain direct
quality/budget comparability while their wall-clock fields are explicitly
observational-only. Schema v8 retains the schema-v7 budget boundary and adds
the paired class-fidelity count, top-1/top-5, mean target probability, target
NLL, predicted-class coverage, and normalized entropy to the matched direct
rows. It encodes the actual budget basis as matched optimizer steps and training
images (plus dataset, resolution, and effective batch), while permanently
setting equal wall-clock, GPU-hours, FLOPs, and generic compute-matched claims
to false. Time, throughput, and peak VRAM are measured outcomes rather than
pre-equalized budgets. The
full paired-config preflight confirms 62,836,011 vs 62,824,707 parameters
(+0.017993%). See
`docs/records/2026-07-12_generation_matched_compute_accounting.md` and
`docs/records/2026-08-02_generation_gpu_contention_provenance.md`; the explicit
claim boundary is recorded in
`docs/records/2026-08-03_generation_training_budget_claim_boundary.md`.
`cofitok.generation_pair.generation_pair_contract` additionally compares every
shared resolved model field, all data/diffusion/runtime/optimization fields,
the positive primary epsilon loss, and exact factorized-versus-dense identities.
Only token/synthesis/feedback fields and CoFiTok factorization-only auxiliary
losses may differ. Shared stabilization losses, including rollout consistency
and EMA-teacher consistency, must match exactly and may be nonzero for both
members; every remaining dense factorization-only auxiliary weight must be
zero. Both the training-pair validator and promotion/final gates use this
contract. See `docs/records/2026-07-12_generation_pair_contract.md`.

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
and comparison schema v8, including the exact formal sampling protocol fields,
GPU-contention policy, and—under `stability_full`—the source-bound class-fidelity
qualification. The comparison binds both training reports, both 50K metrics
reports, the final gate, the terminal pair monitor, and the stability-full
class-fidelity qualification by authoritative path, byte count, and SHA256;
the completion audit rereads those files before accepting any displayed metric,
class-fidelity result, or cost field.
Missing evidence is `in_progress`, contradictory evidence is `failed`, and only
the full chain is `complete`. See
`docs/records/2026-07-12_large_scale_generation_completion_audit.md`.

Before the legacy 128-channel full 300K path starts, both methods run the same
checkpoint-free training-runtime candidates `16x4`, `32x2`, and `64x1`.
The independent 256-channel stability-full readiness path instead benchmarks
`1x64`, `2x32`, `4x16`, `8x8`, and `16x4`, with `1x64` as
the explicit fail-closed baseline. Selection preserves
effective batch 64, requires both methods to fit below 90% VRAM, and minimizes
the slower method's synchronized optimizer-step time. The selected microbatch
and accumulation are applied to every full-training segment. Every completed
benchmark records the canonical Python/PyTorch/CUDA/GPU/project-lock runtime
environment; all candidates, both methods, and the subsequent full training
reports must share one exact environment SHA. Cached benchmark reuse and the
completion audit both reject environment or tracked-Git drift. Readiness can be
built only once while both formal run directories contain no state. That frozen
report binds both run paths, the complete candidate set and explicit baseline,
the 300K horizon, benchmark horizon, config SHA256 values, clean
branch/revision, dataset identity, and runtime environment.
A missing or drifted readiness source fails before full-training model load.
The completion audit replays the artifact on the later evaluation revision
without requiring the training directories to remain empty, while still
requiring the externally supplied SHA256 used to authorize launch.
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
Routine inference also freezes a source-bound manifest before its first PNG and
atomically advances per-output progress. Exact `--resume` rehashes and preserves
valid outputs, regenerates only missing or corrupt seed/class/prefix identities,
and rejects checkpoint/request/Git/runtime/control-evidence drift. A completed
resume performs no writes. Export smoke completion is accepted only after the
terminal audit independently rehashes the manifest and completed progress chain;
legacy smoke reports are not release evidence.
See `docs/INFERENCE.md` and
`docs/records/2026-07-12_stable_generation_session.md` plus
`docs/records/2026-08-02_generation_inference_exact_resume.md`.
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
For the 256-channel stability-full pair, readiness preflight multiplies the
largest measured 128-channel reference checkpoint by `4.0` before reserving
16 checkpoint slots. The report stores the reference bytes, multiplier,
rounded-up planned bytes, and reserve arithmetic; the stability completion
audit requires schema v2 and refuses a multiplier below `4.0`. The 300K launch
writes a separate `storage_capacity_launch.json`, so disk consumption between
CUDA qualification and launch cannot silently invalidate the earlier reserve.
That file becomes immutable once it is bound into
`full_training_launch_receipt.json`. Receipt creation occurs before any monitor
or trainer starts; failed receipt creation removes the unbound launch-storage
file. Resume requires the exact receipt SHA and writes the fresh capacity result
to `storage_capacity_current.json`. The terminal completion audit reopens the
receipt and all bound sources from the deployment-receipt checkout, allowing
later evaluation code while preventing a same-named config on that later
revision from replacing the training-time source.

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

A source-bound post-training supervisor can be started only after a human has
separately authorized full training and supplied the immutable full launch
receipt SHA256. It does not launch training. It waits for the exact 300K monitor
to pass, then runs formal 50K post-evaluation, validates the full scientific
gate, exports release-authorized EMA artifacts, and invokes the terminal audit.
Execution-stage failures receive bounded retries, while a failed scientific
gate or a failed/incomplete completion report stops permanently. Existing
final-gate or completion files are rehashed and must match the current complete
expectation map before reuse.
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
The final export runbook invokes both artifact preflights and both inference
smoke tests with release authorization required. Their reports carry this
policy bit, and the completion audit fails unless all four production loads
prove that the release gate was enforced before deserialization.
After that terminal audit passes, the completion runbook publishes a
deterministic `release_receipt.json` that binds the audit bytes and its unique
passing inference-artifact evidence to both physical EMA exports. Routine
production inference can require this stronger consumer boundary with
`--completion-receipt` and `--require-completion-authorization`; audit or
artifact drift is rejected before `torch.load`, and the receipt identity is
frozen into resumable inference metadata. This separates the quality-gated
exports needed *by* the terminal audit from artifacts proven to have passed the
entire completion chain. See
`docs/records/2026-08-03_generation_terminal_release_receipt.md`.

The stability completion runbook takes the 50K training and evaluation
revisions/branches as explicit required inputs. It contains no historical
post-evaluation revision constant, so a later v4 source-bound gate cannot be
silently audited against an obsolete checkout identity.
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

During a matched run, shared fixed-validation behavior can be inspected without
turning an intermediate metric into a quality gate using
`scripts/build_generation_matched_training_trajectory.py`. The builder freezes
byte-exact JSONL prefixes through an exact shared cutoff, checks Git, dataset,
runtime, pair-contract, logging, finite-metric, sample-accounting, and validation
provenance identities, then compares only the paired validation epsilon stream.
It explicitly prohibits total-loss and wall-clock comparisons and keeps quality,
promotion, formal-50K substitution, and full-training authorization false. The
first stability report through 10K is recorded in
`docs/records/2026-08-02_generation_stability_matched_10k_training_trajectory.md`.
The schema-v2 schedule-aware extensions at 12K and the checkpoint-aligned 20K
boundary are recorded in
`docs/records/2026-08-03_generation_stability_matched_12k_schedule_trajectory.md`
and
`docs/records/2026-08-03_generation_stability_matched_20k_schedule_trajectory.md`.
Through 20K, the 20 matched validation events split evenly by lower epsilon MSE
and the ratio of means differs by only `-0.291554%` for CoFiTok; this is evidence
of a closely matched, finite training trajectory, not free-rollout or generation
quality evidence. EMA-teacher consistency has not reached its 30K start in that
window, and exact matched 50K completion plus formal EMA post-evaluation remains
required.
