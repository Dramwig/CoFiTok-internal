# Matched Min-SNR 50K generation pilot preparation (2026-08-25)

## Decision source

The authoritative 100K terminal system remains an operational pass with a
scientific `hold`. The matched terminal point estimates favor CoFiTok over the
dense identity control, but absolute quality and class fidelity remain too weak
to prove a generation advantage. The completed 1K epsilon-stability sampling
diagnostic selected no shared sampler-recovery candidate.

The source-bound post-diagnostic decision is:

- decision: `prepare_fresh_matched_min_snr_training_pilot`
- decision SHA256:
  `b9ce02d19a01e48652277eecc82c12fa193b036b54e7d57edc52225105a34aff`
- verification SHA256:
  `e00837848be5ef785009d0321a096403e73d132c242889e22723eb7d4ce119a9`
- terminal status: `hold`
- `generation_advantage_proven=false`

The decision is preparation evidence, not an execution authorization. The user
subsequently supplied the standing instruction
`之后不要我授权你直接运行需要的实验`, meaning that needed experiments should run
directly without repeated authorization prompts. The pilot execution gate must
bind that exact instruction and its own narrow scope; arbitrary non-empty text
is rejected. A fresh, source-bound live execution gate remains mandatory.

## Controlled treatment

The pilot changes one shared training field only:

```text
loss.min_snr_gamma: 0.0 -> 5.0
prediction target: epsilon
weight: min(SNR, gamma) / SNR
```

The gamma-zero path retains the exact legacy scalar MSE reduction. The gamma-5
path computes a per-sample epsilon MSE in float32 and applies the standard
epsilon-prediction Min-SNR weight. `min_snr_gamma` is part of the matched pair
contract, so CoFiTok and dense cannot silently use different weighting.

No CoFiTok token, synthesis, feedback, model, data, diffusion, optimizer,
stabilization, or sampling field changes.

## Precommitted matched protocol

- dataset: full `imagenet_256`, identity
  `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
- methods: CoFiTok K8 and same-backbone `dense_identity`
- initialization: fresh for both methods; changed-config resume is forbidden
- seed: `2027`
- micro-batch / effective batch: `64 / 64`
- gradient accumulation: `1`
- scheduler: unchanged 100K cosine horizon
- pilot stop: exact step `50,000`, or `3,200,000` images per method
- recovery: same-config exact resume only
- controls: physically audited legacy gamma-zero step-50K checkpoints
- evaluation: four serial arms, `legacy/pilot x CoFiTok/dense`
- sampling: EMA, DDIM-100, CFG 1.5, no rescale, batched CFG, bf16
- sample count: 10,000 per arm
- shared evaluation seed / index interval: `20260825 / [0, 10000)`
- class schedule: balanced modulo over 1,000 ImageNet classes
- mechanism replay: 256 images for each arm; four random orders for CoFiTok

The precommitted selection requires at least 5% relative FID improvement for
both methods, bounded precision/recall and class-fidelity regressions, minimum
class-support thresholds, and retained CoFiTok zero/shuffle/order/coarse-energy
mechanism gates. A selected candidate still requires a separate future gate for
continuation beyond 50K.

## Recovery and evidence boundaries

The sole serial controller acquires one flock and runs CoFiTok training, dense
training, then the four evaluation arms. It revalidates GPU idleness before
every GPU stage. Existing training state is accepted only through exact
same-config resume and cannot exceed step 50K.

Each sampling preflight is bound to the exact physical checkpoint and execution
source. A failed or stale canonical preflight is moved to an immutable rejected
attempt path before a new preflight is run. An evaluation arm is reusable only
after structured validation of its preflight, sampling/metrics report, class
fidelity report, and checkpoint-mechanism report. The final result binds every
source by bytes and SHA256.

The preparation, execution gate, controller status, and result permanently keep
the following false:

```text
continuation_beyond_50000_allowed
full_training_launch_allowed
full_300k_launch_allowed
promotion_allowed
inference_export_allowed
release_allowed
process_signals_allowed
generation_advantage_proven
```

## Implementation identity

- branch: `scale/generation-min-snr-matched-pilot-v1-20260825`
- direct base: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- implementation revision: `f8c746d7549d010ae67fd8a7892951e887eddaa7`
- implementation tree: `ca9129938919d341fb5f3248934d354b34621cea`
- preparation runbook SHA256:
  `3b07366bae28e1dafae750199c38985f1d1db75a04213521937ec129f0aef8a1`
- execution-gate runbook SHA256:
  `bae29c1bcb9d5e6484589d15f36ca78eb3af50254497da2abda64bc4946b6cdb`
- serial controller runbook SHA256:
  `55cbf51e89c27da6c5c0fbf1e9a9e0c07b4b261b2342392c0e17c3bd846d6656`
- Min-SNR contract module SHA256:
  `7278e612ce7e581ae00f56add1d5e700abc01248d1922b25f6bacfd96ce0938b`

## Local verification

- focused Min-SNR suite: `7 passed`
- complete collection: `1,084` tests across `140` test files
- complete suite: no code/test failure; the isolated worktree initially lacked
  its parent-level `paper/` sibling, causing four path-only structure failures
- after adding a D:-side read-only junction to the existing paper tree, the four
  structure tests passed
- Python compilation of all changed Python entrypoints/modules: pass
- all three runbooks: `bash -n` pass
- `git diff --check`: pass

No remote checkout, execution gate, output root, training process, sampling
process, promotion, or release was created by this preparation commit.

