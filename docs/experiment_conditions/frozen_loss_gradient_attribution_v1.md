# Frozen loss-gradient attribution: bounded diagnostic specification

Status: executable implemented and under pre-execution validation;
**not an execution authorization** (2026-09-13).

## Question

Does either shared auxiliary objective substantially oppose the main epsilon
gradient in the class-conditioning path, and how often is clipped-x0 rollout
supervision locally saturated? This follows the validated frozen class-support
diagnostic, whose AMI is weak but nonzero and whose held-out relabeling failed.
Do not repeat label-index or generated-label-contingency diagnostics as if new.

## Frozen sources to bind before execution

- The completed class-support result and receipt at hashes `9d1a5709...` and
  `6ad46a2c...`; full identities are in the September 13 result record.
- The original frozen 100K CoFiTok/dense quality-bridge checkpoints and native
  sidecars, not the 10K screen outputs. Read embedded configs, model/EMA state,
  runtime/source Git and exact checkpoint hashes before selecting executable code.
- A new immutable selection of 32 previously unused validation images/classes,
  one per class, from the authoritative ImageNet-256 validation manifest. Select
  by deterministic seed 2031, excluding all images in previous sensitivity and
  intervention probes. Preserve paths, WNIDs/labels, bytes and image SHA256.
- Exact clean analysis executable/checkouts and the unchanged relevant training
  source blobs; bind all hashes in the preparation. Never borrow authority from
  the failed terminal screen or the prior classifier-only approval.

## Measurement protocol

Use one image per microbatch, timesteps 100/500/700/900, matched per-image noise
and recorded RNG state across methods. Timestep rows are repeated measurements;
the independent unit is the validation image. Use frozen online weights for
training-loss attribution and the original EMA only as its detached teacher.
No optimizer, scaler update, EMA update, checkpoint write or new image sampling.

Run exact frozen precision first, not a new training recipe. Preserve and log
actual class-dropout decisions for the main forward, rollout forwards and EMA
teacher. Measure epsilon, rollout, EMA-teacher, and CoFiTok-only auxiliary
gradients **separately with their frozen weights/schedules**, then their sum.
Do not equate logged loss magnitudes with gradient magnitudes.

Implementation clarification before execution: the frozen training batch is 64,
with four teacher-selected images and up to eight rollout-selected images. In a
single-image probe the same ceil-based helper selects 1/1 when active. Thus the
measurement is the weighted gradient of one **selected** image, not the original
minibatch gradient or a replay of its full population/normalization. Report both
the frozen selection counts and probe counts. Do not silently multiply by batch
size or claim this establishes the minibatch gradient's direction. A subsequent
exact-batch check would be required before a claim about optimizer-step conflict.
Mixed-precision per-term and combined backwards can differ through rounding;
record their gradient-sum residual, rather than forcing exact additivity.

Record per image/timestep:

- True-label versus null-label routing and the exact selected-image fractions;
- raw-x0 saturation fraction at every unrolled step, clipped loss, and gradient
  with respect to that step's epsilon prediction;
- gradient norm and cosine to the main epsilon gradient for class embedding,
  conditioning projections, shared trunk, and output heads;
- zero/unused gradient separately from undefined cosine; finite checks; effective
  weighted term contributions, without silently replacing zero denominators;
- optional diagnostic-only paired dropout-mask replay, clearly separated from
  the exact-recipe result. It must not be reported as a trained repair.

The initial 32-image run is exploratory attribution, not a powered generation
quality comparison. Report all images and timesteps, image-level summaries and
paired effects; do not select just unfavorable timesteps or count them as extra
independent samples. Before execution, freeze any rule used to nominate a
follow-up ablation. A gradient conflict can select a hypothesis, not authorize
training or establish that changing it will improve FID/recall.

Before observing real gradients, freeze this exploratory nomination rule:
for the same auxiliary term (rollout or teacher) and same conditioning group
(class embedding or conditioning projections), cosine <= -0.1 and weighted
norm ratio >= 0.1 must hold at >= 3/4 timesteps in >= 24/32 images **in both
methods**. For a shared saturation hypothesis, both rollout steps must each
have >= 50% strictly saturated x0 coordinates at >= 3/4 timesteps in >= 24/32
images in each method. These are operational screening thresholds, not
statistical significance or established effect-size standards. Report every
count even when no route qualifies; no qualifying route is not proof of no
effect. A nominated route still does not authorize any intervention.

The executable is `scripts/frozen_loss_gradient_diagnostic.py`. Its bounded
invocation has exactly 256 primary rows, a 7,200-second own-process deadline,
one atomic output-directory claim and no automatic rerun. It measures online
weights without updating `.grad`, optimizer or EMA; it fingerprints model and
EMA tensors before/after, rehashes source checkpoints before/after, records
physical GPU ownership and preserves partial evidence on failure. Runtime must
exactly match the frozen report before model deserialization/forward.

The independent receipt rehashes every physical row and source, independently
recalculates scalar norms/cosines/sums and image-level summaries, and verifies
recorded tensor preservation. It **does not independently recompute full gradient
vectors**; that limitation is explicit in machine reports and cannot be promoted
to a claim of independent tensor-level replication.

## Falsification and stopping rules

If the auxiliary gradients are neither substantial nor consistently opposed to
epsilon, stop that route rather than automatically removing auxiliaries. If
saturation is rare, do not claim clipped gradients explain collapse. If teacher
conflict is observed at 100K, retain the counterexample that 10K screen collapse
exists before teacher activation: another cause is necessary for that endpoint.

If conditioning gradients are present and aligned while semantic control stays
weak, the next question concerns representational/optimization learnability or
rollout state distribution, not label transport. Any selected intervention must
retain compressed negative-noise tokens, current-token-only restricted synthesis,
and the matched dense epsilon control. Do not silently switch targets to x0/v.

## Authorization and resource boundary

Build and test the executable and physically validate a source-bound preparation
before requesting a distinct exact frozen-checkpoint diagnostic authorization.
Set a bounded invocation budget and idle-GPU/storage preflight in that preparation.
Write only to a new `checkpoints/generation/` diagnostic root. Rehash source
checkpoints before/after and produce an independent receipt. Missing or changed
sources, nonfinite gradients, or an existing output cause a fail-closed stop.

No new training, generated sample set, confirmation, 250M/300K stage, promotion,
export, release or paper integration is permitted by this document. The completed
terminal-SNR screen and all historical runs remain immutable.
