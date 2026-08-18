# Matched generated-sample validation after conditioning-ranking recovery

Date: 2026-08-19

## Purpose

The four-arm 1K conditioning-ranking probe ends with a CPU-only held-out
noise-prediction sensitivity decision.  A shared pass shows that the ranked
objective makes the correct label more useful than wrong or null labels in
both CoFiTok and dense identity, but it does not show that generated images
follow their requested classes.

The current repository therefore has a real execution gap: the postevaluation
can recommend
`consider_separately_authorized_matched_sampling_validation`, but no exact
generated-sample stage implements that recommendation.  This record freezes the
minimum next experiment before implementation.

## Stage identity

```text
stage: conditioning_ranking_four_arm_sampling5k_v1
role: generation_conditioning_ranking_four_arm_sampling_validation
output root:
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_sampling5k_v1
```

The stage remains a four-arm comparison:

1. control CoFiTok K8;
2. ranked CoFiTok K8;
3. control dense identity;
4. ranked dense identity.

It may run only after the source-bound 1K postevaluation reports:

```text
status: completed
shared_semantic_alignment_recovery_supported: true
method_passes.cofitok: true
method_passes.dense_identity: true
cofitok_specific_advantage_claim_allowed: false
recommended_next_action: consider_separately_authorized_matched_sampling_validation
```

Any asymmetric or failed postevaluation must cause the sampling waiter to exit
without GPU work.

## Sampling contract

Each arm uses its exact step-1,000 EMA checkpoint and the same generated-sample
protocol:

```text
dataset: imagenet_256_10pct training source; ImageNet-256 validation real set
samples per arm: 5,000
class schedule: balanced modulo, exactly 5 images per ImageNet class
sample indices: 0..4,999
seed: 406020
sampler: DDIM
sample steps: 50
num train timesteps: 1,000
CFG scale: 1.5
guidance rescale: 0.0
CFG batch mode: batched
eta: 0.0
x0 clipping: enabled
precision: bf16
weights: EMA
CoFiTok prefix budget: 8
dense prefix budget: 1
```

The per-global-index random stream must be identical across all four arms and
remain invariant to batch size and resume.  The seed differs from the existing
formal and milestone streams.  Five thousand samples per arm provide five
independent requested images per class and enough paired units to detect a
shared semantic-direction change without pretending to be a formal 10K/50K
quality evaluation.

Batch size is not frozen until all four exact checkpoints pass real sampling
preflight.  One shared batch must be selected; any arm requiring a smaller
batch forces all arms to use that smaller value.

## Generated-sample evidence

The stage must produce, for every arm:

- sampling preflight with exact checkpoint sidecar validation;
- immutable sampling manifest, progress, and report;
- complete numbered PNG set and sample-set SHA256;
- FID and Inception Score on the same ImageNet-256 validation real set;
- fixed ResNet-50 V2 requested-class evaluation;
- runtime environment, sampling time, throughput, CUDA peak memory, checkpoint
  SHA256, and sample-set SHA256.

Precision/recall are omitted at this diagnostic 5K budget.  FID and IS are
trend/safety measurements, not formal paper or release metrics.

## Paired class-fidelity evidence

Aggregate Top-1/Top-5 counts are too insensitive near chance and cannot exploit
the matched random stream.  A new source-bound paired evaluator must retain one
row per generated sample and arm with:

- global sample index and requested class;
- control/ranked target log probability;
- control/ranked target probability;
- control/ranked predicted class;
- control/ranked Top-1 and Top-5 correctness;
- ranked-minus-control target-log-probability delta.

The evaluator must verify identical sample indices, requested labels, sampling
protocol, random stream, runtime environment, training Git, checkpoint step,
classifier bytes/SHA256, and image integrity before comparison.

Within each method, the independent unit is the generated sample.  A one-sided
exact sign test is computed over non-tied target-log-probability deltas.

## Predeclared method gates

For each of CoFiTok and dense identity, the ranked arm passes only when all
conditions hold:

1. mean ranked-minus-control target log probability is at least `0.02` nats;
2. the one-sided paired sign test for a positive target-log-probability delta
   has `p < 0.05`;
3. ranked mean target probability is at least `1.02x` control;
4. ranked Top-1 accuracy is not lower than control;
5. ranked Top-5 accuracy is not lower than control;
6. ranked predicted-class fraction is no more than `0.05` below control;
7. ranked FID is no more than `1.10x` control FID;
8. every checkpoint, sampling, image-set, evaluator, and runtime provenance row
   passes exact validation.

These thresholds test whether semantic improvement reaches generated pixels
without a large distribution-quality or support regression.  They do not claim
that a 1K model is usable.

## Shared decision

- Both methods pass: `shared_generated_class_alignment_recovery_supported` and
  recommend `prepare_separately_bound_matched_5k_training_recipe_confirmation`.
- Exactly one method passes: reject the objective as a shared fair repair and
  report method asymmetry; do not claim a CoFiTok advantage.
- Neither passes: recommend revising the training-time semantic-alignment
  objective.

Difference-in-differences between CoFiTok and dense remains descriptive only.
The experiment is designed to validate a shared conditioning repair, not to
manufacture a CoFiTok-specific win.

## Execution and authorization boundary

Implementation must use a new immutable preparation and execution receipt that
bind:

- the exact clean revision and branch;
- the completed 1K postevaluation path, bytes, and SHA256;
- all four training reports, checkpoint sidecars, and checkpoint SHA256 values;
- the standing authorization record;
- the exact stage and output root;
- the shared preflight-selected batch and complete sampling protocol;
- five consecutive idle-GPU polls with no unrelated compute.

The stage is permanently non-authorizing.  It cannot:

- launch more training;
- promote a checkpoint;
- authorize full 100K/300K training;
- replace the active full-data quality bridge or its terminal gate;
- authorize inference release;
- support broad generation superiority or a CoFiTok-specific advantage claim.

The active full-data bridge remains authoritative and must not be interrupted
while this validation is implemented and rehearsed.
