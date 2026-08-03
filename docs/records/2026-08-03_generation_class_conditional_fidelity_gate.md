# Generation class-conditional fidelity gate

Date: 2026-08-03
Branch: `scale/generation-large-capacity`
Base revision: `322e36ab11facc96fac9f98982d689b8df5c6814`

## Motivation

The formal ImageNet-256 sampler is class conditional, but FID, Inception Score,
precision, and recall do not prove that a generated image follows its requested
class. A model can ignore, collapse, or consistently permute labels while still
producing a plausible unconditional distribution. The formal stability gate
therefore needs an independent, source-bound class-fidelity check over the same
EMA sample set used by the distribution metrics.

This change does not modify training, the restricted synthesis operator, or any
active checkout. It does not authorize or launch full 300K training.

## Fixed evaluator contract

`scripts/evaluate_generation_class_fidelity.py` evaluates each formal PNG with
the torchvision ResNet-50 ImageNet-1K V2 classifier. The classifier is not
downloaded implicitly and is accepted only at the following identity:

- path default:
  `/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth`
- bytes: `102540417`
- SHA256:
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- weights enum: `ResNet50_Weights.IMAGENET1K_V2`
- classes: `1000`, with a pinned category-order SHA256
- preprocessing: pinned V2 resize/crop/mean/std/interpolation/antialias values

The requested class is reconstructed as
`int(zero_based_png_stem) mod 1000`. The evaluator first replays the formal
sampling manifest/progress/report and physical PNG digest, requires EMA weights,
balanced-modulo classes, and the selected K8/K1 prefix budget, then records:

- top-1 and top-5 requested-class accuracy;
- mean requested-class probability and requested-class negative log-likelihood;
- requested and predicted class support;
- predicted-class coverage and normalized predicted-class entropy.

The output is component-locked and supports only an exact completed-result
replay under identical sample, classifier, Git, runtime, and request identities.

## Paired qualification and thresholds

`scripts/build_generation_class_fidelity_qualification.py` requires CoFiTok K8
and dense K1 reports to share the exact formal EMA sampling protocol, evaluator
Git/runtime, fixed classifier, and sample count. It independently recomputes all
absolute and paired checks.

| stage | samples | DDIM | top-1 | top-5 | class coverage | normalized entropy | max CoFiTok top-1/top-5 regression |
|---|---:|---:|---:|---:|---:|---:|---:|
| scaling | 10,000 | 100 | 0.01 | 0.05 | 0.25 | 0.50 | 0.05 absolute |
| full | 50,000 | 250 | 0.10 | 0.25 | 0.50 | 0.70 | 0.05 absolute |

Both matched methods must satisfy the absolute floors. CoFiTok must also remain
within the paired top-1 and top-5 regression limits. A qualification can be a
source-valid `hold`, but only `pass` satisfies a new stability gate.

## Gate, comparison, and completion integration

- Generation gate schema v5 requires the paired qualification for both
  `stability_scaling` and `stability_full`. Gate provenance additionally binds
  both raw class-fidelity reports and replays their physical identities.
- The scaling and full post-evaluation runbooks execute the two classifier
  evaluations and qualification from the same formal sample roots. The full
  path places all three high-cost/restart-sensitive operations under immutable
  stage receipts.
- Strong-baseline comparison schema v8 adds the qualification as the seventh
  `stability_full` source and carries the paired class metrics into the two
  matched direct rows. Contextual D-AR/MAR/ReTok rows remain in the separate
  official-pretrained tier.
- Both generic and stability completion audits reopen the qualification, its
  two raw reports, checkpoint/sample identities, and displayed comparison rows.
  Tampered paired deltas or row metrics fail closed.
- Historical gate schemas v2-v4 remain replayable. The new requirement applies
  to newly built schema-v5 stability evidence and does not rewrite locked runs.

## Claim boundary

Class fidelity answers whether formal class-conditional samples respond to the
requested ImageNet labels. It does not measure unconditional distribution
quality and cannot replace FID, IS, precision, recall, EMA rollout stability,
mechanism diagnostics, visual review, or the complete release chain. Every
qualification fixes:

- `standalone_generation_quality_claim_allowed=false`
- `full_training_launch_allowed=false`
- `release_authorization_allowed=false`

## Local validation

- complete pytest: `1041 passed, 6 skipped in 304.28s`
- `python -m compileall -q src scripts tests`: pass
- both new direct CLI `--help` entrypoints: pass
- `git diff --check`: pass

The active pro6000 dense stability trainer, post-evaluation waiter, readiness
waiter, and supplemental waiter were not modified, signalled, or restarted.
