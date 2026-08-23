# 100K post-reconciliation experiment decision

Date: 2026-08-24

## Outcome

The DDIM-50/2,048 versus DDIM-100/10,000 ranking conflict is resolved. A
CPU-only replay using the existing PNGs showed that the observed CoFiTok/dense
ranking reversal is driven by sampler step count, not sample count:

- DDIM-50/2,048: dense identity has lower FID;
- DDIM-100/first-2,048: CoFiTok has lower FID;
- DDIM-100/10,000: CoFiTok has lower FID.

This does not turn the terminal result into a scientific pass. The authoritative
quality screen remains `hold` because `cofitok_absolute_fid`,
`cofitok_recall_floor`, and `class_fidelity` fail. Both methods exceed the
absolute FID threshold and both have recall far below the floor. The matched
quality tolerances and every factorization-mechanism check pass.

The versioned post-reconciliation decision therefore classifies the remaining
problem as a mixed shared absolute-quality/support and class-conditioning
failure. It explicitly records:

```text
matched_quality_only_failure=false
factorization_mechanism_failure=false
class_only_failure=false
generation_advantage_proven=false
terminal_status=hold
```

Consequently neither the historical factorization-regression supervisor nor
the conditioning-only supervisor is eligible. Their canonical decision paths
and execution-authorization paths remain untouched.

## Selected minimum-information experiment

The audited choices were frozen-checkpoint sampling controls, matched capacity,
matched exposure, and a fresh training-recipe intervention. The decision selects
only a matched 100K epsilon-stability sampling discriminator for preparation.

The selected screening family keeps both immutable step-100K EMA checkpoints,
DDIM-100, matched random streams, balanced ImageNet classes, and shared case
selection. It may compare the legacy terminal start/hard clip with bounded
controls for:

- nonterminal sampling start;
- schedule-sigma initialization;
- dynamic `x0` thresholding;
- recomputing epsilon after the `x0` constraint.

Each method/case is limited to 1,000 screening samples with exactly one sample
per ImageNet class. Small-sample FID and precision/recall are non-formal. A
candidate would still require an independent matched 10,000-sample confirmation
and then a new formal gate.

This decision does not bind an executable repair revision, fresh random stream,
output root, evaluator/classifier identities, or execution receipt. A new
source-compatible execution gate must bind all of those before GPU work.

## Immutable sources

- authoritative pre-reconciliation decision SHA256:
  `ae989b2e0b0f06a2b28346aae629f5368079a45a9a5b53d90c2d6a20c137175a`;
- cross-protocol reconciliation SHA256:
  `6fc639caec0320225c2e8d26316490cde2d765d49c6062a18e0e671f79384d19`;
- terminal quality result SHA256:
  `15752e05611fa888e15352934c1627ccb95418df0339f9ad120e195bc088d165`;
- terminal training-exposure report SHA256:
  `d95c4d9c41327be36cb462af3076064123803b53d9f2f9e927ac89351513300b`;
- training revision/tree:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a` /
  `6cef27723196fd363379bca2e7b85b1678ebd777`.

The builder physically rehashed both terminal metrics reports, both
class-fidelity reports, both checkpoint payloads, both integrity sidecars, and
both canonical `latest.json` files before writing the decision.

## Implementation and remote replay

- code branch: `analysis/generation-100k-post-reconciliation-decision-v1`;
- code revision: `2ebecf56c5e4c86fc9021b617588dc94bd72a475`;
- code tree: `2c2693c705e9786b6bcd7ae23a5a555c6d2771dc`;
- remote checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/100k-post-reconciliation-decision-2ebecf5/CoFiTok-internal`;
- decision:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/100k_post_reconciliation_decision_v1_20260824/post_reconciliation_decision.json`;
- decision SHA256:
  `45817b27c33866f9c48eddc557c921e48c69c708d9fd47ee09695e086a1c47f4`;
- independent verification SHA256:
  `19512853989d8d001bacd063e20485ecaa57d96bdae75e40da069f8a0cb9352d`.

The exact replay was run twice; the second invocation reused and revalidated
both immutable outputs. Local and Linux related suites each completed `44
passed`; the runbook passed Linux `bash -n`, Python compilation, and `git diff
--check`.

After replay the GPU remained `0 MiB / 0%` with no compute process. No execution
authorization was created. The historical terminal-support, exposure,
factorization, conditioning, and random-token guarded processes retained their
exact PID/start-tick identities and no process signal was sent.

## Authorization boundary

Every execution or release capability remains false:

```text
sampling_launch_allowed=false
evaluation_launch_allowed=false
training_launch_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
legacy_factorization_route_allowed=false
legacy_conditioning_route_allowed=false
promotion_allowed=false
export_allowed=false
release_allowed=false
process_signals_allowed=false
new_source_bound_execution_gate_required=true
```

