# Frozen scaling class-fidelity launch prerequisite

Date: 2026-08-03
Branch: `scale/generation-large-capacity`

## Gap

The frozen stability post-evaluation checkout
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` produces the formal matched 10K
EMA DDIM-100 sample sets, but it predates the class-conditional fidelity gate.
The later frozen supplemental adds precision/recall and EMA rollout evidence,
yet it still does not prove that either class-conditional generator follows the
requested ImageNet label. Consequently, the previous schema-v3 full-launch
receipt could satisfy its quality prerequisites without class fidelity.

The first class-fidelity report schema also required the sample-generation Git
identity to equal the evaluator Git identity. Re-evaluating the immutable
`c1efb12...` sample sets from a later clean evaluator checkout would therefore
be rejected even though preserving both identities is the scientifically
correct operation.

## Cross-revision evidence contract

Raw class-fidelity report schema v2 keeps two explicit identities:

- `sample_provenance.git`: the exact frozen sampler revision and branch;
- `git`: the clean later evaluator revision and branch.

The paired qualification copies those identities into `sampling_git` and
`evaluator_git`, binds the evaluator runtime-environment SHA256, and still
requires CoFiTok and dense to share the same sampler Git, evaluator Git,
runtime, classifier, formal sampling protocol, and sample count. Schema v1
remains replayable only under its original same-Git rule.

## Dormant follow-up runbook

`generation_stability_frozen_50k_class_fidelity_after_supplemental.sh` is a
quality-only follow-up. It:

1. requires the exact successful frozen post-evaluation status;
2. requires the source-bound supplemental waiter and physical supplemental
   qualification to pass, then independently replays that supplemental;
3. revalidates the original `stability_scaling` promotion gate;
4. refuses to start while any GPU compute process exists;
5. evaluates the existing CoFiTok K8 and dense K1 10K EMA sample trees with the
   fixed torchvision ResNet-50 ImageNet-1K V2 checkpoint;
6. builds the scaling qualification and accepts only an exact `pass`;
7. wraps both GPU evaluations and the qualification file in immutable
   `run_generation_stage_once.py` receipts.

The qualification stage declares only its output file. Its receipt lives in a
sibling `stage_receipts/` directory, so it does not violate the stage runner's
rule that receipt state must remain outside declared outputs.

## Independent replay

`scripts/verify_generation_stability_frozen_class_fidelity.py` does not trust a
summary `pass`. It reopens and rehashes both raw reports, recomputes the
evaluator runtime hash, validates the fixed classifier and formal EMA
DDIM-100/balanced-modulo protocol, recomputes all metric arithmetic, paired
deltas, and all ten threshold decisions, and binds checkpoint/sample-set
SHA256 values back to the original promotion gate. The qualification, both raw
reports, and promotion gate are rehashed again after validation to close
time-of-check/time-of-use drift.

## Full-launch integration and boundary

Full-launch receipt schema v4 adds the qualification as source 12 and requires
the standalone replay to pass. The full runbook, readiness revision bridge,
post-training supervisor, and terminal stability audit all carry and replay the
same binding. A missing, held, failed, forged, or drifted qualification blocks
receipt creation.

This evidence remains non-authorizing:

```text
class_fidelity_passed=true
supplemental_non_authorizing=true
required_for_full_training_launch=true
full_training_launch_allowed=false
```

It neither launches nor authorizes full 300K, does not rerun CoFiTok, and does
not replace FID, IS, precision, recall, EMA rollout stability, visual review,
readiness, deployment identity, fresh storage, or separate human authority.
The currently running dense 50K trainer and all remote waiters were left
untouched while this control-plane change was developed.

## Validation

Focused class-fidelity, launch-receipt, readiness-bridge, full-runbook,
supervisor, completion-audit, and runbook/CLI contract tests passed locally.
`compileall` over `src`, `scripts`, and `tests`, the new runbook's embedded
Python syntax, and `git diff --check` also passed. The final complete local
suite collected `1061` tests and finished with `1055 passed, 6 skipped` in
`237.5s`.

Isolated CPU-only Linux validation is recorded after the exact implementation
revision is committed.
