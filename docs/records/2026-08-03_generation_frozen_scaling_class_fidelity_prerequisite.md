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

The exact implementation revision is
`d1ce0330c1b0fc9573570103763c9614d130ee2b`. A prerequisite-aware bundle from
the already attested remote base
`b45c490429c4e7f7ebeb1cc7eeca9ac66eaf2402` contains exactly the two later
commits, advertises only `d1ce033...` as `HEAD`, and is `37,579` bytes with
SHA256
`acff97f4170b9f803863ff2af15a10db2e29b28d8c870f5cd6ae8b20ab1207fe`.
Local and remote `git bundle verify` passed. It was fetched only into the new
isolated checkout
`/tmp/cofitok-stability-frozen-class-fidelity-d1ce033/CoFiTok-internal`, which
remained clean on `scale/generation-large-capacity@d1ce033...`; neither the
active training checkout nor any deployment checkout moved.

The first isolated full-suite invocation exposed four outer-layout failures:
the checkout had the inner repository but not the project-level `paper/`
sibling required by `test_aaai27_experiment_structure.py`. No implementation
or test was changed. The two exact files were copied into the isolated outer
root from the clean, previously attested
`/tmp/cofitok-launch-current-state-b45c490` source after matching the
authoritative project copies:

- `paper/venues/aaai27/main.tex`: `27,183` bytes, SHA256
  `d63ae7b2509166ba222f6fa63b3c1667794ab80b3a2424eae67116bd0107646a`;
- `paper/latex/main.tex`: `26,647` bytes, SHA256
  `d73e829739db55dee86047357c98c439bf25941e385a033a7beb058cf2b1d27b`.

With the exact outer layout restored, the complete Linux suite passed with
`1061 collected / 1059 passed / 2 skipped / 0 failed`. The rehearsal used
Python `3.10.20`, PyTorch `2.7.1+cu128`, `CUDA_VISIBLE_DEVICES=""`, and
`torch.cuda.is_available() == false`; common BLAS/OpenMP thread counts were
fixed to one. `compileall -q src scripts tests` passed, all `105/105` tracked
generation runbooks passed native `bash -n`, and direct `--help` imports passed
for the frozen class-fidelity verifier, qualification builder, full-launch
receipt builder, and full-launch receipt validator. The fixed ResNet-50
classifier was rehashed without deserialization or GPU use as `102,540,417`
bytes / SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

Small machine evidence is retained under
`artifacts/reports/generation/stability_frozen_class_fidelity_linux_rehearsal_2026-08-03/`:

- `linux_rehearsal_summary.json`: `3,144` bytes / SHA256
  `8a32392b688780d68bef587ad26a55cdc8345794b22c5dec80e46c8cdf0af7a4`;
- `pytest_d1ce033.log`: `1,200` bytes / SHA256
  `348ce468a3bee2de04532feea6a2ade14c23cecc8c6624efda04b2d9816fcb47`;
- `runbook_syntax_d1ce033.json`: `8,817` bytes / SHA256
  `3e7b708722d60edc8e66dbc712672e755322f37ea971fc2bf64a55587c415864`.

This rehearsal performed no GPU work, did not deserialize a training
checkpoint, did not deploy or start the dormant class-fidelity runbook, did
not modify any trainer or waiter, and retained
`full_training_launch_allowed=false`.

## Concurrent training boundary after rehearsal

The final read-only check at `2026-08-03T04:34:31Z` found the authoritative
dense recovery at step `30,050`, `1,923,200` samples, and `602` metric rows.
The pair monitor was `running` in `dense_identity_training` with no issues;
both rollout-consistency and EMA-teacher schedule audits covered all `602`
rows with no missing or mismatched steps. The exact 30K boundary row retained
EMA-teacher scale zero, while step 30,050 increased it to approximately
`0.005` with finite consistency `0.0004730072`, matching the configured 10K
warmup after start step 30K.

`checkpoint_step_00030000.pt` is `1,006,120,214` bytes with SHA256
`8d4a4e098febe9c61c687267488f5dcb2339bbad2ac03bc42c3f142ea9d21c45`;
its required sidecar and `latest.json` bind the same step, bytes, SHA, dataset,
runtime, and clean immutable
`scale/generation-stability-50k-preflight@2c2c1f5166b73d4f28df93b276901671ac1a7836`.
The pair monitor independently reported the binding as `metadata_verified`
with no pending integrity issues.

The GPU still contained only trainer leader PID `541878` (`77,970 MiB`),
with `19,271 MiB` free and no unrelated compute process. The filesystem had
`182,812,155,904` free bytes. Recovery/controller, pair monitor, watchdog,
post-evaluation waiter, readiness waiter, and frozen supplemental waiter kept
their expected identities; the three waiters remained waiting with no child.
Supplemental remained non-authorizing and readiness retained
`full_training_launch_allowed=false`. No process was signalled, paused,
restarted, or modified; CoFiTok was not rerun and full 300K was neither
authorized nor launched.
