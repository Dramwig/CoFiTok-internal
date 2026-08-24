# Matched epsilon-stability sampling design (2026-08-24)

## Purpose and current state

The epsilon-stability controls now have a deterministic matched diagnostic
design instead of an executable GPU runbook. This distinction is deliberate.
At design time the 100K terminal chain was still running and no exact terminal
route receipt existed, so this change cannot launch sampling or infer that a
new experiment is permitted.

The design is produced by:

```text
scripts/build_generation_epsilon_stability_sampling_design.py
```

and replayed by:

```text
scripts/validate_generation_epsilon_stability_sampling_design.py
```

Both operate on CPU and create no execution authorization.

## Matched sampling contract

Every case is defined for both immutable 100K EMA checkpoints with the same
DDIM-100, CFG 1.5, balanced-modulo class order, one sample per ImageNet class,
fresh shared seed, zero-based sample indices, real set, evaluator, and classifier. CoFiTok
uses prefix budget 8 and dense identity uses prefix budget 1. Each method/case
has 1,000 screening samples, so the complete design contains eight cases and
16,000 images.

The cases form explicit parent-child comparisons:

| case | parent | incremental control |
|---|---|---|
| legacy terminal hard clip | none | matched reference |
| start-975 unit hard clip | legacy | nonterminal start |
| start-975 sigma hard clip | start-975 unit | schedule-sigma initialization |
| terminal dynamic threshold | legacy | x0 constraint |
| terminal dynamic threshold + recompute | terminal dynamic | constrained-x0 epsilon consistency |
| terminal hard clip + recompute | legacy | constrained-x0 epsilon consistency |
| start-975 sigma dynamic threshold | start-975 sigma hard clip | x0 constraint |
| start-975 sigma dynamic threshold + recompute | previous case | combined recovery upper bound |

The cosine schedule values bound in the design are:

| start | alpha_bar | sqrt(alpha_bar) | sigma | SNR |
|---:|---:|---:|---:|---:|
| 999 | 2.428e-9 | 4.927e-5 | 1.000000 | 2.428e-9 |
| 975 | 0.001398 | 0.037387 | 0.999301 | 0.001400 |

Timestep 975 is conservative: it reduces the initial epsilon-to-x0
amplification from roughly 20,000 to roughly 27 while keeping the schedule
sigma close to one.

## Causal and claim boundary

This design only studies sampling controls on frozen checkpoints. It does not
test Min-SNR training and cannot be used to claim that Min-SNR fixes the
system. A Min-SNR conclusion requires a separate fresh matched CoFiTok/dense
training experiment, output root, checkpoint chain, and authorization.

The 1,000-sample rows are screening evidence only. Even strict matched
improvement for both methods can only nominate one protocol for an independent
matched 10,000-sample confirmation. It cannot replace terminal evidence,
authorize 300K training, or support promotion/export/release.

## Execution blockers

The deterministic design keeps these fields deliberately unbound:

- terminal route receipt identity;
- separate execution authorization identity;
- exact repair Git revision and tree;
- both checkpoint payload/sidecar identities;
- real-set, evaluator, classifier, dataset, and runtime identities;
- fresh random-stream seed with evaluator-compatible `start_index=0`;
- versioned non-overlapping output root.

Until all are bound by a later execution receipt, every launch, evaluation,
training, promotion, export, release, and process-signal permission is false.

## Deterministic artifact and verification

The committed design artifact is:

```text
artifacts/reports/generation/epsilon_stability_sampling_design_v1/design.json
```

- bytes: `21,902`
- SHA256: `bb5231ad7645f0a0e3055a437b90c3e10a0866c931a8ae3806112652eb52bbb1`

The validator rebuilt it byte-semantically from source and matched the bound
SHA256. Targeted repair/design tests passed, Python `compileall` and
`git diff --check` passed, and the full repository suite completed as
`1,097 passed, 6 skipped` in `265.03s`.
