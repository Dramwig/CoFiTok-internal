# 2026-08-18 quality-bridge 31K first post-teacher validation

## Outcome

The active full-data 100K quality bridge reached optimizer step `31,000` and
persisted its first scheduled validation event after EMA-teacher consistency
became active. A source-bound CPU-only audit passed.

The exact validation result is:

```text
step: 31,000
samples seen: 1,984,000
EMA-teacher scale: 0.10000000149011612
EMA-teacher loss: 0.0007146265706978738
validation event index: 30
validation batch index: 30
validation noise seed: 102030
validation images: 64
validation epsilon MSE: 0.025744637474417686
```

The row is finite, has complete validation provenance, satisfies
`samples_seen == step * 64`, and follows the configured linear EMA-teacher
schedule exactly.

## Source-bound audit

The existing read-only consistency transition auditor was executed from the
isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-ema-transition-2a80f6e
```

It verified the active training checkout at:

```text
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
branch: scale/generation-stability-quality-bridge-100k
tracked dirty: false
```

The audit covers every canonical metrics row through step `31,000`, including
the exact disabled boundary at 30K and the 20 active rows from `30,050` through
`31,000`. It does not load a checkpoint, use GPU, or signal any process.

Authoritative report identities:

```text
audit report bytes: 31,913
audit report SHA256: f044ab66d45f3c585126c3c375753f98bf5d129ba063f7efd557523ff437051e
audit status bytes: 1,342
audit status SHA256: 73c775222cadd75d21fa9c9ecbd7c8d3053753fef5e5d8194879c1c75d5f4e64
metrics snapshot bytes: 582,857
metrics snapshot SHA256: 58dd174c21f14bd5919c213a23cec57a7bbf715ddb106286c30a8f0ea306fa1f
```

The first invocation omitted the isolated checkout's `src` directory from
`PYTHONPATH` and exited at module import. It did so before argument parsing,
lock creation, report creation, GPU access, or any interaction with training.
The corrected invocation set `PYTHONPATH` explicitly and produced the passing
report above.

## Descriptive stability context

The 30K boundary validation MSE was `0.024968227371573448`. The 31K event is
`3.1096%` higher, but these are different scheduled validation batches and the
comparison is not a causal estimate.

Across the final ten pre-teacher scheduled events at steps 21K through 30K:

```text
mean: 0.028952131047844887
median: 0.029547957703471184
population standard deviation: 0.0020478149313135838
minimum: 0.024968227371573448
maximum: 0.032176531851291656
31K z-score: -1.5663005110377495
```

The 31K value lies inside the observed pre-teacher range and below its mean.
This is evidence against an immediate one-event validation discontinuity. It
is not evidence of temporal convergence because scheduled events use different
validation batches.

The historical 10%-data stability run logged the same event identity at step
31K with MSE `0.025718122720718384`. The full-data value is only `0.1031%`
higher. Event index, batch index, noise seed, image count, step, samples seen,
and EMA-teacher scale match exactly. This is useful descriptive context, but
the different training dataset and Git revision prevent treating it as a
causal control.

## Live continuity

Training continued past the audited event to step `31,050`. The runbook, pair
monitor, watchdog, trainer, 40K warmup waiter, and matched-uncertainty waiter
remained alive. Pair monitor and watchdog status were `running` with no health
issues. GPU compute contained only trainer PID `619775` using `85,284 MiB`;
there were no unrelated GPU processes.

## Claim boundary

This evidence proves that the first post-activation validation event was
finite, provenance-complete, schedule-correct, and not an obvious one-event
discontinuity. It does not establish EMA-teacher causal benefit, sample
quality, matched superiority over dense, promotion readiness, release
readiness, or any authorization for 300K training.

The compact evidence receipt is:

```text
artifacts/reports/generation/quality_bridge_31k_first_post_teacher_validation_evidence_2026-08-18.json
bytes: 7,090
SHA256: e2ef050c8475bca446cb28b32d637452d8d474a0efd185146e06b79d435af826
Git blob OID: 9dbfdaeed4a7b04cb79e734aabefac131d2753f1
```
