# Stability sampling-confirmation random-stream independence

Date: 2026-08-05

## Superseded by the class-complete candidate

This file preserves the random-stream correction at revision `0a1b886`, but
that revision must not be executed. Its 512-sample selector covered only
ImageNet classes 0--511, and its declared `[10000, 20000)` confirmation could
not pass the then-zero-based-only metrics evaluator. The replacement code
candidate is `aed68853914dc0ecf4b3f15d5d090ddee740d1df`, documented and
rehearsed in
`2026-08-05_generation_stability_sampling_recovery_class_complete.md`.

## Outcome

The dormant 10K sampling confirmation no longer reuses the samples that select
its CFG/rescale protocol. The exact, non-authorizing execution candidate is:

```text
development branch: scale/generation-stability-sampling-recovery-v1
revision: 0a1b88630e2b24e27b3f084dfc1490381e906655
tree: 83636d4e488f87be83135b03ab3c7ef463019003
execution branch identity: scale/generation-large-capacity
```

This revision supersedes `1940788522c787bcb12f01dc9487b9490a34326a`.
Neither revision has been deployed to the formal checkout or authorized for
execution.

## Corrected scientific contract

The previous confirmation inherited `seed=0,start_index=0` from the diagnostic.
That reused all 512 protocol-selection random streams and the complete frozen
formal 10K stream. A quality result on those same samples would be optimistically
biased after protocol selection and could not be called independent.

The source-bound plan now predeclares:

```text
selection diagnostic global indices: [0, 512)
frozen formal global indices:         [0, 10000)
new confirmation global indices:      [10000, 20000)
seed formula:                          (seed + global_index) mod 2^63
seed:                                  0
```

Because the three stages use seed zero, the corresponding derived-seed
intervals are the same half-open intervals. The confirmation interval is
disjoint from both source intervals, does not wrap modulo `2^63`, begins at a
multiple of 1,000, and contains 10,000 samples. Balanced-modulo scheduling
therefore covers every ImageNet class exactly ten times.

`build_generation_stability_sampling_recovery.py` now fails closed on missing
or extra confirmation fields, non-integral/negative values, incomplete class
coverage, overlap in either global-index or derived-seed space, and seed
wraparound. It emits a recomputable `random_stream_independence` object with
all three windows and four exact disjointness checks.

`build_generation_stability_sampling_confirmation.py` binds that evidence to
the confirmation protocol and revalidates it before building a report. The
runbook reads count, seed, start index, and sampling/metrics batch parameters
from the validated preflight instead of hardcoding the frozen formal stream.
Existing outputs with `start_index=0` would be rejected rather than resumed.

## Local verification

At exact candidate `0a1b886`:

- approval/recovery/confirmation/runbook targeted suite: `41 passed`;
- full repository excluding the four path-dependent paper tests:
  `1,113 collected`, `1,107 passed`, `6 skipped`, `0 failed`;
- the exact candidate paper test blob matched the canonical test blob, and all
  four tests passed in the real sibling `paper/` layout;
- changed Python files passed `py_compile`, the plan parsed and validated, both
  builder CLIs passed `--help`, and `git diff --check` passed;
- direct reconstruction produced confirmation `[10000,20000)` and all four
  global-index/derived-seed disjointness checks as `true`.

## Isolated Linux rehearsal

The prerequisite-aware bundle was:

```text
bytes: 40189537
sha256: 043a0509646a5adaf80d3b7ed9fc74cd2526979ffe07d02bfd53530c89d8b860
advertised refs: one, HEAD=0a1b88630e2b24e27b3f084dfc1490381e906655
prerequisites: 1ebcc15210e63a776a2ba448481cbd8bb94a4066,
               58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

In `/tmp/cofitok-sampling-recovery-rehearsal-0a1b886/CoFiTok-internal`, with
`CUDA_VISIBLE_DEVICES` empty:

- revision, tree, and execution branch were exact;
- both runbooks passed native Linux `bash -n`;
- changed Python files passed `py_compile`;
- the 41 targeted tests passed;
- the physical plan independently reproduced the three intervals and four
  exact `true` checks;
- tracked and full porcelain remained empty after tests.

The remote checkout and bundle were removed afterward. The formal checkout
remained exactly `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`, with the same full-porcelain SHA256 before and after
the rehearsal. The complete receipt is:

```text
artifacts/reports/generation/sampling_confirmation_stream_independence_rehearsal_2026-08-05.json
```

## Live and authorization boundary

At the final read, the frozen gate remained SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`,
`fail/hold`, with sole failed gate `absolute_fid_quality` and CoFiTok/dense FID
`138.29702495267782/151.4476773464495`. Unrelated FieldScope PID `433140`
still owned about `15412 MiB` of GPU memory and was not modified or signaled.
Recovery and confirmation output roots remained absent.

No approval sentinel was created and no recovery, confirmation, training,
100K bridge, or full-300K process was launched. A future recovery run still
requires the user's exact recovery-only approval for revision `0a1b886...` and
an idle GPU. Recovery approval does not authorize confirmation. Confirmation
requires its own later approval bound to the completed physical recovery
summary. Both stages remain permanently non-authorizing.
