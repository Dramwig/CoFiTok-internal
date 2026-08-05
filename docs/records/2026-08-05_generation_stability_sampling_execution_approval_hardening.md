# Stability sampling execution approval hardening

Date: 2026-08-05

## Superseded again by the class-complete candidate

This file remains historical evidence for revision `1940788`. Neither
`1940788522c787bcb12f01dc9487b9490a34326a` nor its former replacement
`0a1b88630e2b24e27b3f084dfc1490381e906655` may be executed. The 512-sample
selection interval covered only ImageNet classes 0--511, and the former 10K
confirmation used a nonzero global-index interval that the metrics evaluator
could not actually consume. Both issues are corrected and independently
rehearsed at code candidate
`aed68853914dc0ecf4b3f15d5d090ddee740d1df`. See
`2026-08-05_generation_stability_sampling_recovery_class_complete.md`.

## Superseded execution candidate

This file preserves the approval-hardening rehearsal history. Revision
`1940788522c787bcb12f01dc9487b9490a34326a` must not be executed: its 10K
confirmation reused seed/global-index interval `[0, 10000)`, which overlaps
both the 512-sample protocol-selection sweep and the frozen formal 10K
evaluation. It is superseded by the independently rehearsed execution
candidate `0a1b88630e2b24e27b3f084dfc1490381e906655`, whose confirmation interval
is `[10000, 20000)`. See
`2026-08-05_generation_stability_sampling_confirmation_stream_independence.md`.

## Outcome

The frozen matched sampling-recovery and 10K confirmation entrypoints now
enforce explicit, stage-specific execution authority. The implementation
candidate is:

```text
development branch: scale/generation-stability-sampling-recovery-v1
revision: 1940788522c787bcb12f01dc9487b9490a34326a
tree: 7616103494bbe720747a39f5915dc4c51cf1f826
execution branch identity: scale/generation-large-capacity
```

This commit supersedes `d314c8d` as the sampling diagnostic candidate. It has
not been deployed to the formal checkout and is not authorized for execution.

## Enforced authority

Recovery requires all of:

```text
SAMPLING_RECOVERY_EXECUTION_ALLOWED=true
scope=stability_50k_sampling_recovery_v1_execution_only
approval text=Approve the non-authorizing matched 512-sample sampling-recovery diagnostic only.
```

Confirmation separately requires:

```text
SAMPLING_CONFIRMATION_EXECUTION_ALLOWED=true
scope=stability_50k_sampling_confirmation_10k_v1_execution_only
approval text=Approve the non-authorizing matched 10000-sample sampling confirmation only.
```

The user-created sentinel is bound by bytes/SHA256 to the physical plan or
recovery summary, exact clean Git revision and branch, exact output root,
timezone-qualified approval time, exact approval text, and an exact
authorization boundary. Recovery authority cannot authorize confirmation.
Both scopes keep training, full training, full 300K, release, frozen-gate
replacement, and automatic formal-protocol changes false. The validator also
rejects extra top-level authority, a non-user approver, and timezone-free or
otherwise ambiguous records. Validation happens before output-root creation or
GPU-lock acquisition.

No approval sentinel was created in this work.

## Verification

Local evidence:

- static Python compile, diagnostic-plan JSON parse, validator `--help`, and
  `git diff --check`: pass;
- approval/recovery/confirmation/runbook targeted tests: `32 passed`;
- full repository excluding the four parent-layout paper tests:
  `1,098 passed, 6 skipped` (`1,104` collected);
- the exact four candidate paper tests replayed against the canonical parent
  layout: `4 passed`.

The prerequisite-aware bundle was `40,180,101` bytes with SHA256
`79baa686ba248e1472cfb9459e80a991361a6d5a7af0c55cf80b1559803b6a2f`.
It advertised only candidate HEAD `1940788`; both prerequisites were present
on the server and `git bundle verify` passed.

In an isolated pro6000 checkout, with `CUDA_VISIBLE_DEVICES` empty:

- checkout revision/tree/branch were exact;
- both runbooks passed `bash -n`;
- the same targeted suite passed `32/32`;
- tracked state remained clean.

The formal checkout remained exactly
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`. Its full porcelain SHA256 was identical before and
after rehearsal. The remote bundle and isolated checkout were removed after
verification. The complete receipt is:

```text
artifacts/reports/generation/sampling_execution_approval_hardening_rehearsal_2026-08-05.json
```

## Live boundary

The frozen gate is still SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`,
`fail/hold`, with sole failed gate `absolute_fid_quality` and CoFiTok/dense FID
`138.2970/151.4477`. FieldScope PID `433140` still owns unrelated GPU compute
using about `15,412 MiB`; it was not modified or signaled. Available storage
was `299,912,142,848` bytes. Recovery, confirmation, and 100K-bridge output
roots remain absent.

Therefore this evidence authorizes nothing by itself. A future recovery run
still requires the exact candidate to be placed in a separate clean checkout,
an idle GPU, and a new user-created recovery sentinel plus the exact boolean
opt-in. A recovery approval does not authorize the later 10K confirmation, and
neither stage authorizes training or full 300K.
