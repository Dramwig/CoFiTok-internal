# Full-data 100K claim-language guard waiter

Date: 2026-08-18 (Asia/Shanghai)

## Finding

The deployed full-data 100K claim-qualification waiter is correctly
non-authorizing and source-bound, but its historical report schema describes a
positive paired block-KID result as a statistically supported lower FID. FID is
only a matched point estimate in this protocol; the statistical uncertainty
evidence comes from paired block-KID.

The metric-semantics guard at revision `0f12a3c` corrects that distinction, but
before this change it required a manual invocation after the upstream
qualification finished. That was an avoidable terminal reporting gap.

## Implementation

An independent CPU-only post-source waiter was added:

```text
branch:   analysis/generation-quality-bridge-claim-language-guard-waiter-v1
revision: 62e96a56171c186e335c9baee4b19f15c608fe2e
tree:     e90f8ea8ec08313eb3de2ab950ff5a1959056ae4
subject:  Add quality claim language guard waiter
```

Source:

```text
scripts/run_generation_quality_bridge_claim_language_guard_waiter.py
bytes:  18,151
sha256: 44c30a64c4716c1b80735c1936e712c0d98ce912efcb12ea05213ea36335dbb7
```

The waiter:

- waits only for the exact deployed quality-bridge claim waiter;
- binds its PID, control revision/tree/branch, output root, terminal status,
  and immutable qualification identity;
- requires the source process to exit and its PID file to disappear;
- rejects source report identity or status-summary drift;
- invokes the existing SHA-bound metric-semantics guard exactly once;
- publishes FID as a point-estimate direction and paired block-KID as the
  statistical uncertainty evidence;
- propagates either `pass` or `hold` without changing the scientific decision;
- cannot launch training, sampling, export, release, or signal any process.

It uses a separate output root and exclusive lock. It does not replace, restart,
or modify the deployed source waiter.

## Validation

Local project-specific Python 3.10 validation:

- focused new waiter suite: `8 passed`;
- integrated quality/capacity claim, uncertainty, and language-guard suite:
  `102 passed, 1 skipped`;
- the skip is the optional local `torch_fidelity.metric_kid` import path;
- Python compile: pass;
- `git diff --check`: pass.

At code commit `62e96a5`, remote Linux rehearsal and deployment were pending.
Both stages must use an isolated checkout, hide CUDA, preserve the active
trainer and all existing waiters, and leave the formal checkout unchanged.

## Isolated Linux rehearsal

The complete incremental bundle from the deployed source-waiter prerequisite
`c42ac96c6628ff71f7c67c8957c86ea0523aea0b` to control revision
`c0fde4284c8a611b80fe02b97499e20393486264` was verified on `pro6000`.
The control revision includes the code commit above and this initial audit
record.

```text
bundle: /tmp/claim-language-guard-waiter-c42ac96-c0fde42.bundle
bytes:  24,886
sha256: b782c385683f3aa31c9aaa3357ce37afd76337a9f4aaa44d563c5db595b6e3b1
target tree: ccd971e8462c42fd63f2c9664525c7fb4a767243
```

The bundle advertised exactly one branch head, required exactly the deployed
`c42ac96` prerequisite, and passed `git bundle verify`. It was checked out
only in:

```text
/tmp/cofitok-claim-language-guard-waiter-rehearsal-Gg0l3X/CoFiTok-internal
```

Linux CPU-only validation used `CUDA_VISIBLE_DEVICES=-1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`:

- focused new waiter suite: `8 passed`;
- integrated claim, uncertainty, and metric-semantics suite: `103 passed`;
- Python compile and `git diff --check`: pass;
- rehearsal checkout tracked state: clean.

At 2026-08-18 07:32 CST, the only GPU compute process remained the expected
trainer PID `619775`; CoFiTok advanced normally to step `37,850`. The formal
checkout remained tracked-clean at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

No deployed waiter was changed or restarted. Persistent deployment remains
pending an integrity-bound deployment receipt and initial-state validation.
