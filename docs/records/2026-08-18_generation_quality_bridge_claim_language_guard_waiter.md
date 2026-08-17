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

Remote Linux rehearsal and deployment remain pending at this record revision.
They must use an isolated checkout, hide CUDA, preserve the active trainer and
all existing waiters, and leave the formal checkout unchanged.
