# Statistical claim metric-semantics hardening

Date: 2026-08-18 (Asia/Shanghai)

## Finding

The terminal matched-uncertainty audit uses a paired block-KID contrast with a
shared real reference. It reports a 95% bootstrap interval over disjoint
generated-sample blocks and an exact one-sided sign test. FID is retained as the
matched point-estimate direction.

Those two roles must not be conflated: significance of the paired KID contrast
is not a confidence interval or hypothesis test for FID itself. The rigorous
positive wording is therefore:

> Under the exact bound matched protocol, CoFiTok K=8 obtained a lower FID point
> estimate than dense_identity, and a paired block-KID analysis supported the
> same distribution-quality direction.

It is not valid to describe the paired KID result as a statistically significant
FID difference.

## Guard

`scripts/build_generation_statistical_claim_language_guard.py` binds either the
full-data 100K quality-bridge claim qualification or the future capacity-full
300K claim addendum by SHA256. It reopens and verifies the exact paired
uncertainty report and publishes separate machine-readable roles for:

- FID: matched point-estimate direction only;
- paired block-KID: statistical uncertainty evidence;
- combined matched distribution-quality language.

The guard always sets `fid_statistical_significance_claim_allowed=false` and
`fid_confidence_interval_claim_allowed=false`. It remains non-authorizing and
cannot launch training, sampling, export, release, or process signaling. It does
not modify or replace the already deployed waiters or their immutable evidence.

No positive guard artifact can be produced before the source quality screen and
paired uncertainty report both pass. At implementation time those terminal
sources do not yet exist, so this change prepares the final reporting boundary
without asserting an experimental result.

## Verification

- focused language-guard suite: `9 passed`;
- integrated quality/capacity qualification and matched-uncertainty suite:
  `70 passed, 1 skipped`;
- the one skip is the optional local `torch_fidelity.metric_kid` import path;
- Python compile: pass;
- `git diff --check`: pass;
- no server process, waiter, checkpoint, sample, or formal checkout was changed.
