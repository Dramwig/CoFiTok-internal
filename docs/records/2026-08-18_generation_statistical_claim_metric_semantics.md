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

## Isolated Linux rehearsal

The exact incremental bundle from prerequisite
`c42ac96c6628ff71f7c67c8957c86ea0523aea0b` to target
`0f12a3c20a6795a219f498c98ba8f5143daa26b8` was rehearsed on `pro6000`
at 2026-08-18 07:18 CST without moving or modifying the formal checkout.

- bundle bytes: `15,563`;
- bundle SHA256:
  `5c40e145c6c1761da6f7d13eafea690fea940f98ea34407caa82a16c18aa930c`;
- target tree: `635dde1757f7ec5fe22e13434a8e245c80b3aab9`;
- rehearsal checkout:
  `/tmp/cofitok-claim-language-guard-rehearsal-3k0kbu/CoFiTok-internal`;
- focused language-guard suite: `9 passed`;
- integrated quality/capacity claim, terminal-uncertainty, and language-guard
  suite: `95 passed`;
- Python compile and `git diff --check`: pass;
- rehearsal checkout tracked state after verification: clean;
- formal checkout remained at
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
  `scale/generative-system` with zero tracked changes;
- the only GPU compute process remained the expected CoFiTok trainer PID
  `619775`; training advanced normally to step `37,550` while the CPU-only
  rehearsal ran.

The bundle and rehearsal checkout exist only under remote `/tmp`. No deployed
waiter was restarted or replaced, and no training process was signalled.
