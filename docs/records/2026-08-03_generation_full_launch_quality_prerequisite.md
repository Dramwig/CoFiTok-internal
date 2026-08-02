# Full 300K launch quality prerequisite

Date: 2026-08-03

## Problem

The frozen 50K supplemental chain adds the missing matched EMA free-rollout and
precision/recall distribution-support evidence, but its first implementation
was a side report. The existing schema-v2 full-training launch receipt bound
the original promotion gate, readiness, bridge, deployment, runtime, configs,
and storage without consuming the supplemental. Consequently, a later operator
could create a 300K launch receipt even when the newer sample-quality evidence
was absent or failed.

That is not a usable quality-gated generation system. Evidence that can be
ignored by the scale transition is diagnostic documentation, not a decision
boundary.

## Change

The full-training launch receipt is now schema v3. Its exact source set grows
from ten to eleven reports by adding:

```text
stability_scaling_50k_ema_teacher/
  reports/frozen_posteval_supplemental/supplemental_qualification.json
```

Receipt creation and replay require an externally supplied SHA256 for this
report. The verifier then:

1. Rehashes the supplemental report itself.
2. Rehashes its four direct reports: the original promotion gate,
   post-evaluation verification, distribution-support qualification, and EMA
   rollout qualification.
3. Rehashes the raw post-evaluation waiter and every nested distribution and
   rollout source.
4. Requires the supplemental's promotion-gate identity to be the exact gate
   used by the full launch.
5. Revalidates the schema-v1 combined contract against the physical gate
   provenance and clean supplemental builder Git identity.
6. Requires the exact distribution-support checks and rollout-stability gates
   to be present and passing under the frozen thresholds/protocol.

The receipt stores the verified prerequisite under
`quality_prerequisites.frozen_stability_supplemental` and separately keeps its
source identity in `source_reports`.

## Authorization semantics

This is a necessary negative prerequisite, not a new positive authorizer:

```text
supplemental_non_authorizing=true
required_for_full_training_launch=true
full_training_launch_allowed=false
```

Thus:

- missing/failing/held/drifted supplemental -> no launch receipt;
- passing supplemental alone -> still no launch receipt;
- launch additionally requires the promotion gate, readiness, readiness
  revision bridge, matching deployment, runtime/config contract, fresh storage,
  absent training state, and separate human launch authority.

No milestone audit can authorize full 300K through this path.

## Downstream replay

The full runbook requires the exact supplemental SHA before creating or
replaying its immutable launch receipt. The readiness revision bridge recognizes
only these new quality-binding lines as a controlled preamble upgrade; the
training execution suffix must remain byte-identical and every training-critical
Git blob must still match the source readiness revision.

The post-training supervisor now requires schema v3, rehashes all eleven launch
sources, and cross-checks the receipt's prerequisite identity and pass checks.
The terminal completion audit independently reconstructs the launch receipt
from the deployed training checkout and the physical supplemental source.

The launch-time verifier is deliberately implemented as the standalone
`scripts/verify_generation_stability_frozen_supplemental.py`. It depends only
on the stable reporting helper plus the standard library; it does not require
the later `cofitok.generation.frozen_supplemental` module. This keeps the
quality binding eligible for a control-plane-only target without changing the
trainer package or checkpoint semantics.

## Readiness bridge compatibility audit

The active readiness waiter is bound to source revision
`5dd3488ac9b30274f4960195e252cc9fdb161002`. A direct ancestry audit over all 50
commits through pre-change HEAD `e448b7e5ee93da38573f6da2a45dd540ee0613f7`
found:

- latest revision with both the source training-critical manifest and training
  execution suffix unchanged:
  `27edb632d8cb9a642040ba6fcd3b1a0d682e6c79`;
- latest revision with only the training execution suffix unchanged:
  `5029487a9a3c91c5447e3da3f9ee9ab0ff8b7425`;
- first training-critical manifest drift:
  `08d48504ac4898321b0c873574599bcfa063813f`;
- first training-execution drift:
  `00549fe3ec4ea5334c76e7249537224104ee44be`.

Therefore current branch HEAD is **not** a valid direct bridge target for the
active `5dd3488...` readiness. A later launch must either use a dedicated
control target whose training-critical blobs and execution suffix still match
the readiness source, or obtain a fresh readiness receipt on the final target.
This change does not claim or synthesize such a bridge.

## Verification

- Python compile check passed for every modified control script.
- 42 focused receipt/bridge/runbook/audit/supervisor tests passed.
- Full local suite passed: 1,010 passed and 6 skipped (1,016 collected).
- The modified full-training runbook passed remote Linux `bash -n` on
  `pro6000` after LF-normalized index streaming.
- `git diff --check` passed.

## Active-run boundary

This change does not deploy, start, stop, pause, or restart any remote process.
It does not run GPU evaluation and does not grant current full-training
authorization. The active dense 50K trainer, frozen post-evaluation waiter, and
readiness waiter remain untouched. The prerequisite can become satisfiable only
after those existing stages finish and the separate supplemental evaluation
produces a passing source-bound report.

The final read-only snapshot for this change observed dense step `20,300/50,000`
with exact `samples_seen=1,299,200`, finite monotonic metrics, one CoFiTok GPU
trainer, and no monitor issues. The new 20K recovery checkpoint and integrity
sidecar were present; `latest.json` bound SHA256
`86b4f9bd3a76168d94c9c3420346ca1c66dbab9a024dd2d45221edad20162f1f`.
The post-evaluation/readiness waiters remained waiting and readiness retained
`full_training_launch_allowed=false`.
