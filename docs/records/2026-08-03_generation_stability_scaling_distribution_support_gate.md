# Stability-scaling distribution-support gate

Date: 2026-08-03

## Motivation

The formal stability 50K post-evaluation already computes matched 10K-sample
FID, Inception Score, precision, and recall. Its scaling decision, however, only
blocked on absolute/relative FID plus endpoint and mechanism evidence. A gate
could therefore pass while CoFiTok precision or recall had collapsed, which is
not sufficient evidence for spending the next large training budget.

## Change

- Generation-gate schema v4 adds a blocking
  `scaling_precision_recall_quality` row for the `stability_scaling` source
  profile.
- CoFiTok precision and recall must each be at least `0.10`.
- CoFiTok precision and recall must each remain within `0.05` absolute of the
  matched dense member.
- The authorization validator independently recomputes all four inequalities
  from the summary, rejects weaker declared thresholds, and cross-checks the
  named row's values against both the summary and thresholds.
- The formal stability 50K post-evaluation runbook passes all four thresholds
  explicitly. Generic scaling gates keep their previous contract; the full
  gate retains its existing `0.30` floors and `0.05` retention limits.
- Historical schema-v2 and schema-v3 evidence remains replayable. The new
  requirement applies only to newly built schema-v4 `stability_scaling` gates.

## Scientific interpretation

The `0.10` values are non-collapse readiness floors, not quality targets and
not a claim of competitive ImageNet generation. FID, precision, and recall
measure different failure modes; this change prevents acceptable aggregate FID
from masking inadequate fidelity support or coverage. A pass still does not
authorize full 300K training: it only strengthens the scientific evidence
available to the separately receipt-bound readiness decision.

## Active-run boundary

The active remote dense 50K trainer and its post-evaluation/readiness waiters
remain immutable and untouched. The frozen post-evaluation checkout at
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` will still produce its historical
schema. This local change prepares future or supplemental decision-grade gate
replay after the matched metrics exist; it does not replace a remote process,
move a checkout, or authorize full training.

## Verification contract

- Focused tests cover passing distribution support, both absolute floors, both
  matched-retention limits, weakened-threshold rejection, summary
  recomputation, schema-v2/v3 replay, and the explicit runbook flags: `85`
  focused tests passed.
- The complete repository suite collected `983` tests and completed with `977`
  passed / `6` skipped in `257s`.
- The modified post-evaluation runbook passed Linux `bash -n` on `pro6000`
  from its Git clean-filter blob (`600586b5e783f3c4ad18272e53067333f1bf0900`).
