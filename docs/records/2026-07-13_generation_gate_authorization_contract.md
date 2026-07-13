# Generation gate authorization contract

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

The generation gate builder already enforced an absolute CoFiTok FID ceiling of
100.0 for the 10% ImageNet-256 scaling decision and 20.0 for final readiness.
However, the downstream shell pipeline consumed a saved report by checking only
`status=pass` and the decision string. The final completion audit applied strict
threshold validation to the full gate but did not apply the equivalent contract
to the scaling gate. A weakened or partially edited promotion report therefore
could have authorized the multi-week 300K stage.

## Contract

`cofitok.generation_gate.validate_generation_gate_authorization` is now the
single authorization contract for both stages. It requires:

- gate schema v1, the exact stage, passing status, and canonical decision;
- unique, passing, stage-specific required checks;
- scaling thresholds of at least 10K samples, at most 5% relative FID and
  endpoint-MSE regression, and absolute CoFiTok FID at most 100.0;
- full thresholds of at least 50K samples, absolute FID at most 20.0,
  precision/recall at least 0.30, and no more than 0.05 relative degradation;
- finite and mathematically valid summary metrics that satisfy those bounds;
- internally consistent FID, endpoint, ordered-prefix, zero-token, shuffle, and
  full precision/recall evidence.

`scripts/validate_generation_gate_report.py` exposes this contract to shell
runbooks. Both `generation_complete_pipeline_after_10pct.sh` and
`generation_full_matched_300k_after_gate.sh` call it before full training. The
large-scale completion auditor imports the same validator for both scaling and
full gates.

## Verification

Focused gate, runbook CLI, transition, and completion-audit tests pass. Negative
tests cover weakened sample/FID thresholds, missing required checks, summary
values that violate the declared threshold, and a weakened full precision floor.
