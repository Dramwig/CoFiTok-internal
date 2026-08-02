# Formal EMA rollout-stability gate

Date: 2026-08-03

## Motivation

The stability 50K post-evaluation already measured matched EMA DDIM samples,
10K FID/IS/precision/recall, and timestep-500 mechanism diagnostics. It did not
measure free-rollout stability with the same EMA weights and sampler protocol.
Consequently, another distribution-quality failure could not immediately
distinguish rollout high-frequency amplification or tail-energy concentration
from a broader representation-quality shortfall.

## Change

- `build_generation_stability_qualification.py` now accepts `--weights ema`
  while retaining raw `model` as its compatibility default. Training and
  evaluation Git identities are explicit and may differ; both matched
  evaluator reports must still share the exact clean evaluation revision and
  branch.
- The stability scaling post-evaluation now evaluates both final 50K
  checkpoints on 64 fixed validation images with EMA, DDIM-100, CFG `1.5`,
  clipped `x0`, bf16, and seed `2029` before building the formal gate.
- The dormant stability-full post-evaluation applies the same diagnostic at
  DDIM-250 and receipts both GPU evaluations plus the qualification as
  resumable high-cost stages.
- Schema-v3 stability gates require the passing qualification as a blocking
  `rollout_stability_diagnostic` row. The evidence must match the final
  checkpoint SHA256 values, checkpoint step, 1,024-image mechanism reports,
  clean evaluator revision/branch, pair contract, and formal sampling
  protocol.
- The qualification report is separately bound into the gate by authoritative
  path suffix, bytes, and SHA256. Gate-source replay rehashes it along with the
  existing six scientific sources.
- Schema-v2 gates remain accepted for immutable historical evidence. This is
  deliberate compatibility for the already frozen remote post-evaluation
  checkout; it is not a way for newly built schema-v3 stability gates to omit
  the diagnostic.

## Scientific interpretation

This diagnostic does not replace FID or relax any quality threshold. It makes
the gate stricter: a distribution-quality pass cannot promote a checkpoint if
the matched EMA rollout qualification fails. If FID fails, the qualification's
existing high-frequency, reconstruction-amplification, component-energy,
zero-token, and shuffle gates provide source-bound failure attribution for the
next recipe decision.

## Active-run boundary

The active remote 50K training and its post-evaluation waiter remain pinned to
the immutable `c1efb12c6640f2d2d62ac7e9982c8804d96e7289` evaluation checkout.
This local change does not replace, restart, or modify that waiter. It prepares
the next formal execution/follow-up diagnostic; full 300K training remains
unauthorized.

## Verification

- Focused gate, provenance, qualification, and both stability post-evaluation
  runbook suites pass (`88 passed`).
- The complete repository suite passes: `967 collected / 961 passed / 6
  skipped in 259.6s`.
- Both modified shell runbooks pass Linux `bash -n` on `pro6000` after line-end
  normalization.
