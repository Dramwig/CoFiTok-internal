# 250M capacity scaling source-replay decision waiter

Date: 2026-08-14

## Purpose

The bounded matched 250M capacity probe intentionally stops both fresh runs at
step 10,000. Its result is non-authorizing and can only recommend a new
source-compatible scaling decision. The repository previously had no consumer
for that recommendation, so a successful probe would have stopped at a manual
decision gap.

This change adds a CPU-only decision builder, independent replay verifier, and
waiter. They rebuild the four-arm capacity result from its physical source
reports, verify the standing experiment instruction, and emit one exact next
stage.

## Decision policy

When both base-256 methods strictly improve FID over their matched base-128
references and the CoFiTok mechanism invariants remain valid, the decision may
authorize only:

- exact resume of the existing base-256 CoFiTok and dense checkpoints;
- source step 10,000 to required stop step 50,000;
- ImageNet-256, effective batch 64, unchanged 250M matched configurations;
- a paired 2,048-sample EMA/DDIM-50 milestone and 256-image CoFiTok mechanism
  audit at step 50,000.

The decision does not authorize fresh training, configured completion to step
100,000, full 300K training, promotion, formal claims, release, or changes to
the frozen evidence.

If capacity is not supported, no GPU or training authorization is emitted. A
shared FID improvement with failed mechanism invariants routes specifically to
factorization-mechanism recovery; lack of shared improvement routes to a
recipe/objective intervention.

## Files

- `src/cofitok/generation/capacity_scaling_decision.py`
- `scripts/build_generation_capacity_scaling_decision.py`
- `scripts/verify_generation_capacity_scaling_decision.py`
- `scripts/wait_for_generation_capacity_scaling_decision.py`
- `artifacts/runbooks/generation_capacity_scaling_decision_after_probe.sh`
- `tests/test_generation_capacity_scaling_decision.py`
- `tests/test_generation_capacity_scaling_waiter.py`

The waiter uses no GPU and launches no training. It writes a content-addressed
decision only after the exact capacity result exists and replays byte-equivalent
from all physical sources.

## Validation

- Targeted decision, waiter, capacity-probe, follow-up, supervisor, and CLI
  contract tests passed locally.
- `git diff --check` passed.
- The waiter runbook is statically asserted to contain no training entrypoint,
  GPU query, full-300K launch, promotion, or release path.
- The first isolated deployment attempt exited before reading any result because
  the runbook did not export the checkout-local `PYTHONPATH`. The failure log was
  retained, the runbook now exports both the project and `src/` roots, and the
  repaired entrypoint is verified through the real Linux launch environment.

## Isolated deployment

The repaired waiter is deployed from an isolated clean checkout; the formal
remote repository was not fetched, checked out, or modified.

- execution revision: `c18a606953e34725f0bae6f6f568db3ebf7c42c6`
- execution tree: `a7b1c4a9da1492c4114c1eccfc6a0c260c89c36a`
- execution branch: `scale/generation-capacity-scaling-decision-waiter-v1`
- checkout: `/root/autodl-tmp/CoFiTok/checkouts/capacity-scaling-decision-c18a606/CoFiTok-internal`
- incremental repair bundle: 1,221 bytes, SHA256
  `6528fe56727745fde020fec1a485ba7c916118498a4a6b99ed478d6a5a44f961`
- active waiter PID at deployment: `191931`, nice level 19
- status: `waiting_for_exact_capacity_probe_result`

The initial `e7d09a1` deployment failure is preserved in
`capacity_scaling_decision_waiter.log`; it exited at import and performed no
result read, GPU work, or training. The repaired checkout passed 11/11 targeted
Linux tests and a real runbook smoke that reached the expected waiting state
with GPU and training permissions both false.
