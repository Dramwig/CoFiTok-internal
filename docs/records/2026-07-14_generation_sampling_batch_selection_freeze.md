# Formal sampling batch selection freeze (2026-07-14)

## Finding

Both formal post-evaluation runbooks invoked the shared sampling batch selector
on every restart. Preflight caches were provenance-checked, but the aggregate
selection was recomputed and rewritten. If a 10K or 50K sample set was already
partial and measurement noise changed the selected batch, the immutable
sampling manifest would correctly reject resume. That protected provenance but
could permanently strand an otherwise resumable formal sample set.

The selector also checked runtime-environment equality within each CoFiTok/dense
candidate but not across successful candidates. Timings from two different
environments could therefore influence one ranking.

## Contract

`select_generation_sampling_batch.py` now receives both matched formal output
directories. Before either contains state it may benchmark and publish a shared
selection. After either contains any state it cannot run preflight subprocesses
or rewrite the report; it must validate and return the frozen batch.

The lock binds:

- both formal sample-set output paths;
- ordered candidates `16,32,64,128`, baseline 32, and the 90% memory ceiling;
- both checkpoint paths, SHA256 values, steps, and integrity sidecars;
- CoFiTok/dense prefix budgets 8/1;
- EMA, bf16, CFG 1.5, zero guidance rescale, batched CFG, and 2/5 warmup/measured
  forwards;
- clean Git revision/branch and the benchmark root.

Reuse revalidates every successful embedded preflight and deterministically
recomputes the complete selection. All successful candidate pairs must share
one canonical runtime-environment SHA. Existing sampling state without a lock,
or any checkpoint/protocol/candidate/Git drift, fails before model loading.

## Terminal evidence

The large-scale completion audit requires the full 50K selection lock, all four
candidates, exact formal output paths, both 300K checkpoint identities, and the
fixed EMA/bf16/CFG preflight protocol. It independently recomputes the selection
and rejects environment drift in non-selected candidates as well as the selected
candidate.

## Verification

Focused tests cover empty-state selection, read-only partial-sample reuse,
missing-lock failure, candidate/checkpoint/protocol drift, cross-candidate
environment drift, terminal lock tampering, and non-selected preflight drift.

- implementation commit: `459250b0aed0137353263bac0eef4734bf0fe036`;
- complete local suite: `558/558` passed;
- complete isolated Linux suite in the real sibling-`paper/` layout: `558/558`
  passed;
- tracked Linux runbook syntax audit: `40/40` passed;
- formal remote target-added-path scan: `150` paths, `0` conflicts;
- pinned `781a01444fddbf0d48a427ba58bdeed50167b5be` to implementation
  bundle: `427,363` bytes, SHA256
  `b9e21bc005dad4f8271e8bcb786222f99c1003e1fd72d0195cf2731336d48bfe`;
- formal remote HEAD remained pinned at `781a01444fddbf0d48a427ba58bdeed50167b5be`
  with a clean tracked worktree before and after verification.
