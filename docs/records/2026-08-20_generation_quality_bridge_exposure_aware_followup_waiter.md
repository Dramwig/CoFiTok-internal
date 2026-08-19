# Exposure-aware quality-bridge follow-up waiter

Date: 2026-08-20

## Evidence gap

The original quality-bridge follow-up router could send an improving but still
low-quality terminal result directly toward a capacity qualification. That
route did not bind the physical terminal training exposure and could therefore
silently treat about five full-data epochs as evidence that model capacity,
rather than training exposure, was the limiting variable.

This is a scientific routing problem, not a training failure. The existing v1
decision remains immutable evidence, but it is not sufficient on its own to
select the next experiment when absolute quality remains below threshold.

## Implementation

Commit `85e3ece1196fd318cd6823439824e19fca4275a3` on branch
`analysis/generation-quality-bridge-exposure-routing-v1`, tree
`1bf21fa0c44cec90011287b78f6f2ec14ed5eb17`, adds a separate exposure-aware v2
decision and a persistent CPU-only waiter.

The v2 builder must physically reopen and replay:

1. The terminal `quality_bridge_result.json` and every source it binds.
2. The terminal training-exposure report and every mutable training,
   preparation, milestone, result, and execution-status source it binds.
3. Exact matched completion at 100K steps, effective batch 64, 6.4 million
   images per method, and approximately 4.995 full-data epochs.
4. Exact terminal checkpoint binding for CoFiTok and `dense_identity`.

Only after those checks can it write
`followup_experiment_decision_exposure_aware_v2.json`. Insufficient exposure is
preserved as a live hypothesis. The existing capacity route ID is retained for
schema compatibility, while the decision also names
`prepare_matched_training_exposure_qualification` as the explicit fallback.

## Verification

The local full code suite, excluding four known tests that require the sibling
paper checkout absent from this isolated worktree, passed 1,239 tests with six
skips and no failures or errors. Its JUnit artifact is 182,585 bytes with
SHA256 `7d5a23a9cfc6227726dfb23a62dde9cd742fd82e554d4e0aff4c56070eb88234`.

The Linux CUDA-hidden focused suite passed all 66 tests. Its JUnit artifact is
10,628 bytes with SHA256
`2aebcdf6f2a2c0c85805fca516a977cf9355aacac542cb9fef00718a5b990478`.
Both new runbooks passed native Linux `bash -n`.

The bundle was verified locally and remotely:

```text
local: C:/Users/17194/AppData/Local/Temp/cofitok-exposure-route-85e3ece.bundle
remote: /tmp/cofitok-exposure-route-85e3ece.bundle
bytes: 43656415
sha256: 0f3520380ec8a8b2d93cdfdc7b5d95575377a42d1e91df491a1f3fa9a602c82a
advertised head: 85e3ece1196fd318cd6823439824e19fca4275a3
prerequisites: 1ebcc15210e63a776a2ba448481cbd8bb94a4066,
               58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

## Deployment

The clean remote checkout is:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-exposure-routing-85e3ece/CoFiTok-internal
```

The waiter runs as PID `587876`, parent PID 1, with
`CUDA_VISIBLE_DEVICES=-1`, one OMP/MKL thread, and an exclusive lock. Its exact
argv hashes to `a1d67f8eefae4dbba73f35a259362fd859b32e90aa85cb6b12559c52acf356cd`.
The PID file, `/proc` cwd, executable, start ticks, script SHA256, checkout
revision/tree, and single matching process were all reverified.

At deployment evidence capture, the waiter status was `waiting` with detail
`waiting_for_quality_bridge_result`. The active pair monitor was healthy at
CoFiTok step 51,650 and dense step 50,000. The only GPU process was the active
CoFiTok trainer, and no signal or mutation was applied.

The formal checkout remained at `scale/generative-system@1ebcc152...` with a
clean tracked worktree and index. Its 301 untracked paths were preserved; the
full checkout was therefore intentionally not described as clean.

## Claim boundary

This deployment improves the reliability of the next scientific decision. It
does not prove generation advantage, alter the active quality bridge, launch a
follow-up experiment, authorize full 300K, or release a model. The v2 decision
is parallel non-authorizing evidence and does not automatically supersede v1.
