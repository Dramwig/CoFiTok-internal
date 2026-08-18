# Generation terminal evidence-chain live continuation

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The active full-data matched 100K quality bridge remains healthy and its first
50K checkpoint/evaluation/dense-transition path is statically ready. No new
control-chain gap requiring an active-run change was found. Scientific terminal
qualification remains pending because CoFiTok has not reached 100K, dense has
not started, and the bound quality, paired-uncertainty, claim-language, visual,
and runtime terminal reports do not yet exist.

This continuation is read-only evidence. It did not replace, signal, pause, or
restart a controller, trainer, monitor, guard, or waiter, and it did not launch
sampling, evaluation, later training, release, or full 300K work.

## Live snapshot

At `2026-08-18T12:35:46+08:00`:

```text
pair status/stage: running / cofitok_training
pair issues:       []
CoFiTok:           44,250 / 100,000
images seen:       2,832,000
dense:             0 / 100,000
```

The latest bound checkpoint remained step 40,000:

```text
bytes:  1,010,937,514
SHA256: f6b0b3d285d4565d7d163dedc4727d5d9580581cf857e4856df6801dfff283a2
```

The CoFiTok step-50K physical-integrity waiter was still
`waiting / checkpoint_missing`, as expected before publication. The exact
controller guard remained `observing`, with zero identity-loss polls and no
mismatches.

## 50K transition evidence

The independent transition audit at revision
`7c7bc42e26c80df809180339fa69279d238eb24f` established that the deployed
foreground order is:

1. CoFiTok training to exact 50K and physical checkpoint verification;
2. synchronous CoFiTok 2,048-sample DDIM-50 milestone evaluation;
3. dense training to exact 50K and physical checkpoint verification;
4. synchronous dense milestone evaluation;
5. validated paired step-50K report;
6. continuation toward exact 100K only after the paired milestone succeeds.

Checkpoint publication is atomic, the 50K/100K checkpoints are protected from
rolling retention, and completed milestone evidence is reusable only under the
same bound checkpoint, protocol, runtime, and revision. No blocking publication,
retention, exact-resume, serial-transition, or recovery gap was found.

## Terminal and runtime boundaries

The terminal evidence contract audited at `8cc6ab6dedad` remains unchanged:
positive large-scale statistical support requires a lower CoFiTok FID point
estimate, a paired block-KID bootstrap interval whose upper bound is below zero,
an exact one-sided sign-test p-value at most 0.05, and a passing absolute-quality
screen. Different protocols cannot be pooled, and paired KID evidence cannot
override failed absolute quality.

Runtime ranking remains observational-only for this active run. The immutable
GPU-observation history contains a long gap, and CoFiTok's recovery adjustment
is a physical elapsed-time lower bound. Future safeguard revision
`0d1fa8db971c9b33b16f943f511efaf5e51b0ace` additionally requires exact elapsed
evidence for both methods before permitting direct wall-clock, throughput, or
cost-efficiency ranking; it was deliberately not deployed into the active
waiter chain.

## Current scientific conclusion

The allowed conclusion is still:

- ordered restricted factorization and prefix-control advantages are
  established;
- two disjoint 10K streams provide repeated directional FID evidence over the
  matched dense baseline;
- terminal matched large-scale generation-quality superiority is not yet
  established;
- broad generation, SOTA, release, runtime-efficiency, or later-training claims
  remain disallowed.

Machine-readable snapshot:

```text
artifacts/reports/generation/
terminal_evidence_chain_live_continuation_2026-08-18/continuation_receipt.json
bytes:  3,540
SHA256: 8a3409ff81882edade30acb7f32128209bdc64cfe8e47f935f88a337da7f0ad8
```
