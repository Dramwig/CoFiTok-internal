# Statistical claim language: single-stream replication boundary

Date: 2026-08-23

## Problem

The full-data 100K terminal uncertainty stage re-analyzes the exact terminal
10K CoFiTok/dense sample sets with paired block KID, bootstrap intervals, and a
sign test. These are additional analyses of one bound matched terminal stream;
they are not a second independently generated 10K replication.

The existing language guard correctly limited claims to the exact bound sample
stream, but it did not expose a machine-verifiable prohibition against phrases
such as "two independent 10K streams replicated the result."

## Change

`build_generation_statistical_claim_language_guard.py` now:

- requires the uncertainty report to bind a verified execution-manifest stream,
  both sample-set digests, checkpoints, and the exact sample-index window;
- cross-checks the quality-bridge qualification stream ID against that bound
  uncertainty stream;
- emits `bound_terminal_stream_count=1`,
  `independent_replication_count=0`, and
  `independent_replication_supported=false`;
- sets independent-replication and multiple-independent-stream claim
  permissions to false; and
- states explicitly that paired KID is a re-analysis of the same terminal
  sample sets, not an independent generation replication.

The change is non-authorizing. It does not permit training, sampling, 300K
scaling, promotion, export, release, or process signaling.

## Verification

CUDA-hidden, low-priority tests were run on `pro6000` from an isolated `/tmp`
snapshot using the non-symlink Python 3.10 executable:

```text
67 passed
```

The test selection covered the statistical language guard, its waiter and
deployment receipt, quality-bridge statistical qualification and waiter,
terminal uncertainty waiter, and matched uncertainty report logic.

## Live-chain boundary

This branch is isolated evidence and has not replaced or signaled any live
waiter. The running 100K training/controller chain remains pinned to its
authorized checkout and revision. Any later canonical integration must preserve
the single GPU chain and bind this supplement without claiming an independent
replication or authorizing a new sample stream.
