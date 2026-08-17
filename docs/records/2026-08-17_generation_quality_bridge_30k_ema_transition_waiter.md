# 2026-08-17 quality-bridge 30K EMA-teacher transition waiter

## Purpose

The active full-data matched 100K quality bridge enables EMA-teacher
consistency at optimizer step `30,000` with a `10,000`-step linear warmup and
weight `0.25`. A checkpoint audit alone can verify payload integrity at 30K but
cannot prove that the first active training rows actually follow the configured
schedule.

`scripts/wait_generation_consistency_schedule_transition.py` adds a reusable,
CPU-only, read-only transition waiter. It never loads a checkpoint, uses no GPU,
and cannot signal either training or unrelated processes.

## Exact 30K contract

The planned CoFiTok transition audit is source-bound to:

```text
training revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
training tree: 6cef27723196fd363379bca2e7b85b1678ebd777
training branch: scale/generation-stability-quality-bridge-100k
effective batch: 64
target training steps: 100,000
EMA-teacher start: 30,000
EMA-teacher warmup: 10,000
EMA-teacher weight: 0.25
metrics log interval: 50
first active row: 30,050, expected scale 0.005
audit target row: 30,250, expected scale 0.025
minimum active rows: 5
```

The waiter verifies:

1. exact training checkout revision, branch, tree, and tracked-clean state;
2. checked-in config and resolved `run_manifest.json` schedule equality;
3. strictly increasing canonical metrics and `samples_seen == step * 64`;
4. exact step-30K disabled boundary with zero scale and zero loss;
5. five exact active rows from 30,050 through 30,250;
6. exact linear scales, strictly increasing warmup, positive finite teacher loss,
   and finite total/epsilon/gradient values;
7. content identities for the waiter, config, run manifest, metrics snapshot, and
   final report.

The report is diagnostic and non-authorizing. It cannot change the quality
bridge, launch another training stage, promote a model, or support a broad
generation-quality claim.

## Local validation

Development uses an isolated clean worktree:

```text
C:/qbschedule
branch: analysis/generation-quality-bridge-ema-transition-v1
base: cf0e5faa94bf4ab38d947b921935b3b765b5537a
```

Current targeted validation:

```text
10 tests passed
68 transition + training-progress + loss + trajectory tests passed
ruff format/check passed
Python compile passed
git diff --check passed
```

The tests cover the exact disabled-to-active boundary, scale drift, premature
loss, zero active loss, missing target row, samples-seen drift, canonical tail
readiness, resolved run-manifest binding, checked-in config binding, and the
non-authorizing waiting status.

## Execution boundary

The script must be committed, bundled, Linux-rehearsed with CUDA hidden, and
launched from a separate clean checkout before the 30,250 row is reached. Its
output belongs under the active quality-bridge `reports/` tree. It must not
modify the active training checkout or compete for GPU resources.
