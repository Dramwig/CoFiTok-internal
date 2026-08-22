# 2026-08-22 quality-bridge terminal completion audit

## Purpose

Add one permanently non-authorizing, CPU-only terminal completion audit for the
full-data matched 100K quality bridge. The audit runs only after all upstream
terminal evidence is ready and does not alter the independent terminal system
guard's `pass` or `hold` decision.

The audit closes the remaining physical/source-binding gap by:

- exactly rebuilding `quality_bridge_result.json` and the strong-baseline
  comparison;
- exactly rebuilding the comparison JSON, Markdown, and CSV presentation;
- physically replaying CoFiTok and dense 90K/95K/100K checkpoint payload,
  integrity-sidecar, historical `latest.json`, and metrics-prefix evidence;
- requiring the three independent dense checkpoint replay receipts and their
  source-bound waiter result;
- requiring canonical final `latest.json` and `train_metrics.jsonl` state to
  stop exactly at 100K / 6,400,000 samples seen per method;
- physically revalidating the terminal 10K PNG sets, sampling
  manifest/progress/report, real set, checkpoint payload/sidecar, and class
  fidelity through an exact quality-result replay;
- rehashing the checkpoint chain and exact terminal sources again after the
  long replay, and failing closed on any source or Git drift.

An operational audit `status=pass` preserves a separate
`terminal_status=pass|hold`. In particular, a terminal `hold` remains
`generation_advantage_proven=false`.

## Immutable code identity

- branch:
  `analysis/generation-quality-bridge-terminal-completion-audit-v1-20260822`
- revision:
  `c3bfc3dcb4ae732d555191d9d813248b6c0c4941`
- tree:
  `7a33f1642223b0508e22f1432c1b2d177f99d869`
- waiter source SHA256:
  `278ea157ba36eb7006a0fbbafb6d66f3b36e692d74227424d124235bdb4b67ab`
- builder source SHA256:
  `f2fa5f3de5cbd49a47280eaa03cfb0363363793bd473b3ec33de530eb9762a4d`
- incremental bundle:
  `D:/cofitok-bundles/quality-bridge-terminal-completion-audit-c3bfc3d.bundle`
- bundle bytes: `50,266`
- bundle SHA256:
  `d8112759112872cedbbf93f58c2f3285e19ed3ccdc39a7a8d4d50e5f757ddde0`
- bundle prerequisite:
  `f7cfa34719797bca13cae95217ffd50bf59e2f09`

## Validation

Windows:

- Ruff check: pass.
- Ruff format check: pass.
- Python compile: pass.
- new terminal-completion tests: `20/20` pass.
- associated quality/comparison/checkpoint suite: `92/92` pass.
- repository suite excluding the four paper-layout tests whose paper tree is
  intentionally outside the isolated `D:` worktree: `1,146 passed, 4 skipped`.
  The excluded four failures were only missing `D:/paper/...` fixtures, not
  code or assertion failures in this change.

Linux isolated rehearsal on `pro6000`:

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-terminal-completion-audit-c3bfc3d/CoFiTok-internal`
- exact revision/tree/branch verified;
- CUDA hidden, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `nice=10`, and
  `ionice=idle` for test execution;
- Python compile: pass;
- associated suite: `92/92` pass;
- all tracked `artifacts/runbooks/*.sh`: `bash -n` pass;
- tracked status remained clean.

## Deployment

The unique detached waiter was deployed under the same immutable checkout and
is currently waiting for upstream terminal sources.

- PID at deployment: `707023`
- PPID: `1`
- start ticks: `1705025753`
- cwd:
  `/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-terminal-completion-audit-c3bfc3d/CoFiTok-internal`
- `CUDA_VISIBLE_DEVICES=`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `PYTHONPATH=<checkout>/src:<checkout>`
- nice: `10`
- ionice: `idle`
- status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/terminal_completion_audit_v1/waiter_status.json`
- deployment receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/terminal_completion_audit_v1/deployment_receipt.json`
- deployment receipt SHA256:
  `a175c2f3de7acca1f1b7ddcd66fdaae7ed7689a77b1c1ee7b2a25320183a5994`
- initial status SHA256:
  `cc3b4ca185ce15eb3eb7126e870a60bb160908366f57b7a68221dd7db575d7f9`

The initial state was `waiting / waiting_for_exact_terminal_completion_sources`.
At deployment, the only GPU compute owner remained the existing dense trainer
PID `219593`; the new waiter did not appear in the GPU process list.

Three launch-preflight failures occurred before the waiter could create a status
or deployment receipt:

1. missing checkout `PYTHONPATH` caused an immediate `ModuleNotFoundError`;
2. the first detach wrapper reached the Python detached check before its SSH
   parent exited;
3. the next wrapper's shell-escaped parent check stayed idle without executing
   Python and was terminated only after exact PID/start-tick/cwd/cmdline/GPU
   verification.

All three failure logs and PID files are retained with `failed_*` suffixes in
the terminal-completion report directory. No checkpoint, sample set, training
process, or upstream artifact was modified.

## Restart rule

Rediscover the live PID every time; `707023` is only the deployment identity.
Restart only if all of the following are true:

- `terminal_completion_audit.json` does not exist;
- canonical waiter status is not `pass`;
- the exact live waiter process is absent;
- no duplicate command line exists;
- the waiter lock is free;
- the immutable audit/training/physical-auditor/replay checkout identities and
  both source SHA256 values still match;
- the existing deployment receipt normalizes only PID/runtime and otherwise
  matches the exact requested launch contract.

The waiter is permanently non-authorizing. It may not launch training,
sampling, 300K scaling, promotion, export, release, or send process signals.
