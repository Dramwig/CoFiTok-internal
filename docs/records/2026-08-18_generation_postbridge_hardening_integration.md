# Generation post-bridge hardening integration

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The post-bridge hardening branches are integrated and fully validated in the
isolated `integration/generation-postbridge-hardening-v1` worktree. The result
combines EMA-teacher schedule-transition evidence, exact 50K milestone ordering,
terminal evidence-chain guards, capacity continuity, exact controller/runbook
identity, and runtime-claim fail-closed behavior.

No active trainer, controller, monitor, waiter, or remote checkout was modified,
signaled, restarted, or replaced. No GPU sampling, evaluation, training,
deployment, release, or full-300K launch was started by this integration.

## Integrated sources

- `analysis/generation-quality-bridge-ema-transition-v1` at
  `b5c610674120e7334e16c51d18ea9884ccf42631`;
- `analysis/generation-quality-bridge-50k-transition-readiness-v1` at
  `b300e1b9d28bb15f594c70b7feab5e1521361113`;
- `analysis/generation-terminal-evidence-chain-audit-v1` at
  `36b49a6fbe854e84b8c92a796888150d7dcde5a1`.

The validated integration parent is
`b51a8a4f189d3419bd66a8ed11b2c1dd994e7415`, tree
`fde0cc5af1aa65e63c88d5c958aa8488dc136a23`.

## Integration defects found and corrected

The combined suite exposed three integration-only defects without weakening
production validation:

1. Capacity claim-addendum fixtures still described pre-schema-v9 training
   sources. They now provide exact training-cost fields and empty, verified
   resume-compute bindings instead of bypassing the new fail-closed contract.
2. Several deployment-evidence tests hashed Windows CRLF checkout bytes even
   though receipts bind immutable Linux/Git bytes. They now reopen the exact
   implementation-commit blob recorded by each receipt.
3. The requested-class visual-audit waiter imported POSIX `fcntl` at module
   import time. Linux locking is unchanged; non-POSIX execution is importable
   for tests and fails closed before acquiring a waiter lock.

The AAAI structure test also now resolves the shared project root through the
Git common directory, so it works in isolated linked worktrees without copying
or rewriting the paper tree.

## Validation

All validation was local and CPU-only with `CUDA_VISIBLE_DEVICES=-1`.

```text
targeted quality/capacity/terminal/runtime group: 403 passed
full pytest suite:                              1618 passed, 9 skipped
full pytest elapsed:                            669.97 seconds
runbook bash -n:                                140/140 passed
Python compileall (scripts/src/tests):           passed
git diff --check:                               passed
```

## Live training continuity

At `2026-08-18T13:31:41+08:00`, the authoritative pair monitor reported:

```text
status/stage:        running / cofitok_training
issues:              []
CoFiTok:             45,400 / 100,000
samples seen:        2,905,600
dense:               0 / 100,000
latest checkpoint:   step 45,000
checkpoint SHA256:   703a89752ad8f6937f78fff2b446062c4b3cbe801c871fd2ed9dd0fae805ef2b
unrelated GPU work:  none
free storage:        321,601,044,480 bytes
```

The only GPU compute process remained the bound CoFiTok trainer, PID `619775`.

## Scientific boundary

This integration strengthens reproducibility, transition ordering, provenance,
and terminal claim enforcement. It does not itself add scientific quality
evidence. Ordered restricted factorization and prefix-control advantages remain
established, and the two disjoint 10K streams remain repeated directional FID
evidence. Terminal matched large-scale generation superiority remains pending
until the active full-data CoFiTok/dense 100K pair and the bound terminal quality
and paired-uncertainty reports complete.

Machine-readable receipt:

```text
artifacts/reports/generation/postbridge_hardening_integration_2026-08-18/
integration_receipt.json
bytes:  3,954
SHA256: 711d2776f0b1f65810940eb03840c466f9498c301b4ada0a17d36a8683248363
```
