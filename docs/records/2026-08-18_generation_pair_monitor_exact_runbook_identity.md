# Exact runbook identity for future generation pair monitors

Date: 2026-08-18 (Asia/Shanghai)

## Purpose

The active full-data matched 100K quality bridge exposed an ambiguity in the
legacy pair monitor's runbook-process classification. Its filename-substring
pattern reports both:

```text
618821  bash artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
832803  python ... --active-runbook .../generation_stability_full_data_quality_bridge_100k_execute.sh
```

PID `618821` is the real runbook controller. PID `832803` is the independent
runtime-compute-fairness waiter; its argv merely contains the active runbook
path. A pattern-only monitor can therefore retain a false controller candidate
if the real controller disappears before the matched pair reaches a terminal
result.

This change hardens the reusable pair monitor for future runs. It does not
replace or modify the monitor, controller, recovery supervisor, trainer, or
waiters already running for the current quality bridge. The active run remains
protected by the separately deployed exact controller identity guard recorded
in `2026-08-18_generation_quality_bridge_controller_identity_guard.md`.

## Implementation identity

```text
branch:   fix/generation-monitor-exact-runbook-identity-v1
revision: ab2999d6e2fd8a6e4bacca1f8962469f5c778a06
tree:     82e011bf06bc85e53c84983a0bff6e149a54a556
subject:  Bind pair monitor to exact runbook identity
```

Changed files:

```text
scripts/monitor_generation_pair.py
tests/test_monitor_generation_pair_exact_runbook_identity.py
```

The monitor now accepts an optional complete controller binding:

```text
--runbook-process-pid
--runbook-process-start-ticks
--runbook-process-executable
--runbook-process-cwd
--runbook-process-cmdline-sha256
```

When any exact field is requested, all five fields are required. The monitor
reads the bound PID directly from `/proc`, verifies PID-reuse-resistant start
ticks, executable, cwd, and the SHA256 of the raw NUL-separated cmdline, and
writes the result under `runbook_identity` in every monitor report.

In exact mode the legacy filename pattern is diagnostic only. Its candidates
are retained as evidence, including non-bound PIDs and explicit substring
contamination, but they cannot substitute for the bound controller. Loss or
mutation of the exact identity before pair completion fails the monitor closed.
After a genuinely completed pair has already reached `pass`, normal controller
exit does not rewrite that completed result as a failure. Supplying no exact
fields preserves the historical pattern-only CLI behavior.

## Bound live identity used for rehearsal

```text
pid:             618821
start_ticks:     1660953145
executable:      /usr/bin/bash
cwd:             /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
argv:            bash artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
cmdline SHA256:  b774556b5b21e8005598614df1e54c9f55b8fa52b5a7d1c878a2e39d8a2dbbb6
```

The pattern candidates during both rehearsals were PIDs `618821` and `832803`.
The exact binding classified only `618821` as the controller and recorded
`832803` as non-bound contamination.

## Validation

Local Windows validation used the project-specific `C:/qbfd5/.venv`, explicit
project `PYTHONPATH`, and `CUDA_VISIBLE_DEVICES=-1`:

```text
focused and related tests: 33 passed, 2 skipped
skips: CUDA-required generation-system regressions
Python compile: pass
git diff --check: pass
```

Linux validation used the exact isolated checkout with
`CUDA_VISIBLE_DEVICES=-1`, one OMP/MKL thread, and `PYTHONPATH=.:src`:

```text
focused and related tests: 33 passed, 2 skipped
skips: CUDA-required generation-system regressions
Python compile: pass
tracked status: clean
```

The selected suite covered the new exact-identity tests plus generation live
audit, runbook entrypoint, and generation-system tests. The eight dedicated
tests verify contamination isolation, missing-controller failure, PID reuse via
start-tick mismatch, pre-completion fail-closed behavior, post-completion normal
exit, complete binding validation, normalization, and legacy compatibility.

## Bundle and isolated checkout

Deployment/rehearsal bundle:

```text
bytes:        40,236,601
SHA256:       aba20a9f42e5d557f8d78bb226831dde8e74a229f92b8b2f966962e5404e233b
advertised:   ab2999d6e2fd8a6e4bacca1f8962469f5c778a06
prerequisite: 1ebcc15210e63a776a2ba448481cbd8bb94a4066
prerequisite: 58d83bfce2770eab2565b8c89a5f9a06201a0c86
remote path:  /tmp/exact_runbook_monitor_ab2999d.bundle
local archive: C:/Users/17194/.codex/bundles/CoFiTok/exact_runbook_monitor_ab2999d.bundle
```

`git bundle verify` passed on the server and the bundle advertises only the
target branch head.

Remote isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-monitor-exact-runbook-ab2999d/CoFiTok-internal
```

The checkout remains clean and fixed at `ab2999d`. The formal checkout at
`/root/autodl-tmp/CoFiTok/CoFiTok-internal` was not fetched, merged, moved, or
written by this work.

## Positive live rehearsal

Authoritative report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
reports/exact_runbook_monitor_rehearsal_v1/report.json
```

Identity:

```text
bytes:  19,795
SHA256: 35b3ccee507d0a67354092dba5169a51e1d287cd39ea3807410c7e2f176f6545
status: running
stage:  cofitok_training
issues: []
```

The report observed all five exact fields without mismatch. Its diagnostic
pattern evidence was:

```text
pattern PIDs:                           [618821, 832803]
bound PID reported by pattern:          true
non-bound PIDs:                         [832803]
substring candidate contamination:      true
exact runbook identity status:           active
```

This proves that the exact controller remains active while the waiter remains
visible only as diagnostic contamination.

## Negative live rehearsal

Authoritative report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
reports/exact_runbook_monitor_negative_rehearsal_v1/report.json
```

Identity:

```text
bytes:  19,824
SHA256: 9d61ba676b359661fbd9fdd3333548ec9bfbbe4fc7ab00805bc55931990eb960
status: failed
expected start_ticks: 1660953146
observed start_ticks: 1660953145
issue: exact runbook identity is unavailable before pair completion: start_ticks
```

Only the expected start tick was deliberately changed by `+1`. The same two
pattern candidates remained visible, yet PID `832803` could not replace the
real controller and the incomplete pair failed closed. This directly exercises
the ambiguity that motivated the change.

## Deployment and claim boundary

This is future-run monitor hardening plus a CPU-only live rehearsal. It is not
an active-monitor replacement and it makes no training or scientific claim.
During this work:

- no GPU process was launched;
- no checkpoint payload was loaded or rewritten;
- no controller, trainer, monitor, supervisor, or waiter was signaled,
  restarted, or replaced;
- no active output lock was taken over;
- no formal checkout revision was moved;
- no sampling, evaluation, promotion, release, 300K training, or follow-up
  stage was authorized.

The current 100K pair continues to use its immutable historical monitor. Its
separate exact controller guard remains the authority for controller identity
during this run. The `ab2999d` monitor interface should be used when constructing
future matched-pair runbooks so that exact controller identity is native rather
than supplied by a sidecar.
