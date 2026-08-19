# Capacity-probe supervisor child-heartbeat audit

Date: 2026-08-19

## Outcome

The deployed capacity control chain has two related prospective liveness gaps. The
10K capacity-probe execution supervisor at `3a7dc9d` publishes a running
status once and then calls blocking `child.wait()`. Its status can therefore
become stale while the bounded training and evaluation runbook remains alive.
Its wrapper also opens and locks fd 8 before `exec Python`, but deployed PID
`11853` retains only fd 0/1/2. A nonblocking probe acquired the same lock while
PID `11853` was alive, proving that the lock does not cover the supervisor
lifetime.

The deployed 50K scaling, 100K completion, full-300K readiness, full-300K
training, formal post-evaluation, and finalization supervisors do not have this
gap. Their exact deployed sources poll the child and publish status at every
configured interval, capped at 60 seconds for the full stages.

The machine-readable audit is:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/capacity_probe_supervisor_child_heartbeat_audit.json
```

## Fix

The fix uses `cofitok.process_monitoring.wait_for_child_with_heartbeat` for the
10K supervisor. Each heartbeat reports the child PID, current execution stage,
related capacity processes, and GPU rows. A heartbeat read or write exception
is reported once per consecutive failure sequence but cannot detach the
already-launched child; the helper continues waiting and returns the child's
real exit code.

The wrapper now uses `flock --no-fork --conflict-exit-code 75` to execute the
Python supervisor. This keeps the same PID while retaining the lock for the
complete process lifetime. A Linux test starts the wrapper with a bounded fake
child and verifies that a competing lock request exits exactly 75.

Two source-bound candidates exist:

| role | revision | tree | parent |
|---|---|---|---|
| aggregate integration candidate | `88d2c445a18610893bfbf6f0578f89b06001ef06` | `5ccbd82c3ea7b8acd7c568d8c6258aa95d29efc0` | `85c5dbdba40bb31ec0af1c311f1780fa34714d57` |
| exact deployed-parent candidate | `23a1c054f40940a9d14778671e2dbae4820ad3a6` | `a7b71675a14d68aa033ea379251e22d3ae3c7c9e` | `3a7dc9db6950055829db00c8ffdd6e906501fbb4` |

The exact-parent bundle is 6,951 bytes with SHA256
`b902679ea59918c2ba39ae5095f0bb2ad40cf1cf881af7b92acc12bb739b6eb1`.
It advertises only the exact fix branch and requires deployed revision
`3a7dc9db6950055829db00c8ffdd6e906501fbb4`.

An earlier integration-parent bundle was rejected before fetch because the
formal remote repository did not contain its `85c5dbd` prerequisite. No remote
HEAD moved as part of that negative prerequisite check.

## Verification

- Local aggregate focused tests: `14 passed`.
- Local aggregate capacity/control-chain tests: `248 passed, 1 Linux-only skipped`.
- Local aggregate full suite before adding the Linux-only lock test:
  `1,639 passed, 9 skipped`.
- Local exact-parent capacity tests: `58 passed, 1 Linux-only skipped`.
- Linux exact-parent capacity tests: `59 passed` with CUDA hidden, one CPU
  thread, `nice 10`, and idle I/O priority.
- The Linux lock-lifetime test observed competing exit code `75`.
- Exact supervisor runbook: `bash -n` passed.
- Incremental bundle prerequisite verification: passed in an isolated clone.
- The isolated clone and deployed checkout were tracked-clean after rehearsal.

The first dirty-worktree full-suite run correctly caused three Git-identity
tests to fail, and its subprocess CLI inherited the main editable environment
instead of the isolated `src`. After committing the candidate and setting the
isolated `PYTHONPATH`, all four nodes passed and the complete suite passed.

## Live boundary

At `2026-08-19T09:54:13.521064+00:00`, deployed supervisor PID `11853` was
still `waiting_for_source_bound_capacity_preparation`, with `child_pid=null`
and attempt zero. It remained on exact revision `3a7dc9d`, and its checkout was
tracked-clean. The active quality bridge was not modified; dense identity was
at step 34,150 and remained the only GPU workload.

This audit deliberately did not signal, restart, or replace the deployed
supervisor. A loaded Python process cannot consume the patch merely because a
new checkout exists. Because the deployed lock is not retained, starting a
second supervisor would create a race. A later replacement must therefore be
explicitly source-bound, stop only the verified waiting supervisor, prove it
has no child, and acquire the corrected lifetime lock before it can proceed.

This audit does not authorize the 10K capacity probe, full-300K training,
promotion, or release.
