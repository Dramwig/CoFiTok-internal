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

## Source-decoupled deployment

The first exact-parent fix could not be deployed directly: using its checkout
for both control and execution would have changed the eventual 10K training
revision from the already-bound `3a7dc9d` to the supervisor fix revision. The
deployed replacement therefore separates the two identities:

| role | revision | tree | checkout |
|---|---|---|---|
| supervisor control | `acedc6d32b4f77875f83948d8cb0a2024bd69a84` | `7afcf46846f2d079a026c6aeccd2850a5ee7e2de` | `/root/autodl-tmp/CoFiTok/checkouts/capacity-supervisor-acedc6d` |
| capacity execution | `3a7dc9db6950055829db00c8ffdd6e906501fbb4` | `2b187e1bc6a36342ef2d803b26b4ae6f92b1a9f1` | `/root/autodl-tmp/CoFiTok/checkouts/capacity-execution-3a7dc9d` |

The child runbook, working directory, sanitized `PYTHONPATH`, and target
revision/tree/branch environment all remain execution-checkout bound. The
supervisor status exposes the control identity separately as `supervisor_git`
while retaining `expected.execution_git=3a7dc9d`.

The final incremental bundle is 10,029 bytes with SHA256
`438f13cfd1a234debceece57f950dca929047ed24601e344e9c70e24782bed44`.
It requires `3a7dc9d` and advertises only `acedc6d`. Linux verification with
CUDA hidden passed `64/64` selected capacity and runbook-contract tests;
`bash -n` passed. A real isolated process rehearsal refreshed its heartbeat
after 60 seconds, retained the lock on fd 3, and made a competing nonblocking
lock request return exactly `75` while keeping `child_pid=null`.

Immediately before replacement, PID `11853` was revalidated by PID, start
time, cwd, command-line SHA256, fresh status, absent preparation, null child,
and attempt zero. Its pre-replacement status was frozen with SHA256
`3c9e49c2679f06e59ec9a663baf7a37321228c81dcfa20132beaab77164210c8`.
Only that PID was terminated. Replacement PID `667226` started at
`2026-08-19T18:59:55+08:00`, remained waiting with no child, and published a
source-decoupled status snapshot with SHA256
`0037a4b064556a4356be47a76b13f9225035da5bbd5971646c66aabea09bac68`.

The recovery-aware lineage observer accepted PID `667226`, reported the stage
healthy and waiting, and retained zero issues. The sole GPU compute process
remained the active quality-bridge dense trainer PID `79894`; the replacement
started no GPU child. The immutable remote deployment receipt is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_supervisor_acedc6d_deployment_receipt.json
bytes: 5094
SHA256: a2ccc1448a508e9988dcba3ba7326778a9ad58e6fedd393d63e3369eeb2fe611
```

The local receipt copy is:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/capacity_supervisor_acedc6d_deployment_receipt.json
```

This replacement preserves the existing source-bound 10K capacity-probe
scope. It does not authorize 100K continuation, full-300K training, promotion,
release, or any unrelated process modification.

## Scientific state after deployment

At `2026-08-19T11:08:28+00:00`, the active quality bridge remained healthy at
`dense_identity_training` with no issues. CoFiTok was at its intentional 50K
stop, while dense identity's source log had reached step 36,000 and 2,304,000
images. Its fixed-validation epsilon MSE at that step was
`0.027402421459555626`, and the shared EMA-teacher consistency scale was `0.6`.
The source-bound 40K full-warmup waiter remained healthy and waiting for the
target step.

These are single-run progress values, not a matched quality result. The next
scientific evidence remains, in order: the exact dense 40K warmup report, the
healthy matched 50K completion, and the paired EMA sampling/uncertainty gate.
The supervisor deployment itself provides no generation-advantage evidence.
