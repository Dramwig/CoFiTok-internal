# Capacity-probe supervisor child-heartbeat audit

Date: 2026-08-19

## Outcome

The deployed capacity control chain has one prospective liveness gap. The
10K capacity-probe execution supervisor at `3a7dc9d` publishes a running
status once and then calls blocking `child.wait()`. Its status can therefore
become stale while the bounded training and evaluation runbook remains alive.

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

Two source-bound candidates exist:

| role | revision | tree | parent |
|---|---|---|---|
| aggregate integration candidate | `3d2bca465fe3f775b36fe48341a5fe47fa8d1abc` | `b8224074be4ccfefa6d9b05b5df3907b98f7c77d` | `85c5dbdba40bb31ec0af1c311f1780fa34714d57` |
| exact deployed-parent candidate | `6bbfa806353bb2095890b5b883ea380f08e43116` | `3033960bf34fc50a2502ea7bd91cf5710b119bcf` | `3a7dc9db6950055829db00c8ffdd6e906501fbb4` |

The exact-parent bundle is 5,551 bytes with SHA256
`f8e23f44851168ed56b6a5d3a17fdb76b0f371497623b752467f966c49f85241`.
It advertises only the exact fix branch and requires deployed revision
`3a7dc9db6950055829db00c8ffdd6e906501fbb4`.

An earlier integration-parent bundle was rejected before fetch because the
formal remote repository did not contain its `85c5dbd` prerequisite. No remote
HEAD moved as part of that negative prerequisite check.

## Verification

- Local aggregate focused tests: `14 passed`.
- Local aggregate capacity/control-chain tests: `248 passed`.
- Local aggregate full suite: `1,639 passed, 9 skipped`.
- Local exact-parent capacity tests: `58 passed`.
- Linux exact-parent capacity tests: `58 passed` with CUDA hidden, one CPU
  thread, `nice 10`, and idle I/O priority.
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
new checkout exists. A later replacement must therefore be explicitly
source-bound and prove that the old supervisor cannot race the replacement.

This audit does not authorize the 10K capacity probe, full-300K training,
promotion, or release.
