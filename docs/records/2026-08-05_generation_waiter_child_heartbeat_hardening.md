# Generation waiter child-heartbeat hardening

Date: 2026-08-05

## Scope

This change hardens the coordination status emitted while the frozen 50K
post-evaluation, frozen supplemental, and full-readiness waiters are waiting
for a child runbook. It does not launch, restart, signal, or authorize any
training or evaluation stage.

The non-authorizing full-data 100K quality-bridge execution target remains
exactly `cf0e5faa94bf4ab38d947b921935b3b765b5537a`; this hardening is a separate
future-facing commit and is not part of that immutable execution target.

## Observed failure mode

The frozen 50K post-evaluation waiter wrote one `running` status and then
blocked in `child.wait()` for the whole formal EMA post-evaluation. A
downstream supplemental waiter correctly enforced a finite coordination
silence window, but the upstream status did not refresh while its child was
healthy. The supplemental waiter therefore exited with
`stability 50K post-evaluation status is stale` before the upstream child
completed.

The authoritative remote evidence remained fail-closed:

- formal EMA post-evaluation later completed with `status=pass`,
  `detail=formal_ema_postevaluation_completed`, and `child_exit_code=0`;
- the supplemental waiter stayed terminal `failed` and
  `supplemental_non_authorizing=true`;
- no physical supplemental qualification or class-fidelity qualification was
  produced;
- readiness retained `full_training_launch_allowed=false`.

## Change

`cofitok.process_monitoring.wait_for_child_with_heartbeat` polls
`Popen.wait(timeout=poll_seconds)`. Each `TimeoutExpired` publishes a fresh
status through the waiter's existing atomic JSON writer. The helper is used by
the three waiters that can own long child stages:

- `run_generation_stability_50k_posteval_waiter.py`;
- `run_generation_stability_frozen_supplemental_waiter.py`;
- `run_generation_stability_full_readiness_waiter.py`.

Child exit codes and terminal status handling remain unchanged. The heartbeat
contains the same source-bound identity and non-authorizing fields as the
initial `running` record. No gate, launch receipt, promotion decision, or 300K
authorization behavior changes.

## Verification

- Changed Python entrypoints compiled successfully.
- 29 focused process-monitoring and waiter tests passed.
- The isolated short-path worktree collected 1,079 repository tests. It
  produced 1,069 passes, 6 skips, and 4 failures caused only by the test
  worktree being rooted at `C:\\cwh805`, while the four paper-layout tests
  intentionally locate `paper/` from the canonical project parent.
- Those same 4 paper-layout tests passed in the canonical project layout.
- `git diff --check` reported no patch errors.

No remote checkout, checkpoint, report, lock, GPU process, or other project's
state was modified while preparing this hardening.
