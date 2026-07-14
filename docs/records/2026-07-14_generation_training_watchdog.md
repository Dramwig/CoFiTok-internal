# Generation training watchdog (2026-07-14)

## Problem

The read-only generation-pair monitor could identify a stalled/failed queue, but
the formal training shell still waited on `train_generation.py`. A failed
monitor or a stalled metrics stream therefore required an operator to discover
the report and terminate the training process manually.

## Implementation

`cofitok.training.watchdog.run_training_watchdog` now owns one training command
and consumes the pair monitor's atomic JSON plus PID file. It requires a fresh
report from the configured monitor name and fails closed on:

- exit `86`: monitor status is `failed` or `stalled`;
- exit `87`: no fresh monitor report before startup grace expires;
- exit `88`: a previously fresh report stops updating;
- exit `89`: the monitor process disappears beyond its grace period.

An ordinary child failure preserves the normalized child exit code. On POSIX,
the child starts a new session and watchdog shutdown signals its entire process
group, including DataLoader workers, with `SIGTERM` followed by `SIGKILL` after
the configured grace period. Every invocation atomically writes a bounded
status/history artifact carrying command, PIDs, monitor summary, timestamps,
reason, and both child/watchdog exit codes.

The compressed 10% matched runbook writes one
`training_watchdog.json` per method run. The segmented full 300K runbook writes
`training_watchdog_step_<target>.json` for each method and milestone so a later
resume cannot erase the reason an earlier segment stopped.

Windows process liveness uses `OpenProcess` plus `GetExitCodeProcess`. It must
not use `os.kill(pid, 0)`, because signal value zero is `CTRL_C_EVENT` on
Windows and would turn a read-only liveness probe into a signal sent to the
training console group.

## Scope

The active pinned legacy 10% pair remains on revision `781a014` and is not
modified. The watchdog becomes active only after the controlled upgrade bundle
is deployed and the true-compressed 10% pair starts from one clean target
revision.

## Verification

Local unit coverage exercises successful completion, child failure passthrough,
fresh monitor failure, stale-report rejection, startup timeout, report silence,
and monitor-process disappearance. The runbook entrypoint contract also checks
that the wrapper and nested training CLI options remain live. A Linux isolated
checkout rehearsal is required before the deployment waiter may target this
revision.
