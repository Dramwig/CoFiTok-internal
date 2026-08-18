# Generation quality-bridge exact controller identity guard

Date: 2026-08-18 (Asia/Shanghai)

## Purpose

The active full-data matched 100K quality bridge uses a legacy pair-monitor
runbook pattern:

```text
[g]eneration_stability_full_data_quality_bridge_100k_execute.sh
```

The pattern matched both the real controller and the runtime-fairness waiter,
because the latter carries the runbook path in its `--active-runbook` argument.
The same filename-substring classification is present in the already deployed
bounded recovery supervisor. If the real controller disappeared before the
terminal result, the legacy monitor and supervisor could therefore retain a
false-positive controller candidate.

The deployed historical supervisor source is byte-bound by its immutable
deployment receipt. It was intentionally left unchanged. This record adds an
independent CPU-only sidecar that binds the real controller by exact Linux
process identity and cannot restart or signal anything.

## Bound controller

```text
pid:             618821
start_ticks:     1660953145
executable:      /usr/bin/bash
cwd:             /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
argv:            bash artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
cmdline SHA256:  b774556b5b21e8005598614df1e54c9f55b8fa52b5a7d1c878a2e39d8a2dbbb6
```

At deployment, the legacy pair monitor reported runbook candidates `618821`
and `832803`. The guard records `618821` as the bound PID and explicitly records
`832803` as a non-bound substring-contamination candidate.

The guard invocation does not contain the runbook filename. It receives only
the expected raw-cmdline SHA256 plus PID, start ticks, executable, and cwd, so
the guard cannot become another match for the legacy runbook pattern.

## Implementation identity

```text
branch:   analysis/generation-quality-bridge-controller-identity-guard-v1
revision: 57b2897a4cfbcde09bb24f3082d7fe3c67b83960
tree:     bcc79d4a6c2cba1c8481a0f49956e35739922e29
subject:  Guard exact quality bridge controller identity
```

Primary files:

```text
scripts/run_generation_quality_bridge_controller_identity_guard.py
tests/test_generation_quality_bridge_controller_identity_guard.py
```

The guard checks every poll:

- exact PID and `/proc/<pid>/stat` start ticks;
- exact `/proc/<pid>/exe` and `/proc/<pid>/cwd`;
- SHA256 of the raw NUL-separated `/proc/<pid>/cmdline`;
- the expected two-argument bash command shape;
- immutable control-checkout Git identity and binding-manifest identity;
- exact training revision/branch in the execution status and pair monitor;
- whether the execution reached its terminal `completed` status and produced
  the quality-bridge result.

One missing or changed observation enters a confirmation state. Two consecutive
identity-loss polls before terminal completion produce a terminal failed guard.
The guard never sends a signal and never launches or restarts a controller,
trainer, evaluator, or waiter.

## Validation

Local Windows validation from the clean isolated worktree:

```text
focused and related tests: 36 passed
repository collection: pass
Python compile: pass
```

Linux isolated-checkout validation with `CUDA_VISIBLE_DEVICES=-1`:

```text
focused and related tests: 36 passed
Python compile: pass
tracked status: clean
```

Deployment bundle:

```text
build path:   C:/qbfd4/controller_identity_guard_57b2897.bundle (transferred; not tracked)
bytes:        40,229,554
SHA256:       bc68a174d723d8cc58c039ea2f2f893e357753c532fc8ac9dc22c5ff0c7a2882
prerequisite: 1ebcc15210e63a776a2ba448481cbd8bb94a4066
prerequisite: 58d83bfce2770eab2565b8c89a5f9a06201a0c86
advertised:   57b2897a4cfbcde09bb24f3082d7fe3c67b83960
```

Remote isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-controller-identity-guard-57b2897/CoFiTok-internal
```

The formal checkout at
`/root/autodl-tmp/CoFiTok/CoFiTok-internal` was not moved or modified.

## Active sidecar

Authoritative output root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
reports/controller_identity_guard_v1
```

Artifacts created by the exact live rehearsal and reused immutably by the
persistent guard:

```text
controller_binding.json
  bytes:  2,506
  SHA256: 147f5bae9d4a9681709d241afa2f21d8bb0696423ed0154461b1398381c335c4

deployment_receipt.json
  bytes:  2,645
  SHA256: d2e9416e6e32c0267e2399cd0661dc002bcfabad05ab816467c0e0813489e3e8
```

Persistent guard process:

```text
guard PID: 921082
status:    observing
detail:    exact_bound_controller_identity_is_active
loss polls: 0
mismatches: []
```

The status source is:

```text
.../reports/controller_identity_guard_v1/guard_status.json
```

At deployment the CoFiTok run remained healthy at step `39,550`, and neither
the trainer nor any pre-existing monitor/waiter was restarted, signaled, or
replaced.

## Claim and authorization boundary

This sidecar is operational evidence only. Its scope permanently states:

```text
cpu_only: true
gpu_execution_allowed: false
checkpoint_payload_loading_allowed: false
process_signals_allowed: false
controller_restart_allowed: false
training_process_signals_allowed: false
unrelated_process_signals_allowed: false
promotion_authorization_allowed: false
release_authorization_allowed: false
full_training_launch_allowed: false
full_300k_launch_allowed: false
diagnostic_non_authorizing: true
```

It does not replace the pair monitor, training reports, terminal quality
result, matched uncertainty report, promotion gate, or completion audit. Its
only claim is that the exact bound controller remains present, or that its
identity was lost before terminal completion.
