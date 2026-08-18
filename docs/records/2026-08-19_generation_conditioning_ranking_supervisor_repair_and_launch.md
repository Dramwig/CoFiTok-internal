# Conditioning-ranking supervisor repair and standing-authorization launch

Date: 2026-08-19

## Scope

This record covers the CPU-only supervisor that waits for the full-data quality
bridge to select the exact
`run_class_conditioning_fidelity_diagnostic` route.  The supervisor does not
modify the active bridge, signal any process, authorize promotion or release,
or authorize full 300K training.  It may launch only the already prepared
matched four-arm 1K conditioning-ranking probe after the bound terminal sources
exist and five consecutive GPU-idle polls pass.

Standing authorization source:

```text
/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json
SHA256: 5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
```

## Initial launch failure

The first launch used implementation revision
`7484d981ad63c227a0103d7748b61afb76a66d9a`.  It wrote only its PID receipt and
then exited before writing a status report:

```text
TypeError: _write_status() missing 1 required positional argument: 'path'
```

All eight `_write_status(...)` call sites omitted the required output path.  No
execution authorization was created, the probe output root remained absent,
and no GPU work was launched.  The active dense trainer continued normally.

## Repair

The repair passes the resolved `status_output` path at every status-write call
and adds an AST regression test that rejects any future `_write_status` call
without a positional or named path argument.

```text
branch: scale/generation-label-ranking-standing-authorization-v1
revision: fc47dbfda9758bf0aa440f1c23dd058e446c751d
tree: f6243c8750ddf6835c2cca76d64f3b32a5b7b642
subject: Fix ranking probe supervisor status writes
```

Incremental bundle:

```text
D:/cofitok-bundles/conditioning-ranking-supervisor-fix-fc47dbf.bundle
bytes: 1,814
SHA256: 5e89c6d0d7e0eb358a12793a269f9de48b6cd3dee7d455cd5014c7c07bd6824e
prerequisite: 7484d981ad63c227a0103d7748b61afb76a66d9a
advertised revision: fc47dbfda9758bf0aa440f1c23dd058e446c751d
```

The bundle passed local and remote `git bundle verify`.  It was applied only to
the new isolated checkout:

```text
/tmp/cofitok-label-ranking-standing-auth-fc47dbf
```

The formal checkout was not moved.  At launch audit time it remained
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066` with no
tracked or staged changes.  The newly cloned server currently contains 89
pre-existing untracked paths in that formal checkout; this repair did not
create or modify them.

## Validation

Local validation in the D-drive project worktree passed `git diff --check`,
Python compilation, and a direct AST check of all eight status calls.  The
fresh local D-drive environment did not contain Torch, so import-based pytest
collection was not treated as evidence.

Authoritative Linux validation in the exact clean isolated checkout completed:

```text
targeted pytest: 10 passed
Python compileall: pass
probe runbook bash -n: pass
postevaluation runbook bash -n: pass
post-test Git porcelain: empty
```

A foreground fail-closed smoke used missing follow-up sources and a 0.12-second
timeout.  It exited with code 4 and successfully wrote the expected
`timeout_before_probe_launch` status, directly exercising the repaired write
path without querying or using the GPU.

## Immutable preparation

The preparation was regenerated twice from the repaired Linux checkout and was
byte-identical:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_standing_auth_v1/fc47dbfda9758bf0aa440f1c23dd058e446c751d/preparation.json
bytes: 8,937
SHA256: c7df83f31691b337e37807e5c6c6c029b69a434a3d5af9788206cb150036305d
status: pass
valid: true
gpu_execution_authorized: false
```

It binds the unchanged output root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_probe1k_v1
```

## Live supervisor

The repaired supervisor was launched once as PID `205564` from the exact clean
isolated checkout.  Its first authoritative status was:

```text
status: waiting
detail: waiting_for_quality_bridge_followup_decision
child_pid: null
idle_gpu_polls: 0
```

Bound live artifacts:

```text
supervisor.pid.json
bytes: 322
SHA256: 73ec1d3137a9134916b3fecfe6224c0b31c26ef3eb560547746ed535a78ad57f

supervisor_status.json
bytes: 1,583
initial SHA256: e624ade1d7d9bdf01f3342b8829c9aa2b833703d163b384789c8926716dc9ac9
```

The status file is intentionally mutable while the supervisor waits, so its
hash above identifies only the initial observed snapshot.

At the same audit time, the quality bridge remained healthy:

```text
stage: dense_identity_training
dense step: 9,400 / 50,000 current segment
pair-monitor issues: []
GPU: 100% utilization, 77,983 MiB used
free storage: 320,198,512,640 bytes
probe output root: absent
execution authorization: absent
```

The supervisor will exit without GPU work if the terminal decision selects any
other route.  If the class-ranking route is selected, it still requires the
terminal-system guard, requested-class visual evidence, exact source hashes,
the repaired revision and output binding, and five consecutive idle-GPU polls
before starting the four matched 1K arms.
