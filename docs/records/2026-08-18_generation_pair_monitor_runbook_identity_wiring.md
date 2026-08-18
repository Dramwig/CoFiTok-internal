# Exact controller identity wiring for matched generation runbooks

Date: 2026-08-18 (Asia/Shanghai)

## Purpose

The reusable generation pair monitor already supports an exact Linux process
identity for its runbook controller. This follow-up wires that identity into
the matched 50K, full-data 100K quality-bridge, and two full 300K runbooks so
future runs do not fall back to filename-substring controller detection.

The active full-data matched 100K quality bridge is not changed by this work.
It remains on its immutable historical monitor and is protected by the
separately deployed exact controller identity guard. This change is future-run
hardening only.

## Implementation identity

```text
branch:   fix/generation-monitor-exact-runbook-identity-v1
revision: 261fbca92e6259b678b9f3fc498942695cefc763
tree:     7a83ee671b2f8ef4da6b150c2534f598ee9e7697
subject:  Wire exact controller identity into matched runbooks
```

The implementation adds a shared `/proc` identity module and a NUL-delimited
CLI-to-Bash bridge:

```text
src/cofitok/process_identity.py
scripts/print_generation_process_identity.py
artifacts/runbooks/lib/generation_exact_runbook_identity.sh
```

The helper captures the controller PID, start ticks, executable, cwd, and raw
NUL-separated cmdline SHA256 before a monitor is launched. All five fields are
passed to `monitor_generation_pair.py`. Reuse of an existing monitor is allowed
only when its report is bound to the same exact controller identity. Terminal
monitor pass checks also require an active, mismatch-free exact identity.

The following runbooks now use this contract:

```text
artifacts/runbooks/generation_stability_ema_teacher_matched_50k_after_gate.sh
artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
artifacts/runbooks/generation_stability_ema_teacher_full_matched_300k_after_gate.sh
artifacts/runbooks/generation_full_matched_300k_after_gate.sh
```

Pattern-only monitor mode remains backward compatible when no exact identity
arguments are supplied.

## Bundle and isolated checkout

Incremental bundle:

```text
local:        C:/Users/17194/.codex/bundles/CoFiTok/exact_runbook_monitor_wiring_261fbca.bundle
remote:       /tmp/exact_runbook_monitor_wiring_261fbca.bundle
bytes:        14,548
SHA256:       061b37602797620a4aa95c803914ecda7afe5cc6fab650661f361a86d2dc5fa1
prerequisite: ab2999d6e2fd8a6e4bacca1f8962469f5c778a06
advertised:   261fbca92e6259b678b9f3fc498942695cefc763
```

Both local and remote bundle verification passed. The remote rehearsal
checkout is:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-monitor-runbook-wiring-261fbca/CoFiTok-internal
```

It is tracked clean at `261fbca`. The earlier monitor-only rehearsal checkout
remains tracked clean and fixed at `ab2999d`:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-monitor-exact-runbook-ab2999d/CoFiTok-internal
```

The formal checkout remains at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`; it was not fetched, merged,
checked out, or otherwise modified.

## Validation

Local Windows validation used `C:/qbfd5/.venv`, project `PYTHONPATH`, and
`CUDA_VISIBLE_DEVICES=-1`:

```text
targeted tests: 62 passed, 2 skipped
Python compile: pass
Git Bash syntax: pass
git diff --check: pass
```

The targeted suite covers the quality-bridge runbook, runbook entrypoint and
syntax contracts, generation-system contracts, stability live audit, exact
monitor identity, the process-identity CLI, and exact-controller runbook
wiring.

Linux validation in the isolated checkout produced:

```text
targeted tests: 62 passed, 2 skipped
Python compile: pass
native bash -n: pass
tracked status: clean
```

The first full-suite invocation used an incomplete isolated layout and
reported:

```text
1140 passed, 2 skipped, 4 failed
```

All four failures were caused solely by the absent sibling `paper/` directory.
After reproducing the real parent layout with a temporary sibling link, the
full suite reported:

```text
1144 passed, 2 skipped
```

The temporary `paper` link was removed after validation.

## Real `/proc` rehearsal

A short-lived Bash controller in the exact remote checkout was captured as:

```text
pid:             938958
start_ticks:     1666834224
executable:      /usr/bin/bash
cwd:             /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-monitor-runbook-wiring-261fbca/CoFiTok-internal
cmdline SHA256:  db131d99db4790229bcf9d178183cfe8b88aa65d06d7e961eed4a5802db1fa6e
```

The positive report passed and had SHA256:

```text
47132195c1dbea45a1f10fa9ab154f2feee4487e002a583f8d337f645573da2d
```

A negative rehearsal changed only the expected start tick by `+1`; it was
rejected. This exercises PID-reuse resistance and proves that a pattern match
cannot replace the bound controller. Temporary rehearsal reports were removed
after their identities were recorded.

## Rehearsal receipt

Machine-readable receipt:

```text
artifacts/reports/generation/
quality_bridge_monitor_runbook_wiring_2026-08-18/rehearsal_receipt.json
bytes:  4,231
SHA256: 0f9539868bf60950201745af32fc15e2bcc1c60aa511bdc918899d15237b65ad
status: pass
```

The receipt binds the implementation revision/tree, incremental bundle,
remote checkout, local/Linux validation, live `/proc` positive and negative
cases, and the non-interference snapshot.

## Active-run non-interference snapshot

At receipt creation, the unrelated active quality bridge remained:

```text
CoFiTok: 41,050 / 100,000
dense:   0 / 100,000
stage:   cofitok_training
status:  running
issues:  []
trainer PID: 619775
trainer GPU memory: 85,286 MiB
```

Its exact controller guard was `observing` with
`identity_loss_polls=0`. The CoFiTok step-50,000 physical-integrity waiter was
still `waiting / checkpoint_missing`.

## Authorization and deployment boundary

This work is diagnostic, CPU-only, and non-authorizing. It did not:

- deploy into or replace the active pair monitor;
- signal, restart, pause, or replace the active controller, trainer, monitor,
  supervisor, guard, or waiter;
- load, hash, rewrite, or publish an active checkpoint payload;
- launch training, sampling, evaluation, promotion, release, or full 300K;
- move the formal checkout or either existing remote rehearsal checkout;
- grant a scientific quality claim or any later-stage authorization.

Future matched runbooks should use the native exact identity binding. The
current 100K quality bridge continues under its historical monitor plus the
separate exact controller guard until its immutable runbook reaches a terminal
result.
