# Full-data quality bridge pin-memory recovery v2

Date: 2026-08-17 (Asia/Shanghai)

## Scope

This change hardens only the already-authorized full-data matched 100K quality
bridge at training revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. It does not authorize full 300K,
release, promotion, or any change to the CoFiTok/dense training contract.

Authoritative output root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

## Incident

The first CoFiTok controller stopped after metric step 20,200. The latest
physical recovery checkpoint was step 20,000:

```text
checkpoint: checkpoint_step_00020000.pt
bytes: 1,010,933,866
sha256: 93174b4ac19f03c48edc3940eb909c51864ba2760bc5ad4e514bfa73e749df03
```

The failure signature was:

```text
FileNotFoundError in multiprocessing/resource_sharer.py
RuntimeError: Pin memory thread exited unexpectedly
```

The host had no OOM, CUDA Xid, shared-memory exhaustion, or unrelated GPU
process. Exact resume reconciled the post-checkpoint metric rows into:

```text
train_metrics_orphaned_at_resume_00020000_6b3c4bca8ffa.jsonl
```

The resumed canonical trajectory began at step 20,001 and passed the former
failure point. At the final check for this record it was at step 20,550 with
1,315,200 images seen.

## Recovery policy

Tracked implementation commit:

```text
c58a233148cbc1afa16c1144b852cea12307c0f5
branch: fix/quality-bridge-ipc-recovery-v2
parent training revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
```

The record commit is `564b751b31e0a75fb5a8756d12325cdcc515c531`.
The incremental deployment bundle requires `cf0e5fa`, advertises only
`564b751`, is 13,359 bytes, and has SHA256
`b886d92ad9d0d2943aeb7555e73a36e3b8c4b7be5103aad81bf0cfbcc5ffeffa`.

The source-bound deployment receipt is tracked by commit
`e663470f534f0b1fd2bab477009a5baaaed6f172` at:

```text
artifacts/reports/generation/stability_full_data_quality_bridge_ipc_recovery_v2/deployment_receipt.json
bytes: 3,714
sha256: c20cf6193db929adbecd2dbe2822deb907f6aedfb29095175317524b1a6cc075
```

The byte-identical immutable server copy is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/recovery_incident_2026-08-17_pin_memory/recovery_supervisor_v2_deployment_receipt.json
```

## Step-25K physical integrity waiter

The resumed run had not yet reached step 25,000, so a separate read-only,
source-bound waiter was deployed from commit
`172388fc4d873bb1001313f979442516ea7b5069`. It runs at nice level 10 and
idle I/O priority with `CUDA_VISIBLE_DEVICES=""`; it cannot signal the trainer,
authorize promotion, or use the GPU. The waiter performs a full physical
checkpoint SHA256 replay and verifies the sidecar, exact `latest.json` binding,
training revision/tree/branch, dataset/runtime identities, strictly increasing
metrics, and `samples_seen == step * 64`.

```text
source bytes: 14,973
source sha256: eafc4e1f7b33f8b890883ac4878944a1d57f41aca02dc0483b9e82d1259fca8b
bundle bytes: 10,770
bundle sha256: 1e7ad070b85d93faacaf789579b19b5ff45a52cafd53563170f6fd5e05bbea2b
clean checkout: /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-checkpoint-waiter-172388f
waiter PID: 650982
initial status: waiting / checkpoint_missing
```

The deployment receipt is:

```text
artifacts/reports/generation/stability_full_data_quality_bridge_checkpoint_25k_waiter/deployment_receipt.json
bytes: 3,781
sha256: 8d9f54cf1d5ceef8a506c3d1d10700beb10c25caa1c3c0968610715a6b7d2577
```

Its byte-identical server copy, source, and bundle are preserved in the same
incident evidence directory. Windows targeted validation passed 11 tests; the
exact Linux checkout passed 10 tests, and the final waiter/receipt validation
passed 4 Linux tests. The only GPU compute PID after launch remained the
training process `619775`.

Deployed standalone supervisor identity:

```text
path: /tmp/cofitok-quality-bridge-execution-cf0e5fa/recovery_supervisor_v2.py
sha256: bc920eacc7d95141396c4257fe537bd77ca8dec5e0a7d7337d064ddfb30c6447
pid: 630988
status: observing
attempt: 0
```

The supervisor permits at most three additional recoveries. Exit code 1 is
retryable only when all of the following are simultaneously present:

1. watchdog status is `failed` with `reason=child_failed`;
2. watchdog and child exit codes are both 1;
3. controller output contains the exact pin-memory, resource-sharer, and
   `FileNotFoundError` markers;
4. monitor evidence contains no checkpoint-integrity, non-finite,
   non-monotonic, revision, branch, or samples-seen issue;
5. the exact training checkout, runbook, preparation, approval, standing
   authorization, storage, and GPU-idle bindings remain valid.

Arbitrary exit code 1, scientific gate failures, integrity failures, and
configuration/provenance drift remain non-retryable.

The formal checkout is read-only and is not used as a mutable control surface.
At deployment it was tracked-clean at revision `1ebcc15210e63a776a2ba448481cbd8bb94a4066`;
its informational full-porcelain snapshot was 89 paths with SHA256
`18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`.

## Verification

- Windows targeted pytest: `5 passed`.
- Linux targeted pytest from the exact tracked checkout: `5 passed`.
- Deployment-receipt validation: Windows `2 passed` (`7 passed` together with
  the supervisor tests), Linux `2 passed`; both observed receipt SHA256
  `c20cf6193db929adbecd2dbe2822deb907f6aedfb29095175317524b1a6cc075`.
- Linux `py_compile`: pass.
- Linux built-in supervisor self-test: pass.
- Remote `git bundle verify`: pass; the isolated clean checkout is
  `/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-ipc-recovery-564b751`.
- Source SHA matched byte-for-byte locally, in `/tmp`, and in the immutable
  incident evidence directory; it also matched the tracked file in the remote
  `564b751` checkout.
- Replaying the preserved real incident classified it as
  `bounded_pin_memory_resource_sharer_failure`.
- Replaying the same log with a checkpoint-integrity issue was rejected as
  `exit_1_has_nontransient_monitor_issue`.
- A source-bound `--once` rehearsal observed the existing controller without
  launching a duplicate.

Preserved failure evidence and the deployed supervisor source are under:

```text
reports/recovery_incident_2026-08-17_pin_memory/
```

## Scientific boundary

This recovery changes no model, optimizer, sampler, data order, worker count,
pin-memory setting, micro-batch, accumulation, or loss schedule. It only
restores bounded control-plane continuity. Full-data CoFiTok/dense terminal
quality evidence remains incomplete until both 100K trainings and matched
post-evaluation finish.
