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
