# SHA-bound lightweight retention runway checks

Date: 2026-08-01 CST

## Problem

The historical checkpoint retention inventory was correctly built and
independently replayed with physical SHA256 verification of 54 checkpoints
(`54,336,513,724` payload bytes). The dynamic runway helper reused the full
inventory verifier, so every capacity refresh implicitly repeated the same
physical checkpoint hashing. That is appropriate for establishing the inventory
trust boundary, but inappropriate for frequent monitoring beside an active GPU
training run.

One attempted post-dedup refresh exposed this behavior. It created no report and
was stopped by exact maintenance PIDs `741042` and `741041`; trainer PID `319202`
and readiness waiter PID `622788` remained alive and unchanged. No checkpoint,
sample, receipt, or training process was modified.

## Trust-boundary change

Commit `1604dc45c21ec73d9debf8a81e3d7367cd9fb220` separates the two operations:

1. Inventory construction and `validate_generation_checkpoint_retention_inventory.py`
   retain full physical replay.
2. `check_generation_retention_runway.py` now requires
   `--expected-retention-inventory-sha256`. It rejects an invalid expected SHA,
   a changed inventory file, a report that differs from the persisted bytes, or
   any inventory that weakens the read-only/zero-currently-reclaimable policy.
3. A successful dynamic report records
   `retention_inventory_verification=expected_sha256_binding` and
   `physical_checkpoint_hashes_replayed=false`.
4. The full audit runbook computes the SHA only after its physical validator
   succeeds, then passes that exact digest to the lightweight runway stage.

This does not claim that checkpoint bytes are unchanged forever. It means a
frequent capacity check is explicitly bound to the previously physically
validated immutable inventory artifact instead of silently repeating the
expensive validation. Any new inventory requires another physical audit and a
new expected SHA.

## Verification and deployment

Local targeted tests and the complete pytest suite passed. Exact Linux
deployment used:

```text
revision: 1604dc45c21ec73d9debf8a81e3d7367cd9fb220
checkout: /root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-1604dc4
bundle bytes: 39499901
bundle sha256: 0baaf43696a778cb8fc39202752bed04bfbee0a65556fc77a5458bed62c628b0
deployment receipt sha256: 38b14b3f20d478f23919cb748f208966dec5669198f7431266728f9f5684431e
Linux pytest: 874 passed / 2 skipped / 0 failures / 0 errors
runbook syntax: 101/101 passed
```

Independent deployment receipt replay passed. The formal repository remained
clean at `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`; the receipt retains
`readiness_executed=false`, `full_training_launch_allowed=false`, and
`formal_generation_completion_claimed=false`.

The deployment's repeated common Git pack/index was then hardlinked to the
existing canonical inode under the same fail-closed plan/apply protocol:

```text
plan sha256: cc7cefef40e63ce2fac831b4f3a7d1a2778c44887b2c6fe04346138de46f1d75
result sha256: e65e2d4749496d26789f991cd1f27a6e8e49d1c1e1754810568f3462a3b726cb
expected bytes saved: 136940968
observed free-byte delta: 136941568
```

Independent result replay passed. The final plan reported `22 already_linked`,
`0 eligible`, `26 indeterminate`, and `8 required` files across 14 checkouts,
with `currently_saved_bytes=1,506,350,648` and
`potential_physical_bytes_saved=0`; its SHA256 is
`da850209109baa1b8aae2103056aa47ee281d1959ba659deeaeb860c2afb20d6`.

## Final post-dedup runway

The pinned inventory remains:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/retention_audit_2026-08-01/checkpoint_retention_inventory.json
bytes: 220769
sha256: 62ae7388983e046fc0d0afbb7db2059e1afb5c2674880881e20411cd66f68cd5
```

The final lightweight report completed without reading checkpoint payloads:

```text
status: pass
filesystem free bytes: 186542624768
required free bytes: 180880415360
current headroom bytes: 5662209408
currently reclaimable bytes: 0
potential archive candidate bytes: 0
physical checkpoint hashes replayed: false
report sha256: f23de67d99ebe5b4a58f5f66875d163ec857b10b581d7ad14bb1eff39988bcf9
```

The margin remains narrow and must be refreshed before any readiness or launch
decision. This report authorizes neither deletion nor archive, CUDA readiness,
nor full 300K training. The final live snapshot for this pass was
`running/cofitok_training/issues=[]`, CoFiTok `35,500/50,000`, dense
`0/50,000`, trainer PID `319202`, and readiness waiter
`waiting_for_passing_stability_gate` with
`full_training_launch_allowed=false`.
