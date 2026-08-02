# Stability dense-identity 50K recovery launch (2026-08-02)

## Authorized scope

The completed CoFiTok stability member remained immutable. This operation was
authorized only to launch the missing matched dense-identity member, restore
the existing post-evaluation/readiness waiting chain, and monitor it. It did
not authorize a CoFiTok rerun or full ImageNet-256 300K training.

## Independent launch boundary

After FieldScope released the GPU, an independent prelaunch audit verified:

- controller `5f57757a2162507c6166dbfe976246df6ae1af91` on
  `scale/generation-stability-dense-recovery-control-00549fe`, tracked-clean;
- immutable training revision
  `2c2c1f5166b73d4f28df93b276901671ac1a7836` on
  `scale/generation-stability-50k-preflight`, tracked-clean;
- exact clean post-evaluation and receipt-bound readiness checkouts;
- no related queue process and an absent authoritative dense run directory;
- CoFiTok step-50K payload SHA256
  `ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a`
  and sidecar SHA256
  `4f3f6f3f401f34131f902b016baf41897b35f95d242bb023981feff4a36226a8`;
- frozen runtime-selection SHA256
  `f7befbcdbc6644fc066a71b85cb1df5cae5b922c526f09fe0000ae252b324663`
  with micro-batch 64, accumulation 1, effective batch 64;
- `186,847,199,232 / 118,385,312,804 / 68,461,886,428` bytes for
  free/required/headroom;
- GPU idle both before and after the independent checks.

The serialized recovery controller was launched under `nohup` as PID `511801`.
It repeated its trust-boundary, pair-contract, and storage checks under the
non-blocking fd-6 lock before starting the dense member.

## Live training identity

The live chain was independently verified as exactly one of each:

```text
root recovery controller  511801
pair monitor              530064
training watchdog         537704
GPU trainer leader        541878
post-eval waiter          717145
readiness waiter          748183
```

The trainer leader cwd is the immutable training checkout. DataLoader children
inherit the trainer argv, but `nvidia-smi` reported exactly one compute PID,
`541878`. The run manifest binds training revision/branch, dataset identity,
runtime environment SHA256
`d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`,
62,824,707 trainable parameters, bf16, and 50,000 target steps. The controller
lock probe returned 1, proving the active controller still held the lock.

At the first authoritative pair-monitor refresh, status was `running`, stage
was `dense_identity_training`, issues were empty, and dense was at step 100 /
6,400 images. Initial metrics contained steps 1, 50, and 100, were finite and
strictly increasing, and satisfied `samples_seen = step * 64`.

## Restored waiting chain

The exact post-evaluation waiter was restored from
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` and reported
`waiting_for_completed_training_pair`. It remains the only component allowed to
build `pair_summary.json` from the original locked `reports/config_validation.json`.

Only after that waiter wrote a new `waiting` status was the receipt-bound
readiness waiter restored from
`5dd3488ac9b30274f4960195e252cc9fdb161002`. It reported
`waiting_for_passing_stability_gate` and
`full_training_launch_allowed=false`.

The heartbeat `cofitok-dense-recovery-launch-monitor` was changed from the
temporary 5-minute GPU-handoff cadence to a 30-minute dense-progress monitor.
It may observe and diagnose but must not kill/restart training, rerun CoFiTok,
or authorize full 300K.

Evidence:

`artifacts/reports/generation/stability_scaling_50k_ema_teacher/dense_recovery_launch_2026-08-02/`
