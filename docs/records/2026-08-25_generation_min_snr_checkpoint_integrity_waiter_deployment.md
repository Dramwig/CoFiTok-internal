# Min-SNR pilot checkpoint-integrity waiter deployment

Date: 2026-08-25 CST

## Purpose and boundary

The fresh matched Min-SNR gamma-5 pilot is running as one serial controller:
CoFiTok to step 50,000, dense identity to step 50,000, then four terminal
DDIM-100 evaluation arms. A source-bound CPU-only waiter now performs physical
checkpoint replay at serial milestones without loading a model, allocating GPU
memory, signaling any process, changing training state, or authorizing another
stage.

Every status and audit emitted by this waiter keeps training, sampling,
evaluation, continuation beyond 50K, full 300K, promotion, export, release, and
process-signal permissions false. A passing milestone audit proves checkpoint
integrity only. It does not prove sample quality or a generation advantage.

## Pinned implementation

```text
local worktree: D:/cofitok-min-snr-checkpoint-integrity-monitor-20260825
branch: analysis/generation-min-snr-checkpoint-integrity-monitor-v1-20260825
revision: 0d312e1248592522aa982afa52c822b01464a4e9
tree: e4f63c781b3f1861ff598f4411870643cd904538
remote checkout: /tmp/cofitok-min-snr-checkpoint-integrity-0d312e1/CoFiTok-internal
waiter source bytes: 17419
waiter source SHA256: 98b4ef54c0442213a1ca642f9c36c4553840b776a0b8ca91f4ceac9ec0253f39
test source bytes: 7748
test source SHA256: 567ef5544b99f671e19af4f3c425663c268361c35f8abdfe616b8d77585a8a49
```

The Linux source hash above is authoritative. The Windows checkout uses CRLF
and therefore has a different byte-level source hash.

Validation and transport evidence:

```text
local focused tests: 4 passed
remote related tests: 28 passed
Python compilation: pass
CLI help: pass
git diff --check: pass
bundle: D:/cofitok-bundles/min-snr-checkpoint-integrity-waiter-0d312e1.bundle
bundle bytes: 7475
bundle SHA256: caaa709c33147776d17ad6f212f19fa7c37aab52b02495c46b39fc89fab0d2af
```

## Training identity

The waiter is bound to the clean training checkout and cannot accept another
revision, branch, tree, dataset, runtime, or effective batch:

```text
checkout: /root/autodl-tmp/CoFiTok/checkouts/min-snr-matched-pilot-842a341/CoFiTok-internal
revision: 842a34130e82f241330707118a05bf6ed01e263e
tree: 3fd4c4538d15b85233b1f8b582dce0f185dedba2
branch: scale/generation-min-snr-matched-pilot-v1-20260825
dataset identity SHA256: 6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659
runtime environment SHA256: d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
effective batch: 64
```

## Physical 5K replay

The first audit completed with `status=pass` and
`physical_sha256_verified=true`:

```text
checkpoint payload bytes: 1010933994
checkpoint payload SHA256: b6f25451d2a21eda42e879437f2c8d7101274f73dbba72a147af82c292f1985b
integrity sidecar bytes: 599
integrity sidecar SHA256: fa5670714e673de44541f332434395807c2d9b47fa1ddc841bce25322754dfa4
latest.json bytes: 669
latest.json SHA256: 58b4e1be3a2ee452fc6e229bd88d7849ed58f0b8b3670183eac0b981949c70f2
audit bytes: 5627
audit SHA256: c5d797b25ebe80c357fed384e800665e053cabc14d8fefad247093a9480c0042
status SHA256: 13ba42cfc5e5ae000a020b92a6bd86be6727c73ebb9043b8375ea8545711d709
```

The audit re-read the physical checkpoint payload and sidecar, exact
`latest.json` binding, clean Git identity, dataset/runtime hashes, and canonical
metrics. The metrics were finite and strictly increasing, satisfied
`samples_seen == step * 64`, contained the required Min-SNR numeric fields, and
bound the step-5,000 row to 320,000 images.

Canonical paths:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/checkpoint_integrity_waiter_v1/cofitok_step_00005000_physical_integrity_audit.json
/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/checkpoint_integrity_waiter_v1/cofitok_step_00005000_waiter_status.json
```

## Unique 10K waiter

The only live milestone waiter targets CoFiTok step 10,000:

```text
PID at deployment: 431253
PPID after verified detach: 1
start ticks: 1731054760
nice: 19
ionice: idle
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
initial status: waiting / checkpoint_missing
```

The transient launch wrapper PID `431252` was verified by PID, start ticks,
cwd, and exact command identity before it exited. The waiter was then confirmed
reparented to PID 1. Its lock was held only by `/proc/431253/fd/3`, and the GPU
process list contained only the existing CoFiTok trainer.

Deployment receipt:

```text
path: /root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/checkpoint_integrity_waiter_v1/deployment_receipt.json
bytes: 7064
SHA256: 37a07f4e6819b5e6c4e21e409cb9cd5346708198a6bca299da877a8d3ac5054b
```

At the independent 2026-08-25 20:28 CST replay, the pilot had reached step
5,550 with finite, strictly increasing metrics and exact image accounting. The
health monitor reported `running`, `issues=[]`, `terminal_status=hold`, and
`generation_advantage_proven=false`; the 10K checkpoint and audit were
correctly absent. Training, monitor, and waiter checkouts remained tracked
clean, the waiter retained its exact process identity and CPU-only environment,
and free disk space was `262,910,423,040` bytes.

## Serial successor rule

No second milestone waiter may run concurrently. After the 10K waiter passes,
its audit must be independently re-read and replayed, including Min-SNR fields
and the all-false authorization boundary. Only after PID `431253` is absent,
its lock is free, and no duplicate waiter exists may the same pinned source
launch one successor for step 15,000. The same one-at-a-time rule continues
through CoFiTok 50K and then dense 50K under the existing controller.
