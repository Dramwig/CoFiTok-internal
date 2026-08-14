# 2026-08-14 terminal requested-class visual-audit waiter

## Outcome

A single CPU-only waiter now follows the matched full-data base128 100K quality
bridge. It does nothing until the exact terminal `quality_bridge_result.json`
exists with the locked completed, non-authorizing schema. It then revalidates
the two terminal sampling reports and automatically builds the deterministic
real/CoFiTok/dense requested-class panels for global indices `0..15`.

The waiter does not participate in the quality screen or follow-up decision,
does not use the GPU, does not launch training, and cannot authorize full 300K,
promotion, or release.

## Exact implementation

- Branch: `scale/generation-terminal-visual-audit-waiter-v1`.
- Revision: `c1abf65fdafb8e198a8f1ac59c83b3038a9702b5`.
- Tree: `3362a6dc939ae5d907103211db41eaa88851f1d2`.
- Isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/terminal-visual-audit-waiter-c1abf65/CoFiTok-internal`.
- Incremental bundle: `5,221` bytes, SHA256
  `ce241d46193cc67b2e93eeb5929581209ba42ab3d76105e4629e567534aac330`.
- Bundle prerequisite:
  `2e86ed24925fa958c86dc9bcfa8bc5f029b3a1d8`.
- Focused visual, class-fidelity, and quality-bridge regression suite:
  `43/43 passed` on the exact Linux checkout.

## Runtime binding

- Waiter PID: `182562`.
- PID file:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/requested_class_visual_audit_waiter.pid`.
- Status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/requested_class_visual_audit_waiter_status.json`.
- Log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/requested_class_visual_audit_waiter.log`.
- Exclusive lock:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/requested_class_visual_audit_waiter.lock`.
- Poll interval: `60 s`.
- Niceness: `19`.
- Forced child environment: `CUDA_VISIBLE_DEVICES=''`, four OMP/MKL/OpenBLAS
  threads.

At deployment the status was `waiting`, detail
`waiting_for_exact_quality_bridge_terminal_result`, with poll count `0`. The
quality bridge itself had not launched because the unrelated FieldScope CUDA
process remained active.

## Terminal sources and output

The trigger is:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/quality_bridge_result.json`

After that source validates, the waiter reads the two terminal DDIM-100
sampling reports and their `prefix_8` / `prefix_1` directories under the
CoFiTok and dense `terminal_100k/samples_10000_ddim100_cfg15/` roots. The
calibrated real reference remains the source-bound 1,000-class validation
calibration report with SHA256
`a4c68b9b1c4dffda89f622887455f46d554fdb0c306d467fecb51d8f5abe6132`.

The resulting report and two panels will be written under:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/requested_class_visual_audit_terminal_100k_v1`

Every restart or pre-existing output is handled with the visual builder's
source-revalidating `--resume` path. A second waiter cannot acquire the same
file lock.
