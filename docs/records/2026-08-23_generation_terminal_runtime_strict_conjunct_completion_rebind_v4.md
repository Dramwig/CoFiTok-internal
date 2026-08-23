# Terminal runtime strict conjunct completion rebind v4 (2026-08-23)

## Reason

The live v2 and v3 conjunct waiters were correctly non-authorizing but still
pointed at the immutable `terminal_completion_audit_v1` source. The corrected
terminal completion replay is versioned as
`terminal_completion_audit_v2_runtime_strict`, so those waiters could never
consume the corrected terminal evidence.

The builder and waiter now require an explicit, validated terminal completion
directory component. The default remains `terminal_completion_audit_v1` for
backward compatibility. A non-default component must use the
`terminal_completion_audit_v` prefix, remain a single safe path component, and
bind exactly to `waiter_status.json` and `terminal_completion_audit.json` in the
same directory.

## Control identity and verification

- Branch: `analysis/generation-terminal-runtime-strict-conjunct-completion-rebind-v3-20260823`
- Revision: `b9c03a70204fe7e89af39c041894539f31097086`
- Tree: `c3d0435f15d5b47c6f2df8093c25fcee0475cbc7`
- Waiter SHA256: `e2bbe54e973c9bc6fe33fc9ba938c344e4d2ead89d3a2d667c9ac15efdf6bd69`
- Builder SHA256: `bf16d8f82fe4476a16d7d87a9e1fcf6b8793985b98ecdbf11a9f53df6bb325ff`
- Local targeted tests: `21 passed`
- Remote targeted tests: `21 passed`
- Bundle: `D:/cofitok-bundles/terminal-runtime-strict-conjunct-completion-rebind-b9c03a7.bundle`
- Bundle bytes: `11021`
- Bundle SHA256: `a4143f3f00bc3e6ecd5218a812184850c6766405dbc21a574beb6c66008df9b0`

## Canonical corrected waiter

- Checkout: `/root/autodl-tmp/CoFiTok/checkouts/terminal-runtime-strict-conjunct-completion-rebind-b9c03a7/CoFiTok-internal`
- Output: `reports/terminal_runtime_strict_conjunct_v4_runtime_strict`
- Terminal source: `reports/terminal_completion_audit_v2_runtime_strict`
- Runtime source: `reports/runtime_compute_claim_guard_strict_comparison_v2`
- Deployment PID: `131907`
- Start ticks: `1714809667`
- Deployment receipt SHA256: `3434edff929285a28b82bdfff303c80717e2235f5d83da8617321cb81d632b0e`
- Initial waiter-status SHA256: `bfd53e94ee85695cec4c23aa6d4bfd9e3306b915b80a3f3dc0fd3384633783ff`

At the deployment snapshot the process was unique for the corrected completion
source, had PPID 1 and the exact checkout/cmdline/start-tick identity, empty
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, nice 10, and
idle I/O priority. It was `waiting / waiting_for_terminal_completion_audit`;
the strict runtime comparator was already ready. The deployment did not signal,
overwrite, or reinterpret the existing v1, v2, or v3 conjunct evidence.

The conjunct is CPU-only and permanently non-authorizing. Training, sampling,
checkpoint payload loading, 300K scaling, promotion, export, release, and all
process signals remain forbidden. A terminal `hold` remains a hold and cannot set
`generation_advantage_proven=true`.
