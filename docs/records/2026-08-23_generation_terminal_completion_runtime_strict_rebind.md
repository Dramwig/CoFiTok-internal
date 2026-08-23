# Terminal completion runtime-strict rebind

Date: 2026-08-23

The corrected terminal-system guard and strong-baseline comparison are immutable,
versioned reports. The original terminal completion waiter and its physical replay
builder accepted only the v1 report directory names, so pointing only the waiter at
the corrected sources would still fail inside the builder.

This change adds explicit expected directory-name arguments for the terminal guard,
comparison, and completion output. All default to the original v1 names. A
non-default value must remain a single lower-case report-directory component with
the corresponding role prefix, and every source/output path must still match its
declared directory exactly.

The audit remains CPU-only and permanently non-authorizing. It cannot launch
training or sampling, load GPU work, authorize 300K scaling, promote, export,
release, or signal any process. Existing v1 and failed evidence remains immutable.

## Verification

- Code commit: `107f739035a71297fa9d970281097e04fa1a0b45`
- Tree: `e57229608c3ce72ab94cfe3a31251f87c3fba89e`
- Branch: `analysis/generation-terminal-completion-runtime-strict-v2-20260823`
- Local targeted tests: `28 passed`
- Remote targeted tests: `28 passed`
- Deployment bundle: `D:/cofitok-bundles/terminal-completion-runtime-strict-v2-107f739.bundle`
- Bundle bytes: `15069`
- Bundle SHA256: `e3fc3faab3b6af41b5df22e80c3cbafa51a04c4f463ac449861897c529ddf1a3`

## Canonical corrected deployment

- Remote checkout: `/root/autodl-tmp/CoFiTok/checkouts/terminal-completion-runtime-strict-107f739/CoFiTok-internal`
- Waiter source SHA256: `5f348afdc5e1ce8d91df91c41ad0ccffd74f7863f36248f26d32be6dfbd05932`
- Builder source SHA256: `015e51c27eda1f0bad53e659b646f92eb8db07d204ba343bd8e3866b204162c3`
- Canonical directory: `reports/terminal_completion_audit_v2_runtime_strict`
- Corrected terminal guard: `reports/terminal_system_claim_guard_v2_runtime_strict`
- Corrected comparison: `reports/quality_bridge_comparison_v3_runtime_strict`
- Deployment receipt SHA256: `eb1ee6cdee214cbefa1a2e3bafe3e9cf58eeb267c9dbf444eebcc1efef200c3a`
- Initial waiter-status SHA256: `1eadb576c3634e08a07fe77677faa83032b866a2d08a743473b7e75e7101f180`
- Deployment PID: `125152`

The live deployment was re-read on 2026-08-23 at 22:56 CST. PID `125152`
still had parent PID 1, start ticks `1714687556`, the exact corrected command
line and checkout cwd, empty `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, nice 10, and idle I/O priority. Its mutable status was
`waiting / waiting_for_exact_terminal_completion_sources`; the completion output
did not yet exist. The status correctly named the not-yet-produced corrected
terminal guard and comparison artifacts as its only missing sources.

The deployment receipt fixes all launch, scaling, promotion, export, release, and
process-signal permissions to false. An operational audit pass may retain either a
scientific terminal `pass` or `hold`; it cannot promote a hold or set
`generation_advantage_proven=true` unless the corrected terminal guard itself
proves the complete terminal claim conjunction.
