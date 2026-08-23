# Terminal runtime strict conjunct v3 deployment (2026-08-23)

## Control identity

- local branch: `analysis/generation-terminal-runtime-strict-conjunct-fail-closed-v2`
- code revision: `0a64804ea698c9708afa1368198274395f34c15b`
- tree: `5f71d8706dd11628b10005799aa4bc661038c813`
- bundle: `D:/cofitok-bundles/terminal-runtime-strict-conjunct-fail-closed-0a64804.bundle`
- bundle bytes: `5714`
- bundle SHA256: `fd869e9bc791847ea601690ef3b88f05edae797c18d2049daa8c0d82511c2150`
- required prerequisite: `06ac9b4f034c4cc7ce8961d9499937e0dbe6dfa3`

The remote bundle SHA256 and prerequisite were verified before an independent
checkout was created.  No existing checkout was moved.

## Remote checkout and tests

- checkout: `/root/autodl-tmp/CoFiTok/checkouts/terminal-runtime-strict-conjunct-fail-closed-0a64804`
- branch/revision/tree match the control identity above
- tracked state: clean
- waiter source SHA256: `7fbd68022c8e29288345ea3334aa3f9e53577bfe7d75cb948af7d7c200057ff0`
- builder source SHA256: `95ab962164cc92ff8358e6c15c3e8bd160af734f558d0325efd80f5272399c1d`
- targeted remote tests: `32 passed`

The corrected builder was also run directly against the live strict comparator
v2 JSON and waiter status.  It accepted the fail-closed result with:

- `canonical_runtime_claim_trusted=false`
- `strict_runtime_guard_controls=true`
- `strict_recovery_binding_required=true`
- `strict_recovery_binding_verified=true`
- `direct_runtime_ranking_allowed=false`

## v3 waiter

- canonical output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/terminal_runtime_strict_conjunct_v3`
- live deployment PID at the deployment snapshot: `114402`
- start ticks: `1714429204`
- PPID: `1`
- cwd: exact checkout above
- cmdline SHA256: `0609c964a7c3b24296b9c821aca67a49bc74f21925621a525155754795f5f03f`
- `CUDA_VISIBLE_DEVICES=""`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- nice: `10`
- ionice: `idle`
- deployment receipt SHA256: `f53c72e54d8b05abaec128e5b61c224ff76ee4724e07a7cda1a45fda0a80cf26`
- initial waiting status SHA256: `3d390c693e3a1edfb1f1e50b1459b0bc3dd1926639bf55c7edc1dd113565f202`
- runtime comparison SHA256: `541ee6ea5ad9dcc1a15d6e01a1418522d04eb40fbd6574b359a73dc01e36d6b5`
- runtime comparison waiter status SHA256: `0a899a8b21128e0b0612a975bcdbfb52cd8d86e6ff9d8f249297bb600b31cdab`

The v3 waiter is waiting only for the terminal completion audit.  It is
CPU-only and permanently non-authorizing.  It did not signal or overwrite the
existing v1/v2 waiters or artifacts.  Training, sampling, promotion, release,
export, and full-300K permissions remain false.

