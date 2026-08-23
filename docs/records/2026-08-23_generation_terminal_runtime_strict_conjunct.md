# Terminal runtime strict conjunct deployment (2026-08-23)

## Purpose

The live terminal completion audit did not consume the independently replayed
runtime strict comparator. This left a provenance gap: terminal quality evidence
could complete without an explicit, immutable conjunction with the fail-closed
runtime-claim boundary.

This change adds a CPU-only, permanently non-authorizing conjunct. It publishes
only after both of these exact upstreams pass:

- `reports/terminal_completion_audit_v1/terminal_completion_audit.json`
- `reports/runtime_compute_claim_guard_strict_comparison_v1/runtime_claim_guard_comparison.json`

It preserves terminal `pass` versus `hold`, never promotes a hold into an
advantage, and permits recovery-adjusted elapsed only as an observational
physical lower bound. Direct wall-clock, throughput, or cost-efficiency ranking
remains prohibited.

## Source identity

- Code revision: `ebd0a49fe17a43cbf4159ad1ad1932ad8fd96ec1`
- Tree: `60b652debc76d0d1555127c3b4eb0eb8dacf2275`
- Branch: `analysis/generation-terminal-runtime-strict-conjunct-v1`
- Builder SHA256: `df7a9558e33d59693a81da636fe63e94b79896eaa16c4fd98813f8a8acc1b6e6`
- Waiter SHA256: `7fbd68022c8e29288345ea3334aa3f9e53577bfe7d75cb948af7d7c200057ff0`
- Incremental bundle: `D:/cofitok-bundles/terminal-runtime-strict-conjunct-ebd0a49.bundle`
- Bundle bytes: `11,916`
- Bundle SHA256: `114e0440dbcd27a2e2ef37c0674d877a6a2ecfbfe8611040005f796d54b60d74`
- Bundle prerequisite: `2764089f97713ebc52d2a67f0e1b60b08cca3d50`

## Verification

Windows targeted regression:

```text
26 passed
```

Linux targeted regression in the exact deployed checkout:

```text
26 passed
```

The broader suite's first failure in the isolated `D:` checkout was an existing
absolute-path assumption for `D:/paper/venues/aaai27/main.tex`; it occurred
outside the three files in this change.

## Remote deployment

- Checkout: `/root/autodl-tmp/CoFiTok/checkouts/terminal-runtime-strict-conjunct-ebd0a49/CoFiTok-internal`
- Output: `stability_full_data_100k_base128_quality_bridge_v1/reports/terminal_runtime_strict_conjunct_v1/`
- Initial PID: `927815`
- Initial start ticks: `1710019403`
- Parent PID: `1`
- `CUDA_VISIBLE_DEVICES`: empty
- `OMP_NUM_THREADS`: `1`
- `MKL_NUM_THREADS`: `1`
- nice: `10`
- ionice: `idle`
- Deployment receipt SHA256: `04486178c180167fcf4110bfbebc2ce97defd32b1548aa4a9b20cc5b36b8e40b`
- Initial status: `waiting_for_terminal_completion_audit`

The deployment receipt records runtime policy rather than a transient PID, so a
server-restart relaunch can reuse the immutable receipt while the live status and
PID file record the new process identity. Relaunch remains fail-closed: require
the old process to be absent, the output not to be pass, exactly no duplicate,
an idle output lock, and exact checkout/source/target identities.

## Authorization boundary

This artifact does not authorize training, sampling, checkpoint loading,
inference export, promotion, release, process signals, full training, or 300K.
It does not modify either upstream decision. `generation_advantage_proven`
remains false until the matched terminal evidence and all claim guards pass.

At deployment, dense training was still the sole GPU workload and had reached
step `90,700`; dense 90K physical integrity and historical replay had passed.
