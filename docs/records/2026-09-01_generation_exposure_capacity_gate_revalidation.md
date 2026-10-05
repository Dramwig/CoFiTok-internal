# Exposure/capacity gate revalidation (2026-09-01)

## Scope

This is a CPU-only evidence revalidation. It does not launch training or
sampling, modify the remote checkout, promote a checkpoint, export an
inference artifact, release a model, or send process signals.

## Live source state

At `2026-09-01T08:17:15+08:00`, the remote `pro6000` checks were:

- standing authorization SHA256:
  `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`;
- locked bridge revision/tree:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a` /
  `6cef27723196fd363379bca2e7b85b1678ebd777`;
- pair monitor: `status=pass`, `stage=complete`, `issues=[]`;
- GPU: `0 MiB`, no compute application; no training or sampling process;
- `/root/autodl-tmp`: `233 GB` free (`77%` used).

The source-bound terminal artifacts were rehashed as follows:

| artifact | SHA256 |
| --- | --- |
| `pair_monitor.json` | `78306a02a47d96be78f325ec2cd708ff4741f248c544fbc861dc9acfa829a2eb` |
| `quality_bridge_result.json` | `15752e05611fa888e15352934c1627ccb95418df0339f9ad120e195bc088d165` |
| `quality_bridge_comparison_v4_authoritative/quality_bridge_comparison.json` | `991600ffddc66d3d3f81d7294f202b636f71acc9e7de52fb33418f16f4f882e8` |
| `terminal_completion_audit_v4_authoritative/terminal_completion_audit.json` | `2c4d924573f99e6fc62cf7bf83a37a942fa683bb6523b01e2cdd4b867fac088f` |

Both methods remain at 100,000 steps and 6,400,000 images seen. The formal
terminal result is still `terminal_status=hold` and
`generation_advantage_proven=false`.

## Candidate gate revalidation

The current preparation was independently validated with ten source files:

```text
artifacts/reports/generation/exposure_capacity_disambiguation_2026-08-31/preparation.json
SHA256=b34d0e0a057bed34511c88f83671c80039b86ea45cf548636daff02f91dcb074
```

The two source-bound candidate gates preserved in the isolated D: evidence
fixture are:

| arm | artifact SHA256 | result |
| --- | --- | --- |
| `capacity_qualification` | `57390cd2d776eb20bb35d41418332a3a098a1551d6021c2ea5cdc5f4d990fafe` | `status=prepared`, `execution_ready=false` |
| `exposure_continuation` | `8b8f35a442d999baa30466f6dd47f6c3d59bcae8d469d8eab5aca044be3e8faa` | `status=prepared`, `execution_ready=false` |

The gate builder binds the locked bridge checkout, dataset/runtime identities,
idle-GPU snapshot, free-storage check, versioned output root, and (for the
exposure arm) the exact CoFiTok 100K checkpoint, sidecar, and `latest.json`.
Both gates retain the exact stage-authorization requirement and set every
training, sampling, evaluation, GPU, 300K, promotion, export, release, and
process-signal permission to `false`.

## Verification

The following project-local CPU suites passed:

```text
tests/test_generation_exposure_capacity.py
tests/test_generation_exposure_capacity_gate.py
tests/test_process_monitoring.py
tests/test_generation_training_completion.py
tests/test_generation_stability_sampling_recovery.py
tests/test_large_scale_generation_completion_audit.py
tests/test_generation_stability_completion_audit.py
tests/test_generation_sampling_preflight.py
```

The preparation validator reported `status=pass`, `source_count=10`; the gate
tests reject permission tampering, wrong source checkout, changed checkpoint
payload/sidecar/latest identities, GPU contention, and insufficient storage.

## Boundary

The candidate gates are preparation artifacts, not execution authorization.
The current source-compatible stage gate still requires a separately recorded
exact stage authorization. No GPU stage is selected, no 300K escalation is
allowed, and the terminal scientific hold remains unchanged.
