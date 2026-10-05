# Source-Compatible Gate Revalidation (2026-09-01 22:15 CST)

## Scope

This was a CPU-only, read-only revalidation. It did not modify the
authoritative bridge checkout, create a candidate output root, launch training
or sampling, send process signals, replace the terminal hold, or grant
promotion/release permission.

## Source and gate identities

The isolated remote checkout was:

```text
/tmp/cofitok-exposure-capacity-source-compatible-20260901
```

The preparation validator accepted:

```text
preparation SHA256: e9652b12912c399d9e462475f3777e40cb27b16d0221734f11ce11b032aa524c
schema: cofitok_generation_exposure_capacity_preparation_v1
status: pass
source_count: 11
```

The two exact gate files were rehashed and validated:

| arm | SHA256 | validation |
| --- | --- | --- |
| `capacity_qualification` | `f078c5b376c4c8f57315d9a7176ff4665640f8b4b31d590f765482dfbfe276d4` | `status=pass`, `execution_ready=false`, source checkpoint verified |
| `exposure_continuation` | `171512fa8ca3d18208f5a5dc4be1213661fb45735b38b62ef0a5cd2d065ab16c` | `status=pass`, `execution_ready=false`, source checkpoint verified |

Both gates retain `stage_authorization.status=not_authorized`,
`terminal_status=hold`, and false training, sampling, GPU, 300K, promotion,
export, release, and process-signal permissions.

## Runtime checks

The authoritative bridge remains `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
with tree `6cef27723196fd363379bca2e7b85b1678ebd777` on
`scale/generation-stability-quality-bridge-100k`. The standing authorization
SHA256 remains
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.

At the final check the RTX PRO 6000 reported `0 MiB` and no compute
application. Both candidate output roots were absent. The full-data terminal
result remains `generation_advantage_proven=false`.

## Verification

The source-compatible gate/process test subset passed (35 tests). The current
main-tree gate/metrics/process subset also passed. No code or locked evidence
was changed by this revalidation.

