# Generation Current-State Revalidation (2026-09-02)

## Scope

This was a CPU-only, read-only revalidation. It did not launch training or
sampling, select a candidate arm, create a candidate output root, modify the
authoritative checkout, send process signals, replace the terminal hold, or
grant promotion/release permission.

## Live bridge and terminal decision

The authoritative bridge remains revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree
`6cef27723196fd363379bca2e7b85b1678ebd777`, on
`scale/generation-stability-quality-bridge-100k`. Both matched methods remain
complete at 100,000 steps and 6,400,000 images seen. The authoritative terminal
completion audit is operationally `pass`, while `terminal_status=hold` and
`generation_advantage_proven=false` remain unchanged. The failed quality checks
are CoFiTok absolute FID, CoFiTok recall floor, and class fidelity.

The RTX PRO 6000 was idle (`0 MiB`, no compute applications) during the
revalidation. Existing CPU-only supervisors remained outside GPU execution.

## Source-compatible candidate gates

The isolated source-compatible preparation was revalidated with the tracked
validator and returned `status=pass`, `source_count=11`, and both candidate arm
IDs. Its SHA256 is:

```text
e9652b12912c399d9e462475f3777e40cb27b16d0221734f11ce11b032aa524c
```

Both source-bound execution gates were revalidated and returned
`status=pass`, `source_checkpoint_verified=true`, and `execution_ready=false`:

| arm | gate SHA256 | stage authorization |
| --- | --- | --- |
| `capacity_qualification` | `f078c5b376c4c8f57315d9a7176ff4665640f8b4b31d590f765482dfbfe276d4` | `not_authorized` |
| `exposure_continuation` | `171512fa8ca3d18208f5a5dc4be1213661fb45735b38b62ef0a5cd2d065ab16c` | `not_authorized` |

The gates retain false training, sampling, GPU, full-training, 300K,
promotion, export, release, and process-signal permissions. Their output roots
and execution locks remain absent.

## Verification

The remote preparation and both gate validators passed. The local focused
regression suite also passed: `19 passed, 1 skipped`; the skip is the expected
Windows limitation for real POSIX `flock` probing.

## Decision

No candidate arm is selected. The next GPU action requires a new exact
candidate-specific stage authorization. The terminal scientific hold and all
locked evidence remain unchanged.
