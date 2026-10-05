# Source-compatible exposure/capacity replay recheck (2026-09-01)

## Scope

This is a CPU-only, non-authorizing replay check. It does not modify the
authoritative remote checkout, create a candidate output root, launch
training or sampling, send process signals, replace the terminal hold, or
grant promotion/release permission.

## Source-compatible fixture

The isolated source-compatible branch is:

```text
analysis/generation-exposure-capacity-source-compatible-20260901
HEAD eb6af52110ab743332858882be2e33cd2dbb90fe
tracked tree clean
```

It contains the preparation builder (`bc1cd95`), the source-compatible
preparation artifact (`195244a`), and the remote replay record (`eb6af52`).
The preparation schema explicitly carries `objective_reassessment` inside
`evidence` and records eleven hashed sources, including that objective
report. This resolves the older isolated-validator mismatch without changing
the locked 100K quality evidence.

## Replay results

The remote Linux checkout at
`/tmp/cofitok-exposure-capacity-source-compatible-20260901` was revalidated
with the recorded preparation SHA
`e9652b12912c399d9e462475f3777e40cb27b16d0221734f11ce11b032aa524c`.
The validator returned:

```text
schema_version=cofitok_generation_exposure_capacity_preparation_v1
status=pass
source_count=11
```

The local D: checkout was also validated using its actual Windows working
tree bytes (`9d07a00bbd89cb36d38abe9bf7b6271381590a8e32dd87af2af5e24cdf74096f`);
it returned the same `status=pass` and `source_count=11`. The differing hash
is caused by the checkout's historical CRLF materialization and is not used
to weaken the strict hash check. The remote replay keeps the LF artifact and
its own content-addressed SHA.

The source-compatible focused tests passed:

```text
17 passed, 1 skipped
```

The recorded capacity and exposure remote replay gates both remain
`status=prepared`, `execution_ready=false`, `terminal_status=hold`, and
`stage_authorization.status=not_authorized`. Every training, sampling,
evaluation, GPU, 300K, promotion, export, release, and process-signal field
remains false.

## Decision

The source-compatible preparation replay is now reproducible on both the
remote Linux fixture and the local isolated checkout. It is still only a
preparation/replay artifact. The authoritative bridge remains at revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`; no candidate-specific exact-stage
authorization exists, so no exposure or capacity GPU stage is selected. The
scientific terminal status remains `hold` and
`generation_advantage_proven=false`.
