# Frozen supplemental evaluation checkout attestation

Date: 2026-08-03

## Purpose

The frozen stability post-evaluation still needs a later source-bound quality
supplemental: matched EMA checkpoint diagnostics, matched EMA free rollouts,
and distribution-support replay. The implementation already existed locally,
but executing it from an unpinned working tree would weaken checkpoint and EMA
evaluation reproducibility. This step prepares one exact Linux checkout while
leaving the active dense trainer and both existing waiters untouched.

The prepared checkout is an evaluation artifact, not a large-capacity training
deployment and not a launch authorization.

## Exact identities

The existing clean post-evaluation checkout was used only as the prerequisite
object source:

```text
path:     /tmp/cofitok-stability-schedule-audit-c1efb12
revision: c1efb12c6640f2d2d62ac7e9982c8804d96e7289
branch:   scale/generation-stability-50k-posteval-v4
```

The separately prepared checkout is:

```text
path:     /tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal
revision: c212b9e2b64d1b302b17a9d4e30a296d773d4215
branch:   scale/generation-large-capacity
```

Both source and target were tracked-clean after checkout construction and after
the Linux attestation. The target was made with a no-hardlinks clone and a
single-head prerequisite-aware incremental bundle; the source checkout was not
fetched into, merged, checked out, or otherwise changed.

## Transfer evidence

The temporary bundle was `650,894` bytes with SHA256
`4ead1d6ebb4e320a975515fb71faba936cddc07af6ee7ab3d1cc173cc256a915`.
It contained `64` incremental commits and `972` objects, advertised exactly one
head (`c212b9e` on `scale/generation-large-capacity`), and required exactly the
existing `c1efb12` prerequisite. `git bundle verify` passed locally and on
`pro6000`. Both temporary bundle copies were deleted after the target checkout
was verified; the user-owned `.codex-bundles/` directory was not used or
modified.

The deployed waiter entrypoint and runbook are byte-bound in
`checkout_attestation_receipt.json`. Their SHA256 values are respectively:

```text
f937cafeac19b84bf768c72c8c073149a78526e0aaea117c87fd37e149c6a887
1d780d06a4391d5f80aa1a1e2dfcd46929dbdf3fb3f0a17d8a73f895ee33d8e6
```

## Linux CPU-only attestation

With `CUDA_VISIBLE_DEVICES` empty, the checkout passed:

- waiter `py_compile`;
- `70` focused tests with `0` failures, `0` errors, and `0` skips in
  `34.006` seconds;
- syntax checking for all `104/104` tracked runbooks with `0` failures.

The synchronized JUnit, pytest log, syntax report, and machine-readable receipt
are in:

```text
artifacts/reports/generation/stability_frozen_supplemental_checkout_2026-08-03/
```

## Operational boundary

No supplemental waiter or supplemental evaluation process was started. The
only GPU compute process remained the pre-existing CoFiTok dense trainer PID
`541878`; it was not signaled, paused, restarted, or modified. No other
project's process was touched.

This attestation proves only that the exact evaluation checkout is prepared and
reproducible. It does not prove supplemental sample quality, replace the frozen
post-evaluation or readiness results, create a launch receipt, or authorize or
launch full 300K. The checkout remains dormant until the existing dense 50K,
formal post-evaluation, and queued readiness GPU stage become terminal in the
order already enforced by the waiter.
