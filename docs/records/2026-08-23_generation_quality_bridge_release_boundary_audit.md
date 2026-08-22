# Quality bridge stable-inference and release-boundary audit

Date: 2026-08-23 04:50 CST

## Purpose

This CPU-only audit verifies that the active full-data ImageNet-256 matched
100K quality bridge cannot be converted into, or consumed as, a released
generation system. It also hardens the generic release-receipt boundary so a
non-authorizing terminal audit remains non-authorizing even if its other fields
are made to resemble a full completion audit.

The work was performed only in the isolated D: worktree. It did not modify the
active remote checkout, controller, trainer, waiter chain, checkpoint, sample
set, scientific decision, or authorization state.

## Live source state

The remote state was reread before the audit:

- training checkout:
  `/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal`;
- revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`;
- tree: `6cef27723196fd363379bca2e7b85b1678ebd777`;
- branch: `scale/generation-stability-quality-bridge-100k`;
- full Git porcelain: empty;
- standing authorization SHA256:
  `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`;
- dense continuation controller SHA256:
  `f29f8cda690febcdc4ddbd565c3a28f56549c59878e18bf5e688d4c532b8731d`;
- original terminal runbook SHA256:
  `f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73`.

At the source snapshot, CoFiTok was complete at 100,000 steps and dense was
advancing at 83,600/100,000 steps. The pair monitor was `running` in
`dense_identity_training` with `issues=[]`. Dense trainer PID `219593` was the
only GPU compute owner, using 79,132 MiB. Free project storage was
301,481,611,264 bytes.

The dense 90K/95K/100K physical-integrity waiters, replay waiter, runtime
fairness/claim waiters, terminal guard, comparison waiter, terminal completion
waiter, factorization supervisor, and random-token waiter were all present with
their expected parent/cwd/environment/nice/ionice identities. No second trainer,
controller, or GPU experiment was launched.

## Existing release contract

The existing implementation already provided several independent barriers:

1. A non-formal checkpoint may be exported only without a release gate. Passing
   a full release gate for such a checkpoint is rejected.
2. A formal checkpoint export requires a validated `full` gate whose decision
   is `large_scale_generation_ready`.
3. A released artifact is bound to its physical artifact, integrity sidecar,
   export manifest, source checkpoint, source Git/runtime/training
   authorization, and full release-gate authorization.
4. Production completion-authorized loading requires a generation release
   receipt and validates it before deserializing model weights.
5. A release receipt requires a passing full or stability completion profile,
   both methods' step-300K EMA artifacts, export manifests, preflight/smoke
   evidence, and the physical artifact identities recorded by the completion
   audit.
6. The quality-bridge terminal completion artifact itself declares
   `promotion_or_release_allowed=false`, `full_300k_launch_allowed=false`,
   `release_authorization_allowed=false`, and
   `inference_export_authorization_allowed=false`.

## Gap and hardening

The release-receipt profile validator previously rejected the current
quality-bridge terminal audit indirectly because that artifact is not shaped as
a full completion audit. The validator did not, however, explicitly consume
the audit's non-authorizing role and boundary fields. A later schema extension
or an incorrectly copied completion structure could therefore weaken the
fail-closed intent even though the current 100K checkpoint and step-300K checks
would still prevent a successful receipt.

The validator now rejects before any receipt publication when:

- the audit role is
  `generation_quality_bridge_terminal_completion_audit`;
- `authorization_boundary.release_authorization_allowed=false`;
- `authorization_boundary.inference_export_authorization_allowed=false`; or
- `claim_policy.promotion_or_release_allowed=false`.

These checks are also replayed by receipt verification, so changing the bound
completion audit to an explicitly non-releasing state invalidates an existing
receipt before artifact deserialization.

The code hardening commit is:

```text
revision: d4758f6ce3005ec8abe6ec3d25f4d5b0dbdcde18
tree:     85b8ffba04f6c6b745d6d7655568cbbe17619bb9
subject:  Harden quality bridge release boundary
```

New regression coverage proves that:

- a non-formal step-100K quality-bridge checkpoint cannot bind a full release
  gate;
- an unreleased step-100K inference artifact cannot enter completion-authorized
  inference and is rejected before `torch.load`;
- a quality-bridge terminal audit cannot publish a release receipt even if its
  remaining fields are copied from an otherwise passing full audit; and
- an explicit non-release boundary invalidates receipt consumption.

## Verification

Local CUDA-hidden targeted and related suites:

```text
tests/test_generation_inference_artifact.py: 31
tests/test_generation_quality_bridge_terminal_completion_audit.py: 34
tests/test_generation_session.py: 15
tests/test_large_scale_generation_completion_audit.py: 82
total: 162 passed
```

`git diff --check` and Python compilation passed. The project-local environment
does not contain the optional `ruff` package, so ruff was not used as evidence.

The exact code commit was exported with `git archive` and transferred to an
isolated server directory. Archive evidence:

```text
local path: D:/cofitok-bundles/quality-bridge-release-boundary-d4758f6.tar.gz
remote path: /tmp/quality-bridge-release-boundary-d4758f6.tar.gz
bytes: 139,684,666
sha256: 38a8f807e4beb1364c1db92479eeb62e29b88b3b5f2368b6c520b4a88db42f81
embedded commit: d4758f6ce3005ec8abe6ec3d25f4d5b0dbdcde18
```

The first archive-only run correctly showed that provenance tests require a Git
repository; all failures were `git rev-parse` failures caused by the missing
`.git` directory. The extracted tree was then initialized as an isolated test
repository without changing source files, and the same 162-test CUDA-hidden
suite passed on Linux. The rehearsal used `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; it launched no GPU process and did
not touch the active training checkout.

## Authorization and scientific decision

This hardening is not deployed into the active quality-bridge chain. It cannot
authorize export, release, promotion, another GPU diagnostic, full training, or
300K scaling. Current state remains:

```text
generation_advantage_proven=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
promotion_or_release_allowed=false
```
