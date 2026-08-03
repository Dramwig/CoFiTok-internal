# Readiness-compatible schema-v4 full control target

Date: 2026-08-03

## Outcome

The active full-readiness waiter remains bound to the clean source checkout
`5dd3488ac9b30274f4960195e252cc9fdb161002`. The earlier compatible control
target `9019dd3f0f504e799c03496ec41a653b61deaa02` only consumed the schema-v3
frozen supplemental contract, so it could not honestly consume the later
mandatory frozen scaling class-fidelity qualification.

A new local-only control target now exists:

```text
branch  scale/generation-stability-full-control-quality-v4-27ed
commit  cac762ee147f645185259f45fdf212e1872838cb
parent  9019dd3f0f504e799c03496ec41a653b61deaa02
```

It ports only two control-plane boundaries onto the readiness-compatible
target:

- the `b45c490` launch-time replay of the current deployment checkout and
  runtime environment;
- the `d1ce033` schema-v4 requirement that frozen scaling class fidelity pass
  and remain source-bound to the original promotion gate.

The standalone verifier is byte-equivalent to the audited `d1ce033` verifier.
It rehashes the qualification, both raw reports, the promotion gate, the
classifier identity, runtime fingerprint, sampler/evaluator Git split,
checkpoint and sample-set digests, metric arithmetic, and every threshold
check. It always returns `full_training_launch_allowed=false`; it is a required
input to a later launch receipt, never an authorization by itself.

No class-fidelity sampler/evaluator implementation, training module, training
configuration, or post-`monitor_report_passes()` execution logic was imported
into the target.

## Exact compatibility evidence

The committed target was replayed against the source revision through
`build_generation_full_readiness_bridge.py`:

- `5dd3488...` is an ancestor of `cac762e...`;
- all 66 training-critical Git blobs are byte-identical;
- the full-training execution suffix is byte-identical;
- execution suffix SHA256 remains
  `1a4e559c3c8828ceee9907f45e2de2d4513a77f37e4458eb50755ee53ab6382c`;
- source and normalized-target preamble SHA256 both equal
  `3c196801c05153da13501c265da8480714f2ef119b686db8dbc3e153558d2d07`;
- the controlled preamble now requires the readiness bridge, frozen stability
  supplemental, frozen scaling class fidelity, schema-v4 launch receipt, and
  the existing `116,640`-sample completion reserve;
- `src/cofitok/generation_gate.py` and
  `src/cofitok/generation_gate_sources.py` have the same Git blobs as frozen
  post-evaluation revision `c1efb12...`, so the standalone verifier replays the
  original schema-v2 scaling gate without changing trainer semantics.

The complete 66-entry manifest and runbook proof are in:

```text
artifacts/reports/generation/readiness_compatible_quality_control_target_v4_2026-08-03/compatibility_manifest.json
```

## Verification

Local short sibling layout:

- focused control tests: `50 passed`;
- full suite: `877 collected / 873 passed / 4 skipped / 0 failed`;
- modified scripts and standalone verifier passed `compileall`;
- `git diff --check` passed.

Exact Linux isolated checkout on `pro6000`:

- Git: clean
  `scale/generation-stability-full-control-quality-v4-27ed@cac762e...`;
- full suite: `877 tests / 875 passed / 2 skipped / 0 failed / 0 errors`;
- JUnit runtime: `124.566` seconds;
- all `100/100` runbooks passed `bash -n`;
- the verifier, launch receipt builder, readiness bridge builder, and
  post-training supervisor CLI help contracts passed;
- tests ran with an empty `CUDA_VISIBLE_DEVICES` and did not touch the active
  trainer.

## Incremental bundle rehearsal

An ephemeral prerequisite-aware bundle was created from the active readiness
source through the new target:

```text
source prerequisite  5dd3488ac9b30274f4960195e252cc9fdb161002
advertised head       cac762ee147f645185259f45fdf212e1872838cb
advertised refs       1
bundle bytes          77,285
bundle SHA256         1da88d84baeca066daa9a6d9a0886f5613b378145b43cbe9c61505aaba69ebfd
incremental commits   10
incremental objects   147
```

Local and remote `git bundle verify` passed. Remote verification used the
clean authoritative `5dd3488...` checkout only as the prerequisite repository;
no fetch or merge entered that checkout. The bundle was fetched only into an
isolated `/tmp` clone for Linux tests. Before and after, the authoritative
checkout retained the same HEAD, branch, and empty tracked status. The only GPU
compute process remained the active dense trainer PID `541878` at `77,970 MiB`.

The remote bundle, remote isolated checkout, JUnit file, and local temporary
bundle were deleted after evidence capture. The exact machine-readable receipt
is:

```text
artifacts/reports/generation/readiness_compatible_quality_control_target_v4_2026-08-03/incremental_bundle_linux_rehearsal_receipt.json
```

## Boundary

The target remains local-only. Its objects were not fetched, merged, deployed,
or executed in any authoritative remote checkout. No deployment receipt,
readiness bridge, launch receipt, or full-training authorization was created.
Full 300K remains forbidden.

The active dense 50K run, frozen post-evaluation, frozen supplemental, and
frozen class-fidelity waiter must finish first. Even if all quality evidence
passes, a later explicit deployment/bridge/receipt sequence and separate user
authority are still required before any full-training launch.
