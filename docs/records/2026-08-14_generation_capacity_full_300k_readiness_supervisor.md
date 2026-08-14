# 2026-08-14 Capacity-full 300K readiness supervisor

## Outcome

A decision-bound readiness artifact builder and GPU-idle supervisor were
implemented, validated at the exact deployed revision, and launched as a
single waiting controller. This stage does **not** launch training, resume the
capacity 100K checkpoints, launch the fresh 300K pair, promote a result, or
release anything.

The capacity 100K pair and the target full 300K pair are not exact-resume
compatible. Their resolved configurations differ in loss schedules, runtime
horizon/evaluation cadence, checkpoint retention, and optimizer scheduling.
Consequently:

- capacity 100K checkpoints are qualification evidence only;
- `capacity_100k_checkpoint_resume_allowed=false`;
- any selected matched 300K run must start freshly at step 0;
- a separately source-bound training launch receipt is required after the
  readiness artifact passes.

## Exact code identity

- Branch:
  `scale/generation-capacity-full-300k-readiness-supervisor-v1`
- Revision:
  `87b043339df073437e8130e20a032d367a2e1649`
- Tree:
  `53d034ffd597efa4cff3ea5564f3748d28c19fe9`
- Prerequisite decision revision:
  `70b7470a6ca8f43a544cf41dd988802dc5f8c9f1`
- Isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-full-readiness-87b0433/CoFiTok-internal`

Incremental bundle:

- Remote path:
  `/tmp/cofitok-capacity-full-readiness-87b0433.bundle`
- Bytes: `23,208`
- SHA256:
  `1b77355724074a5e96725e7063c300a806816f94c59b870eb4686045a638a1e4`
- Advertised ref count: one
- Advertised revision: exact deployed revision
- Required prerequisite: exact decision revision above

The bundle was verified against the prior isolated decision checkout. The
formal checkout intentionally does not contain the prerequisite object and was
not fetched or moved.

## Implemented boundary

The readiness runbook:

- physically replays the exact post-100K decision and all bound sources;
- requires the fresh matched full-training directories to be absent;
- uses the actual matched base-256 100K checkpoints only as byte-size
  references for storage planning;
- reserves 16 checkpoints, 116,640 256-KiB samples, 16 GiB additional space,
  and 64 GiB safety margin;
- benchmarks matched candidates `1x64,2x32,4x16,8x8,16x4` with `1x64` as the
  mandatory baseline, eight benchmark steps, and two warmup steps;
- rechecks that the GPU is idle immediately before any CUDA allocation;
- builds and physically replays the readiness report;
- never calls a full-training runbook.

The supervisor:

- waits for the exact decision waiter to finish;
- validates the standing authorization and decision-waiter deployment receipt;
- requires five consecutive idle-GPU polls;
- supervises only the readiness runbook;
- retries only bounded launch-race exit code 9;
- does not signal any process;
- records that it performed neither training nor a full-300K launch, even if a
  later readiness artifact permits a separate launch stage.

## Validation

Local validation:

- targeted changed-code tests after the final validator hardening: `10 passed`;
- readiness-core tests after the tamper-test addition: `4 passed`;
- earlier implementation-worktree capacity/quality-bridge regression:
  `208 passed`;
- full-readiness/runtime/storage regression: `39 passed`;
- runbook entrypoint contract: `2 passed`.

Exact deployed Linux revision validation:

- Python compilation: passed;
- both new runbooks: `bash -n` passed;
- targeted readiness/decision/waiter/runbook tests: `15 passed`;
- capacity/quality-bridge files: `208 passed`;
- full-readiness/runtime/storage files: `39 passed`;
- runbook entrypoint contract: `2 passed`, logged exit code `0`.

The deployment receipt originally summarized the local broad regression as
`208 passed`. That run was on the implementation worktree immediately before a
final validation-only hardening change, so it is not the authoritative
exact-revision broad replay. This distinction is preserved in an immutable
clarification sidecar; the exact deployed Linux `208 passed` replay is the
authoritative broad result.

## Deployment

- Supervisor PID: `264718`
- Nice level: `19`
- Status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_supervisor_status.json`
- Initial state: `waiting`
- Initial detail:
  `waiting_for_source_bound_capacity_full_readiness_decision`
- Target readiness report:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_v1/reports/capacity_full_300k_readiness.json`

Deployment receipt:

- Path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_supervisor_deployment_receipt.json`
- Bytes: `8,371`
- SHA256:
  `679b3de48432363bdd26c48e7c2b0d971613de143eb6d6e76b4def538a74c1c4`

Validation clarification:

- Path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_supervisor_deployment_validation_clarification.json`
- Bytes: `1,812`
- SHA256:
  `3149f2f4d2c97b035c9e3b275631e9cafca003c8c789744f355c327e07f61e86`

At deployment, the full target root was absent. The only GPU process was the
unrelated FieldScope PID `910099` using 2,256 MiB; it was not signaled or shared.
The readiness supervisor therefore remains upstream-blocked and does not run a
GPU benchmark yet.

## Formal checkout preservation

After deployment:

- formal HEAD:
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- porcelain entry count: `87`
- porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`

No fetch, checkout, merge, cleanup, or tracked-file modification was performed
in the formal checkout.
