# Generation release consumer source policy

Date: 2026-08-20

## Outcome

The terminal release boundary now states its source-retention threat model in
machine-readable form. Routine inference remains fail closed for the compact
deployment package without becoming dependent on archived 300K training
checkpoints or every upstream completion input.

This is release-integrity and reproducibility hardening. It does not create a
release-authorized model, change any training or sampling recipe, establish a
generation-quality advantage, or authorize a GPU stage.

## Threat model

The terminal completion audit is a content-addressed attestation produced only
after its profile-specific auditor has replayed the required training,
evaluation, gate, comparison, export, preflight, and smoke evidence. The release
receipt binds the physical audit bytes and reconstructs all cross-provenance
contracts from that immutable audit.

Routine inference must live-verify:

- the deterministic release receipt reconstruction;
- the bound terminal completion audit;
- the selected EMA inference artifact bytes;
- the adjacent artifact integrity sidecar; and
- the adjacent inference export manifest.

Routine inference deliberately does not reopen:

- the full source training checkpoint or its integrity sidecar;
- the training-authorization gate or capacity launch receipt;
- the final release gate and its nested source reports; or
- readiness, launch, deployment, and other upstream completion sources.

Those historical objects were verified before terminal publication and remain
bound by the audit, export manifest, artifact sidecar, and authorization
descriptors. Their later archival must not make the smaller EMA release artifact
unusable. A retained copy can still be re-audited separately, but it is not a
live dependency of every production inference request.

## Implementation

`cofitok.generation.release` now emits release receipt schema v2 with an exact
`consumer_source_policy` object. It records:

- `trust_model=content_addressed_terminal_snapshot_v1`;
- the four live-consumption evidence classes;
- the historical source classes that are not reopened; and
- that historical-source archival preserves receipt validity.

Receipt consumption reconstructs this object from code together with the audit
payload. Removing, adding, reordering, or weakening a policy field makes the
receipt differ from the completion audit reconstruction and is rejected before
`torch.load`. The verified policy is propagated through the completion
authorization metadata used by sessions, preflight, inference manifests,
progress, and reports.

## Regression coverage

The inference artifact regression now deletes the source training checkpoint,
its integrity sidecar, the full release gate, and all gate source reports after
receipt publication. The EMA artifact remains consumable because the compact
receipt/audit/artifact/sidecar/export-manifest package is intact. A separate
negative test mutates the frozen consumer policy and proves rejection before
artifact deserialization.

The focused release test file collected and passed `64/64` tests. The broader
completion/release selection collected and passed `184/184` tests across:

```text
tests/test_generation_inference_artifact.py
tests/test_generation_stability_completion_audit.py
tests/test_generation_capacity_full_posteval.py
tests/test_large_scale_generation_completion_audit.py
tests/test_generation_capacity_full_training_launch.py
tests/test_generation_full_launch_receipt.py
```

## Operational boundary

The change was developed only in the clean local worktree
`C:/qbfinalreleaseaudit`. It did not deploy to a remote checkout, access or
deserialize a production checkpoint, use the GPU, signal a process, modify the
active matched 100K queue, or alter any standing training authorization.
