# Capacity control-process fail-closed relaunch

Date: 2026-08-15

## Outcome

The quality-bridge-to-capacity control chain now has a source-bound recovery
executor for a future host/process restart. It may relaunch CPU-only waiters,
supervisors, and the lineage observer only after an independent readiness
assessment proves that every process signature in the immutable 17-process
snapshot is absent.

The production assessment on 2026-08-15 returned `not_ready` because all 17
original process signatures were still alive. It therefore wrote no approval,
launched no process, sent no signal, made no GPU query/allocation, and did not
modify the formal checkout.

## Recovery contract

The recovery flow is split into three separately recorded roles:

1. readiness re-verifies the immutable process snapshot, the standing
   authorization, exact implementation Git state, and the complete formal
   checkout fingerprint before scanning `/proc` by absolute CWD plus resolved
   entrypoint;
2. approval is issued only when all selected signatures are absent, there are
   no invalid/hold/failure status rows, and at least one active or missing stage
   needs recovery;
3. execution re-runs the complete readiness assessment after approval, then
   launches only the selected CPU control processes as detached sessions with
   `/dev/null` stdin and append-only stdout/stderr.

Stages whose status is already successful are skipped. A hold or failure is a
terminal result and cannot receive a relaunch approval. If a multi-process
relaunch fails part way through, rollback targets only the process groups
created by that execution; unrelated or pre-existing processes are never
signaled.

The production runbook is inert unless
`CONTROL_PROCESS_RELAUNCH_ALLOWED=true` is supplied explicitly. This switch
does not replace the source/readiness/standing-authorization checks. The
recovered supervisors also retain their own existing stage authorization and
GPU-idle gates, so relaunching the control plane does not directly authorize or
launch training, sampling, evaluation, promotion, export, or release.

## Implementation identity

- branch: `scale/generation-capacity-control-process-relaunch-v1`;
- revision: `7836359feb823fd93840255f6aeae5478008bb1c`;
- tree: `490ee96a290786d22c07847f371cd3b33b6e17c4`;
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-relaunch-7836359/CoFiTok-internal`.

The incremental deployment bundle is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_process_relaunch_v1/deployment/cofitok-control-process-relaunch-7836359.bundle`;
- bytes: `17,109`;
- SHA256:
  `6310f46fdb35d1aef733644aa7922bd13599cb90ef374910e286eaf495f30983`;
- prerequisite: `4eb041d6d385df905a120fba1e5d0fcab575d170`;
- advertised head: `7836359feb823fd93840255f6aeae5478008bb1c`.

## Production fail-closed assessment

The immutable inputs were:

- process manifest: `106,174` bytes,
  SHA256 `455ddc919802c697afa4a80d43e70d79fc3d72057c8a194460772cb15f1c08c3`;
- standing authorization: `865` bytes,
  SHA256 `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`;
- expected process count: `17`.

The resulting report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_process_relaunch_readiness_live_v1/readiness.json`;
- bytes: `20,485`;
- SHA256:
  `36eeb3de7996338fafc53d72b73d6090dcf18b9a63e623ef1386b15e0b6abd5f`;
- status/detail: `not_ready / matching_control_processes_are_still_alive`;
- live matches: `17/17`;
- `approval_allowed=false`;
- selected active stages: `17`;
- successful skipped stages: `0`;
- effects: zero launches, signals, GPU operations, and formal-checkout
  modifications.

The source-only snapshot verification embedded in that report passed all 17
entrypoint and checkout identities before the live scan.

## Validation

- focused readiness/execution safety suite: `6 passed` locally;
- process snapshot, static continuity, pipeline lineage, and runbook syntax
  compatibility: `30 passed` locally;
- complete runbook CLI/entrypoint contract: `2 passed` locally;
- the same focused Linux compatibility set: `30 passed` with CUDA hidden;
- Python compilation, `bash -n`, `git diff --check`, exact revision/tree, and
  clean tracked isolated checkout: passed.

The negative tests cover live-process refusal, missing/active selection,
successful-stage skipping, terminal hold refusal, approval-bound execution,
partial-launch rollback, and the explicit production switch.

## Preserved live state

At the final audit, the recovery supervisor remained PID `155032`, status
`waiting`, attempt `0`, idle polls `0/5`, with detail
`initial_waiter_disappeared_before_launch:waiting_for_gpu_idle`. The lineage
observer remained PID `319645`, status `waiting`, with no issues. The only GPU
process was unrelated FieldScope PID `910099` at approximately `2,256 MiB`; it
was neither shared nor signaled.

The formal checkout remained exactly:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`;
- tree: `659fa94726c4aec0afef49904f82b828bb62872b`;
- branch: `scale/generative-system`;
- porcelain count: `87`;
- porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

The existing recovery supervisor will continue waiting and will launch the
already-authorized quality bridge only after five consecutive GPU-idle polls.
No additional user authorization is required for that experiment chain.
