# Capacity-full 300K postprocessing supervisors

Date: 2026-08-15

## Purpose and boundary

This revision closes the automatic path after the source-bound fresh matched
300K training pair. It adds two independent waiting controllers:

- the post-evaluation supervisor may run the exact matched 50K DDIM-250
  protocol only after the training supervisor reports an exact pass;
- the finalization supervisor may export inference artifacts, run the
  completion audit, build the release receipt, and claim completion only after
  the post-evaluation supervisor records a passing final quality gate.

A held final gate terminates the chain without export. Neither controller can
launch training, signal another process, modify an unrelated GPU process, or
reinterpret the experimental training receipt as a quality-promotion gate.

The frozen failed scaling gate remains unchanged and is not replaced. The
capacity-full lineage uses the separate source profile `capacity_full` and the
training authorization identity
`capacity_full_experimental / authorize_fresh_matched_300k_training`.

## Exact code identity

- branch: `scale/generation-capacity-full-postprocessing-v1`
- revision: `5e04dbe488ee166d2756bc99c71ee5fffbe0bd83`
- tree: `29d4e24f5ae9c294401902d554a1f2388a61ef64`
- training prerequisite:
  `d75dfea37ddcb504cfe8ecba25774ee36f522480`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-full-postprocessing-5e04dbe-v2/CoFiTok-internal`
- incremental bundle:
  `/tmp/cofitok-capacity-full-postprocessing-5e04dbe.bundle`
- bundle bytes: `57,876`
- bundle SHA256:
  `be6d1d5b548db6fe694d52bd1fd19a2f6e47bddaff8466bbef71d7e39b16fbb1`

The bundle advertises only the postprocessing branch at the exact revision
above and requires the exact training prerequisite. The formal checkout was
not fetched, merged, or moved.

## Post-evaluation contract

The capacity wrapper reuses the locked stability-full formal protocol with
parameterized output identities:

- runs: `cofitok` and `dense_identity`;
- 50,000 generated samples per method;
- DDIM-250, EMA weights, bf16, CFG 1.5;
- exact checkpoint evaluation, rollout-stability qualification,
  FID/IS/precision/recall, class-fidelity qualification, and prefix visual
  audit;
- final schema-v5 full gate and source-bound strong-baseline comparison.

The supervisor physically replays the training launch receipt and matched
training reports, requires the exact completed pair monitor, checks five
consecutive idle-GPU polls, and supervises only the formal post-evaluation
runbook. A valid quality hold is recorded as `hold`, not as a pass.

## Finalization contract

The finalization supervisor validates the post-evaluation deployment, terminal
status, final gate, comparison, and training authorization before acting. It
waits for five idle-GPU polls and then runs, in order:

1. release-authorized CoFiTok and dense EMA inference exports plus preflight
   and smoke inference;
2. the fail-closed capacity-full completion audit;
3. the release receipt bound to the passing completion audit and both
   inference artifacts.

The completion profile is `capacity_full_generation_system_v1`. Its audit
replays training, progress, milestone, formal 50K sample and real-set files,
runtime selection, visual diagnostics, final gate provenance, comparison, and
both exported artifacts. A formal completion claim is emitted only after that
audit and release receipt both verify.

## Validation

Local validation:

- capacity postprocessing and workspace paths: `19 passed`;
- gate, comparison, export, completion, and runbook compatibility:
  `212 passed`;
- capacity readiness/training/postprocessing chain: `40 passed`;
- complete runbook entrypoint contract: `2 passed`;
- Python compile, `bash -n`, and `git diff --check`: passed.

Exact Linux isolated revision with CUDA hidden:

- source-scoped capacity, gate, comparison, export, completion, syntax, and
  entrypoint suite: `254 passed`;
- Python compile: passed;
- all changed and new runbooks: `bash -n` passed.

Both controllers also completed real short-timeout waiting-state rehearsals.
Each returned the expected timeout status with its launch flag false, no
inference export or completion claim, and no capacity-full target root.

## Deployment

Post-evaluation supervisor:

- PID: `310354`
- initial status: `waiting`
- detail: `waiting_for_passed_capacity_full_training`
- deployment receipt bytes: `7,985`
- deployment receipt SHA256:
  `1dd2326555fce015be4c28063edc8f34d45a66c8e311555f929273d9755b6e1c`

Finalization supervisor:

- PID: `310989`
- initial status: `waiting`
- detail: `waiting_for_capacity_full_final_gate`
- deployment receipt bytes: `8,735`
- deployment receipt SHA256:
  `5948f31da3ea6b4614e893ccf2cab79297e56a9d39eaed9deaf246ebc89635a5`

Both deployment receipts are under:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/`

At the final deployment snapshot (`2026-08-15 00:14:44 CST`), the upstream
readiness supervisor and training supervisor were also alive and waiting. The
capacity-full output root, training launch receipt, and export root were all
absent. The sole GPU process was unrelated FieldScope PID `910099` using
approximately 2,256 MiB, and `/root/autodl-tmp` had `213,927,219,200` free
bytes.

The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`, with 87 porcelain entries and preserved snapshot
SHA256
`a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.
