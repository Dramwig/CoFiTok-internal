# Capacity-full fresh matched 300K training supervisor

Date: 2026-08-14

## Purpose and boundary

This layer closes the post-readiness execution gap for the capacity-selected
250M ImageNet-256 path.  It can launch a fresh matched CoFiTok/dense-identity
300K experiment only after the exact capacity-full readiness artifact passes.
It does not reinterpret or replace the frozen failed scaling promotion gate.

The training authorization is deliberately identified as
`capacity_full_experimental / authorize_fresh_matched_300k_training` and is
bound into every checkpoint payload and integrity sidecar.  Its fixed boundary
states that it is not a quality-promotion gate, does not authorize a formal
generation claim or release, and requires a separate final quality gate.

The source capacity checkpoints are storage-size evidence only.  They cannot
be used to resume this stage.  CoFiTok and dense identity must both initialize
fresh at step 0; exact resume is allowed only from checkpoints written under
the new target configs and output root.

## Exact code identity

- branch: `scale/generation-capacity-full-300k-execution-v1`
- revision: `d75dfea37ddcb504cfe8ecba25774ee36f522480`
- tree: `e96059ab0e77942e716c23a69364fc6c8ad9048d`
- readiness prerequisite: `87b043339df073437e8130e20a032d367a2e1649`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-full-execution-d75dfea/CoFiTok-internal`
- incremental bundle:
  `/tmp/cofitok-capacity-full-300k-execution-d75dfea.bundle`
- bundle bytes: `29,894`
- bundle SHA256:
  `71e89c0ec46538ec7bcf8a4da298cb9406d63837361c842765429b5bfc65991f`

The exact readiness-to-training diff contains only the deployment record,
authorization/checkpoint metadata, new orchestration, and tests.  Target
configs and model/data/diffusion compute paths are byte-identical to the
runtime-qualified readiness revision.

## Execution contract

- output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_v1`
- runs: `cofitok` and `dense_identity`
- dataset: `imagenet_256`
- configured horizon: `300,000` steps per method
- effective batch: `64`, using the readiness-selected micro-batch and
  accumulation pair
- paired milestones: `50K`, `100K`, `200K`, `300K`
- protected milestone checkpoints and integrity sidecars are mandatory
- each milestone runs the existing matched 2,048-sample DDIM-50 trend
  evaluation and checkpoint diagnostic
- the final training-pair validator consumes the same experimental training
  receipt for both methods

The supervisor waits for the exact terminal readiness status, physically
replays every readiness and deployment source, builds one immutable launch
receipt, confirms five consecutive idle GPU polls, and launches only the new
runbook.  It has no process-signaling capability.  Retryable runbook exits can
resume only from the target lineage.

## Validation

Exact Linux revision, with `CUDA_VISIBLE_DEVICES=-1` for tests:

- Python compile: passed
- both new runbooks: `bash -n` passed
- capacity chain: `173 passed`
- quality bridge: `28 passed`
- authorization, inference, training-pair, and completion audit: `151 passed`
- new direct entrypoint help replay: `3 passed`

Local validation:

- new launch/authorization tests: `4 passed`
- gate, pair, completion, and inference compatibility: `44 passed`
- large completion audit: `82 passed`
- complete runbook entrypoint contract: `2 passed`

A real short-timeout waiting-state rehearsal returned its expected exit code
`1` with `training_launch_performed=false`,
`formal_generation_completion_claimed=false`, and no target output root.

For audit transparency, two monolithic Linux validation attempts exceeded the
client's 604-second command limit and produced no usable result.  They are not
counted as evidence.  The residual CPU-only entrypoint pytest process was
precisely terminated, and the source-scoped split suites plus exact new
entrypoint replays above were used instead.

## Deployment

- supervisor PID: `296091`
- current status: `waiting`
- current detail: `waiting_for_passed_capacity_full_readiness`
- status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_training_supervisor_status.json`
- deployment receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_training_supervisor_deployment_receipt.json`
- deployment receipt bytes: `12,158`
- deployment receipt SHA256:
  `2ffb1518c14034f4879c5500f933768e7406c44b52bf0d66ac75b3c3b31696ce`

At deployment, the full target root remained absent, no training receipt or
training state existed, and the sole GPU process was unrelated FieldScope PID
`910099`.  The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` with the preserved 87-entry
porcelain snapshot SHA256
`a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.
