# 2026-08-18 capacity pipeline authority and launch boundary

## Outcome

The capacity control plane is deployed and waiting correctly, but no 250M
capacity-probe training or full 300K training has started.

The authoritative lineage observer is v2:

```text
PID: 765120
status: running
issues: []
current stage: quality_bridge_100k_recovery
detail: superseded_by_source_bound_exact_resume:existing_quality_bridge_execution_is_active
revision: 1ff6bb3db932ef9e43ddfafc067db779dab797d9
tree: bf0d2e8208ea3bce0ab04c7f1991ea17882a3ea2
```

Its deployment receipt is immutable:

```text
bytes: 12,867
SHA256: 925e62a420bb9b6f5d84e161cc31f62ae7def0f69ec8582794ccdd4066f9de9a
```

The recovery-supersession contract passed and binds the exact-resume path:

```text
bytes: 3,568
SHA256: 28318ff89c009be65899117bbb9d45f533bc116f63b6bd335e21463818d5f609
resume step: 20,000
trusted physical checkpoint step: 25,000
successor recovery PID: 630988
```

At the audit snapshot the active base-128 quality bridge had reached step
`31,450`; its pair monitor remained `running / cofitok_training` with no
issues. Dense had not started.

## Preserved v1 observer

The old v1 observer PID `12692` remains alive and its report still says:

```text
detail: lineage_health_failure:quality_bridge_100k_recovery
stage status: failed
failure: nonretryable_recovery_controller_exit_1:exit_1_is_not_retryable
issue: formal_checkout_snapshot_changed
```

This is preserved historical observer evidence. The v2 deployment receipt
explicitly records that v1 was alive and was not signaled or modified. V1 is
superseded by the source-bound v2 observer and must not be treated as the
current capacity-pipeline verdict.

## Exact 250M 10K probe state

The capacity output root exists only as a control-plane root. Current state:

```text
source-bound reports/preparation.json: absent
preparation waiter: waiting_for_quality_bridge_followup_decision
execution supervisor PID: 11853
execution supervisor: waiting_for_source_bound_capacity_preparation
attempt: 0
training child PID: absent
idle polls: 0 / 5
checkpoint payload count: 0
capacity run directories: absent
```

The file

```text
reports/execution_supervisor_launch_receipt.json
```

does exist, but its role is
`generation_capacity_probe_execution_supervisor_launch_receipt` and its status
is `active_waiting_for_source_bound_preparation`. It proves only that the
waiting supervisor was deployed. It is not an experiment-training launch
receipt and cannot be cited as evidence that the 250M probe started.

Static receipt identity:

```text
bytes: 4,401
SHA256: 9c04632adf19fa74120be5430220e8dd46635610924d68393106f4e6e0a6ae6d
```

## Full 300K boundary

The configured full output root is still absent:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_v1
```

Therefore the readiness artifact and training launch receipt under that root
are also absent.

Current readiness supervisor:

```text
PID: 12247
status: waiting
detail: waiting_for_source_bound_capacity_full_readiness_decision
decision: null
readiness: null
launch performed: false
idle polls: 0 / 5
```

Current training supervisor:

```text
PID: 12259
status: waiting
detail: waiting_for_passed_capacity_full_readiness
attempt: 0
child PID: null
training launch performed: false
training launch receipt: null
idle polls: 0 / 5
```

The future training stage is locked to:

```text
revision: d75dfea37ddcb504cfe8ecba25774ee36f522480
tree: e96059ab0e77942e716c23a69364fc6c8ad9048d
branch: scale/generation-capacity-full-300k-execution-v1
tracked dirty: false
```

Training-supervisor deployment receipt:

```text
bytes: 12,158
SHA256: 2ffb1518c14034f4879c5500f933768e7406c44b52bf0d66ac75b3c3b31696ce
```

Standing authorization remains active and content-addressed:

```text
exact text: 之后不要我授权你直接运行需要的实验
bytes: 865
SHA256: 5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
```

It permits automatic progression after the source-bound gates pass. It does
not bypass readiness, does not replace the separate exact training launch
receipt, and does not bypass the required five consecutive idle-GPU polls.

## Required progression chain

Before a 300K child can be spawned, the chain must still:

1. finish and evaluate the active matched full-data quality bridge;
2. produce its source-bound follow-up decision;
3. prepare and complete the exact matched 250M 10K probe;
4. pass the source-replayed capacity scaling decision;
5. complete and bind the configured capacity 50K and 100K stages;
6. build and pass source-bound full-300K readiness;
7. create a separate receipt bound to the exact revision, stage, configs,
   source checkpoints, and output root;
8. observe five consecutive idle-GPU polls.

Only then may the waiting training supervisor launch. Nothing in this audit
authorizes promotion, release, or a scientific quality claim.

Compact evidence:

```text
artifacts/reports/generation/capacity_pipeline_authority_launch_boundary_2026-08-18.json
bytes: 9,397
SHA256: fb762f0579e1e91833017d7017766f9ac8939880b61735608a49f4f46dc83cb4
```
