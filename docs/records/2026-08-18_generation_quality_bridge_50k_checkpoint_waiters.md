# 2026-08-18 quality-bridge 50K checkpoint waiters

## Purpose

The quality-bridge runbook calls the checkpoint verifier before each milestone,
but that runtime check does not itself preserve an independent physical-payload
audit report. Reproducible milestone evidence requires durable reports binding
the checkpoint bytes and SHA256 to its integrity sidecar, `latest.json`, Git,
dataset, runtime, and canonical metrics.

Two existing source-locked physical-integrity waiters were deployed for the
CoFiTok and dense 50K checkpoints.

## Locked implementation

```text
checkout: /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-checkpoint-waiter-172388f
revision: 172388fc4d873bb1001313f979442516ea7b5069
tree: d7e54a8ea9ff5b6c6a3eea342ec787605d0ebc50
source bytes: 14,973
source SHA256: eafc4e1f7b33f8b890883ac4878944a1d57f41aca02dc0483b9e82d1259fca8b
```

The waiters are bound to:

```text
training revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
training tree: 6cef27723196fd363379bca2e7b85b1678ebd777
training branch: scale/generation-stability-quality-bridge-100k
dataset identity: 6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659
runtime identity: d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
effective batch: 64
```

## CoFiTok 50K waiter

```text
PID: 822372
checkpoint step: 50,000
poll interval: 30 seconds
timeout: 604,800 seconds
initial status: waiting / checkpoint_missing
```

Target report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/checkpoint_audits/cofitok_checkpoint_step_00050000_physical_integrity_audit.json
```

Initial status identity:

```text
bytes: 1,170
SHA256: 7eec19ea104b861aed271e30ed943cbe0a26074c2383de3d74e4ce495637efc1
```

## Dense 50K waiter

```text
PID: 822373
checkpoint step: 50,000
poll interval: 30 seconds
timeout: 604,800 seconds
initial status: waiting / checkpoint_missing
```

Target report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/checkpoint_audits/dense_checkpoint_step_00050000_physical_integrity_audit.json
```

Initial status identity:

```text
bytes: 1,157
SHA256: eb82004307430b67b3dfb05e882c5b181399ce6e3c62838439c919b9fa914b0e
```

## Audit contract

After an exact 50K checkpoint becomes ready, each waiter verifies:

1. physical checkpoint bytes and SHA256 against the integrity sidecar;
2. exact 50K `latest.json` filename, step, byte count, SHA, sidecar, Git,
   dataset, and runtime binding;
3. exact training checkout revision, branch, tree, and tracked-clean state;
4. strict canonical metric ordering and `samples_seen == step * 64`;
5. the presence of the exact 50K metric row.

The resulting report embeds identities for the checkpoint payload, sidecar,
latest pointer, metrics snapshot, and target row.

## Resource boundary

Both waiters run with CUDA hidden, one OMP/MKL thread, `nice 10`, and idle I/O
priority. They do not read checkpoint payloads before the exact target becomes
ready. At the deployment snapshot CoFiTok had continued to step `31,250`, and
GPU compute still contained only trainer PID `619775` using `85,284 MiB`.

Physical hashing may overlap the runbook's own verifier or milestone sampling,
but idle I/O priority yields to those foreground operations. The waiters cannot
launch, restart, stop, or signal training; cannot authorize promotion or model
release; and cannot authorize 300K training.

Compact deployment evidence:

```text
artifacts/reports/generation/quality_bridge_50k_checkpoint_waiters_deployment_2026-08-18.json
bytes: 4,966
SHA256: a44d7d752d066a1abdb35438eb11d3352d0cbf9248c09180836bcbd6a6c5031c
Git blob OID: a8583e3d0979b04bad10bcc7f1ccefe22bc6c51a
```
