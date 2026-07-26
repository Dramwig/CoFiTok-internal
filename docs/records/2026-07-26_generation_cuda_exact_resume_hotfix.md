# CUDA exact-resume sampler hotfix

Date: 2026-07-26

## Incident

The frozen fixed-basis v3 CoFiTok run was intentionally paused after its
verified step-25,000 checkpoint. A user-authorized resume launched the
existing completion supervisor with `--resume auto`. The training process
exited before metrics reconciliation or an optimizer step:

```text
TypeError: RNG state must be a torch.ByteTensor
```

The checkpoint, integrity sidecar, model, EMA, optimizer, scheduler, global
RNG, dataset identity, and runtime identity remained unchanged. The canonical
metrics file still contained the seven stopped rows after step 25,000, and no
new checkpoint was written.

## Root cause

`load_training_checkpoint(..., map_location=device)` correctly restored model
and optimizer tensors to CUDA, but also remapped the CPU sampler generator
state to CUDA. `torch.Generator.set_state` requires a CPU `torch.uint8`
tensor. Global CPU and CUDA RNG restoration already normalized serialized
state tensors to CPU; `StatefulRandomSampler.load_state_dict` did not.

The failed child also exposed a separate lock-lifetime issue. The detached
pair monitor inherited the supervisor and pipeline lock descriptors, so an
otherwise valid bounded retry could not reacquire the pipeline lock.

## Fix

The hotfix is based directly on
`58d83bfce2770eab2565b8c89a5f9a06201a0c86` and changes only:

- sampler RNG restoration now validates `torch.uint8` and moves state to CPU;
- CUDA regression coverage reproduces a map-located sampler state;
- training accepts one explicit clean ancestor through
  `--resume-source-revision` and binds the source checkpoint identity and
  source-to-target transition into every later checkpoint and report;
- the frozen runtime selector accepts the same explicit compatible ancestor
  without rerunning or rewriting the original 64x1 benchmark, and writes a
  compatibility receipt;
- the 10% runbook permits only the fixed 58d source checkpoint, rejects any
  other revision, and closes inherited lock descriptors in the detached
  monitor.

The source checkpoint remains byte-identical:

```text
checkpoint: checkpoint_step_00025000.pt
bytes:      1,006,351,466
sha256:     5b056311d7651f3a222ce10b5bdd1a1652b4446f0a88f26185ef89ab09a7a542
revision:   58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

The resumed CoFiTok final checkpoint and the fresh dense baseline must both
use the hotfix target revision on the remote
`scale/generative-system` branch. The transition is operational
compatibility only; it does not change model structure, losses, data order,
optimizer math, diffusion math, or the matched training contract.

## Verification

Verification results are recorded after local and isolated remote tests. The
formal run must not resume until the CUDA regression passes on `pro6000`, all
runbooks pass syntax checks, the incremental bundle verifies from 58d, and
the deployment receipt binds the new target revision.
