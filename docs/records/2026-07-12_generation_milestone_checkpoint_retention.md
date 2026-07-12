# Generation milestone checkpoint retention (2026-07-12)

## Problem

The full ImageNet-256 queue evaluates CoFiTok and dense at 50K, 100K, 200K,
and 300K optimizer steps. The previous rolling policy kept only the latest
three checkpoints. Although each milestone report bound the evaluated bytes by
SHA256, later training would delete the underlying 50K/100K/200K weights and
make those diagnostics impossible to rerun.

## Policy

`RuntimeConfig.protected_checkpoint_steps` defines checkpoints that pruning may
never remove. Full CoFiTok and dense configurations protect:

```text
50,000; 100,000; 200,000; 300,000
```

`prune_checkpoints` retains the union of these protected steps and the newest
three ordinary recovery checkpoints. It removes the integrity sidecar exactly
when it removes an unprotected checkpoint. Config validation requires the
protected list to be sorted, unique, within the run, and aligned with checkpoint
cadence (the true final step is also valid).

At the end of the matched full queue, the read-only training auditor requires
all four protected checkpoints for both methods. A reached protected step that
is missing marks the run invalid; future protected steps are not required from
an in-progress run.

## Storage

The live 10% CoFiTok checkpoint is about 1.01 GB. Retaining four milestones plus
the rolling recovery tail for each full method is expected to consume roughly
low tens of GB, well below the observed project-disk headroom. Checkpoints and
sample outputs remain under `/root/autodl-tmp/CoFiTok/checkpoints/generation/`
and are never committed to Git.

## Verification

Tests verify the protected/recent set union, paired sidecar pruning, full-config
parity, missing-reached milestone failure, and future-milestone tolerance. The
full runbook emits final CoFiTok and dense training audits with required steps
50K/100K/200K/300K before post-evaluation can proceed.
