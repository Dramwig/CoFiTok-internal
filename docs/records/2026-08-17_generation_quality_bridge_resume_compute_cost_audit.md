# Full-data quality bridge resume-compute cost audit

Date: 2026-08-17

## Purpose

The active full-data ImageNet-256 100K quality bridge resumed exactly from the
trusted CoFiTok step-20K checkpoint after a failed attempt reached step 20,200.
Canonical metrics correctly discard the rolled-back rows, but the physical GPU
work already consumed by those rows must remain visible in the final matched
training-cost comparison.

This change is audit-only. It does not alter the model, checkpoint, training
trajectory, quality metrics, active trainer, promotion authority, release
authority, or any full-300K launch boundary.

## Physical recovery adjustment

Authoritative remote report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
recovery_incident_2026-08-17_pin_memory/resume_compute_adjustment.json
```

Bound lower-bound adjustment:

- orphaned physical compute: `560.6116147041321` seconds;
- orphaned optimizer steps: `200`;
- orphaned training images: `12,800`;
- resume checkpoint: step `20,000`;
- orphan archive SHA256:
  `6b3c4bca8ffa99e814ee77559d140f3a0db6cf7b78a8d78e58ef1b716e2f1cec`;
- adjustment report SHA256:
  `7e4c2362ee38b4b973034dbe492a8ac64044c17f805e6f3cd66b7421c48af740`.

The final physical lower-bound CoFiTok elapsed time is therefore:

```text
training_report.elapsed_seconds + 560.6116147041321
```

Throughput must be recomputed from this adjusted elapsed time. The canonical
metrics timeline remains unchanged and continues to describe the exact resumed
training trajectory.

## Implementation

Code commit:

```text
2b3c3975255f92cc9613f9990592f37ba2cfa968
tree 261187b87f85f8b0bf06cd6bac910c17dc0bc95b
branch fix/quality-bridge-ipc-recovery-v2
```

The implementation:

- discovers and verifies physical orphan-metrics archives;
- verifies canonical replacement-row continuity and samples-seen bindings;
- requires an adjustment whenever reconciled or physically discovered orphaned
  compute exists;
- changes the generation gate schema to `6` and the final comparison schema to
  `9`;
- propagates reported elapsed time, recovery adjustment, physical lower-bound
  elapsed time, corrected throughput, orphaned steps, and orphaned images through
  gate, comparison, and terminal completion audit;
- fails closed when an adjustment identity is omitted, changed, incomplete, or
  inconsistent with the physical sources.

## Validation

Windows isolated worktree:

- recovery/gate/comparison/completion targeted suite: `190 passed`;
- completion-audit suite: `86 passed`;
- inference-artifact suite: `25 passed`;
- full code suite excluding the canonical-layout paper test:
  `1,096 collected`, all passed with the existing expected skips;
- the four AAAI-27 layout tests passed separately from the canonical project
  directory;
- `compileall`: pass;
- `git diff --check`: pass.

Linux isolated rehearsal used `CUDA_VISIBLE_DEVICES=''` and the exact deployed
branch identity. All `215` affected tests passed, along with `compileall` and
`git diff --check`.

## Audit-only deployment

Persistent isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
recovery-cost-audit-2b3c397/CoFiTok-internal
```

Deployment receipt:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
recovery_incident_2026-08-17_pin_memory/
recovery_cost_audit_code_deployment_v1/deployment_receipt.json
```

Receipt identity:

```text
bytes: 7387
SHA256: 7c805de473633d4073e95facdb0beea3b0e84278e057e073026c404c4c649899
```

The receipt verifies:

- deployed commit/tree/branch are exact and tracked-clean;
- the formal checkout remains at
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` and tracked-clean;
- the active training checkout remains at
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a` and tracked-clean;
- trainer, runbook, and pair monitor remain active;
- no process signal was sent;
- the pair monitor remained `running/cofitok_training` with `issues=[]`;
- no promotion, release, full-training, or full-300K authorization was created.

At receipt time CoFiTok was at step `23,200`; dense training had not started.
