# Generation checkpoint retention read-only audit (2026-08-01)

## Purpose

The large-capacity completion plan requires `180,880,415,360` free bytes. The
live filesystem had only about 6.58 GB of headroom above that requirement, while
the historical stability probes occupied about 54 GB. This change adds a strict
read-only inventory before any archive or deletion decision.

The audit never moves, truncates, deletes, or loads a checkpoint. It scans text
evidence, validates integrity sidecars, physically hashes checkpoint bytes when
requested, and classifies each checkpoint as `required`,
`candidate_for_archive`, or `indeterminate`.

## Fail-closed policy

- The latest checkpoint, configured protected steps, and authoritative evidence
  references are `required`.
- Operational monitor references, ambiguous basename-only references, incomplete
  run state, missing physical hashes, or invalid integrity remain
  `indeterminate`.
- `candidate_for_archive` is advisory only. Reports always set
  `archive_or_deletion_authorized=false` and `currently_reclaimable_bytes=0`.
- The runway check never counts unapproved candidate bytes as current capacity.
- Validation rebuilds the inventory from live sources and physically rehashes
  checkpoint bytes when the source report did so.

Implementation:

```text
src/cofitok/checkpoint_retention.py
scripts/build_generation_checkpoint_retention_inventory.py
scripts/validate_generation_checkpoint_retention_inventory.py
scripts/check_generation_retention_runway.py
artifacts/runbooks/generation_checkpoint_retention_readonly_audit.sh
```

## Real probe result

The audit ran at idle I/O priority against:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
```

Result:

```text
checkpoint count: 54
checkpoint bytes: 54,336,513,724
integrity valid: 54/54
required: 33 checkpoints / 33,205,581,950 bytes
indeterminate: 21 checkpoints / 21,130,931,774 bytes
candidate_for_archive: 0 checkpoints / 0 bytes
currently_reclaimable_bytes: 0
archive_readiness: blocked
```

There are 100 unresolved reference occurrences. They are not missing checkpoint
files. They are legacy basename-only references such as
`checkpoint_step_00000500.pt` or `checkpoint_step_00003750.pt` that cannot be
uniquely assigned across multiple runs retaining the same filename. The affected
checkpoints therefore remain `indeterminate`.

The dynamic runway result was:

```text
free bytes: 187,463,581,696
required free bytes: 180,880,415,360
current headroom: 6,583,166,336
status: pass
potential archive bytes counted as current capacity: false
```

This pass is narrow and is not full-training authorization. No historical
checkpoint is approved for archive or deletion by this result.

## Verification

- Local full suite: passed with four expected skips.
- Linux isolated suite: passed with two expected skips.
- New runbook: `bash -n` passed in the Linux isolated checkout.
- Real inventory build and independent replay produced identical summaries.

The active stability 50K queue and readiness waiter were not changed. During
the final audit snapshot CoFiTok was at step 22,900/50,000, dense was at 0, and
the pair monitor reported no issues.
