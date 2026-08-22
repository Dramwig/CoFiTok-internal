# Generation checkpoint integrity-audit replay

Date: 2026-08-22

## Motivation

The source-bound checkpoint waiter writes a strong physical-integrity report at
the moment a milestone is the active `latest.json` target. Training then keeps
appending `train_metrics.jsonl`, and later checkpoints atomically advance
`latest.json`. A verifier that simply compares those mutable files with their
old whole-file identities can no longer replay the valid historical audit.

`scripts/verify_generation_checkpoint_integrity_audit_replay.py` adds a second,
independent verification path. It does not call the original waiter's audit
function. It instead:

- rehashes the checkpoint payload and integrity sidecar;
- rechecks exact Git, dataset, runtime, step, bytes, and SHA bindings;
- reconstructs the historical `latest.json` byte identity from the embedded
  canonical JSON snapshot;
- accepts either the exact historical latest pointer or a strictly newer,
  metadata-consistent pointer;
- reconstructs the exact metrics byte prefix through the audit's recorded last
  byte count and SHA, even when new metrics have been appended;
- validates strict step ordering, finite floats, and
  `samples_seen == step * effective_batch` over the observed metrics;
- samples only a complete-line current JSONL prefix, so an in-flight append
  cannot invalidate an already proven historical prefix;
- rehashes the original waiter source, verifies its exact clean Git checkout,
  and replays the waiter-status/report binding;
- requires canonical run-directory, checkpoint, sidecar, `latest.json`, and
  `train_metrics.jsonl` paths and stable sidecar parsing;
- rehashes the report, status, auditor source, auditor checkout, and training
  checkout again after the physical checkpoint verification to close the
  replay-time mutation window;
- rehashes the checkpoint payload and stably reparses its integrity sidecar a
  second time before publishing the receipt;
- writes a source- and Git-bound replay receipt.

The replay remains read-only and permanently non-authorizing. It is checkpoint
integrity evidence only, not generation-quality, promotion, release, full
training, or 300K authorization evidence.

## Initial target

The first real replay target is the post-restart dense 55K checkpoint in:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

The original audit report is:

```text
reports/checkpoint_audits/dense_checkpoint_step_00055000_physical_integrity_audit.json
```

Its SHA256 is:

```text
8e6920575e6f3259fc52946f0f1b0723614b36c720d43602118e6d21f027c1f4
```

The same verifier is intended for the already queued physical 90K, 95K, and
100K checkpoint audit reports.

## Local validation

Validation was run from the isolated
`analysis/generation-checkpoint-audit-replay-v1` worktree with pytest temporary
files redirected to `D:` because the local `C:` volume had insufficient free
space:

- replay verifier focused tests: pass;
- original checkpoint waiter plus replay verifier: `11/11` pass;
- checkpoint, exact-resume, metrics-resume, retention, migration, and training
  progress regression group: pass;
- complete Windows suite: `1,088 passed / 6 skipped / 1 failed`.

The sole complete-suite failure is the pre-existing deployment-receipt test for
`run_generation_quality_bridge_ipc_recovery_supervisor_v2.py`: the Windows
working-tree file is CRLF-expanded while the immutable receipt records the LF
Linux deployment bytes. The file is tracked-clean and unrelated to this
three-file change. The authoritative validation for this verifier remains the
clean Linux CUDA-hidden rehearsal, where Git checkout bytes use the recorded LF
form.
