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

## Linux rehearsal and dense 55K execution

The verifier implementation was committed and deployed only to an isolated
checkout:

```text
revision: da67abeb7a78064beb0b975f7b11f7d4ba5fc809
tree:     c0077bae46969fff9f7a5ba7148adbf93a6d1777
branch:   analysis/generation-checkpoint-audit-replay-v1
source SHA256:
a9a8cb46e0734ba6f2dadf5c7b62db7838d3a57ff202b6dcedcca565b51b0459
checkout:
/root/autodl-tmp/CoFiTok/checkouts/checkpoint-audit-replay-da67abe/CoFiTok-internal
```

The prerequisite-aware bundle required exact base
`172388fc4d873bb1001313f979442516ea7b5069`, advertised only the verifier
branch above, and had:

```text
bytes:  11,421
SHA256: 05392bff8ccf0899d177b77c0e762afa8063ce87a765a6a7a2c02e2b33c0a9b8
```

The isolated Linux checkout passed Python compilation, the original waiter plus
replay focused suite (`11/11`), and the complete pytest suite with exit code
zero under:

```text
CUDA_VISIBLE_DEVICES=
PYTHONPATH=src
```

The real dense 55K replay then ran with `ionice -c3`, `nice -n 19`, and CUDA
hidden. It did not deserialize a checkpoint, use the GPU, or signal training.
The authoritative result is:

```text
path:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/checkpoint_audits/dense_checkpoint_step_00055000_physical_integrity_audit_replay.json
bytes:  9,265
SHA256: 79c1f09e2f69c468edc242aea048ab51a8f103fb2ed098c08274aaee1e6b3cb8
status: pass
```

The replay physically rehashed the checkpoint payload twice and reproduced:

```text
checkpoint bytes:  1,010,735,510
checkpoint SHA256: 13b74b044ff69ac9190d59ff3412d776b63d2256a1c402b42cea3f0ab59521db
audit metrics bytes:  972,808
audit metrics SHA256: 986da342fe8a3809e1c7e4c236d2a0ec83c7cd4395d6e8896a4abe0c852018a5
audit metrics last step: 55,150
current observed last step during replay: 56,750
latest relation: exact_historical_pointer_still_current
```

Every non-authorizing boundary remained false, including sampling,
full-training, full-300K, promotion, and release authorization. Immediately
after execution the pair monitor remained `running`, stage
`dense_identity_training`, with `issues=[]`; the sole GPU compute process was
the existing dense trainer.

Matched validation through step 56K remained essentially identical rather than
showing a quality advantage: 56 events, CoFiTok/dense means
`0.0294013844 / 0.0293894343`, CoFiTok delta `+0.040661%`, Pearson
`0.9994107`, lower-event counts `26 / 30`, and last-10 delta `-0.011732%`.
These are optimization-space diagnostics only and do not alter the existing
conclusion that generation-quality superiority is unproven pending terminal
matched sampling.
