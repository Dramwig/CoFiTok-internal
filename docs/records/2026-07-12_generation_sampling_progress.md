# Atomic generation sampling progress (2026-07-12)

## Motivation

Formal 10K and 50K DDIM sampling can run for many hours. PNG-level resume was
already deterministic, atomic, and batch-size invariant, but the only live
progress was console output. A monitor could not distinguish active, failed,
or completed sampling, estimate remaining time, or verify that a progress file
belonged to the immutable sampling protocol.

## Progress contract

Every `generate_samples.py` run now maintains an atomic
`sampling_progress.json` bound to the SHA256 of `sampling_manifest.json`. It
records:

- manifest SHA, requested range, prefix budgets, and total sample count;
- completed samples/fraction and last completed global index;
- invocation number, invocation and cumulative elapsed seconds;
- samples/second, ETA, hostname, and PID;
- running/completed/failed state and structured failure details;
- final per-budget sample counts and sample-set SHA256 values.

The file is updated after every completed batch. A resumed invocation validates
the immutable identity and accumulates elapsed time instead of resetting it.
Exceptions publish `failed` with the last durable completed count. Abrupt
process death leaves the last atomic `running` update; rerunning with the same
manifest resumes from valid PNG batches.

`sampling_report.json` is schema version 6 and includes the manifest SHA,
progress path, invocation elapsed time, and cumulative elapsed time. It is
written before progress transitions to `completed`, so a crash cannot expose a
completed progress state without a sampling report.

## Formal evaluation boundary

`evaluate_generation_metrics.py` independently recomputes the manifest SHA and
requires progress to:

- resolve to the same sample-run directory;
- have `status=completed`;
- bind the same manifest SHA and prefix budgets;
- report exactly the requested sample count;
- contain sample-set digests identical to the sampling report.

The promotion/final gate also requires completed progress for both CoFiTok and
dense. FID cannot be published from a merely present PNG directory or stale
running/failed progress file.

The gate additionally requires finite positive cumulative sampling time. The
final large-scale comparison publishes sampling elapsed time, throughput, and
invocation count alongside training cost and generation quality.

The terminal completion audit does not rely only on these historical JSON
claims. It re-enumerates the exact `000000.png` through `049999.png` set for
each formal method, rejects symlinks or extra/missing entries, recomputes the
sample-set SHA256 from the physical bytes, and reopens the sampling report,
immutable manifest, and completed progress files. Their schemas, hashes,
checkpoint/runtime/Git identity, selected output directory, prefix budget, and
sample-set digests must still agree before completion can pass.

## Verification

Unit tests cover atomic JSON publication, elapsed-time accumulation, manifest
mismatch, completion requirements, and structured failures. A local 16x16 CPU
integration generated four EMA samples and reran the same command with
`--resume`; the second invocation produced:

```text
status: completed
invocation: 2
completed: 4
manifest_same: true
sample_sha_same: true
```

Formal sampling remains server-only.
