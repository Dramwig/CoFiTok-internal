# Terminal 100K training-exposure binding waiter

Date: 2026-08-19

## Evidence gap

The active full-data quality bridge will eventually build a source-bound
`quality_bridge_result.json`, but that result carries the exposure plan from
preparation rather than an independently recomputed, dataset-normalized summary
of the final mutable training reports. A terminal quality comparison therefore
still needed a separate artifact proving that CoFiTok and `dense_identity`
actually reached the same 100K steps, images seen, effective batch, equivalent
epochs, Git identity, and checkpoint-bound evaluation stage.

## Implementation

Commit `e6f083d01062d594c32b1d595a77f0feb81a0ac4` on branch
`analysis/generation-terminal-exposure-binding-v1`, tree
`f5691656c0d66a8b4802e2c5f477c8cca0274143`, extends the existing exposure
builder and snapshot waiter without changing the active training runbook.

The terminal binding requires all of the following before it can complete:

1. Both training reports are exactly complete at 100K and have matched formal
   dataset identity, effective batch, images seen, and equivalent epochs.
2. The paired 100K milestone is replayed and its two checkpoint SHA256 values
   equal the final training reports.
3. The terminal quality result has the exact quality-bridge role, stage, Git,
   source set, non-authorizing boundary, and matched-training contract.
4. Every JSON source bound by the terminal result is reopened and checked by
   path, bytes, and SHA256. The logical terminal result is rebuilt from those
   sources and must be byte-equivalent as structured data.
5. The terminal method checkpoints equal both the training and 100K milestone
   checkpoints.
6. The active execution status must state that terminal evidence verification
   completed and must continue to deny full training, 300K, and promotion-gate
   authority.

Only then are the two training reports, preparation, 100K milestone, terminal
result, and execution status copied into one atomically published snapshot. An
existing snapshot is accepted only after full replay.

## Verification

- Focused local regression: 48 passed.
- Full local suite: 1,215 passed, 6 skipped, and 4 unrelated path failures.
  The four failures are the AAAI paper-layout tests resolving `../paper` to
  `D:/paper` from the isolated worktree; the required sibling paper checkout is
  absent there.
- Ruff lint: pass.
- Ruff format check: pass.
- Python compile: pass.
- Linux CUDA-hidden focused regression: 48 passed.
- Linux runbook `bash -n`: pass.

Bundle:

```text
D:/cofitok-bundles/terminal-exposure-binding-e6f083d-from-d8749a6.bundle
bytes: 20075
sha256: fd4b5dc3251a29657c0c58ee7d65d114b2a95421e4191293a9c22a00be4dfda2
prerequisite: d8749a6aea927545eb194f290a3b46017d7d99dd
advertised head: e6f083d01062d594c32b1d595a77f0feb81a0ac4
```

## Deployment

The exact clean Linux checkout is:

```text
/root/autodl-tmp/CoFiTok/checkouts/terminal-exposure-waiter-e6f083d
```

The waiter runs as PID `414533` with nice level 10, idle I/O priority,
`CUDA_VISIBLE_DEVICES=-1`, and one OMP/MKL thread. Its status path is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
training_exposure_terminal_100k_waiter_status.json
```

At deployment it was correctly waiting for the CoFiTok 100K report. The active
pair was still in the first dense 50K segment at step 25,950, with no health
issues or unrelated GPU process. The formal checkout remained at
`scale/generative-system@1ebcc152...` with clean tracked worktree and index.

The eventual immutable output is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
training_exposure_terminal_100k
```

This waiter cannot launch training, sampling, evaluation, 300K scaling, or a
release. It binds evidence only and cannot turn a failed or held quality screen
into a generation-quality claim.

