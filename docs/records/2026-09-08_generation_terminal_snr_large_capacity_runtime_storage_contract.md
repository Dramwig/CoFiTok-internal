# Terminal-SNR large-capacity runtime and storage contracts (2026-09-08)

## Outcome

The fresh endpoint-0.975 base-256 / 300K path now has fail-closed validators
for the two measured prelaunch reports that follow matched-pair validation:

- a shared real-forward training-runtime selection; and
- a full-stage storage-capacity preflight.

These validators do not collect evidence or authorize execution.  They define
the exact reports that a later source-bound execution authorization may
consume after the terminal-SNR frozen confirmation passes.

## Runtime selection

The measured candidate set is fixed to effective batch 64:

- `1x64` (required conservative feasibility baseline);
- `2x32`;
- `4x16`.

Every candidate benchmarks both CoFiTok and dense for eight optimizer steps
after two warmup steps.  Selection minimizes the slower method's measured
optimizer-step time subject to at most 90% peak device-memory use.  The
validator recomputes runtime-environment and ImageNet-256 dataset identities,
per-method timing and memory arithmetic, candidate eligibility, the selected
candidate, baseline speedup, config hashes, fixed run layout, clean Git
binding, and the freeze-on-training-state lock.  Benchmarks must remain
outside the future result root.

## Storage reserve

The storage contract reserves:

- 14 large checkpoints: eight protected 50K/100K/200K/300K artifacts plus
  three rolling recovery checkpoints per method;
- a 4.25x multiplier over the measured base-128 checkpoint for base-256
  payloads and schema overhead;
- 116,384 generated images: both formal 50K sets plus every matched
  2,048-sample milestone evaluation;
- 256 KiB per generated image, 32 GiB additional working space, and a
  separate 128 GiB safety margin.

The validator independently recomputes all byte arithmetic, filesystem usage,
required capacity, and remaining headroom while binding the report to the
exact clean execution Git and generation storage root.

## Authorization boundary and verification

Neither report launches GPU work by itself.  Pair validation, runtime,
storage, and a later idle-GPU/live-output snapshot must all be content-bound
by a separate execution authorization and immutable launch receipt.  Resume,
training, full-300K launch, sampling, evaluation, promotion, export, release,
and process signalling remain unauthorized.

The idle snapshot contract and POSIX-only builder are now implemented.  The
builder replays the exact runtime and storage reports, both config identities,
the clean execution Git, current runtime and ImageNet-256 identities, the
single-GPU inventory, compute-process inventory, output-scoped process scan,
exact sibling lock availability, output/run-directory absence, and current
filesystem headroom.  Its schema is evidence-only and keeps every execution,
training, sampling, evaluation, promotion, export, release, and process-signal
permission false.  It refuses to overwrite a prior snapshot and contains no
training or detached-process launch path.  No live snapshot is created before
the frozen terminal-SNR confirmation passes.

Focused large-capacity, runtime-selection, storage, and preparation tests pass
(`52 passed` before the snapshot addition; the consolidated execution module
now has `29 passed`), together with Python compilation and `git diff --check`.
The complete Windows CPU-only suite after the snapshot addition collected
`1,344` tests and exited successfully (`1,332 passed / 12 skipped`, as shown
by pytest's progress output).  The earlier complete suite at commit `a26a500`
also exited successfully, closing the previously pending full-suite
validation.
No remote checkout, running process, checkpoint, sample, or locked evidence
was modified.
