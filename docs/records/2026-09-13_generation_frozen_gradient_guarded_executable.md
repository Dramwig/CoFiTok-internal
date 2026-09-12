# Guarded frozen selected-image gradient executable

Implemented the pending non-updating diagnostic, without running it on a real
checkpoint. The original terminal screen remains immutable `hold`; classifier
results remain weak conditioning with shared support collapse, not proof that
conditioning is completely ignored.

## New code and scientific boundary

- `scripts/frozen_loss_gradient_diagnostic.py`: separate prepare, independent
  preparation replay, authorization-consumption, run and validation stages.
- `src/cofitok/generation/frozen_gradient_stage.py`: source-bound stage contract.
- `src/cofitok/generation/frozen_gradient_readout.py`: independent scalar
  arithmetic and image-level readout, without importing the gradient core.
- Core checks now also reject image-shape drift and changed preexisting `.grad`
  contents; the CPU test fixture restores its original thread count.

No command creates a user approval. A distinct human approval must bind the
preparation identity, exact evaluator checkout, 256-row protocol and all denied
later permissions. The prior classifier-only approval is rejected by schema.
The executable refuses missing admission before runtime allocation/model load,
source drift, existing output (even partial), non-idle GPU or insufficient disk.
The output directory is the single-invocation claim. A 7,200-second deadline
targets only this evaluator process; no unrelated process is signalled.

Model, rollout/loss, EMA and validation preprocessing sources are compared with
the original `cf0e5faa94bf4ab38d947b921935b3b765b5537a` training revision. The
only accepted compatibility differences are the already reviewed default-1
cosine-endpoint field/schedule extension at `5ac6192`; all other 14 of 16 checked
production files are byte-identical. Every executable Python blob is compared
with its checkout Git blob, including files hidden by index optimization flags.
The recorded original runtime must match exactly before model deserialization;
payload config/Git/runtime/dataset/EMA metadata are checked before model forward.

The measurement still uses single selected images, not batch-64 gradients. Fixed
validation preprocessing deliberately has no random horizontal flip and is not
called an exact historical training-batch replay. All 32 images and all four
timesteps must be present, with matched per-image/timestep noise/dropout seeds
across methods. Original online and detached EMA tensors are fingerprinted
before/after; no optimizer, scaler or EMA update occurs.

The pre-observation operational nomination rule is in the protocol/specification:
same auxiliary term and conditioning group, cosine <= -0.1 and weighted norm
ratio >= 0.1, >= 3/4 timesteps for >= 24/32 images in each method. Saturation has
an analogous explicitly exploratory rule. Thresholds nominate hypotheses only;
they are not validated effect-size standards, causal proof or training permission.
All counts and undefined statistics remain visible. Repeated timesteps are not
treated as independent samples.

The adjacent immutable receipt independently checks scalar arithmetic, source
bytes, every saved row and all image-level summaries. It explicitly states that
full parameter-gradient vectors were **not independently recomputed**. A passing
receipt is technical validity of this bounded diagnostic, not generation-quality
qualification and not resolution of the old screen failure.

## Verification at implementation time

Local project CPU focused tests: **62 passed**, covering the previous core/toy
semantics/rollout suites, real tiny-core output against the independent scalar
validator, missing/prior/expanded authorizations, refusal before `torch.load`,
GPU ownership, no-overwrite behavior, finite arithmetic, seed/order/row coverage,
component assignment and image-level nomination behavior. `py_compile` and
`git diff --check` pass. This is not a full-repository or real-checkpoint GPU test.

## Exact remote rehearsal and completed preparation

Implementation revision `ad237efa810df8677b4187110a723279d47f9d4b`, tree
`e5b0def15e93df13b845abbc905cff7eda7bc486` was transferred as a single-head
incremental bundle requiring `5ac619228b051d12d87a2d790a3676d31c8d7b72`:

```text
/tmp/cofitok-frozen-gradient-stage-ad237ef-from-5ac6192.bundle
bytes: 36604
sha256: 9c2951d22a7d414f62d9500a913f6047f4e252e0deb775985102510d4437b8f0
```

Local/remote bundle verification passed. Four new isolated clean checkouts were
created without moving any existing checkout:

```text
/tmp/cofitok-frozen-gradient-preparation-ad237ef
/tmp/cofitok-frozen-gradient-authorization-ad237ef
/tmp/cofitok-frozen-gradient-evaluator-ad237ef
/tmp/cofitok-frozen-gradient-validator-ad237ef
```

Each is at the exact revision/tree above with its own
`analysis/frozen-gradient-<role>-v1-20260913` branch. Each independently passed
the same **62 CPU tests** with `CUDA_VISIBLE_DEVICES=''`, bytecode writing and
pytest cache disabled. All remained clean.

The complete immutable preparation now exists:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/.frozen_loss_gradient_attribution_v1.control/preparation.json
bytes: 51660
sha256: 8cc706a237812d137389b44985c599789bbca38d3cfa02e8ceb8c07c26cd422b
mode: 0444
mtime_ns: 1789233150586218883
```

The preparation builder performs its own physical replay, and the distinct
exact validator checkout then returned `status=pass` for `replay-preparation`.
The receipt-independent preparation mtime remained unchanged. The bound prior
catalog was replayed from its original clean `5ac6192` builder, including the
32 image bytes, original checkpoints/sidecars/reports and exclusion census.

At handoff, `stage_approval.json`, `execution_authorization.json` and
`checkpoints/generation/frozen_loss_gradient_attribution_v1/` are all absent.
The executable has never deserialized a real checkpoint or performed the
gradient measurement. A passing preparation is not GPU/runtime execution,
scientific evidence, user consent or a later-stage permission. The next step
requires a distinct human approval of this exact 256-row, at-most-7,200-second,
non-updating frozen diagnostic. All other permissions remain false.
