# Deployable EMA inference artifacts (2026-07-12)

## Separation of concerns

The full training checkpoint is the authority for reproducibility. It retains
model, EMA, optimizer, scheduler, scaler, RNG, sampler, metrics, and cumulative
cost state. It must never be rewritten or replaced by a smaller deployment file.

After the final generation gate passes, each method also exports a separate
EMA-only inference artifact:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/exports/imagenet256_full_300k/
  cofitok_k8_ema_inference.pt
  dense_identity_ema_inference.pt
```

These files are for inference only and cannot resume training.

## Format and trust boundary

Artifact type `cofitok_generation_inference`, format version 2, contains only:

- resolved experiment config;
- EMA-applied model state;
- training step;
- source checkpoint path/SHA/format provenance;
- source training runtime-environment SHA256 and Git revision/branch/dirty state.

The adjacent integrity sidecar binds artifact filename, byte size, SHA256,
format, step, source training-checkpoint SHA256, environment SHA, and Git
identity. `load_generation_model`
reads the sidecar type before deserialization and dispatches to either the exact
training-checkpoint verifier or the inference-artifact verifier. An artifact can
only be requested as EMA; `weights=model` fails closed.

Export uses an fsynced temporary file followed by atomic replacement, then writes
and re-verifies the integrity sidecar. Re-running export reuses an existing
verified artifact only when its source SHA, environment, and Git identity all
match exactly. Export from an unprovenanced checkpoint is rejected.

## Final verification

The completion pipeline exports both CoFiTok and dense artifacts, runs repeated
real-forward preflights, and performs short DDIM class-conditional inference:

- CoFiTok: seeds 0/1 at prefix budgets 1 and 8 (four PNGs);
- dense: seeds 0/1 at prefix budget 1 (two PNGs).

Every smoke PNG is atomically published and hashed. Completion requires export
size to be smaller than its training checkpoint, source SHA/environment/Git
agreement with the full training report, artifact/preflight/smoke identity
agreement, exported-EMA load, and the exact expected smoke output counts.

The terminal audit also opens the fixed deployment paths and runs the artifact
integrity verifier again. A stale report cannot mask a deleted, truncated, or
replaced EMA artifact. The same audit independently rehashes the two physical
step-300K exact-resume checkpoints, preserving both deployable inference and
reproducible training assets.
