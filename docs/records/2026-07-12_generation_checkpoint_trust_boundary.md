# Generation checkpoint trust boundary (2026-07-12)

## Finding

Formal sampling, sampling preflight, and checkpoint mechanism evaluation shared
`load_generation_model`, but the loader previously calculated a new SHA256 and
then deserialized the file without consulting the checkpoint integrity
sidecar. A modified checkpoint that still deserialized could therefore become
a new self-reported provenance identity instead of being rejected.

## Strict loader

`cofitok.generation.load_generation_model` now calls
`verify_training_checkpoint` before `torch.load`. It requires:

- the adjacent `<checkpoint>.integrity.json` file;
- the expected integrity schema version and checkpoint filename;
- exact checkpoint byte size and SHA256;
- matching checkpoint format version and optimizer step between sidecar and
  deserialized payload.

Only after these checks does the loader instantiate the configured model and
copy model or EMA weights. The resolved integrity-manifest path is part of
`LoadedGenerationModel` and is propagated into preflight, sampling manifest,
sampling report, generation-metrics provenance, checkpoint mechanism
evaluation, milestone reports, and promotion/final gate evidence. Mechanism
and sampling evidence must match both checkpoint SHA and integrity-manifest
path.

The 10% queue was launched from legacy revision `781a014`, so its 50K
checkpoints must first pass the byte-preserving integrity migration already at
the start of the post-eval runbook. Full-training checkpoints are written with
integrity sidecars atomically by the upgraded checkpoint code.

## Verification

CPU preflight tests now build a format-versioned checkpoint plus sidecar and
exercise the shared EMA/CFG path. Flipping one byte after sidecar creation must
raise `Checkpoint SHA256 mismatch` before forward execution. Generation-metric
tests also require the sampling report to name the sidecar corresponding to the
checkpoint filename.
