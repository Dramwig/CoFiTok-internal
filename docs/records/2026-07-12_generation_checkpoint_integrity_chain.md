# Generation checkpoint integrity chain

Date: 2026-07-12

Branch: `scale/generative-system`

## Contract

Each newly saved production checkpoint now publishes three atomically ordered
artifacts:

1. `checkpoint_step_XXXXXXXX.pt`, saved to a temporary path, fsynced, and
   atomically replaced;
2. `checkpoint_step_XXXXXXXX.pt.integrity.json`, containing checkpoint byte
   count, SHA256, optimizer step, checkpoint format, and schema version;
3. `latest.json`, containing the same integrity fields plus the sidecar name.

`latest.json` is published last. An interruption before sidecar or pointer
publication leaves the previous resume pointer valid. Temporary checkpoint
files are removed on failure, and checkpoint pruning removes matching sidecars.

## Resume validation

`--resume auto` now rejects:

- a missing or malformed sidecar;
- a path-bearing or non-local checkpoint name in `latest.json`;
- byte-count or SHA256 mismatch;
- disagreement between `latest.json` and the sidecar;
- disagreement between sidecar step/format and the deserialized payload.

The full generation gate requires the final training report's checkpoint hash
to match the EMA checkpoint used for formal samples. The active 10% matched
queue remains on legacy commit `781a014`; its exact checkpoint bytes will still
be hashed by sampling and checkpoint evaluation, but sidecar enforcement begins
with subsequent full-data training after the scaling gate.

## Verification

- Unit roundtrip restores model, EMA, optimizer, scheduler, sampler-adjacent
  extra state, and Python/NumPy/PyTorch RNG state.
- Same-size single-byte checkpoint mutation fails SHA256 verification before
  `torch.load`.
- A tampered `latest.json` step is rejected.
- Pruning removes the old checkpoint and its integrity sidecar together.
- Real CPU smoke resumed from step 1 to step 2 with `--resume auto`; the final
  report marked training complete and published a new verified hash.
