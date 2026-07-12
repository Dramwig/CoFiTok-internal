# Atomic sample and report publication

Date: 2026-07-12

Branch: `scale/generative-system`

## Failure mode

Resumable generation previously wrote each PNG directly to its final numbered
path. A process interruption during image encoding could leave a truncated file
that still existed, causing batch-completion checks to skip it on resume. JSON
manifests and reports also replaced their previous contents in place, so a
serialization interruption could destroy the last valid state.

## Contract

- PNG encoding now targets a hidden same-directory `.part` path. `os.replace`
  publishes the numbered final file only after encoding returns successfully.
- Failed PNG writes remove their partial path and never create the final path.
- Existing numbered PNGs are skipped on resume only after successful decode,
  PNG verification, and exact size/mode checks. A damaged file is regenerated
  from its original per-sample RNG stream and atomically replaces the damaged
  path.
- `write_json_report` writes and fsyncs a same-directory temporary file, then
  atomically replaces the destination.
- Failed JSON serialization removes the temporary file while preserving any
  previously valid destination report.

The same-directory requirement ensures the replacement remains an atomic
filesystem rename rather than a cross-device copy.

## Verification

- Fault-injected PNG encoding leaves neither a final file nor a partial file.
- A successful PNG is invisible at its final path until encoding completes.
- A corrupted numbered PNG makes its batch incomplete and is regenerated on
  resume; formal metric validation rejects corruption before torch-fidelity.
- Sampling provenance records `[C, H, W]`, and the matched gate checks it
  against each checkpoint's training configuration.
- Fault-injected JSON serialization leaves the prior valid JSON unchanged.
- Successful JSON replacement parses to the complete new payload.

The active 10% matched training queue remains pinned to commit `781a014`; this
inference/reporting change will be synchronized only after both training runs
finish.
