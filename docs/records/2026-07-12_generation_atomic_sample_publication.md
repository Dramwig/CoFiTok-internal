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
- `write_json_report` writes and fsyncs a same-directory temporary file, then
  atomically replaces the destination.
- Failed JSON serialization removes the temporary file while preserving any
  previously valid destination report.

The same-directory requirement ensures the replacement remains an atomic
filesystem rename rather than a cross-device copy.

## Verification

- Fault-injected PNG encoding leaves neither a final file nor a partial file.
- A successful PNG is invisible at its final path until encoding completes.
- Fault-injected JSON serialization leaves the prior valid JSON unchanged.
- Successful JSON replacement parses to the complete new payload.

The active 10% matched training queue remains pinned to commit `781a014`; this
inference/reporting change will be synchronized only after both training runs
finish.
