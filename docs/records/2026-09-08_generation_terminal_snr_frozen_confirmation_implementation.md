# Terminal-SNR frozen confirmation implementation and rehearsal

Date: 2026-09-08 (Asia/Shanghai)

## Scope

This record closes the source-implementation gap for the separately gated
terminal-SNR frozen-checkpoint confirmation. It does not authorize or launch
that confirmation. It also does not authorize capacity-readiness work, full
training, 250M/300K execution, export, release, or process signals.

The confirmation remains conditional on a physically replayable passing result
from the active four-arm terminal-SNR endpoint screen. It evaluates the four
exact step-10K screen checkpoints without further training, using a new random
stream (`seed=2028`) and 10,000 EMA DDIM-100 samples per arm.

## Exact source identity

- Implementation commit: `a5199e6eea63817fe4eee59ac187766748d6e772`
- Implementation tree: `2bbe3a1fdc39180f0f5d2f0fd144cebe25828296`
- Branch: `analysis/generation-terminal-snr-confirmation-v1-20260908`
- Required screen-source prerequisite: `89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`
- Prerequisite tree: `46efd20cff489bccd799bb13c4155a0cc79e7649`

The implementation adds strict preparation, stage authorization, execution
authorization, storage/live-snapshot evidence, immutable launch receipt,
controller, per-arm physical validation, fail-closed result, adjacent result
validation, CLI surfaces, runbook, and tests. All future-stage permissions stay
false. A passing confirmation may only prepare a separately source-bound
large-capacity readiness decision.

Every arm must complete a real EMA+CFG forward preflight before formal
sampling. The preflight binds the raw training checkpoint and integrity
sidecar, step, execution Git, originating screen Git through its replayed arm
evidence, runtime SHA256, finite output, elapsed time, and positive CUDA peak
memory. Raw training checkpoints correctly expose `source_git=null`; their
screen origin is bound by the immutable screen-arm validation rather than by
inference-artifact metadata.

## Incremental bundle

- Local bundle:
  `C:/Users/17194/AppData/Local/Temp/cofitok-generation-terminal-snr-confirmation-a5199e6-from-89bcd9a.bundle`
- Server bundle:
  `/tmp/cofitok-generation-terminal-snr-confirmation-a5199e6-from-89bcd9a.bundle`
- Bytes: `53352`
- SHA256: `a0fb234ea7377e442a866be8818a55d1d6a7846106b9477345bd73e2ad54903e`
- Advertised ref only:
  `refs/heads/analysis/generation-terminal-snr-confirmation-v1-20260908`
  at `a5199e6eea63817fe4eee59ac187766748d6e772`
- Required prerequisite only:
  `89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`

Both local and server byte counts and SHA256 values matched. `git bundle
verify` succeeded before the bundle was fetched into an isolated checkout.

## Validation

Local exact implementation checkout:

- Full suite: `1287` tests, `0` failures, `0` errors, `12` expected skips.
- JUnit time: `354.128` seconds.
- `compileall`: passed.
- Confirmation CLI help and runbook-entrypoint coverage: passed.
- `git diff --check`: passed.

Server isolated checkout:

- Path: `/tmp/cofitok-terminal-snr-confirmation-rehearsal-a5199e6-v1`
- Revision/tree matched the exact implementation identity above.
- CUDA was explicitly disabled for the full Linux suite.
- Full Linux suite: `1287` tests, `0` failures, `0` errors, `6` expected skips.
- JUnit time: `204.542` seconds.
- `compileall`: passed.
- Runbook syntax: `112/112` passed with `bash -n`.
- Tracked and untracked status after validation: clean.

## Concurrent live-screen audit

The active screen checkout and evidence were not mutated during development or
rehearsal. At the post-rehearsal read-only audit:

- Screen execution revision:
  `89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`
- Screen execution tree: `46efd20cff489bccd799bb13c4155a0cc79e7649`
- Controller PID: `459216`
- Stage: `training_control_cofitok`
- Trainer/GPU PID: `459302`, a controller descendant and the only compute PID.
- Progress: step `3150/10000`, `201600` samples seen.
- Metrics: `64` rows, strictly increasing, all finite, and every row satisfied
  `samples_seen = step * 64`.
- GPU: `84122 MiB` attributed to PID `459302`; no rehearsal GPU process.
- `/root/autodl-tmp` free bytes: `522129629184`.

No process received a signal. No screen checkpoint, sidecar, report, launch
receipt, authorization, source checkout, or locked evidence was changed. The
confirmation has not launched, and full/300K authorization remains false.
