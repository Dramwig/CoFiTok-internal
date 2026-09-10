# Terminal-SNR label-conditioning pipeline audit

Date: 2026-09-10

## Scope

The completed four-arm terminal-SNR endpoint screen remains immutable at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1
```

Its controller remains in the sibling
`.terminal_snr_endpoint_screen_v1.terminal_snr_screen_control` directory, and
the execution checkout remains the clean
`terminal-snr-execution-89bcd9a@89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`
checkout with tree `46efd20cff489bccd799bb13c4155a0cc79e7649`.

This work adds a standard-library-only, CPU/read-only scientific audit that
tests whether a class-index mapping mismatch can explain the shared generated
support collapse. It does not alter the screen, retrain, sample, evaluate a
model, signal a process, or authorize confirmation/full training.

## Implementation

The standalone builder and independent validator are the two subcommands of:

```text
scripts/build_generation_terminal_snr_label_conditioning_pipeline_audit.py
```

The audit performs the following physical replay:

1. Rechecks the exact clean execution revision and binds nine relevant tracked
   code blobs plus all four tracked screen configs.
2. Streams the 405 MB ImageNet-256 manifest without retaining it in memory and
   verifies every row's numeric label, WNID, path component, split, and 256x256
   resolution.
3. Reconstructs the loader's `sorted(unique WNID)` class order and compares it
   with the execution mapping, both dataset mappings, torchvision categories,
   and timm ImageNet synsets.
4. Replays every arm's finite, strictly increasing metrics and exact
   `samples_seen = step * 64` accounting.
5. Physically rehashes both protected checkpoints (5K and 10K) and their
   sidecars for every arm.
6. Rehashes all 1,000 generated PNG files per arm using the sampler's declared
   `filename_utf8_nul_file_bytes_nul` framing and checks the manifest,
   progress, report, metric, class-fidelity, checkpoint-diagnostic, rollout,
   arm-validation, terminal-result, and controller bindings.
7. Verifies that training labels, sampling labels, CFG conditioning, the model
   class embedding, and class-fidelity requested targets all transport the
   same zero-based integer index.
8. Replays the real-validation classifier calibration and the immutable
   support-collapse causal discriminator.

Both the audit and its validation receipt use exclusive immutable writes,
reject symlinked source/output paths, use stable-read TOCTOU guards, and retain
exact builder/validator Git and script identities.

## Predeclared conclusions and boundary

The report may pass operationally only with all of these conclusions:

```text
class_index_mapping_mismatch = false
mapping_can_explain_support_collapse = false
shared_support_collapse_consistent = true
common_cause_proven = false
```

The terminal screen therefore stays on scientific hold. Every confirmation,
training, 300K, promotion, export, release, and paper-integration permission is
fixed false in both the audit and validation schemas.

## Local verification

Focused tests cover mapping order, category/synset agreement, streamed
manifest label/WNID/path/resolution tampering, code-path tampering, strict
metrics/sample accounting, immutable output writes, fail-closed authorization,
and validation-receipt reproducibility:

```text
14 passed
```

The implementation also passes `py_compile` and direct `--help` execution in
the project `.venv`.

## Remote evidence

The immutable remote audit and independently replayed validation receipt will
be appended after building from distinct exact clean checkouts of the committed
implementation. Until that replay passes, this record does not claim a remote
audit result.
