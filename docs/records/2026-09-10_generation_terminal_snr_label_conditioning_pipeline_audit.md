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

The implementation was committed as
`c38291b14058746712dc0934937539a86bb3800e` with tree
`ae5d00d450cee99777ed1f663e92d179511204ed`. The source-bound incremental
bundle from required prerequisite `2efb3ea84a15f9b812250d09cdd97dcb34b3f802`
is `23,767` bytes with SHA256
`e850cd1ac4fec0d6d6268fca275826b0e6382490bd48a50eb4558feb5587e59c`.
The bootstrap bundle from remote-available prerequisite `94f7dd1` is `65,720`
bytes with SHA256
`7715878c6e14d8ac407a9cd6f1a4cfe663b048e45f856cba872d5f70962c2a8c`.
Both bundles passed remote prerequisite and advertised-head verification.

The production builder and validator were kept in distinct clean checkouts:

```text
/tmp/cofitok-terminal-snr-label-pipeline-audit-builder-c38291b
  analysis/generation-terminal-snr-label-pipeline-audit-builder-v1-20260910
/tmp/cofitok-terminal-snr-label-pipeline-audit-validator-c38291b
  validation/generation-terminal-snr-label-pipeline-audit-v1-20260910
```

Both checkouts independently reported exact revision `c38291b...`, tree
`ae5d00...`, and empty tracked/untracked status. Each passed the focused remote
suite (`14 passed`) and `py_compile` under
`/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10` with GPU visibility
disabled.

The builder then created the new evidence root without touching the immutable
screen or controller:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_label_conditioning_pipeline_audit_v1
```

The immutable outputs are:

| artifact | bytes | SHA256 | mode |
|---|---:|---|---:|
| `label_conditioning_pipeline_audit.json` | 66,258 | `cead3bc3dc13d8d3e6aa8ef0470d415e7083615dbe505ebda08f5b2a4284bc9a` | `0444` |
| `label_conditioning_pipeline_audit.validation.json` | 3,344 | `6f9de1e525c3ec3d79f24a9e542257eb378d61c5d637d26ae7c187ae8aeace86` | `0444` |

The independent validator replayed all `100` source files, rebound the audit's
exact bytes/SHA256, and produced validation-basis SHA256
`7bdb9805f9ac5eb4cb0d27b122958cb674016b31554e137f76826bc7e38b59fb`.
Its recomputed canonical audit SHA256 is
`8072cf1fe014cb861a642a22745644aa78eb27258115ff1aed0db05a15e88ae1`.

The physical replay streamed all `1,331,167` dataset-manifest rows, rehashed
all eight protected checkpoints and adjacent integrity sidecars, and rehashed
all `4,000` generated PNGs. Every arm has `201` finite, strictly increasing
metric rows ending at step `10,000`, exact `640,000 = 10,000 * 64` sample
accounting, a balanced one-sample-per-class request schedule, and a physically
replayed 1,000-image sample-set digest. The exact execution checkout remains
clean at `89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`, tree
`46efd20cff489bccd799bb13c4155a0cc79e7649`.

All three stored label mappings are semantically identical in official
lexicographic WNID order. The torchvision category file and timm synset order
match that mapping, and all nine execution code paths preserve the same
zero-based integer class index through the loader, trainer, sampler, CFG path,
model embedding, and class-fidelity evaluator. The frozen real-validation
calibration remains strong (`top1=0.777`, `top5=0.941`). Therefore the immutable
scientific conclusion is:

```text
class_index_mapping_mismatch = false
mapping_can_explain_support_collapse = false
shared_support_collapse_consistent = true
common_cause_proven = false
```

This diagnostic passes operationally but does not rescue the terminal screen.
The locked screen still fails both relative-FID thresholds (CoFiTok
`-0.11363547384116164`, dense identity `-0.09408838680587096`), and recall is
still exactly zero in all four arms. The screen, audit, and validation receipt
all retain `scientific_status=hold`, `generation_advantage_proven=false`, and
every confirmation, training, 300K, promotion, export, release, and paper
permission false.

The final read-only host check found no GPU compute process; the RTX PRO 6000
reported zero MiB compute allocation and `97,253` MiB free. The generation
filesystem had `189,708,296,192` free bytes. No controller, trainer, sampler,
or evaluator process for the terminal screen remains. One older read-only SSH
inspection subtree (`87710 -> 87723 -> 87725`) is blocked in `sed` waiting on
stdin; it has no GPU ownership and was not signaled because the immutable
authorization explicitly forbids process signals.
