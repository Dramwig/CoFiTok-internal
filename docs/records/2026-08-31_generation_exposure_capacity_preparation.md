# Exposure/capacity disambiguation preparation (2026-08-31)

## Scope

This is a CPU-only, non-authorizing preparation record. It does not launch
training or sampling, modify the remote checkout, send process signals,
promote a checkpoint, export an inference artifact, release a model, or
replace the terminal quality hold.

The machine-readable preparation report is:

```text
artifacts/reports/generation/exposure_capacity_disambiguation_2026-08-31/preparation.json
```

Its SHA256 is:

```text
b34d0e0a057bed34511c88f83671c80039b86ea45cf548636daff02f91dcb074
```

The report is `schema_version=cofitok_generation_exposure_capacity_preparation_v1`,
`status=prepared`, and
`decision=preserve_qualified_objective_prepare_exposure_capacity_disambiguation`.
It records `execution_ready=false`; preparation status is not execution
authorization.

## Source-bound evidence

The builder used independently copied, hashed evidence from
`D:\cofitok-exposure-capacity-evidence-20260831` for the quality bridge,
post-reconciliation decision, cross-protocol reconciliation,
sampling-recovery result, Min-SNR result, pair monitor, both training reports,
and both metrics JSONL files. The report records all ten source identities.

The matched quality bridge remains at revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`, with both methods at 100,000
steps and 6,400,000 images seen. The final validation epsilon MSEs are
`0.029256021603941917` for CoFiTok and `0.029240703210234642` for dense
identity; the preparation records no large primary-loss divergence.
It also binds the CoFiTok 100K resume source to its run directory and payload
identity (`1010937514` bytes, SHA256
`b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e`), while
retaining the distinct dense payload identity for matched provenance.
The gate validators re-read both hashed training reports and recompute these
summaries, so changing an embedded checkpoint descriptor while changing the
preparation hash is rejected.

The quality bridge, post-reconciliation decision, and cross-protocol
reconciliation all preserve `terminal_status=hold`.
Sampling recovery is `no_shared_sampling_recovery_candidate`; Min-SNR gamma-5
is `screening_only` with `no_shared_min_snr_candidate_at_50k`. These results do
not prove a generation advantage. `generation_advantage_proven` remains
`false`.

## Candidate arms

The preparation keeps two source-compatible, mutually exclusive candidates:

| Arm | Controlled change | Source/output constraint |
| --- | --- | --- |
| `exposure_continuation` | Training exposure only | Exact 100K checkpoint resume, same base channels/layout, new output root |
| `capacity_qualification` | Model capacity only | Fresh matched initialization, base channels 128 to 256, new output root |

Both arms forbid objective and conditioning changes and formal quality claims.
The execution order is deliberately undetermined until a new source-compatible
stage gate selects one arm.

## Authorization boundary

Every preparation permission is disabled:

```text
decision_is_execution_authorization=false
remote_mutation_allowed=false
gpu_execution_authorized=false
training_launch_allowed=false
sampling_launch_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
promotion_allowed=false
export_allowed=false
release_allowed=false
process_signals_allowed=false
terminal_hold_replacement_allowed=false
```

The selection policy requires a fresh remote rehash, exact stage
authorization, a matched pair, distinct versioned output roots, and a
fail-closed response. Automatic 300K escalation and selection by current FID
are disabled. Any future GPU work must first create and validate a new
source-compatible execution gate; this preparation alone cannot start it.

## Verification

The focused contract tests, including the checkpoint-summary replay check,
pass. 

The independent validator replays the generated report with
`status=pass`, `source_count=10`, and the canonical all-false authorization
boundary. It also re-hashes every referenced source and rejects a changed
byte count or SHA256 before reporting pass; the successor execution gate
requires the expected preparation SHA256. The preparation validator itself
also requires `--expected-preparation-sha256` and rejects a replacement of
the complete preparation report before parsing it.
