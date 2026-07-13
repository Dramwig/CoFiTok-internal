# Matched formal sampling batch selection (2026-07-12)

## Motivation

Formal evaluation requires two 10K DDIM-100 sample sets at the scaling gate and
two 50K DDIM-250 sets at the final gate. A fixed batch of 32 is conservative,
but can leave most of a 96 GiB GPU unused and turn final sampling into a
multi-day bottleneck.

Sampling batch size does not change the CoFiTok scientific protocol. The sampler
uses a deterministic generator for each global sample index, and its manifest
declares `batch_size_invariant=true` and `resume_index_invariant=true`.

## Measured preflight

`preflight_generation_sampling.py` now supports warmup and repeated synchronized
forwards. It reports mean/median/p95 forward time, requested and effective CFG
throughput, per-forward durations, output finiteness, peak allocated/reserved
VRAM, total device memory, and exact inference-code Git revision, branch, and
tracked-dirty state. Its default remains one measured forward for backward
compatibility.

## Shared selection

`select_generation_sampling_batch.py` measures batches `16,32,64,128` on both
the CoFiTok and dense checkpoints using EMA, bf16, CFG 1.5, and batched CFG.
A candidate is eligible only when:

- both methods pass on the same checkpoint step and sampling protocol;
- each reusable preflight was produced by the selector's current Git revision
  from a clean tracked worktree;
- both preflights contain valid, identical canonical Python/PyTorch/CUDA/GPU/
  project-lock runtime-environment fingerprints;
- both produce finite positive output-images/second;
- both remain below 90% of device memory;
- batch 32, the conservative baseline, also passes.

The shared selected batch maximizes the lower throughput of the two methods.
OOM or inadequate headroom makes a candidate ineligible. The two methods can
never receive different formal sampling batches. Successfully completed
candidates must also share one runtime-environment SHA across the entire
ranking; two internally matched candidates from different environments cannot
be compared for speed.

The selector runs for the 10K promotion sample sets and again for the final 50K
sample sets, because the checkpoints and available memory profile differ. The
2,048-sample milestone diagnostics remain fixed at batch 32, and the small
prefix diagnostic remains batch 16.

The selector is state-aware. Before either matched formal output directory has
state, it may run or refresh preflights and atomically publish the selection.
After either directory contains a manifest, progress report, PNG, or any other
sampling state, it may only validate and return the frozen selection. The lock
binds both output paths, the ordered candidate set, baseline and memory policy,
both checkpoint identities, both prefix budgets, guidance/CFG/weights/precision,
warmup and measured-forward counts, clean branch/revision, and benchmark root.
Missing or drifted selection evidence fails before a preflight model load.

## Completion evidence

The final selection is bound to the deployed Git revision and both checkpoint
SHA256 values. The completion audit verifies that both formal generation reports
actually use the selected batch, that the selected CoFiTok and dense preflight
evidence carries the deployed revision, and that both retain batch-invariant
random streams. It also requires both formal generation reports to carry the
same runtime-environment SHA selected by those preflights. The final matched
comparison publishes sample batch, cumulative sample time, and images/second
together with FID/IS/precision/recall.

The formal sampler independently embeds the same Git state into its immutable
sampling manifest and completed report. Metrics propagate it as sample
provenance. Both promotion/final gates require matched clean CoFiTok/dense
sampling code, and the final completion audit additionally binds it to the full
training revision and `scale/generative-system` branch. This prevents a clean
selector preflight from masking later uncommitted inference-code changes.
The same chain prevents a cached preflight from masking a later interpreter,
package lock, CUDA driver, GPU, or backend-setting change.

Restart-boundary hardening and adversarial verification are recorded in
`2026-07-14_generation_sampling_batch_selection_freeze.md`.
