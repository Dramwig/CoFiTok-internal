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
VRAM, and total device memory. Its default remains one measured forward for
backward compatibility.

## Shared selection

`select_generation_sampling_batch.py` measures batches `16,32,64,128` on both
the CoFiTok and dense checkpoints using EMA, bf16, CFG 1.5, and batched CFG.
A candidate is eligible only when:

- both methods pass on the same checkpoint step and sampling protocol;
- both produce finite positive output-images/second;
- both remain below 90% of device memory;
- batch 32, the conservative baseline, also passes.

The shared selected batch maximizes the lower throughput of the two methods.
OOM or inadequate headroom makes a candidate ineligible. The two methods can
never receive different formal sampling batches.

The selector runs for the 10K promotion sample sets and again for the final 50K
sample sets, because the checkpoints and available memory profile differ. The
2,048-sample milestone diagnostics remain fixed at batch 32, and the small
prefix diagnostic remains batch 16.

## Completion evidence

The final selection is bound to the deployed Git revision and both checkpoint
SHA256 values. The completion audit verifies that both formal generation reports
actually use the selected batch and retain batch-invariant random streams. The
final matched comparison publishes sample batch, cumulative sample time, and
images/second together with FID/IS/precision/recall.
