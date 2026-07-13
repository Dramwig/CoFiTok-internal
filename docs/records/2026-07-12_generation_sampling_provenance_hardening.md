# Generation Sampling and Provenance Hardening (2026-07-12)

## Scope

This change hardens the production evaluation path on
`scale/generative-system`. It does not modify or overwrite the locked paper
evidence, and it is not applied to the active remote 50K training pair while
that pair is running from commit `781a014`.

## Correctness issue

The first production sampler seeded each batch with a value that depended on
the prefix budget and batch start. Consequently, equal sample numbers at
different prefix budgets did not share the same initial noise, and changing the
batch size or resume boundary changed the generated random stream. That was
reproducible only for one exact invocation, but it was not a valid paired prefix
protocol.

## Resolution

- Assign one deterministic generator to each global sample index.
- Recreate the same per-index streams for every prefix budget.
- Keep each stream independent of batch size and resume boundary, including
  stochastic DDIM updates when `eta > 0`.
- Write an immutable `sampling_manifest.json` before image generation. Resume
  is permitted only when checkpoint hash, weights, target count, schedules,
  precision, guidance, RNG contract, and output directories match exactly.
- Make `--resume` skip complete numbered batches and reconstruct missing files
  from their original per-index streams; existing images without a matching
  manifest are rejected.
- Record the stream contract in `sampling_report.json`.
- Require formal generation evaluation to consume that sampling report and an
  exact zero-based numbered PNG set.
- Carry checkpoint SHA256, step, EMA/model selection, selected prefix budget,
  and sampling parameters into the metrics report.
- Bind revision, branch, and tracked-dirty state independently for the sampler
  and the torch-fidelity evaluator. The immutable sample manifest carries the
  sampler state; the metrics report carries evaluator state and preserves the
  sampler state under `sample_provenance`.
- Require the promotion gate to match sampling protocols and cross-check the
  checkpoint hash used by generation metrics against the checkpoint hash used
  by mechanism diagnostics.
- Require promotion/final gates to reject mismatched or dirty sampler/evaluator
  code. Full completion additionally binds both to the full training revision,
  and refuses a final gate missing either named code-provenance check.
- Validate both completed clean training reports before starting the full 50K
  sample evaluation runbook.

## Verification

Targeted tests cover paired prefix streams, batch-size/resume invariance for
both deterministic and stochastic DDIM, stale sample rejection, matched
sampling provenance, and checkpoint hash mismatch rejection. The complete test
suite must pass before this change is synchronized after the active matched
training pair completes.

A real two-step CPU checkpoint smoke generated four samples at prefix budgets 1
and 4, deleted `prefix_4/000002.png`, and resumed with the original manifest.
The original and reconstructed PNG both had SHA256
`5c6fa99d523592e800ce04b2ff1bec351dd468c9a038449e6420a6a664c30e69`.

## Active remote state at discovery

- Remote revision: `781a014`
- Queue PID: `28952`
- CoFiTok training PID: `29127`
- Latest observed training step: `1150 / 50000`
- First checkpoint: not yet available; interval is 5000 steps
- Dense matched run: pending after CoFiTok
