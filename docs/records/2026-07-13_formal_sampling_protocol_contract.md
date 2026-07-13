# Formal sampling protocol contract (2026-07-13)

## Finding

The promotion and final gates previously required CoFiTok and dense sampling
dictionaries to match. That fairness check could still pass when both methods
were identically weakened, for example by using too few DDIM steps or disabling
`x0` clipping. The completed report also omitted the sampler name and clipping
policy from immutable provenance.

## Contract

`cofitok.generation.sampling_protocol_contract` is now the single validator for
formal sampling protocol `cofitok_ddim_sampling_v1`. It verifies the stable
`GenerationSession` API, DDIM sampler, training horizon, requested sample count,
the exact resolved timestep list, CFG parameters, eta, clipping, precision, seed,
class schedule, and batch/resume-invariant random streams.

The locked formal settings are:

- scaling: EMA DDIM-100 over the configured 1,000-step training schedule;
- full: EMA DDIM-250 over the configured 1,000-step training schedule;
- both: CFG 1.5, guidance rescale 0, batched CFG, eta 0, `clip_x0=true`, bf16,
  seed 0, zero start index, and balanced-modulo class labels.

Sampling manifest schema v3 is written before the first image. Completed report
schema v6 must reproduce its Git, checkpoint, integrity sidecar, weights,
sampling, output-directory, and runtime-environment identities exactly. Metrics
reject unsupported or divergent artifacts before computing distribution scores.

## Fail-closed evidence chain

The promotion/final report adds `formal_sampling_protocol`. The completion audit
independently revalidates each full 50K report against its training diffusion
configuration, requires that named gate, and verifies comparison schema v4 rows
against the source sampling dictionaries. A stale gate or a comparison assembled
from altered protocol fields cannot certify completion.

Targeted tests cover valid scaling/full settings, wrong sampler, DDIM-step drift,
wrong resolved timesteps, changed CFG/precision, disabled clipping, immutable
manifest/report drift, and the matched-weakened case where both methods are
changed together.

## Verification

- focused protocol/session/metrics/gate/comparison/completion tests: 100 passed;
- complete local repository suite: 472 passed.
