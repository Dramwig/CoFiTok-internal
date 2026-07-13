# Generation authorization source freshness (2026-07-14)

## Finding

Generation gate CLI validation and terminal completion already rehashed each
gate's six source reports. The lower-level training and release authorization
capture functions, however, validated only the gate's scientific fields and
bound the gate file bytes. A source report could change after gate construction
while the unchanged gate continued to authorize a later 300K resume or EMA
export. Terminal audit would eventually reject the chain, but only after more
expensive work had been performed.

The full 50K post-evaluation runbook also used a small inline completion check
instead of the authoritative matched-training-pair validator. It could begin
checkpoint evaluation and sampling without freshly checking the scaling source
set, full recipe, dataset identity, parameter gap, or authorization equality.

## Contract

Gate source construction and verification now live in
`cofitok.generation_gate_sources`. Report building, CLI validation, terminal
completion, training authorization, and release authorization all import this
single core module.

`capture_generation_gate_binding` reopens and hashes all six bound sources
before constructing an authorization. Consequently:

- every formal 300K training start/resume fails before checkpoint
  deserialization if a scaling source changed;
- formal EMA export fails before `torch.load` if a full-gate source changed;
- the bound gate identity still transitively commits the exact six path/bytes/
  SHA256 identities into training checkpoints and inference artifacts.

The full 50K post-evaluation runbook now validates the scaling gate with the
shared CLI and reruns `validate_generation_training_pair.py` against both final
300K reports before any checkpoint evaluation, runtime preflight, or sampling.

## Verification

Negative tests mutate a scaling source after authorization capture and a full
source before EMA export. The latter monkeypatches `torch.load` to prove source
validation fails before deserialization. Runbook tests require both source-gate
and full matched-pair validation ahead of formal post-evaluation. Final local
and isolated Linux suite counts are recorded after validation.
