# Full-training promotion authorization binding

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

The controlled deployment receipt proves that the pinned 10% training revision
was upgraded to one reviewed full-training revision. The pipeline also validates
the scaling gate immediately before launching the full run. Neither artifact,
however, was embedded into the resulting 300K checkpoints. A later completion
audit could verify a passing scaling gate and a valid full checkpoint without
proving that this exact gate authorized that checkpoint before training began.

## Contract

Formal `imagenet_256` 300K training now requires `--authorization-gate`. The
trainer validates the gate with the shared generation-gate contract before data
loading or model construction and captures:

- canonical full-report gate identity SHA256;
- exact gate file SHA256 and byte count;
- absolute gate path, stage, decision, and validated thresholds.

The complete authorization is written into the run manifest, checkpoint payload
extra state, and training report. The stage, decision, bytes, file SHA, and
canonical identity are also copied into the checkpoint integrity sidecar and
`latest.json`. Resume checks this sidecar before `torch.load`, then verifies the
payload mapping before restoring model, EMA, optimizer, scheduler, scaler, RNG,
or sampler state.

As of 2026-07-14, capturing this authorization also reopens and hashes all six
source reports embedded in the scaling gate. Every milestone start/resume
therefore rejects source drift before deserializing its training checkpoint;
the gate file alone is no longer sufficient when its evidence changed in place.
See `docs/records/2026-07-14_generation_authorization_source_freshness.md`.

The shared generation loader and EMA-export path also validate the payload
authorization against the integrity sidecar before loading model/EMA weights.
This closes the final-checkpoint case where step 300K is sampled directly and no
later training resume would otherwise deserialize it through the training path.

All alternating 50K/100K/200K/300K segments receive the same gate path. The full
runbook validates the completed CoFiTok/dense pair against that gate before any
formal post-evaluation. Pair validation schema v3 rejects missing, mixed, or
different authorizations. The terminal completion audit passes the actual
scaling gate into the pair validator and independently compares authorization
fields from the real step-300K sidecars with each training report pointer.

## Compatibility

The active 10% queue remains on pinned revision `781a014` and does not contain
this field. Its explicit paired legacy validation is unchanged. Runtime
selection benchmarks do not consume a scientific authorization and remain
checkpoint-free. Only formal full ImageNet-256 300K training is required to bind
the promotion gate.

## Verification

Tests cover gate-file capture, checkpoint/sidecar/latest propagation, successful
authorized load, pre-deserialization rejection of a changed gate, full-pair
authorization matching, scaling-gate mutation, report mutation, and sidecar
mutation. The formal runbook CLI contract requires `--authorization-gate` on
every long-training segment.
