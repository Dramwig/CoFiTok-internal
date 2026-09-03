# Exposure result runtime-binding recovery

## Scope

The bounded 100K-to-110K exposure continuation completed both matched training
arms and every requested evaluation, but the revision `5c23141` result builder
failed before creating `exposure_capacity_result.json`.

No training, sampling, metric, checkpoint, authorization, gate, or controller
artifact is modified by this recovery.

## Failure

The immutable execution authorization records the control-plane environment
captured while the authorization command ran:

```text
921e10fab9cda4aa6f5dbfda70a01e9adce2001a8c33499b5efb940eb407c059
```

The exact candidate gate records the training runtime environment:

```text
d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
```

Both immutable 100K source training reports and both completed 110K training
reports use the candidate-gate value. The result builder incorrectly compared
the training reports with the control-plane authorization value and terminated
with `runtime environment differs from authorization`.

The authorization binds the exact candidate gate by bytes and SHA256, so using
the gate's runtime identity for the training-report check preserves the existing
authorization boundary. It does not change the model, objective, data, training
horizon, sampling protocol, evaluation protocol, or any scientific threshold.

## Change

`build_result` now validates both completed training reports against the runtime
identity in the already validated candidate gate. The authorization's separate
live snapshot remains immutable and continues to be validated as control-plane
prelaunch evidence.

A regression test supplies deliberately different control-plane and training
runtime identities and verifies that both training reports are checked against
the candidate-gate identity.

The first physical replay then exposed a second dormant schema mismatch in the
same result-only path. The authorization stores each source checkpoint as a
serialized file identity with `path`, `bytes`, and `sha256`; it does not carry a
redundant `name` field. The builder compared the training report's source
`filename` with that absent field even though the resume-transition check below
already derived the filename from the authorized path. The horizon-extension
check now derives the basename from the same bound path, rejects an empty or
malformed path, and reuses it for both checks. A filesystem-backed regression
test exercises the path-only authorization schema through final-checkpoint
validation.

The next replay reached formal sampling validation and exposed one final
producer/consumer schema mismatch: the sampler records `weights` at report
top-level while its nested `sampling` object contains the DDIM/random-stream
protocol. The builder had required `weights` in both places. It now validates
the authoritative top-level field once, preserves the nested provenance object
byte-for-byte for sampler replay, and adds the verified weight only to the
normalized result protocol used for matched comparison. A regression test
covers the exact top-level-weight layout.

## Recovery boundary

The recovery must run from a clean, committed checkout. It may create the
previously absent result and its adjacent validation receipt, but it must refuse
to overwrite either artifact. The original failed controller status and log are
retained as immutable evidence. The recovery does not authorize capacity
screening, large training, promotion, export, or release.
