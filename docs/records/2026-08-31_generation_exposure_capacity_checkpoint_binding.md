# Exposure/capacity resume checkpoint binding (2026-08-31)

## Scope

This is a CPU-only implementation hardening. It does not build or execute a
successor gate, modify the remote checkout, launch training or sampling, or
change the locked 100K quality evidence.

## Trust-boundary gap

The bounded exposure/capacity gate previously recorded that an exposure arm
would resume from the 100K checkpoint, but did not carry the physical identity
of the checkpoint payload, its integrity sidecar, or `latest.json`. A later
consumer could therefore select a different file while retaining the same
logical step.

## Change

The exposure arm now requires a source-checkpoint binding containing the real
run directory, completed step, payload descriptor, integrity-sidecar descriptor,
and latest-pointer descriptor. The build CLI requires all four source paths and
replays the checkpoint/sidecar/latest contract before writing a candidate gate.
The validation CLI rehashes the three files and replays the latest pointer
before returning `pass`. The capacity arm explicitly rejects any resume source
binding, so the two initialization paths cannot be mixed.

The preparation evidence also records the exact source run directory,
checkpoint filename, byte count, and SHA256 for both matched 100K training
reports. Exposure validation requires the resume binding to match the CoFiTok
entry, so a same-step dense checkpoint or another run cannot be substituted
while preserving only the logical step number.

Both preparation validators also re-read the content-addressed training
reports and recompute those checkpoint/layout summaries before accepting the
preparation. Changing an embedded summary while updating the preparation hash
therefore fails closed instead of creating a self-consistent but detached gate.

The serialized gate remains deliberately non-authorizing:
`status=prepared`, `execution_ready=false`, `terminal_status=hold`, and every
training, sampling, evaluation, 300K, promotion, export, release, and
process-signal permission remains `false`.

## Verification

Focused tests cover both arms, missing or mismatched resume identities,
physical checkpoint replacement, permission tampering, GPU/process contention,
and storage failure. No gate artifact was generated and no remote process was
started by this change.
